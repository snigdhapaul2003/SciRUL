"""Run every prompt through local Hugging Face models and Azure GPT-5.5.

The output preserves the input document and adds a ``model_outputs`` list to
each record.  Every item in that list has this shape::

    {"model_name": "qwen3_8B", "answer_1": "...", ...}

Required environment variables (the same ones used by llm_api_check.py):
AZURE_OPENAI_API_KEY, AZURE_API_BASE, and AZURE_API_VERSION.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

# Must be set before importing torch. This lets CUDA grow allocator segments
# instead of holding unusable reserved fragments between generations.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from accelerate.hooks import remove_hook_from_module
from litellm import completion
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODELS = (
    "olmo_3_7B_Instruct",
    "llama_3_8B_Instruct",
    "gpt-5.5",
)

HF_MODEL_IDS = {
    "olmo_3_7B_Instruct": "allenai/Olmo-3-7B-Instruct",
    "llama_3_8B_Instruct": "meta-llama/Meta-Llama-3.1-8B-Instruct",
}

API_MODELS = {"gpt-5.5"}

# Azure configuration verified by llm_api_check.py. The requested output label
# remains "gpt-5.5", while this is the actual available Azure deployment.
AZURE_GPT_DEPLOYMENT = "azure/gpt-5.4"
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_API_BASE = os.getenv("AZURE_API_BASE")
AZURE_API_VERSION = os.getenv("AZURE_API_VERSION")

# Fallback context length used only if a model config doesn't expose one.
DEFAULT_CONTEXT_LENGTH = 8192
# Always keep this many tokens free for the answer, even on a long prompt.
MIN_NEW_TOKENS = 128


def validate_api_environment() -> None:
    missing_environment = [
        name
        for name, value in (
            ("AZURE_OPENAI_API_KEY", AZURE_OPENAI_API_KEY),
            ("AZURE_API_BASE", AZURE_API_BASE),
            ("AZURE_API_VERSION", AZURE_API_VERSION),
        )
        if not value
    ]
    if missing_environment:
        raise RuntimeError(f"Missing required environment variables: {', '.join(missing_environment)}")


def ask_api_model(
    model_name: str,
    prompt: str,
    *,
    max_retries: int,
    temperature: float,
) -> str:
    """Submit one prompt, retrying transient API failures."""
    for attempt in range(max_retries + 1):
        try:
            # Keep this call in the same minimal form as llm_api_check.py.
            response = completion(
                model=AZURE_GPT_DEPLOYMENT,
                messages=[{"content": prompt, "role": "user"}],
            )
            content = response.choices[0].message.content
            return content or ""
        except Exception:
            if attempt == max_retries:
                raise
            time.sleep(min(2**attempt, 30))
    raise AssertionError("unreachable")


def load_huggingface_model(model_name: str):
    """Load one local Hugging Face model and its tokenizer."""
    model_id = HF_MODEL_IDS[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        # Some Llama tokenizers ship without a pad token; fall back to eos
        # so padding doesn't crash. The real stop condition is handled
        # separately by get_terminator_ids, so this has no effect on when
        # generation ends.
        tokenizer.pad_token = tokenizer.eos_token

    load_options: dict[str, Any] = {
        "device_map": "auto",
        # Pinned explicitly rather than "auto" -- "auto" can silently fall
        # back to fp32 for some architectures, roughly doubling VRAM use.
        "torch_dtype": torch.bfloat16,
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
    }
    if model_name == "llama_3_8B_Instruct":
        # Route Llama attention through PyTorch SDPA. During generation we
        # additionally force a fused kernel so it cannot fall back to the
        # quadratic-memory math implementation on long prompts.
        load_options["attn_implementation"] = "sdpa"
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_options)
    model.eval()
    # The model's default generation_config often ships with a max_length
    # that conflicts with the max_new_tokens we pass at generate() time.
    # Clearing it here removes the "both set" warning and ambiguity.
    model.generation_config.max_length = None
    return tokenizer, model


def get_context_length(model) -> int:
    """Best-effort lookup of the model's max position/context length."""
    config = model.config
    for attribute in (
        "max_position_embeddings",
        "n_positions",
        "seq_length",
        "sliding_window",
    ):
        value = getattr(config, attribute, None)
        if isinstance(value, int) and value > 0:
            return value
    return DEFAULT_CONTEXT_LENGTH


def get_terminator_ids(model_name: str, tokenizer) -> list[int]:
    """Return every token ID that should end generation for this model.

    Llama-3 Instruct ends a turn with <|eot_id|>, but tokenizer.eos_token_id
    typically resolves to <|end_of_text|> instead. If only the latter is
    passed to generate(), the model runs past the real answer and starts
    hallucinating a new turn (garbled text, fake follow-up dialogue). Include
    both IDs so generation stops at whichever is produced first.
    """
    terminators = [tokenizer.eos_token_id]
    if model_name == "llama_3_8B_Instruct":
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id is not None and eot_id != tokenizer.unk_token_id:
            terminators.append(eot_id)
    seen = set()
    unique_terminators = []
    for token_id in terminators:
        if token_id is not None and token_id not in seen:
            seen.add(token_id)
            unique_terminators.append(token_id)
    return unique_terminators


def ask_huggingface_model(
    model_name: str,
    tokenizer,
    model,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float,
) -> str:
    """Generate an answer with a model loaded from Hugging Face."""
    messages = [{"role": "user", "content": prompt}]
    template_options: dict[str, Any] = {
        "add_generation_prompt": True,
        "return_tensors": "pt",
        "return_dict": True,
    }
    # Qwen3 otherwise emits a potentially long hidden-thinking section.
    if model_name == "qwen3_8B":
        template_options["enable_thinking"] = False

    inputs = tokenizer.apply_chat_template(messages, **template_options)

    # Cap max_new_tokens so prompt_length + max_new_tokens never exceeds the
    # model's context window. Without this, a long prompt plus a fixed
    # max_new_tokens can silently sail past max_position_embeddings, which
    # is exactly the "exceeded the model's predefined maximum length"
    # warning -- and beyond that point position ids wrap/misbehave, which is
    # what produces garbled or nonsensical trailing text.
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    context_length = get_context_length(model)
    prompt_length = inputs["input_ids"].shape[1]
    available_for_generation = context_length - prompt_length
    if available_for_generation < MIN_NEW_TOKENS:
        print(
            f"Warning: prompt for {model_name} uses {prompt_length} tokens, "
            f"leaving only {available_for_generation} of {context_length} "
            "for the answer. Truncating may be needed upstream."
        )
    effective_max_new_tokens = max(1, min(max_new_tokens, available_for_generation))

    terminators = get_terminator_ids(model_name, tokenizer)
    generation_options: dict[str, Any] = {
        "max_new_tokens": effective_max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": terminators,
    }
    if temperature > 0:
        generation_options["temperature"] = temperature
    attention_context = (
        # This Windows PyTorch build has no FlashAttention kernel. Its fused
        # cuDNN SDPA kernel is available on the A100 and provides the same key
        # property needed here: it does not materialize the full attention
        # matrix in GPU memory.
        sdpa_kernel(SDPBackend.CUDNN_ATTENTION)
        if model_name == "llama_3_8B_Instruct"
        else nullcontext()
    )
    if model_name == "llama_3_8B_Instruct":
        torch.backends.cuda.enable_cudnn_sdp(True)
    with torch.inference_mode(), attention_context:
        output = model.generate(**inputs, **generation_options)
    generated = output[0, prompt_length:]
    # clean_up_tokenization_spaces=True is meant for WordPiece tokenizers and
    # strips spaces before punctuation for BPE tokenizers (used by both
    # Llama and OLMo), corrupting the output. Disable it explicitly.
    return tokenizer.decode(
        generated,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    ).strip()


def clear_cached_memory(model=None) -> None:
    """Release Accelerate's dispatch hooks (if any), then free CUDA memory.

    With device_map="auto", Accelerate attaches AlignDevicesHook objects to
    submodules for dispatch/offload bookkeeping. Those hooks hold their own
    references to tensors, so a plain `del model` does not necessarily drop
    the refcount to zero -- and torch.cuda.empty_cache() can only reclaim
    memory that is already unreferenced. Stripping the hooks first ensures
    the model is actually collectible.
    """
    if model is not None:
        try:
            remove_hook_from_module(model, recurse=True)
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        for device_index in range(torch.cuda.device_count()):
            with torch.cuda.device(device_index):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()


def write_json(path: Path, data: dict[str, Any]) -> None:
    """Write via a temporary file so interruptions cannot corrupt the output."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def run(
    input_path: Path,
    output_path: Path,
    models: tuple[str, ...],
    max_retries: int,
    temperature: float,
    max_new_tokens: int,
) -> None:
    unknown_models = set(models) - set(HF_MODEL_IDS) - API_MODELS
    if unknown_models:
        raise ValueError(f"Unknown model(s): {', '.join(sorted(unknown_models))}")
    if any(model_name in API_MODELS for model_name in models):
        validate_api_environment()
    document = json.loads(input_path.read_text(encoding="utf-8-sig"))
    records = document.get("records")
    if not isinstance(records, list):
        raise ValueError("Input JSON must contain a 'records' list")

    for model_name in models:
        # The previous local model (including OLMo) has been fully released
        # before the next model, especially before Llama, is loaded.
        clear_cached_memory()
        tokenizer = model = None
        if model_name in HF_MODEL_IDS:
            print(f"Loading {HF_MODEL_IDS[model_name]} from Hugging Face...")
            tokenizer, model = load_huggingface_model(model_name)

        for record_number, record in enumerate(records, start=1):
            if not all(
                isinstance(record.get(f"prompt_{i}"), str) for i in range(1, 4)
            ):
                print(f"Skipping record {record_number}: prompts are missing")
                continue

            existing = record.get("model_outputs", [])
            outputs = {
                item.get("model_name"): item
                for item in existing
                if isinstance(item, dict) and item.get("model_name")
            }
            result = outputs.setdefault(model_name, {"model_name": model_name})
            for prompt_number in range(1, 4):
                answer_key = f"answer_{prompt_number}"
                if result.get(answer_key):
                    continue
                print(
                    f"Record {record_number}/{len(records)} | "
                    f"{model_name} | prompt_{prompt_number}"
                )
                if model_name in API_MODELS:
                    result[answer_key] = ask_api_model(
                        model_name,
                        record[f"prompt_{prompt_number}"],
                        max_retries=max_retries,
                        temperature=temperature,
                    )
                else:
                    result[answer_key] = ask_huggingface_model(
                        model_name,
                        tokenizer,
                        model,
                        record[f"prompt_{prompt_number}"],
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                record["model_outputs"] = list(outputs.values())
                write_json(output_path, document)

            record["model_outputs"] = list(outputs.values())

        if model is not None:
            # Strip Accelerate hooks and release CUDA memory before the next
            # model in the list is loaded.
            clear_cached_memory(model)
            del model
            del tokenizer
            model = tokenizer = None

    write_json(output_path, document)
    print(f"Saved answers to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate answer_1, answer_2, and answer_3 for each model."
    )
    parser.add_argument("--input", type=Path, default=Path("prompts.json"))
    parser.add_argument("--output", type=Path, default=Path("model_answers.json"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.input,
        arguments.output,
        tuple(arguments.models),
        arguments.max_retries,
        arguments.temperature,
        arguments.max_new_tokens,
    )
