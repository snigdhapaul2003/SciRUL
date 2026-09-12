"""Answer every ``final_prompt_1..3`` with three configured models.

The input document is preserved and each item in ``results`` receives a
``model_outputs`` list with this shape::

    {"model_name": "olmo_3_7B_Instruct", "answer_1": "...", ...}

Generation is checkpointed after every answer, so an interrupted run can be
resumed by running the same command again.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from model_run import (
    API_MODELS,
    DEFAULT_MODELS,
    HF_MODEL_IDS,
    ask_api_model,
    ask_huggingface_model,
    clear_cached_memory,
    load_huggingface_model,
    validate_api_environment,
    write_json,
)


def run(
    input_path: Path,
    output_path: Path,
    models: tuple[str, ...],
    max_retries: int,
    temperature: float,
    max_new_tokens: int,
) -> None:
    """Run each selected model over every valid final-prompt variant."""
    unknown_models = set(models) - set(HF_MODEL_IDS) - API_MODELS
    if unknown_models:
        raise ValueError(f"Unknown model(s): {', '.join(sorted(unknown_models))}")
    if any(model_name in API_MODELS for model_name in models):
        validate_api_environment()

    # Prefer an existing output as the checkpoint. This makes rerunning the
    # same command resume completed answers rather than starting over.
    source_path = output_path if output_path.exists() else input_path
    document = json.loads(source_path.read_text(encoding="utf-8-sig"))
    results = document.get("results")
    if not isinstance(results, list):
        raise ValueError("Input JSON must contain a 'results' list")

    # Write the original document immediately so the output path exists even
    # when there are no valid prompts to process.
    write_json(output_path, document)

    for model_name in models:
        clear_cached_memory()
        tokenizer = model = None
        try:
            if model_name in HF_MODEL_IDS:
                print(f"Loading {HF_MODEL_IDS[model_name]} from Hugging Face...")
                tokenizer, model = load_huggingface_model(model_name)

            for result_number, result in enumerate(results, start=1):
                final_prompts = [result.get(f"final_prompt_{i}") for i in range(1, 4)]
                # Accept old generated files as a one-prompt input, while new
                # files are expected to contain all three numbered variants.
                if not any(isinstance(prompt, str) and prompt.strip() for prompt in final_prompts):
                    legacy_prompt = result.get("final_prompt")
                    if isinstance(legacy_prompt, str) and legacy_prompt.strip():
                        final_prompts[0] = legacy_prompt
                if not all(isinstance(prompt, str) and prompt.strip() for prompt in final_prompts):
                    print(
                        f"Skipping result {result_number}/{len(results)}: "
                        "one or more final_prompt_1..3 values are missing"
                    )
                    continue

                existing = result.get("model_outputs", [])
                if not isinstance(existing, list):
                    existing = []
                outputs: dict[str, dict[str, Any]] = {
                    item["model_name"]: item
                    for item in existing
                    if isinstance(item, dict)
                    and isinstance(item.get("model_name"), str)
                }
                model_output = outputs.setdefault(
                    model_name, {"model_name": model_name}
                )
                for prompt_number, final_prompt in enumerate(final_prompts, start=1):
                    answer_key = f"answer_{prompt_number}"
                    if model_output.get(answer_key):
                        continue

                    print(
                        f"Result {result_number}/{len(results)} | "
                        f"{model_name} | final_prompt_{prompt_number}"
                    )
                    if model_name in API_MODELS:
                        answer = ask_api_model(
                            model_name,
                            final_prompt,
                            max_retries=max_retries,
                            temperature=temperature,
                        )
                    else:
                        answer = ask_huggingface_model(
                            model_name,
                            tokenizer,
                            model,
                            final_prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                        )

                    model_output[answer_key] = answer
                    result["model_outputs"] = list(outputs.values())
                    write_json(output_path, document)
        finally:
            if model is not None:
                clear_cached_memory(model)
            del model
            del tokenizer

    write_json(output_path, document)
    print(f"Saved answers to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer final_prompt_1..3 with the three configured models."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("parallel_QA_prompts.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("parallel_QA_model_answers.json")
    )
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
