"""Paragraph-only inference. No claim strings or embeddings are loaded."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from retracted_reward.data import load_judged_data, sentence_windows, split_by_answer
from retracted_reward.prompts import DETECTION_PROMPT, SYSTEM_PROMPT


def load_paragraphs(path, split="test", seed=13):
    root = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    # Native judged dataset: reproduce the training split and expose only answer
    # text to the inference loop. Claims and judgments never reach the model.
    if isinstance(root, dict) and isinstance(root.get("samples"), list):
        _claims, answers = load_judged_data(path)
        split_answers = split_by_answer(answers, seed)
        selected = answers if split == "all" else split_answers[split]
        return [{"paragraph_id": (
                    f"{answer.paper_id}:{answer.sample_type}:{answer.answer_id}"),
                 "paragraph": answer.text} for answer in selected]
    rows = root.get("paragraphs") if isinstance(root, dict) else None
    if not isinstance(rows, list):
        raise ValueError("input must be {\"paragraphs\": [...]}")
    output = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or "paragraph" not in row:
            raise ValueError("each row needs paragraph")
        forbidden = {"claims", "anchor_claims", "claim_embeddings"} & set(row)
        if forbidden:
            raise ValueError(f"claim-side input is forbidden: {sorted(forbidden)}")
        output.append({"paragraph_id": str(row.get("paragraph_id", i)),
                       "paragraph": str(row["paragraph"])})
    return output


def parse_prediction(generated: str) -> list[str]:
    try:
        left, right = generated.find("{"), generated.rfind("}")
        obj = json.loads(generated[left:right + 1])
        return [x for x in obj.get("spans", []) if isinstance(x, str)]
    except (ValueError, json.JSONDecodeError, AttributeError):
        return []


def locate_quotes(text: str, quotes: list[str], base: int = 0) -> list[dict]:
    found = []
    for quote in quotes:
        cursor = 0
        while quote and (position := text.find(quote, cursor)) >= 0:
            found.append({"start": base + position, "end": base + position + len(quote),
                          "text": quote})
            cursor = position + len(quote)
    return found


def arguments(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="claims_and_answers_judged_large_neg.json",
                   help="Judged positive/negative dataset or paragraph JSON")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct",
                   help="Base model for LoRA, or the saved model for full tuning")
    p.add_argument("--adapter", help="LoRA directory; omit for a full checkpoint")
    p.add_argument("--output", required=True)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--split", choices=("train", "validation", "test", "all"),
                   default="test",
                   help="Subset used when input is the judged dataset")
    p.add_argument("--seed", type=int, default=13,
                   help="Must match the training split seed")
    return p.parse_args(argv)


def main(argv=None):
    args = arguments(argv)
    # Copy/pasted multiline PowerShell commands can accidentally preserve a
    # newline inside a quoted path/model argument.
    args.input = args.input.strip()
    args.model = args.model.strip()
    args.output = args.output.strip()
    args.adapter = args.adapter.strip() if args.adapter else None

    model_path = Path(args.model)
    if (model_path.is_dir() and
            (model_path / "adapter_config.json").exists() and
            not (model_path / "config.json").exists()):
        raise ValueError(
            f"{args.model!r} is a LoRA adapter directory, not a complete model. "
            "For raw inference use --model meta-llama/Llama-3.1-8B-Instruct "
            "and omit --adapter. For LoRA inference, supply this directory via "
            "--adapter and keep the base checkpoint in --model.")
    import torch
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rows = load_paragraphs(args.input, args.split, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.clean_up_tokenization_spaces = False
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="auto", dtype=torch.bfloat16)
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"Inference model: {args.model} with LoRA adapter {args.adapter}")
    else:
        print(f"Inference model: raw checkpoint {args.model} (no LoRA adapter)")
    model.eval()
    predictions = []
    progress = tqdm(total=len(rows), desc="Detecting", unit="paragraph")
    with torch.inference_mode():
        for row in rows:
            paragraph = row["paragraph"]
            messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": DETECTION_PROMPT.format(text=paragraph)}]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False,
                add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt",
                               add_special_tokens=False).to(model.device)
            prompt_length = inputs["input_ids"].shape[1]
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.eos_token_id)
            generated = tokenizer.decode(output[0, prompt_length:],
                skip_special_tokens=True, clean_up_tokenization_spaces=False)
            spans = locate_quotes(paragraph, parse_prediction(generated))
            progress.update(1)
            unique = {(s["start"], s["end"], s["text"]): s for s in spans}
            spans = sorted(unique.values(), key=lambda s: (s["start"], s["end"]))
            predictions.append({"paragraph_id": row["paragraph_id"],
                "uses_retracted_claim": bool(spans), "spans": spans})
    progress.close()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"predictions": predictions},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
