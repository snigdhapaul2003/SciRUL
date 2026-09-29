"""Fast, memory-conscious MMLU evaluation for full or PEFT checkpoints."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

LETTERS = "ABCD"


def arguments():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="Checkpoint, LoRA adapter, or run directory")
    p.add_argument("--base-model", help="Optional override for a LoRA base model")
    p.add_argument("--output", default="mmlu_results.json")
    p.add_argument("--subjects", nargs="*")
    p.add_argument("--num-few-shot", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Questions per forward pass; lower this if CUDA runs out of memory")
    p.add_argument("--max-examples", type=int, help="Limit per subject for quick tests")
    p.add_argument("--load-in-4bit", action="store_true",
                   help="Use bitsandbytes NF4 quantization to greatly reduce GPU memory")
    return p.parse_args()


def resolve_checkpoint(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_dir():
        raise ValueError(f"checkpoint directory not found: {path}")
    if (path / "adapter_config.json").exists() or (path / "config.json").exists():
        return path
    found = list(path.glob("checkpoint-*"))
    if not found:
        raise ValueError(f"no checkpoint-* directory found in {path}")
    return max(found, key=lambda p: int(p.name.rsplit("-", 1)[1]))


def question_text(row, answer=False):
    result = [row["question"]]
    result += [f"{letter}. {choice}" for letter, choice in zip(LETTERS, row["choices"])]
    result += [f"Answer: {LETTERS[row['answer']] if answer else ''}"]
    return "\n".join(result)


def prompt_for(row, dev_rows, tokenizer):
    subject = row["subject"].replace("_", " ")
    text = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    text += "\n\n".join(question_text(x, True) for x in dev_rows)
    if dev_rows:
        text += "\n\n"
    text += question_text(row)
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False,
            add_generation_prompt=True)
    return text


def main():
    args = arguments()
    if args.batch_size < 1 or args.num_few_shot < 0:
        raise ValueError("batch-size must be positive and num-few-shot nonnegative")
    import torch
    from datasets import load_dataset
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    checkpoint = resolve_checkpoint(args.checkpoint)
    adapter_file = checkpoint / "adapter_config.json"
    is_adapter = adapter_file.exists()
    adapter_config = json.loads(adapter_file.read_text()) if is_adapter else {}
    base_model = args.base_model or adapter_config.get("base_model_name_or_path")
    if is_adapter and not base_model:
        raise ValueError("LoRA base model is unknown; pass --base-model")
    source = base_model if is_adapter else str(checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint) if is_adapter else source)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"
    label_ids = [tokenizer(x, add_special_tokens=False)["input_ids"] for x in LETTERS]
    if any(len(x) != 1 for x in label_ids):
        raise ValueError("This fast evaluator requires A/B/C/D to each be one token")
    label_ids = torch.tensor([x[0] for x in label_ids])

    load_options = {"device_map": "auto", "torch_dtype": "auto"}
    if args.load_in_4bit:
        load_options["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(source, **load_options)
    if is_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(checkpoint))
    model.eval()

    data = load_dataset("cais/mmlu", "all")
    dev = defaultdict(list)
    for row in data["dev"]:
        dev[row["subject"]].append(row)
    rows = list(data["test"])
    if args.subjects:
        unknown = set(args.subjects) - {x["subject"] for x in rows}
        if unknown:
            raise ValueError(f"unknown subjects: {sorted(unknown)}")
        rows = [x for x in rows if x["subject"] in args.subjects]
    if args.max_examples:
        counts, selected = defaultdict(int), []
        for row in rows:
            if counts[row["subject"]] < args.max_examples:
                selected.append(row); counts[row["subject"]] += 1
        rows = selected

    predictions, stats = [], defaultdict(lambda: [0, 0])
    device = model.get_input_embeddings().weight.device
    with torch.inference_mode():
        for start in tqdm(range(0, len(rows), args.batch_size), desc="MMLU batches"):
            batch = rows[start:start + args.batch_size]
            prompts = [prompt_for(x, dev[x["subject"]][:args.num_few_shot], tokenizer)
                       for x in batch]
            encoded = tokenizer(prompts, padding=True, return_tensors="pt").to(device)
            # One model pass scores all four options for every question. This avoids
            # repeating the long few-shot context four times.
            logits = model(**encoded, use_cache=False).logits[:, -1, :]
            scores = logits.index_select(1, label_ids.to(logits.device)).float().cpu()
            for row, row_scores in zip(batch, scores):
                predicted, gold = int(row_scores.argmax()), int(row["answer"])
                stats[row["subject"]][0] += predicted == gold
                stats[row["subject"]][1] += 1
                predictions.append({"subject": row["subject"], "question": row["question"],
                    "prediction": LETTERS[predicted], "answer": LETTERS[gold],
                    "correct": predicted == gold,
                    "scores": dict(zip(LETTERS, row_scores.tolist()))})
            del encoded, logits, scores

    correct, total = sum(x[0] for x in stats.values()), sum(x[1] for x in stats.values())
    result = {"checkpoint": str(checkpoint), "accuracy": correct / total,
              "correct": correct, "total": total,
              "subjects": {k: {"accuracy": c / n, "correct": c, "total": n}
                           for k, (c, n) in sorted(stats.items())},
              "predictions": predictions}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"MMLU accuracy: {correct}/{total} = {correct / total:.4f}")
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
