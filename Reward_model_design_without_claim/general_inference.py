"""Evaluate a Transformers checkpoint (including a PEFT run) on MMLU."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


LETTERS = ("A", "B", "C", "D")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run an ordinary or LoRA runs checkpoint on MMLU.")
    parser.add_argument(
        "checkpoint",
        help="Model/adapter checkpoint, or a run directory containing checkpoint-*",
    )
    parser.add_argument(
        "--base-model",
        help="Base model for a LoRA checkpoint (normally read from adapter_config.json)",
    )
    parser.add_argument("--output", default="mmlu_results.json")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--subjects", nargs="*", help="Only evaluate these MMLU subjects")
    parser.add_argument("--num-few-shot", type=int, default=5)
    parser.add_argument("--max-examples", type=int, help="Maximum examples per subject")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Number of answer candidates scored per forward pass")
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args(argv)


def resolve_checkpoint(value: str) -> Path:
    """Use a run's exported adapter/model, or its most recent numbered checkpoint."""
    path = Path(value).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    if path.is_file():
        raise ValueError("checkpoint must be a model or adapter directory")
    if ((path / "adapter_config.json").exists() or
            (path / "config.json").exists()):
        return path
    checkpoints = [p for p in path.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        raise ValueError(f"no model files or checkpoint-* directories found in {path}")

    def step(p: Path) -> int:
        try:
            return int(p.name.rsplit("-", 1)[1])
        except ValueError:
            return -1

    return max(checkpoints, key=step)


def format_question(row: dict, include_answer: bool = False) -> str:
    lines = [str(row["question"])]
    lines.extend(f"{letter}. {choice}" for letter, choice in zip(LETTERS, row["choices"]))
    answer = LETTERS[int(row["answer"])] if include_answer else ""
    lines.append(f"Answer: {answer}")
    return "\n".join(lines)


def make_prompt(row: dict, demonstrations: list[dict], tokenizer) -> str:
    subject = str(row["subject"]).replace("_", " ")
    body = f"The following are multiple choice questions (with answers) about {subject}.\n\n"
    if demonstrations:
        body += "\n\n".join(format_question(item, True) for item in demonstrations) + "\n\n"
    body += format_question(row)
    messages = [{"role": "user", "content": body}]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
    return body


def score_answers(model, tokenizer, prompt: str, batch_size: int) -> list[float]:
    """Return summed conditional log likelihood for each A/B/C/D continuation."""
    import torch

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if not prompt_ids:
        raise ValueError("tokenizer produced an empty prompt")
    candidates = []
    for letter in LETTERS:
        answer_ids = tokenizer(letter, add_special_tokens=False)["input_ids"]
        candidates.append((prompt_ids + answer_ids, len(answer_ids)))

    scores = []
    for start in range(0, len(candidates), batch_size):
        chunk = candidates[start:start + batch_size]
        max_len = max(len(ids) for ids, _ in chunk)
        input_ids, attention_mask = [], []
        for ids, _ in chunk:
            padding = max_len - len(ids)
            input_ids.append(ids + [tokenizer.pad_token_id] * padding)
            attention_mask.append([1] * len(ids) + [0] * padding)
        input_ids = torch.tensor(input_ids, device=model.device)
        attention_mask = torch.tensor(attention_mask, device=model.device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        log_probs = logits.log_softmax(dim=-1)
        for index, (ids, answer_len) in enumerate(chunk):
            first = len(ids) - answer_len
            score = 0.0
            for position in range(first, len(ids)):
                score += log_probs[index, position - 1, ids[position]].item()
            scores.append(score)
    return scores


def main(argv=None):
    args = parse_args(argv)
    if args.num_few_shot < 0 or args.batch_size < 1:
        raise ValueError("num-few-shot must be nonnegative and batch-size must be positive")
    if args.max_examples is not None and args.max_examples < 1:
        raise ValueError("max-examples must be positive")

    import torch
    from datasets import load_dataset
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer

    checkpoint = resolve_checkpoint(args.checkpoint)
    adapter_config_path = checkpoint / "adapter_config.json"
    is_adapter = adapter_config_path.exists()
    base_model = args.base_model
    if is_adapter and not base_model:
        config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        base_model = config.get("base_model_name_or_path")
        if not base_model:
            raise ValueError("adapter_config.json has no base_model_name_or_path; use --base-model")
    model_source = base_model if is_adapter else str(checkpoint)

    tokenizer = AutoTokenizer.from_pretrained(
        str(checkpoint) if is_adapter else model_source,
        trust_remote_code=args.trust_remote_code)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_source, device_map="auto", torch_dtype="auto",
        trust_remote_code=args.trust_remote_code)
    if is_adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(checkpoint))
    model.eval()

    dataset = load_dataset("cais/mmlu", "all")
    demonstrations = defaultdict(list)
    for row in dataset["dev"]:
        demonstrations[row["subject"]].append(row)
    rows = list(dataset[args.split])
    if args.subjects:
        requested = set(args.subjects)
        available = {row["subject"] for row in rows}
        unknown = requested - available
        if unknown:
            raise ValueError(f"unknown subjects: {', '.join(sorted(unknown))}")
        rows = [row for row in rows if row["subject"] in requested]
    if args.max_examples is not None:
        counts = defaultdict(int)
        limited = []
        for row in rows:
            if counts[row["subject"]] < args.max_examples:
                limited.append(row)
                counts[row["subject"]] += 1
        rows = limited

    correct_by_subject = defaultdict(int)
    total_by_subject = defaultdict(int)
    predictions = []
    with torch.inference_mode():
        for row in tqdm(rows, desc="MMLU", unit="question"):
            shots = demonstrations[row["subject"]][:args.num_few_shot]
            prompt = make_prompt(row, shots, tokenizer)
            scores = score_answers(model, tokenizer, prompt, args.batch_size)
            prediction = max(range(4), key=scores.__getitem__)
            gold = int(row["answer"])
            subject = row["subject"]
            correct_by_subject[subject] += int(prediction == gold)
            total_by_subject[subject] += 1
            predictions.append({
                "subject": subject, "question": row["question"],
                "prediction": LETTERS[prediction], "answer": LETTERS[gold],
                "correct": prediction == gold,
                "log_likelihoods": dict(zip(LETTERS, scores)),
            })

    total = sum(total_by_subject.values())
    correct = sum(correct_by_subject.values())
    result = {
        "checkpoint": str(checkpoint), "base_model": base_model,
        "split": args.split, "num_few_shot": args.num_few_shot,
        "accuracy": correct / total if total else 0.0,
        "correct": correct, "total": total,
        "subjects": {
            subject: {"accuracy": correct_by_subject[subject] / count,
                      "correct": correct_by_subject[subject], "total": count}
            for subject, count in sorted(total_by_subject.items())
        },
        "predictions": predictions,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"MMLU accuracy: {correct}/{total} = {result['accuracy']:.4f}")
    print(f"Results written to {output}")


if __name__ == "__main__":
    main()
