"""Detection-only BF16 LoRA/full fine-tuning (no claim-memory objective)."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import random
from pathlib import Path

from retracted_reward.data import build_splits


def arguments(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="claims_and_answers_judged.json")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", default="runs/detection_only_lora")
    parser.add_argument("--mode", choices=("lora", "full"), default="lora")
    # Match train_2.py's stage-2 defaults so the only experimental difference
    # is whether claim-memory training occurred before detection training.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--negative-ratio", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args(argv)


def materialize_stage(records: list[dict], epochs: int, seed: int,
                      stage: str = "detection") -> tuple[list[dict], list[dict]]:
    """Match train_2.py's deterministic stage-2 detection ordering."""
    if epochs < 1:
        raise ValueError("epochs must be at least one")
    by_paper: dict[str, dict[str, list[dict]]] = {}
    for record in records:
        if record["task"] == "memory":
            continue
        tasks = by_paper.setdefault(str(record["paper_id"]), {
            "detection": [], "negative_detection": []})
        tasks[record["task"]].append(record)
    scheduled, report = [], []
    for epoch in range(epochs):
        rng = random.Random(f"{seed}:{stage}:{epoch + 1}")
        paper_order = sorted(by_paper)
        rng.shuffle(paper_order)
        epoch_rows = []
        for paper_id in paper_order:
            positives = list(by_paper[paper_id]["detection"])
            negatives = list(by_paper[paper_id]["negative_detection"])
            rng.shuffle(positives)
            rng.shuffle(negatives)
            epoch_rows.extend(positives)
            epoch_rows.extend(negatives)
        scheduled.extend(epoch_rows)
        report.append({"epoch": epoch + 1, "paper_order": paper_order,
                       "examples": len(epoch_rows)})
    return scheduled, report


def compatible_training_arguments(TrainingArguments, options: dict):
    """Match the transformers-version compatibility path in train_2.py."""
    supported = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in supported and "evaluation_strategy" in supported:
        options["evaluation_strategy"] = options.pop("eval_strategy")
    unsupported = sorted(set(options) - set(supported))
    if unsupported:
        raise RuntimeError(f"Unsupported TrainingArguments: {unsupported}")
    return TrainingArguments(**options)


def main(argv=None):
    args = arguments(argv)
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import SequentialSampler
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
        DataCollatorForSeq2Seq, EarlyStoppingCallback, Trainer,
        TrainingArguments, set_seed)

    class SequentialTrainer(Trainer):
        def _get_train_sampler(self, *unused_args, **unused_kwargs):
            return SequentialSampler(self.train_dataset)

    if not torch.cuda.is_available():
        raise RuntimeError("Llama-3.1-8B training requires a CUDA GPU")
    if args.epochs < 1:
        raise ValueError("epochs must be at least one")
    set_seed(args.seed)
    splits = build_splits(args.data, args.seed, args.negative_ratio)
    detection_base = [row for row in splits["train"] if row["task"] != "memory"]
    train_rows, epoch_report = materialize_stage(
        detection_base, args.epochs, args.seed, "detection")
    if any(row["task"] == "memory" for row in train_rows):
        raise AssertionError("detection-only schedule contains a memory example")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if args.mode == "lora":
        model = get_peft_model(model, LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
            bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"]))
        model.print_trainable_parameters()

    def encode(row):
        prompt = tokenizer.apply_chat_template(
            row["messages"][:-1], tokenize=False, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False)
        prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        encoded = tokenizer(full, add_special_tokens=False, truncation=True,
                            max_length=args.max_length)
        if encoded["input_ids"][:len(prompt_ids)] != prompt_ids:
            raise ValueError("chat-template prefix mismatch")
        if len(prompt_ids) >= len(encoded["input_ids"]):
            raise ValueError("max-length truncated the assistant target")
        encoded["labels"] = ([-100] * len(prompt_ids) +
                             encoded["input_ids"][len(prompt_ids):])
        return encoded

    train_dataset = Dataset.from_list(train_rows).map(
        encode, remove_columns=["task", "messages", "paper_id"],
        desc="Tokenizing detection-only train")
    validation_dataset = Dataset.from_list(splits["validation"]).map(
        encode, remove_columns=["task", "messages", "paper_id"],
        desc="Tokenizing validation")

    effective_batch = args.batch_size * args.gradient_accumulation
    total_steps = max(1, math.ceil(len(train_dataset) / effective_batch))
    # Evaluate roughly once per materialized epoch and at the final step.
    eval_steps = max(1, math.ceil(total_steps / args.epochs))
    options = dict(
        output_dir=args.output_dir, num_train_epochs=1,
        learning_rate=args.learning_rate, lr_scheduler_type="cosine",
        warmup_steps=max(1, math.ceil(total_steps * 0.05)),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        bf16=True, tf32=True, gradient_checkpointing=True,
        eval_strategy="steps", save_strategy="steps", eval_steps=eval_steps,
        save_steps=eval_steps, logging_steps=max(1, eval_steps // 5),
        save_total_limit=2, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        optim="adamw_torch", report_to="none", seed=args.seed)
    trainer = SequentialTrainer(
        model=model,
        args=compatible_training_arguments(TrainingArguments, options),
        train_dataset=train_dataset, eval_dataset=validation_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    statistics = {
        "objective": "detection_only",
        "contains_claim_memory_examples": False,
        "base_examples": {
            "positive_paragraphs": sum(r["task"] == "detection"
                                       for r in splits["train"]),
            "hard_negative_sentences": sum(r["task"] == "negative_detection"
                                           for r in splits["train"]),
        },
        "unique_training_examples": len(detection_base),
        "scheduled_training_examples": len(train_rows),
        "epochs": epoch_report,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    main()
