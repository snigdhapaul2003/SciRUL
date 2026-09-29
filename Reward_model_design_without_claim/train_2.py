"""Two-stage training: claim memorization first, span detection second."""
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
    parser.add_argument("--output-dir", default="runs/two_stage_lora")
    parser.add_argument("--mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--memory-epochs", type=int, default=3)
    parser.add_argument("--detection-epochs", type=int, default=5)
    parser.add_argument("--memory-learning-rate", type=float, default=1e-4)
    parser.add_argument("--detection-learning-rate", type=float, default=1e-4)
    parser.add_argument("--negative-ratio", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    return parser.parse_args(argv)


def materialize_stage(records: list[dict], epochs: int, seed: int,
                      stage: str) -> tuple[list[dict], list[dict]]:
    """Create deterministic epoch order without mixing stage objectives."""
    by_paper: dict[str, list[dict]] = {}
    for record in records:
        by_paper.setdefault(str(record["paper_id"]), []).append(record)
    scheduled, report = [], []
    for epoch in range(epochs):
        rng = random.Random(f"{seed}:{stage}:{epoch + 1}")
        paper_order = sorted(by_paper)
        rng.shuffle(paper_order)
        epoch_rows = []
        for paper_id in paper_order:
            rows = list(by_paper[paper_id])
            if stage == "detection":
                positives = [r for r in rows if r["task"] == "detection"]
                negatives = [r for r in rows if r["task"] == "negative_detection"]
                rng.shuffle(positives); rng.shuffle(negatives)
                rows = positives + negatives
            else:
                rng.shuffle(rows)
            epoch_rows.extend(rows)
        scheduled.extend(epoch_rows)
        report.append({"epoch": epoch + 1, "paper_order": paper_order,
                       "examples": len(epoch_rows)})
    return scheduled, report


def compatible_training_arguments(TrainingArguments, options: dict):
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
    if args.memory_epochs < 1 or args.detection_epochs < 1:
        raise ValueError("both stage epoch counts must be at least one")
    set_seed(args.seed)
    splits = build_splits(args.data, args.seed, args.negative_ratio)
    memory_base = [row for row in splits["train"] if row["task"] == "memory"]
    detection_base = [row for row in splits["train"] if row["task"] != "memory"]
    memory_rows, memory_report = materialize_stage(
        memory_base, args.memory_epochs, args.seed, "memory")
    detection_rows, detection_report = materialize_stage(
        detection_base, args.detection_epochs, args.seed, "detection")
    assert all(row["task"] == "memory" for row in memory_rows)
    assert all(row["task"] != "memory" for row in detection_rows)

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

    def make_dataset(rows, description):
        return Dataset.from_list(rows).map(
            encode, remove_columns=["task", "messages", "paper_id"],
            desc=description)

    memory_dataset = make_dataset(memory_rows, "Tokenizing stage 1 memory")
    detection_dataset = make_dataset(detection_rows, "Tokenizing stage 2 detection")
    validation_dataset = make_dataset(
        splits["validation"], "Tokenizing detection validation")
    collator = DataCollatorForSeq2Seq(
        tokenizer, padding=True, label_pad_token_id=-100)
    root = Path(args.output_dir)
    stage_1_dir = root / "stage_1_memory"
    stage_2_dir = root / "stage_2_detection"

    def step_count(dataset):
        effective = args.batch_size * args.gradient_accumulation
        return max(1, math.ceil(len(dataset) / effective))

    # Stage 1: memory only. A fresh optimizer is used and no detection
    # validation is mixed into checkpoint selection.
    memory_steps = step_count(memory_dataset)
    memory_options = dict(
        output_dir=str(stage_1_dir), num_train_epochs=1,
        learning_rate=args.memory_learning_rate, lr_scheduler_type="cosine",
        warmup_steps=max(1, math.ceil(memory_steps * 0.05)),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        bf16=True, tf32=True, gradient_checkpointing=True,
        save_strategy="no", logging_steps=max(1, memory_steps // 10),
        optim="adamw_torch", report_to="none", seed=args.seed)
    memory_trainer = SequentialTrainer(
        model=model,
        args=compatible_training_arguments(TrainingArguments, memory_options),
        train_dataset=memory_dataset, data_collator=collator)
    print("\n=== Stage 1/2: claim memorization ===", flush=True)
    memory_trainer.train()
    memory_trainer.save_model(stage_1_dir)
    tokenizer.save_pretrained(stage_1_dir)
    memory_trainer.save_state()

    # Stage 2: continue from the in-memory stage-1 weights, but reset optimizer
    # and scheduler and train exclusively on detection supervision.
    detection_steps = step_count(detection_dataset)
    eval_interval = max(1, math.ceil(detection_steps / args.detection_epochs))
    detection_options = dict(
        output_dir=str(stage_2_dir), num_train_epochs=1,
        learning_rate=args.detection_learning_rate, lr_scheduler_type="cosine",
        warmup_steps=max(1, math.ceil(detection_steps * 0.05)),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        bf16=True, tf32=True, gradient_checkpointing=True,
        eval_strategy="steps", save_strategy="steps",
        eval_steps=eval_interval, save_steps=eval_interval,
        logging_steps=max(1, eval_interval // 5), save_total_limit=2,
        load_best_model_at_end=True, metric_for_best_model="eval_loss",
        greater_is_better=False, optim="adamw_torch", report_to="none",
        seed=args.seed)
    detection_trainer = SequentialTrainer(
        model=model,
        args=compatible_training_arguments(TrainingArguments, detection_options),
        train_dataset=detection_dataset, eval_dataset=validation_dataset,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)])
    print("\n=== Stage 2/2: closed-book span detection ===", flush=True)
    detection_trainer.train()
    detection_trainer.save_model(stage_2_dir)
    tokenizer.save_pretrained(stage_2_dir)
    detection_trainer.save_state()

    statistics = {
        "method": "strict_sequential_two_stage_training",
        "stage_1": {"objective": "claim_memorization",
                    "unique_examples": len(memory_base),
                    "scheduled_examples": len(memory_rows),
                    "epochs": memory_report,
                    "checkpoint": str(stage_1_dir)},
        "stage_2": {"objective": "paragraph_span_detection",
                    "contains_memory_examples": False,
                    "unique_examples": len(detection_base),
                    "scheduled_examples": len(detection_rows),
                    "epochs": detection_report,
                    "checkpoint": str(stage_2_dir)},
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "training_design.json").write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
