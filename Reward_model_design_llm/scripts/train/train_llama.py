"""Training-only entry point. The inference path never imports this module."""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import math
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from retracted_claims.data import (
    load_dataset,
    sentences_with_offsets,
    split_answers,
)
from retracted_claims.llama import PROMPT


def is_positive(record: dict) -> bool:
    """Return whether the assistant response contains an evidence span."""
    assistant_response = json.loads(record["messages"][1]["content"])
    return bool(assistant_response["steps"])


def records(examples, negative_ratio=None, seed=13):
    """Convert paragraph annotations into claim-sentence training records."""
    output = []
    for example in examples:
        for _start, _end, sentence in sentences_with_offsets(example.paragraph):
            for claim in example.claims:
                matching_spans = [
                    span.text
                    for span in example.spans
                    if claim.claim_number in span.claim_numbers
                    and span.text in sentence
                ]
                steps = [{"span": span} for span in matching_spans]
                user_instruction = PROMPT.format(
                    claim=claim.text,
                    sentence=sentence,
                )
                assistant_response = json.dumps({"steps": steps})
                output.append(
                    {
                        "messages": [
                            {"role": "user", "content": user_instruction},
                            {"role": "assistant", "content": assistant_response},
                        ]
                    }
                )

    if negative_ratio is None:
        return output

    positives = [record for record in output if is_positive(record)]
    negatives = [record for record in output if not is_positive(record)]
    if not positives:
        raise ValueError("training split contains no positive targets")

    random_generator = random.Random(seed)
    random_generator.shuffle(negatives)
    maximum_negatives = round(len(positives) * negative_ratio)
    balanced_records = positives + negatives[:maximum_negatives]
    random_generator.shuffle(balanced_records)
    return balanced_records


def count_classes(rows: list[dict]) -> dict:
    positive_count = sum(is_positive(row) for row in rows)
    negative_count = len(rows) - positive_count
    return {
        "total": len(rows),
        "positive": positive_count,
        "negative": negative_count,
        "negative_to_positive": negative_count / max(1, positive_count),
    }


def write_training_statistics(
    output_dir: Path,
    trainer,
    training_metrics: dict,
    class_balance: dict,
) -> None:
    """Save a readable summary and the complete Trainer log history."""
    output_dir.mkdir(parents=True, exist_ok=True)
    history = trainer.state.log_history

    report = {
        "class_balance": class_balance,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_metric": trainer.state.best_metric,
        "global_steps": trainer.state.global_step,
        "completed_epochs": trainer.state.epoch,
        "training_metrics": training_metrics,
        "history": history,
    }
    (output_dir / "training_statistics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )

    columns = sorted({key for row in history for key in row})
    with (output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(history)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("checkpoint")
    parser.add_argument("output_dir")
    parser.add_argument("--mode", choices=["lora", "full"], required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--negative-ratio", type=float, default=5.0)
    return parser.parse_args()


def main():
    args = parse_arguments()
    if args.epochs not in (1, 2):
        raise ValueError("the specification permits only 1-2 epochs")
    if args.lr > 1e-5:
        raise ValueError("learning rate must be <= 1e-5")

    # Heavy training dependencies remain isolated from the inference path.
    from datasets import Dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype="auto",
    )

    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model

        configuration = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, configuration)

    answer_splits = split_answers(load_dataset(args.input), args.seed)
    train_examples = answer_splits["train"]
    validation_examples = answer_splits["calibration"]

    def render_messages(messages: list[dict], include_assistant: bool) -> str:
        """Render a native instruction-tuning conversation."""
        selected_messages = messages if include_assistant else messages[:1]
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(
                selected_messages,
                tokenize=False,
                add_generation_prompt=not include_assistant,
            )
        if include_assistant:
            return "\n".join(message["content"] for message in selected_messages)
        return selected_messages[0]["content"]

    def encode(row: dict) -> dict:
        # This is conversational supervised fine-tuning for an instruction model.
        # User/chat-control tokens are masked; only the assistant response is learned.
        user_prefix = render_messages(row["messages"], include_assistant=False)
        full_conversation = render_messages(
            row["messages"], include_assistant=True
        )
        prompt_tokens = tokenizer(user_prefix, add_special_tokens=False)
        full_tokens = tokenizer(
            full_conversation,
            add_special_tokens=False,
            truncation=True,
            max_length=2048,
        )
        prompt_length = len(prompt_tokens["input_ids"])
        if prompt_length >= len(full_tokens["input_ids"]):
            raise ValueError(
                "assistant response was removed by truncation; increase max_length"
            )
        if full_tokens["input_ids"][:prompt_length] != prompt_tokens["input_ids"]:
            raise ValueError(
                "chat template user prefix does not align with full conversation"
            )
        full_tokens["labels"] = (
            [-100] * prompt_length + full_tokens["input_ids"][prompt_length:]
        )
        return full_tokens

    train_records = records(train_examples, args.negative_ratio, args.seed)
    validation_records = records(validation_examples)
    class_balance = {
        "train": count_classes(train_records),
        "validation_natural_distribution": count_classes(validation_records),
    }
    print("Class balance:", json.dumps(class_balance, indent=2), flush=True)

    train_dataset = Dataset.from_list(train_records).map(
        encode, remove_columns=["messages"]
    )
    validation_dataset = Dataset.from_list(validation_records).map(
        encode, remove_columns=["messages"]
    )

    batch_size = 1
    updates_per_epoch = math.ceil(len(train_dataset) / batch_size)
    checkpoint_interval = max(1, updates_per_epoch // 4)
    logging_interval = max(1, updates_per_epoch // 20)
    training_options = {
        "output_dir": args.output_dir,
        "num_train_epochs": args.epochs,
        "learning_rate": args.lr,
        "lr_scheduler_type": "cosine",
        "warmup_steps": max(1, math.ceil(updates_per_epoch * args.epochs * 0.05)),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "eval_strategy": "steps",
        "save_strategy": "steps",
        "eval_steps": checkpoint_interval,
        "save_steps": checkpoint_interval,
        "logging_strategy": "steps",
        "logging_steps": logging_interval,
        "logging_first_step": True,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "report_to": "none",
    }

    # Transformers <=4.40 used a longer name for eval_strategy.
    supported_options = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in supported_options:
        if "evaluation_strategy" in supported_options:
            training_options["evaluation_strategy"] = training_options.pop(
                "eval_strategy"
            )
    training_options = {
        name: value
        for name, value in training_options.items()
        if name in supported_options
    }

    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_options),
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100
        ),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    training_result = trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()
    write_training_statistics(
        Path(args.output_dir),
        trainer,
        training_result.metrics,
        class_balance,
    )
    print(
        json.dumps(
            {
                "training_complete": True,
                "best_checkpoint": trainer.state.best_model_checkpoint,
                "best_validation_metric": trainer.state.best_metric,
                "global_steps": trainer.state.global_step,
                "completed_epochs": trainer.state.epoch,
                "statistics_file": str(
                    Path(args.output_dir) / "training_statistics.json"
                ),
                "history_file": str(
                    Path(args.output_dir) / "training_history.csv"
                ),
            },
            indent=2,
        ),
        flush=True,
    )

    manifest = {
        "train_papers": sorted({row.paper_id for row in train_examples}),
        "validation_papers": sorted({row.paper_id for row in validation_examples}),
        "validation_source": "reserved conformal answer from every paper",
        "class_balance": class_balance,
        "monitor": "loss, validation loss, learning rate, and gradient norm",
    }
    manifest_path = Path(args.output_dir) / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
