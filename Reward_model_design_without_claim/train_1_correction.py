"""Efficient joint training with epoch-dependent weighted task sampling."""
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
    parser.add_argument("--data", default="claims_and_answers_judged_large_neg.json")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output-dir", default="runs/weighted_curriculum_lora")
    parser.add_argument("--mode", choices=("lora", "full"), default="lora")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--negative-ratio", type=float, default=0.5)
    parser.add_argument("--initial-memory-probability", type=float, default=0.8)
    parser.add_argument("--final-memory-probability", type=float, default=0.2)
    parser.add_argument("--samples-per-epoch", type=int, default=None,
                        help="Defaults to the number of unique training examples")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--validation-max-new-tokens", type=int, default=256,
                        help="Maximum tokens generated for each validation response")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args(argv)


class WeightedCurriculumSampler:
    """Fixed-budget sampler with a linear memory-to-detection curriculum."""
    def __init__(self, tasks: list[str], epochs: int, samples_per_epoch: int,
                 initial_memory_probability: float,
                 final_memory_probability: float, seed: int):
        self.memory_indices = [i for i, task in enumerate(tasks) if task == "memory"]
        self.positive_indices = [
            i for i, task in enumerate(tasks) if task == "detection"]
        self.negative_indices = [
            i for i, task in enumerate(tasks) if task == "negative_detection"]
        self.detection_indices = self.positive_indices + self.negative_indices
        if (not self.memory_indices or not self.positive_indices
                or not self.negative_indices):
            raise ValueError(
                "weighted curriculum needs memory, positive, and negative examples")
        self.epochs = epochs
        self.samples_per_epoch = samples_per_epoch
        self.initial = initial_memory_probability
        self.final = final_memory_probability
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def memory_probability(self, epoch: int | None = None) -> float:
        epoch = self.epoch if epoch is None else epoch
        progress = epoch / max(1, self.epochs - 1)
        return self.initial + progress * (self.final - self.initial)

    def __iter__(self):
        # Trainer/Accelerate normally calls set_epoch. Advancing here as well
        # keeps the curriculum correct on older releases that do not propagate
        # set_epoch to a custom sampler.
        current_epoch = min(self.epoch, self.epochs - 1)
        rng = random.Random(f"{self.seed}:weighted-epoch:{current_epoch}")
        probability = self.memory_probability(current_epoch)
        memory_count = round(self.samples_per_epoch * probability)
        detection_count = self.samples_per_epoch - memory_count
        positive_fraction = len(self.positive_indices) / len(self.detection_indices)
        positive_count = round(detection_count * positive_fraction)
        if detection_count >= 2:
            positive_count = min(max(positive_count, 1), detection_count - 1)
        negative_count = detection_count - positive_count
        indices = (
            [rng.choice(self.memory_indices) for _ in range(memory_count)]
            + [rng.choice(self.positive_indices) for _ in range(positive_count)]
            + [rng.choice(self.negative_indices) for _ in range(negative_count)]
        )
        rng.shuffle(indices)
        self.epoch = current_epoch + 1
        return iter(indices)

    def __len__(self):
        return self.samples_per_epoch


def main(argv=None):
    args = arguments(argv)
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
        DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)

    if not torch.cuda.is_available():
        raise RuntimeError("Llama-3.1-8B training requires a CUDA GPU")
    if args.epochs < 1:
        raise ValueError("epochs must be at least one")
    for name, probability in (("initial", args.initial_memory_probability),
                              ("final", args.final_memory_probability)):
        if not 0.0 <= probability <= 1.0:
            raise ValueError(f"{name} memory probability must be between 0 and 1")
    set_seed(args.seed)
    splits = build_splits(args.data, args.seed, args.negative_ratio)
    train_rows = splits["train"]
    samples_per_epoch = args.samples_per_epoch or len(train_rows)
    if samples_per_epoch < 1:
        raise ValueError("samples-per-epoch must be positive")
    sampler = WeightedCurriculumSampler(
        [row["task"] for row in train_rows], args.epochs, samples_per_epoch,
        args.initial_memory_probability, args.final_memory_probability, args.seed)

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
        desc="Tokenizing weighted-curriculum train")
    validation_dataset = Dataset.from_list(splits["validation"]).map(
        encode, remove_columns=["task", "messages", "paper_id"],
        desc="Tokenizing validation")

    class WeightedTrainer(Trainer):
        """Use weighted sampling and validate with actual generation."""

        def __init__(self, *trainer_args, generation_rows, generation_tokenizer,
                     generation_max_new_tokens, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.generation_rows = generation_rows
            self.generation_tokenizer = generation_tokenizer
            self.generation_max_new_tokens = generation_max_new_tokens

        def _get_train_sampler(self, *unused_args, **unused_kwargs):
            return sampler

        @staticmethod
        def _parse_generated_json(text):
            left, right = text.find("{"), text.rfind("}")
            if left < 0 or right < left:
                return None
            try:
                value = json.loads(text[left:right + 1])
            except (json.JSONDecodeError, TypeError):
                return None
            return value if isinstance(value, dict) else None

        @staticmethod
        def _character_positions(paragraph, quotes):
            positions = set()
            for quote in quotes:
                if not isinstance(quote, str) or not quote:
                    continue
                cursor = 0
                while (start := paragraph.find(quote, cursor)) >= 0:
                    positions.update(range(start, start + len(quote)))
                    cursor = start + len(quote)
            return positions

        def _generation_metrics(self, metric_key_prefix):
            model = self.model
            tokenizer = self.generation_tokenizer
            was_training = model.training
            model.eval()
            binary_tp = binary_fp = binary_fn = 0
            character_tp = character_fp = character_fn = 0
            valid_json = 0
            try:
                with torch.inference_mode():
                    for row in self.generation_rows:
                        prompt = tokenizer.apply_chat_template(
                            row["messages"][:-1], tokenize=False,
                            add_generation_prompt=True)
                        inputs = tokenizer(
                            prompt, return_tensors="pt", add_special_tokens=False,
                            truncation=True, max_length=args.max_length).to(model.device)
                        prompt_length = inputs["input_ids"].shape[1]
                        output = model.generate(
                            **inputs,
                            max_new_tokens=self.generation_max_new_tokens,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
                        generated = tokenizer.decode(
                            output[0, prompt_length:], skip_special_tokens=True,
                            clean_up_tokenization_spaces=False)
                        prediction = self._parse_generated_json(generated)
                        if prediction is not None:
                            valid_json += 1
                        predicted_quotes = (prediction or {}).get("spans", [])
                        if not isinstance(predicted_quotes, list):
                            predicted_quotes = []
                        target = json.loads(row["messages"][-1]["content"])
                        gold_quotes = target.get("spans", [])
                        paragraph = row["messages"][-2]["content"].split(
                            "\nTEXT:\n", 1)[-1]
                        predicted = self._character_positions(
                            paragraph, predicted_quotes)
                        gold = self._character_positions(paragraph, gold_quotes)
                        if gold and predicted:
                            binary_tp += 1
                        elif predicted:
                            binary_fp += 1
                        elif gold:
                            binary_fn += 1
                        intersection = len(gold & predicted)
                        character_tp += intersection
                        character_fp += len(predicted - gold)
                        character_fn += len(gold - predicted)
            finally:
                if was_training:
                    model.train()

            def f1(tp, fp, fn):
                denominator = 2 * tp + fp + fn
                return 2 * tp / denominator if denominator else 1.0

            count = len(self.generation_rows)
            return {
                f"{metric_key_prefix}_generation_binary_f1": f1(
                    binary_tp, binary_fp, binary_fn),
                f"{metric_key_prefix}_generation_character_f1": f1(
                    character_tp, character_fp, character_fn),
                f"{metric_key_prefix}_generation_json_valid_rate": (
                    valid_json / count if count else 0.0),
            }

        def evaluate(self, *eval_args, **eval_kwargs):
            metrics = super().evaluate(*eval_args, **eval_kwargs)
            metric_key_prefix = eval_kwargs.get("metric_key_prefix", "eval")
            generation_metrics = self._generation_metrics(metric_key_prefix)
            self.log(generation_metrics)
            metrics.update(generation_metrics)
            return metrics

    updates_per_epoch = max(1, math.ceil(
        samples_per_epoch / (args.batch_size * args.gradient_accumulation)))
    total_steps = updates_per_epoch * args.epochs
    options = dict(
        output_dir=args.output_dir, num_train_epochs=args.epochs,
        learning_rate=args.learning_rate, lr_scheduler_type="cosine",
        warmup_steps=max(1, math.ceil(total_steps * 0.05)),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        bf16=True, tf32=True, gradient_checkpointing=True,
        eval_strategy="epoch", save_strategy="epoch",
        logging_steps=max(1, updates_per_epoch // 5), save_total_limit=2,
        load_best_model_at_end=False, optim="adamw_torch", report_to="none",
        seed=args.seed)
    supported = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in supported and "evaluation_strategy" in supported:
        options["evaluation_strategy"] = options.pop("eval_strategy")
    unsupported = sorted(set(options) - set(supported))
    if unsupported:
        raise RuntimeError(f"Unsupported TrainingArguments: {unsupported}")

    trainer = WeightedTrainer(
        model=model, args=TrainingArguments(**options),
        train_dataset=train_dataset, eval_dataset=validation_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer, padding=True, label_pad_token_id=-100),
        generation_rows=splits["validation"],
        generation_tokenizer=tokenizer,
        generation_max_new_tokens=args.validation_max_new_tokens)
    schedule = []
    for epoch in range(args.epochs):
        probability = sampler.memory_probability(epoch)
        memory_count = round(samples_per_epoch * probability)
        detection_count = samples_per_epoch - memory_count
        positive_fraction = (
            len(sampler.positive_indices) / len(sampler.detection_indices))
        positive_count = round(detection_count * positive_fraction)
        if detection_count >= 2:
            positive_count = min(max(positive_count, 1), detection_count - 1)
        schedule.append({"epoch": epoch + 1,
            "memory_probability": probability,
            "detection_probability": 1.0 - probability,
            "memory_samples": memory_count,
            "detection_samples": detection_count,
            "positive_detection_samples": positive_count,
            "negative_detection_samples": detection_count - positive_count})
    statistics = {
        "objective": "weighted_memory_and_detection",
        "unique_training_examples": len(train_rows),
        "unique_positive_detection_examples": sum(
            row["task"] == "detection" for row in train_rows),
        "unique_negative_detection_examples": sum(
            row["task"] == "negative_detection" for row in train_rows),
        "samples_per_epoch": samples_per_epoch,
        "sampling_with_replacement": True,
        "schedule": schedule,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_statistics.json").write_text(
        json.dumps(statistics, indent=2) + "\n", encoding="utf-8")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # Epoch-based evaluation normally covers the final epoch, but evaluate
    # explicitly so resumed or interrupted schedules always record final metrics.
    final_validation_metrics = trainer.evaluate(
        metric_key_prefix="final_validation")
    (output / "final_generation_validation_metrics.json").write_text(
        json.dumps(final_validation_metrics, indent=2) + "\n",
        encoding="utf-8")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    main()
