"""BF16 LoRA/full training for closed-book retracted-claim detection."""
from __future__ import annotations

import argparse
import inspect
import json
import math
import random
from pathlib import Path

from retracted_reward.data import build_splits


MAX_CURRICULUM_REPEATS = 5


def arguments(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="claims_and_answers_judged_large_neg.json")
    p.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    p.add_argument("--output-dir", default="runs/closed_book_lora")
    p.add_argument("--mode", choices=("lora", "full"), default="lora")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--memory-ratio", type=float, default=None,
                   help="Deprecated: paper-local claim:paragraph ratio is data-driven (5:7)")
    p.add_argument("--negative-ratio", type=float, default=0.5,
                   help="Additional hard-negative sentences per training paragraph")
    p.add_argument("--max-length", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--gradient-accumulation", type=int, default=16)
    p.add_argument("--validation-max-new-tokens", type=int, default=256,
                   help="Maximum tokens generated for each validation response")
    p.add_argument("--seed", type=int, default=13)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--resume-from-checkpoint", default=None)
    return p.parse_args(argv)


def curriculum_schedule(records: list[dict], stages: int,
                        seed: int = 13) -> tuple[list[dict], list[dict]]:
    """Shift smoothly from claim retention to closed-book detection.

    With three stages the repeat weights are memory:detection = 3:1, 2:2,
    then 1:3. For longer runs, repetition is capped at 5; for example, a
    20-stage run transitions gradually from 5:1 to 1:5. The source list is
    already grouped by paper, so repeating records in place preserves
    claims-before-paragraphs ordering in every stage.
    """
    if stages < 1:
        raise ValueError("epochs must be a positive integer")
    by_paper: dict[str, dict[str, list[dict]]] = {}
    for record in records:
        paper_tasks = by_paper.setdefault(str(record["paper_id"]), {
            "memory": [], "detection": [], "negative_detection": []})
        paper_tasks[record["task"]].append(record)

    scheduled, description = [], []
    max_repeats = min(stages, MAX_CURRICULUM_REPEATS)
    for stage in range(stages):
        rng = random.Random(f"{seed}:curriculum-stage:{stage + 1}")
        # Divide the full run into evenly sized curriculum bands. For 20
        # stages, each of the five ratios is therefore used for four stages.
        transition_step = min(
            max_repeats - 1, stage * max_repeats // stages
        )
        memory_repeats = max_repeats - transition_step
        detection_repeats = 1 + transition_step
        stage_records = []
        paper_order = sorted(by_paper)
        rng.shuffle(paper_order)
        for paper_id in paper_order:
            tasks = by_paper[paper_id]
            # All claim rehearsal stays before all extraction work for this
            # paper, even though order within each block changes every repeat.
            for _ in range(memory_repeats):
                block = list(tasks["memory"]); rng.shuffle(block)
                stage_records.extend(block)
            for _ in range(detection_repeats):
                block = list(tasks["detection"]); rng.shuffle(block)
                stage_records.extend(block)
            for _ in range(detection_repeats):
                block = list(tasks["negative_detection"]); rng.shuffle(block)
                stage_records.extend(block)
        scheduled.extend(stage_records)
        description.append({"stage": stage + 1,
            "memory_repeats": memory_repeats,
            "detection_repeats": detection_repeats,
            "paper_order": paper_order,
            "examples": len(stage_records)})
    return scheduled, description


def main(argv=None):
    args = arguments(argv)
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
        DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed)
    from torch.utils.data import SequentialSampler

    class CurriculumTrainer(Trainer):
        """Preserve curriculum order and validate with actual generation."""

        def __init__(self, *trainer_args, generation_rows, generation_tokenizer,
                     generation_max_new_tokens, **trainer_kwargs):
            super().__init__(*trainer_args, **trainer_kwargs)
            self.generation_rows = generation_rows
            self.generation_tokenizer = generation_tokenizer
            self.generation_max_new_tokens = generation_max_new_tokens

        def _get_train_sampler(self, *unused_args, **unused_kwargs):
            return SequentialSampler(self.train_dataset)

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

    if not torch.cuda.is_available():
        raise RuntimeError("Llama-3.1-8B training requires a CUDA GPU")
    if args.epochs < 1 or not float(args.epochs).is_integer():
        raise ValueError("--epochs is the integer number of curriculum stages")
    curriculum_stages = int(args.epochs)
    set_seed(args.seed)
    splits = build_splits(args.data, args.seed, args.negative_ratio,
                          args.memory_ratio)
    scheduled_train, schedule_description = curriculum_schedule(
        splits["train"], curriculum_stages, args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16)
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
        encoded["labels"] = [-100] * len(prompt_ids) + encoded["input_ids"][len(prompt_ids):]
        return encoded

    datasets = {}
    dataset_rows = {"train": scheduled_train,
                    "validation": splits["validation"]}
    for name in ("train", "validation"):
        datasets[name] = Dataset.from_list(dataset_rows[name]).map(
            encode, remove_columns=["task", "messages", "paper_id"],
            desc=f"Tokenizing {name}")
    effective_batch = args.batch_size * args.gradient_accumulation
    steps_epoch = max(1, math.ceil(len(datasets["train"]) / effective_batch))
    # Validate repeatedly throughout the materialized curriculum. Training is
    # still allowed to finish; these evaluations are diagnostic and never
    # trigger early stopping or restoration of an intermediate checkpoint.
    eval_steps = max(1, steps_epoch // curriculum_stages)
    # All curriculum stages are materialized into one ordered pass. Calling it
    # one Trainer epoch prevents the Trainer from reshuffling/repeating stages.
    total_steps = steps_epoch
    training_options = dict(
        output_dir=args.output_dir, num_train_epochs=1,
        learning_rate=args.learning_rate, lr_scheduler_type="cosine",
        # Explicit steps work on older Transformers releases that do not have
        # TrainingArguments.warmup_ratio.
        warmup_steps=max(1, math.ceil(total_steps * 0.05)),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        bf16=True, tf32=True, gradient_checkpointing=True,
        eval_strategy="steps", save_strategy="steps", eval_steps=eval_steps,
        save_steps=eval_steps, logging_steps=max(1, eval_steps // 5),
        # Keep complete Trainer checkpoints (including optimizer, scheduler,
        # and RNG state) so --resume-from-checkpoint can continue training.
        save_only_model=False, save_total_limit=2, load_best_model_at_end=False,
        optim="adamw_torch", report_to="none", seed=args.seed)
    supported = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in supported and "evaluation_strategy" in supported:
        training_options["evaluation_strategy"] = training_options.pop("eval_strategy")
    unsupported = sorted(set(training_options) - set(supported))
    if unsupported:
        raise RuntimeError(
            "The installed Transformers version is too old for these training "
            f"options: {unsupported}. Upgrade with: pip install -U transformers")
    training_args = TrainingArguments(**training_options)
    trainer = CurriculumTrainer(model=model, args=training_args,
        train_dataset=datasets["train"], eval_dataset=datasets["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True,
                                             label_pad_token_id=-100),
        generation_rows=splits["validation"],
        generation_tokenizer=tokenizer,
        generation_max_new_tokens=args.validation_max_new_tokens)
    stats = {name: {"total": len(rows), **{
        task: sum(r["task"] == task for r in rows)
        for task in ("memory", "detection", "negative_detection")}}
        for name, rows in splits.items()}
    stats["curriculum"] = {
        "number_of_stages": curriculum_stages,
        "scheduled_training_examples": len(scheduled_train),
        "stages": schedule_description,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir, "dataset_statistics.json").write_text(
        json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    # Step-based evaluation need not land on the final optimizer step. Always
    # assess and record the model after the complete curriculum has run.
    final_validation_metrics = trainer.evaluate(
        metric_key_prefix="final_validation")
    Path(args.output_dir, "final_generation_validation_metrics.json").write_text(
        json.dumps(final_validation_metrics, indent=2) + "\n",
        encoding="utf-8")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()


if __name__ == "__main__":
    main()
