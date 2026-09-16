from __future__ import annotations

import json
from collections import Counter

from .core import Step, parse_derivation


PROMPT = """Determine whether the sentence uses the claim. Return JSON only.
Each step must contain one verbatim sentence substring in the span field.
For no use return {{\"steps\":[]}}.
Claim: {claim}
Sentence: {sentence}
JSON:"""


class LlamaDeriver:
    def __init__(
        self,
        checkpoint: str,
        device_map="auto",
        adapter: str | None = None,
        max_new_tokens: int = 96,
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.tokenizer.pad_token = self.tokenizer.pad_token or self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map=device_map,
            torch_dtype="auto",
        ).eval()
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter).eval()
        self.model.requires_grad_(False)
        assert not self.model.training

    def _render(self, claim: str, sentence: str) -> str:
        content = PROMPT.format(claim=claim, sentence=sentence)
        if getattr(self.tokenizer, "chat_template", None):
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
        return content

    def _once(self, claim: str, sentence: str, sample: bool = False) -> str:
        prompt = self._render(claim, sentence)
        x = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        kwargs = {"max_new_tokens": self.max_new_tokens, "do_sample": sample}
        if sample:
            kwargs["temperature"] = 0.7
        with self.torch.inference_mode():
            y = self.model.generate(**x, **kwargs)
        return self.tokenizer.decode(y[0, x.input_ids.shape[1]:], skip_special_tokens=True)

    def derive(self, claim: str, sentence: str, samples: int = 5):
        parsed_generations = []
        failures = 0
        for _ in range(samples):
            for attempt in range(2):
                raw_output = self._once(claim, sentence, samples > 1)
                try:
                    parsed_generations.append(parse_derivation(raw_output, sentence))
                    break
                except (ValueError, json.JSONDecodeError):
                    if attempt == 1:
                        failures += 1

        if not parsed_generations:
            fallback = [Step("", status="verified_negative")]
            return fallback, failures

        votes = Counter(
            step.span
            for generation in parsed_generations
            for step in generation
        )
        steps = [
            # The denominator is the requested generation count. A generation
            # that remains invalid after retry contributes no vote rather than
            # disappearing and artificially increasing confidence.
            Step(span, count / samples)
            for span, count in votes.items()
        ]
        return steps, failures

    def derive_batch(self, pairs: list[tuple[str, str]], samples=5):
        """Batch independent one-claim prompts; prompts never contain multiple claims."""
        prompts = [self._render(claim, sentence) for claim, sentence in pairs]
        tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.model.device)
        generation_options = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": samples > 1,
            "num_return_sequences": samples,
        }
        if samples > 1:
            generation_options["temperature"] = 0.7

        with self.torch.inference_mode():
            generated_tokens = self.model.generate(**tokens, **generation_options)

        prompt_width = tokens.input_ids.shape[1]
        generated_texts = self.tokenizer.batch_decode(
            generated_tokens[:, prompt_width:],
            skip_special_tokens=True,
        )

        parsed_by_pair = [[] for _ in pairs]
        retry_pair_indexes = []
        for pair_index, (_, sentence) in enumerate(pairs):
            first = pair_index * samples
            last = first + samples
            for text in generated_texts[first:last]:
                try:
                    parsed_by_pair[pair_index].append(
                        parse_derivation(text, sentence)
                    )
                except (ValueError, json.JSONDecodeError):
                    retry_pair_indexes.append(pair_index)

        # Retry all failed generations once in one batch, not serially.
        failures = 0
        if retry_pair_indexes:
            retry_prompts = [prompts[index] for index in retry_pair_indexes]
            retry_tokens = self.tokenizer(
                retry_prompts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            ).to(self.model.device)
            retry_options = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": samples > 1,
            }
            if samples > 1:
                retry_options["temperature"] = 0.7

            with self.torch.inference_mode():
                retry_output_tokens = self.model.generate(
                    **retry_tokens,
                    **retry_options,
                )
            retry_texts = self.tokenizer.batch_decode(
                retry_output_tokens[:, retry_tokens.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            for pair_index, text in zip(retry_pair_indexes, retry_texts):
                sentence = pairs[pair_index][1]
                try:
                    parsed_by_pair[pair_index].append(
                        parse_derivation(text, sentence)
                    )
                except (ValueError, json.JSONDecodeError):
                    failures += 1

        results = []
        for parsed_generations in parsed_by_pair:
            if not parsed_generations:
                results.append([Step("", status="verified_negative")])
                continue
            votes = Counter(
                step.span
                for generation in parsed_generations
                for step in generation
            )
            results.append(
                [
                    Step(span, count / samples)
                    for span, count in votes.items()
                ]
            )
        return results, failures
