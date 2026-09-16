import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from retracted_claims.core import Step, aggregate, conformal_threshold, parse_derivation, step_dict
from retracted_claims.data import load_dataset, sentences_with_offsets, split_answers
from retracted_claims.llama import PROMPT


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.examples = load_dataset(ROOT / "input_file.json")

    def test_input_and_splits(self):
        self.assertEqual(len(self.examples), 99)
        self.assertEqual(sum(len(x.spans) for x in self.examples), 467)
        splits = split_answers(self.examples)
        self.assertEqual([len(splits[k]) for k in ("train", "calibration", "test")], [77, 11, 11])

    def test_sentence_offsets_roundtrip(self):
        for ex in self.examples:
            for start, end, sentence in sentences_with_offsets(ex.paragraph):
                self.assertEqual(ex.paragraph[start:end], sentence)

    def test_strict_derivation_and_output_span(self):
        sentence = "Vitamin D deficiency increased risk."
        steps = parse_derivation(
            json.dumps({"steps": [{"span": "Vitamin D deficiency"}]}),
            sentence,
        )
        steps[0].start, steps[0].end = 0, 20
        self.assertEqual(step_dict(steps[0], sentence)["span"], "Vitamin D deficiency")
        with self.assertRaises(ValueError):
            parse_derivation('{"steps":[{"span":"paraphrase"}]}', sentence)
        with self.assertRaises(ValueError):
            parse_derivation('[]', sentence)
        with self.assertRaises(ValueError):
            parse_derivation('{"steps":[{"span":""}]}', sentence)

    def test_conformal_and_aggregation(self):
        self.assertEqual(conformal_threshold([.1, .2, .3], .5), .2)
        self.assertEqual(aggregate([Step("x", score=.9, status="verified_positive")], .8), "used")
        self.assertEqual(aggregate([], .8), "not_used")
        self.assertEqual(
            aggregate([Step("x", score=.6, status="verified_positive")], .8),
            "abstain",
        )

    def test_prompt_literal_json(self):
        rendered = PROMPT.format(claim="claim", sentence="sentence")
        self.assertIn('{"steps":[]}', rendered)

    def test_training_ratio(self):
        sys.path.insert(0, str(ROOT / "scripts" / "train"))
        from train_llama import records
        rows = records(split_answers(self.examples)["train"], negative_ratio=5, seed=13)
        positive = sum(bool(json.loads(x["messages"][1]["content"])["steps"]) for x in rows)
        self.assertEqual(len(rows) - positive, positive * 5)

    def test_chat_template_is_used(self):
        from retracted_claims.llama import LlamaDeriver
        class FakeTokenizer:
            chat_template = "present"
            def apply_chat_template(self, messages, tokenize, add_generation_prompt):
                self.arguments = (messages, tokenize, add_generation_prompt)
                return "CHAT:" + messages[0]["content"]
        deriver = object.__new__(LlamaDeriver)
        deriver.tokenizer = FakeTokenizer()
        rendered = deriver._render("claim", "sentence")
        self.assertTrue(rendered.startswith("CHAT:"))
        self.assertTrue(deriver.tokenizer.arguments[2])


if __name__ == "__main__":
    unittest.main()
