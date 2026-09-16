import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "eval"))
from diagnose_generation import inspect_response


class DiagnosticTests(unittest.TestCase):
    def test_failure_categories(self):
        sentence = "The exact evidence is here."
        self.assertEqual(inspect_response("not json", sentence)[0], "malformed_json")
        self.assertEqual(inspect_response('{"answer": []}', sentence)[0], "wrong_schema")
        self.assertEqual(inspect_response('{"steps":[{"span":"invented"}]}', sentence)[0], "non_verbatim_span")
        self.assertEqual(
            inspect_response('{"steps":[{"span":"exact evidence"}]}', sentence)[0],
            "valid",
        )


if __name__ == "__main__":
    unittest.main()
