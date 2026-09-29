import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from inference import load_paragraphs, locate_quotes, parse_prediction
from retracted_reward.data import build_splits, load_judged_data
from unify_parent_data import combine_json_files


def test_real_data_and_replay():
    claims, answers = load_judged_data(ROOT / "claims_and_answers_judged.json")
    assert len(claims) == 55
    assert len(answers) == 99
    splits = build_splits(ROOT / "claims_and_answers_judged.json")
    assert any(x["task"] == "memory" for x in splits["train"])
    assert any(x["task"] == "detection" for x in splits["train"])
    assert all(x["task"] == "detection" for x in splits["validation"])


def test_exact_quote_postprocessing():
    raw = 'text {"uses_retracted_claim": true, "spans": ["beta"]}'
    assert locate_quotes("alpha beta", parse_prediction(raw), 10) == [
        {"start": 16, "end": 20, "text": "beta"}]


def test_inference_rejects_claim_side_input(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"paragraphs": [{"paragraph": "x", "claims": []}]}))
    try:
        load_paragraphs(path)
        assert False, "expected claim-side input rejection"
    except ValueError as error:
        assert "forbidden" in str(error)


def test_judged_input_selects_held_out_answers():
    rows = load_paragraphs(ROOT / "claims_and_answers_judged.json", "test", 13)
    assert len(rows) == 11
    assert set(rows[0]) == {"paragraph_id", "paragraph"}


def test_unifier_merges_claim_only_and_answer_only_records(tmp_path):
    claim_record = {
        "anchor_csv_data": {"Record ID": "7", "Title": "Paper"},
        "anchor_claims": [{"text": "Recovered claim"}],
    }
    answer_record = {
        "record_id": "7",
        "anchor_title": "Paper",
        "model_outputs": [{"model_name": "model", "answer_1": "Answer"}],
    }
    (tmp_path / "claims.json").write_text(
        json.dumps({"records": [claim_record]}), encoding="utf-8")
    (tmp_path / "answers.json").write_text(
        json.dumps({"results": [answer_record]}), encoding="utf-8")

    result = combine_json_files(tmp_path, tmp_path / "output.json")

    assert len(result["samples"]) == 1
    assert result["samples"][0]["anchor_claims"] == [
        {"text": "Recovered claim"}]
    assert result["samples"][0]["positive"]["answers"] == {"answer_1": "Answer"}
    assert result["samples"][0]["positive"]["negative_sample"] is False
    assert result["samples"][0]["negative"]["answers"] == {}


def test_unifier_keeps_negative_samples_separate_and_labels_them(tmp_path):
    positive = {
        "record_id": "7",
        "model_outputs": [{"model_name": "model", "answer_1": "Positive"}],
    }
    negative = {
        "record_id": "7",
        "negative_sample": True,
        "negative_sample_type": "summary_valid_claims_only",
        "model_outputs": [{"model_name": "model", "answer_1": "Negative"}],
    }
    (tmp_path / "samples.json").write_text(
        json.dumps({"records": [positive, negative]}), encoding="utf-8")

    result = combine_json_files(tmp_path, tmp_path / "output.json")

    assert len(result["samples"]) == 1
    sample = result["samples"][0]
    assert sample["positive"]["answers"] == {"answer_1": "Positive"}
    assert sample["negative"]["answers"] == {"answer_1": "Negative"}
    assert sample["negative"]["negative_sample"] is True
    assert sample["negative"]["negative_sample_types"] == [
        "summary_valid_claims_only"]
    assert result["statistics"]["positive_samples_written"] == 1
    assert result["statistics"]["negative_samples_written"] == 1
