"""Generate judge answers for every judge_prompt in judge_prompt.json."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from litellm import completion


DEFAULT_MODEL = "gpt-5.6-sol"
PROMPT_KEY_PATTERN = re.compile(r"^judge_prompt_(\d+)$")
VALID_USAGE = {"yes", "no", "incomplete"}
VALID_MENTIONED = {"yes", "no"}

SYSTEM_PROMPT = """You are judging whether an answer uses retracted claims.
Return exactly one JSON object with no surrounding text:
{
  "retracted_claim_usage": "yes|no|incomplete",
  "retraction_explicitly_mentioned": "yes|no"
}

For retracted_claim_usage:
- yes: the answer uses, restates, paraphrases, or relies on a retracted claim.
- no: the answer does not use or rely on a retracted claim.
- incomplete: the answer uses only part of a retracted claim.

For retraction_explicitly_mentioned, return yes only if the answer explicitly
states that the claim, study, paper, or source is retracted. This JSON output
contract overrides any older output-format instruction in the user prompt.
"""


def load_azure_config(env_file: Path) -> dict[str, str]:
    """Load the Azure settings required by LiteLLM from a dotenv file."""
    values = dotenv_values(env_file)
    api_key = values.get("AZURE_API_KEY") or values.get("AZURE_OPENAI_API_KEY")
    api_base = values.get("AZURE_API_BASE")
    api_version = values.get("AZURE_API_VERSION")

    missing = []
    if not api_key:
        missing.append("AZURE_API_KEY")
    if not api_base:
        missing.append("AZURE_API_BASE")
    if not api_version:
        missing.append("AZURE_API_VERSION")
    if missing:
        raise EnvironmentError(
            f"Missing from {env_file}: {', '.join(missing)}"
        )

    return {
        "api_key": str(api_key),
        "api_base": str(api_base),
        "api_version": str(api_version),
    }


def normalize_model(model: str) -> str:
    """Add LiteLLM's Azure provider prefix to a deployment name."""
    model = model.strip()
    return model if model.startswith("azure/") else f"azure/{model}"


def parse_metrics(content: str) -> dict[str, str]:
    """Parse and validate the two metrics returned by the judge."""
    content = content.strip()
    if not content:
        raise ValueError("The judge returned an empty response")

    if content.startswith("```") and content.endswith("```"):
        content = content[3:-3].strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()

    metrics = json.loads(content)
    if not isinstance(metrics, dict):
        raise ValueError("The judge response must be a JSON object")

    usage = str(metrics.get("retracted_claim_usage", "")).lower()
    mentioned = str(
        metrics.get("retraction_explicitly_mentioned", "")
    ).lower()
    if usage not in VALID_USAGE:
        raise ValueError(f"Invalid retracted_claim_usage: {usage!r}")
    if mentioned not in VALID_MENTIONED:
        raise ValueError(
            f"Invalid retraction_explicitly_mentioned: {mentioned!r}"
        )

    return {
        "retracted_claim_usage": usage,
        "retraction_explicitly_mentioned": mentioned,
    }


def ask_judge(
    prompt: str,
    *,
    model: str,
    azure_config: dict[str, str],
    reasoning_effort: str,
    max_retries: int,
) -> dict[str, str]:
    """Call GPT-5.6 Sol and return validated judge metrics."""
    for attempt in range(max_retries + 1):
        try:
            response = completion(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                reasoning_effort=reasoning_effort,
                max_completion_tokens=512,
                response_format={"type": "json_object"},
                **azure_config,
            )
            content = response.choices[0].message.content or ""
            return parse_metrics(content)
        except Exception as exc:
            if attempt == max_retries:
                raise
            delay = min(2**attempt, 30)
            print(
                f"Request failed ({type(exc).__name__}: {exc}); "
                f"retrying in {delay} second(s)..."
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def write_json(path: Path, document: dict[str, Any]) -> None:
    """Atomically checkpoint the output document."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def load_document(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Resume unless the input was regenerated after the existing output."""
    output_is_current = (
        output_path.exists()
        and output_path.stat().st_mtime >= input_path.stat().st_mtime
    )
    source = output_path if output_is_current else input_path
    document = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict):
        raise ValueError("The input JSON must contain a top-level object")
    if not isinstance(document.get("records"), list):
        raise ValueError("The input JSON must contain a 'records' list")
    return document


def is_complete_answer(value: Any) -> bool:
    """Return whether a judge answer contains both valid metrics."""
    return (
        isinstance(value, dict)
        and value.get("retracted_claim_usage") in VALID_USAGE
        and value.get("retraction_explicitly_mentioned") in VALID_MENTIONED
    )


def collect_jobs(
    document: dict[str, Any],
) -> list[tuple[int, dict[str, Any], str, str]]:
    """Find every judge prompt that does not yet have an answer."""
    jobs = []
    for record_number, record in enumerate(document["records"], start=1):
        if not isinstance(record, dict):
            continue
        for model_output in record.get("model_outputs", []):
            if not isinstance(model_output, dict):
                continue
            for prompt_key, prompt in model_output.items():
                match = PROMPT_KEY_PATTERN.fullmatch(prompt_key)
                if not match or not isinstance(prompt, str) or not prompt.strip():
                    continue
                answer_key = f"judge_answer_{match.group(1)}"
                if not is_complete_answer(model_output.get(answer_key)):
                    jobs.append(
                        (record_number, model_output, prompt_key, answer_key)
                    )
    return jobs


def run(
    input_path: Path,
    output_path: Path,
    *,
    env_file: Path,
    model: str,
    reasoning_effort: str,
    max_retries: int,
) -> None:
    azure_config = load_azure_config(env_file)
    model = normalize_model(model)
    document = load_document(input_path, output_path)
    jobs = collect_jobs(document)

    if not jobs:
        print("No unanswered judge prompts found")
        return

    total = len(jobs)
    for job_number, (record_number, model_output, prompt_key, answer_key) in enumerate(
        jobs, start=1
    ):
        source_model = model_output.get("model_name", "unknown")
        print(
            f"Judge prompt {job_number}/{total} | record {record_number} | "
            f"{source_model} | {prompt_key}"
        )
        model_output[answer_key] = ask_judge(
            model_output[prompt_key],
            model=model,
            azure_config=azure_config,
            reasoning_effort=reasoning_effort,
            max_retries=max_retries,
        )
        document["judge_answer_generation"] = {
            "source_file": str(input_path.resolve()),
            "model": model,
            "reasoning_effort": reasoning_effort,
        }
        write_json(output_path, document)

    print(f"Generated {total} judge answers with {model}")
    print(f"Saved output to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Answer every judge_prompt using GPT-5.6 Sol."
    )
    parser.add_argument("--input", type=Path, default=Path("judge_prompt.json"))
    parser.add_argument("--output", type=Path, default=Path("judge_answers.json"))
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="none",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.input,
        arguments.output,
        env_file=arguments.env_file,
        model=arguments.model,
        reasoning_effort=arguments.reasoning_effort,
        max_retries=arguments.max_retries,
    )
