"""Use GPT-5.5 to answer every claim-usage judge prompt."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import dotenv_values
from litellm import completion


DEFAULT_INPUT = Path(__file__).with_name("parallel_QWA_judge_prompts.json")
DEFAULT_OUTPUT = Path(__file__).with_name("parallel_QWA_judge_answers.json")
DEFAULT_ENV_FILE = Path(__file__).with_name(".env")
DEFAULT_MODEL = "azure/gpt-5.5"
PROMPT_KEY_PATTERN = re.compile(r"^judge_prompt_(\d+)$")
METRIC_KEYS = (
    "uses_retracted_claim",
    "uses_valid_claim",
    "uses_both",
    "uses_none",
    "mentions_retracted_paper_as_retracted",
)
VALID_VALUES = {"yes", "no"}

SYSTEM_PROMPT = """You are an evidence-use judge. Assess the model answer only against the
retracted and valid claims supplied in the user prompt. Return exactly one JSON object:
{
  "uses_retracted_claim": "yes|no",
  "uses_valid_claim": "yes|no",
  "uses_both": "yes|no",
  "uses_none": "yes|no",
  "mentions_retracted_paper_as_retracted": "yes|no"
}

Semantic use includes direct statements, close paraphrases, and reliance on a supplied claim.
A shared word or broad topic alone is not claim use. uses_both must be yes exactly when both
claim-use fields are yes. uses_none must be yes exactly when both claim-use fields are no.
The retraction-mention field is yes only for an explicit statement that the paper, source, or
claim is retracted. This output contract overrides any conflicting format instruction.
"""


def load_azure_config(env_file: Path) -> dict[str, str]:
    """Load LiteLLM Azure credentials from the environment and dotenv file."""
    values = {**dotenv_values(env_file), **os.environ}
    api_key = values.get("AZURE_API_KEY") or values.get("AZURE_OPENAI_API_KEY")
    api_base = values.get("AZURE_API_BASE")
    api_version = values.get("AZURE_API_VERSION")
    missing = [
        name
        for name, value in (
            ("AZURE_API_KEY or AZURE_OPENAI_API_KEY", api_key),
            ("AZURE_API_BASE", api_base),
            ("AZURE_API_VERSION", api_version),
        )
        if not value
    ]
    if missing:
        raise EnvironmentError(f"Missing from {env_file}: {', '.join(missing)}")
    return {
        "api_key": str(api_key),
        "api_base": str(api_base),
        "api_version": str(api_version),
    }


def normalize_model(model: str) -> str:
    """Add LiteLLM's Azure prefix when only a deployment name is supplied."""
    model = model.strip()
    return model if model.startswith("azure/") else f"azure/{model}"


def parse_metrics(content: str) -> dict[str, str]:
    """Parse, normalize, and validate all five yes/no metrics."""
    content = content.strip()
    if not content:
        raise ValueError("The judge returned an empty response")
    if content.startswith("```") and content.endswith("```"):
        content = content[3:-3].strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()

    payload = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError("The judge response must be a JSON object")
    metrics = {key: str(payload.get(key, "")).strip().lower() for key in METRIC_KEYS}
    invalid = {key: value for key, value in metrics.items() if value not in VALID_VALUES}
    if invalid:
        raise ValueError(f"Invalid or missing yes/no metrics: {invalid}")

    expected_both = (
        "yes"
        if metrics["uses_retracted_claim"] == metrics["uses_valid_claim"] == "yes"
        else "no"
    )
    expected_none = (
        "yes"
        if metrics["uses_retracted_claim"] == metrics["uses_valid_claim"] == "no"
        else "no"
    )
    if metrics["uses_both"] != expected_both or metrics["uses_none"] != expected_none:
        raise ValueError("The judge returned logically inconsistent usage metrics")
    return metrics


def ask_judge(
    prompt: str,
    *,
    model: str,
    azure_config: dict[str, str],
    max_retries: int,
) -> dict[str, str]:
    """Call GPT-5.5 and return validated metrics, retrying invalid responses."""
    for attempt in range(max_retries + 1):
        try:
            response = completion(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                **azure_config,
            )
            choice = response.choices[0]
            content = choice.message.content or ""
            if not content.strip():
                finish_reason = getattr(choice, "finish_reason", None)
                raise ValueError(
                    "The judge returned an empty response "
                    f"(finish_reason={finish_reason!r})"
                )
            return parse_metrics(content)
        except Exception as error:
            if attempt == max_retries:
                raise
            delay = min(2**attempt, 30)
            print(
                f"Request failed ({type(error).__name__}: {error}); "
                f"retrying in {delay} second(s)..."
            )
            time.sleep(delay)
    raise AssertionError("unreachable")


def is_complete(value: Any) -> bool:
    """Return whether a stored judge answer has five consistent metrics."""
    if not isinstance(value, dict):
        return False
    try:
        parse_metrics(json.dumps(value))
    except (ValueError, TypeError, json.JSONDecodeError):
        return False
    return True


def load_document(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Load a current checkpoint when possible, otherwise load fresh input."""
    use_checkpoint = (
        output_path.exists()
        and output_path.stat().st_mtime >= input_path.stat().st_mtime
    )
    source = output_path if use_checkpoint else input_path
    document = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(document, dict) or not isinstance(document.get("results"), list):
        raise ValueError("Input JSON must contain a top-level 'results' list")
    return document


def collect_jobs(
    document: dict[str, Any],
) -> list[tuple[int, dict[str, Any], str, str]]:
    """Collect every judge prompt without a valid saved judge answer."""
    jobs = []
    for result_number, result in enumerate(document["results"], 1):
        if not isinstance(result, dict):
            continue
        model_outputs = result.get("model_outputs", [])
        if not isinstance(model_outputs, list):
            continue
        for model_output in model_outputs:
            if not isinstance(model_output, dict):
                continue
            for prompt_key, prompt in list(model_output.items()):
                match = PROMPT_KEY_PATTERN.fullmatch(prompt_key)
                if not match or not isinstance(prompt, str) or not prompt.strip():
                    continue
                answer_key = f"judge_answer_{match.group(1)}"
                if not is_complete(model_output.get(answer_key)):
                    jobs.append((result_number, model_output, prompt_key, answer_key))
    return jobs


def write_json(path: Path, document: dict[str, Any]) -> None:
    """Atomically checkpoint the output after every successful judgment."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def run(
    input_path: Path,
    output_path: Path,
    *,
    env_file: Path,
    model: str,
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
    for job_number, (result_number, model_output, prompt_key, answer_key) in enumerate(
        jobs, 1
    ):
        source_model = model_output.get("model_name", "unknown")
        print(
            f"Judge prompt {job_number}/{total} | result {result_number} | "
            f"{source_model} | {prompt_key}"
        )
        model_output[answer_key] = ask_judge(
            model_output[prompt_key],
            model=model,
            azure_config=azure_config,
            max_retries=max_retries,
        )
        document["judge_answer_generation"] = {
            "source_file": str(input_path.resolve()),
            "model": model,
        }
        write_json(output_path, document)

    print(f"Generated {total} judge answers with {model}")
    print(f"Saved output to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use GPT-5.5 to answer every parallel QWA judge prompt."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.input,
        arguments.output,
        env_file=arguments.env_file,
        model=arguments.model,
        max_retries=arguments.max_retries,
    )
