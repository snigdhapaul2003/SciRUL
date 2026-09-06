import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

from litellm import completion


DEFAULT_INPUT_FILE = Path(__file__).with_name("similar_papers_query.json")
DEFAULT_OUTPUT_FILE = Path(__file__).with_name("similar_papers_claims.json")
S2_API_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper"
LLM_MODEL = os.getenv("LITELLM_MODEL", "azure/gpt-4o")
MAX_WORKERS = max(int(os.getenv("CLAIM_MAX_WORKERS", "4")), 1)
REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
MAX_TEXT_CHARS = max(int(os.getenv("CLAIM_MAX_TEXT_CHARS", "120000")), 1000)
S2_MIN_REQUEST_INTERVAL_SECONDS = 1.0
s2_request_lock = threading.Lock()
last_s2_request_time = 0.0

SYSTEM_PROMPT = """You are a careful scientific reader.

DEFINITION
A "scientific claim" is a concise, declarative set of statements of a finding, result, or
methodological fact asserted by the authors. A valid scientific claim:
- can stand alone as a set of factual statements,
- is specific and empirically or theoretically verifiable,
- reflects the scope and conditions of the work when needed,
- does NOT refer to the paper, authors, or presentation (e.g., no "this paper", "we", "the authors").

INDEPENDENCE REQUIREMENT
Every claim must be fully self-contained and cross-reference resolved. This means:
- Do NOT use pronouns or shorthand that point back to the paper's own artifacts
  (e.g., no "the proposed method", "our model", "the introduced dataset", "the
  described framework", "the above approach", "the presented system").
- Instead, replace every such reference with its actual, fully-specified meaning.
  For example:
    BAD : "The proposed architecture achieves better accuracy than baselines."
    GOOD: "A transformer-based architecture with cross-modal attention achieves
           better accuracy than CNN baselines on image-text retrieval tasks."
- If an acronym or shorthand is defined in the paper (e.g., "BERT", "RAG", "our
  loss function L_adv"), expand or describe it within the claim so a reader who
  has never seen the paper can fully understand the claim without any additional
  context.
- Resolve ALL implicit references: domain, task, dataset type, model family,
  condition, or constraint mentioned elsewhere in the paper but relevant to
  the claim must be folded into the claim sentence itself.
- A claim is invalid if its meaning changes or becomes ambiguous when read in
  isolation, without access to the paper.

TASK:
Read the provided paper text and GENERATE (do not quote) the paper's core scientific claims.

OUTPUT RULES
- Write claims as factual statements, NOT as descriptions of what the paper does.
- Do NOT use phrases such as:
  "the paper proposes", "this work shows", "the authors introduce", "we demonstrate".
- Don't mention the numerical results of the paper as claim statement, instead focus on the conceptual findings, methods, or insights.
- Each claim must be EXACTLY ONE 'crisp' consolidated sentence covering all aspects of the claim.
- Each claim must satisfy the INDEPENDENCE REQUIREMENT above before being included.
- Return ONLY valid JSON in the following format:
  {
    "claims": [
      { "text": "<one claim>" },
      { "text": "<one claim>" }
    ]
  }
- Try to keep only one consolidated claim.
- If more claims are there in the paper: return maximum 5 claims.
- If no clear scientific claim is present, return:
  { "claims": [] }

END OF INSTRUCTIONS. Produce only the lines of claims as specified.
"""


def load_env_value(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value.strip().strip('"')
    env_file = Path(__file__).with_name(".env")
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        key, separator, env_value = line.partition("=")
        if separator and key.strip() == name:
            return env_value.strip().strip('"')
    return None


def load_s2_key() -> str | None:
    return load_env_value("S2_API_KEY") or load_env_value("SEMANTIC_SCHOLAR_API_KEY")


def request_bytes(url: str, headers: dict[str, str] | None = None) -> bytes:
    request = Request(
        url,
        headers={
            "Accept": "application/pdf,application/octet-stream;q=0.9,*/*;q=0.1",
            "User-Agent": "claim-extractor/1.0",
            **(headers or {}),
        },
    )
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                data = response.read()
                if not data:
                    raise ValueError("download returned no data")
                return data
        except (HTTPError, URLError, TimeoutError, OSError, ValueError) as error:
            last_error = error
            if attempt < MAX_RETRIES:
                time.sleep(attempt)
    raise RuntimeError(f"download failed: {last_error}") from last_error


def resolve_anchor_pdf_url(anchor: dict[str, Any]) -> str | None:
    global last_s2_request_time
    doi = str(anchor.get("OriginalPaperDOI", "")).strip()
    if not doi:
        return None
    encoded_doi = quote(f"DOI:{doi}", safe=":")
    url = f"{S2_API_BASE_URL}/{encoded_doi}?fields=openAccessPdf,title,abstract"
    headers = {}
    api_key = load_s2_key()
    if api_key:
        headers["x-api-key"] = api_key
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with s2_request_lock:
                elapsed = time.monotonic() - last_s2_request_time
                if elapsed < S2_MIN_REQUEST_INTERVAL_SECONDS:
                    time.sleep(S2_MIN_REQUEST_INTERVAL_SECONDS - elapsed)
                payload = json.loads(request_bytes(url, headers).decode("utf-8"))
                last_s2_request_time = time.monotonic()
            break
        except RuntimeError as error:
            if "HTTP Error 429" not in str(error) or attempt == MAX_RETRIES:
                return None
            time.sleep(attempt * 2)
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None
    else:
        return None
    pdf_info = payload.get("openAccessPdf") or {}
    return str(pdf_info.get("url", "")).strip() or None


def extract_pdf_text(pdf_bytes: bytes) -> str:
    try:
        import fitz

        document = fitz.open(stream=pdf_bytes, filetype="pdf")
        text = "\n".join(page.get_text("text") for page in document)
        document.close()
    except ImportError:
        from PyPDF2 import PdfReader
        import io

        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise ValueError("PDF contained no extractable text")
    return text[:MAX_TEXT_CHARS]


def normalize_claims(payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, dict) or not isinstance(payload.get("claims"), list):
        raise ValueError("LLM response must contain a claims list")
    claims = []
    for item in payload["claims"][:5]:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError("each claim must be an object with string text")
        text = " ".join(item["text"].split())
        if text:
            claims.append({"text": text})
    return claims


def extract_claims(text: str) -> list[dict[str, str]]:
    prompt = "The following is extracted text from one scientific paper.\n\n" + text
    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = completion(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                api_key=load_env_value("AZURE_API_KEY") or load_env_value("AZURE_OPENAI_API_KEY"),
                api_base=load_env_value("AZURE_API_BASE"),
                api_version=load_env_value("AZURE_API_VERSION"),
                temperature=0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content
            return normalize_claims(json.loads(content))
        except (Exception) as error:
            last_error = error
            if attempt < MAX_RETRIES:
                time.sleep(attempt)
    raise RuntimeError(f"claim extraction failed: {last_error}") from last_error


def process_paper(paper: dict[str, Any], pdf_url: str | None) -> dict[str, Any]:
    result = dict(paper)
    result["claims"] = []
    if not pdf_url:
        result["claim_extraction"] = {"status": "skipped", "error": "no downloadable PDF URL"}
        return result
    try:
        text = extract_pdf_text(request_bytes(pdf_url))
        result["claims"] = extract_claims(text)
        result["claim_extraction"] = {
            "status": "success",
            "text_characters_read": len(text),
        }
    except Exception as error:
        result["claim_extraction"] = {
            "status": "error",
            "error": f"{type(error).__name__}: {error}",
        }
    return result


def skipped_similar_paper(paper: dict[str, Any], reason: str) -> dict[str, Any]:
    result = dict(paper)
    result["claims"] = []
    result["claim_extraction"] = {"status": "skipped", "error": reason}
    return result


def process_anchor(record: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
    result = dict(record)
    anchor = dict(record.get("anchor_csv_data") or {})
    anchor_url = resolve_anchor_pdf_url(anchor)
    anchor_result = process_paper(anchor, anchor_url)
    result["anchor_claims"] = anchor_result["claims"]
    result["anchor_claim_extraction"] = anchor_result["claim_extraction"]

    papers = list(record.get("similar_papers") or [])
    anchor_success = bool(anchor_result["claims"])
    if anchor_success:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(process_paper, paper, (paper.get("open_access_pdf") or {}).get("url"))
                for paper in papers
            ]
            result["similar_papers"] = [future.result() for future in futures]
        skip_reason = ""
    else:
        skip_reason = "anchor paper claims were not extracted"
        result["similar_papers"] = [skipped_similar_paper(paper, skip_reason) for paper in papers]

    processed_similar = len(result["similar_papers"]) if anchor_success else 0
    successful_similar = sum(
        item.get("claim_extraction", {}).get("status") == "success"
        for item in result["similar_papers"]
    )
    similar_with_claims = sum(bool(item.get("claims")) for item in result["similar_papers"])
    result["claim_statistics"] = {
        "anchor_claims_extracted": anchor_success,
        "similar_papers_total": len(papers),
        "similar_papers_processed": processed_similar,
        "similar_papers_skipped": len(papers) - processed_similar,
        "successful_similar_claim_extractions": successful_similar,
        "similar_papers_with_claims": similar_with_claims,
    }
    return result, {
        "processed_papers": 1 + processed_similar,
        "successful_claim_extractions": int(anchor_success) + successful_similar,
        "papers_with_claims": int(anchor_success) + similar_with_claims,
        "anchor_papers_processed": 1,
        "anchor_papers_with_claims": int(anchor_success),
        "anchor_papers_without_claims": int(not anchor_success),
        "similar_papers_processed": processed_similar,
        "similar_papers_skipped": len(papers) - processed_similar,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Download papers and extract self-contained scientific claims.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as input_file:
        records = json.load(input_file)
    selected = records[args.start : args.start + args.limit if args.limit is not None else None]

    output_records = []
    totals = {
        "anchor_records": 0,
        "processed_papers": 0,
        "successful_claim_extractions": 0,
        "papers_with_claims": 0,
        "anchor_papers_processed": 0,
        "anchor_papers_with_claims": 0,
        "anchor_papers_without_claims": 0,
        "similar_papers_processed": 0,
        "similar_papers_skipped": 0,
    }
    for index, record in enumerate(selected, start=args.start + 1):
        try:
            result, statistics = process_anchor(record)
        except Exception as error:
            result = dict(record)
            result["claim_processing_error"] = f"{type(error).__name__}: {error}"
            statistics = {
                "processed_papers": 0,
                "successful_claim_extractions": 0,
                "papers_with_claims": 0,
                "anchor_papers_processed": 1,
                "anchor_papers_with_claims": 0,
                "anchor_papers_without_claims": 1,
                "similar_papers_processed": 0,
                "similar_papers_skipped": len(record.get("similar_papers") or []),
            }
        output_records.append(result)
        totals["anchor_records"] += 1
        for key in statistics:
            totals[key] += statistics[key]
        print(f"Processed anchor {index}/{len(records)}")

    totals["failed_or_skipped"] = totals["processed_papers"] - totals["successful_claim_extractions"]
    output = {
        "source_file": str(args.input),
        "records": output_records,
        "statistics": totals,
    }
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(output, output_file, ensure_ascii=True, indent=2)
        output_file.write("\n")
    print(f"Saved claims for {totals['processed_papers']} papers to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())