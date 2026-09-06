import argparse
import csv
from http.client import HTTPException
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from litellm import completion


DEFAULT_INPUT_FILE = Path(__file__).with_name("experimental_dataset.csv")
DEFAULT_OUTPUT_FILE = Path(__file__).with_name("similar_papers_query.json")
S2_API_BASE_URL = "https://api.semanticscholar.org"
LLM_MODEL = os.getenv("LITELLM_MODEL", "azure/gpt-4o")
MAX_WORKERS = max(int(os.getenv("LLM_MAX_WORKERS", "8")), 1)
RESULT_COUNT = 50
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3
S2_MIN_REQUEST_INTERVAL_SECONDS = 1
s2_request_lock = threading.Lock()
last_s2_request_time = 0.0
PAPER_FIELDS = (
    "paperId,title,abstract,authors,venue,year,publicationDate,externalIds,"
    "url,openAccessPdf,publicationTypes,citationCount"
)


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


def load_api_key() -> str | None:
    api_key = os.getenv("S2_API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        return api_key.strip().strip('"')

    env_file = Path(__file__).with_name(".env")
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() in {
            "S2_API_KEY",
            "SEMANTIC_SCHOLAR_API_KEY",
        }:
            return value.strip().strip('"')
    return None


def anchor_text(row: dict[str, str]) -> str:
    return (
        f"Title: {row.get('Title', '').strip()}\n"
        f"Subject labels: {row.get('Subject', '').strip() or 'Not available'}\n"
        f"Abstract: {row.get('Abstract', '').strip() or 'Not available'}"
    )


def generate_query(row: dict[str, str]) -> dict[str, str]:
    prompt = f"""You are an expert academic information-retrieval specialist.
Create one optimized Semantic Scholar search query for papers in the SAME SPECIFIC,
NARROW research subfield as the anchor paper below.

{anchor_text(row)}

The query must target the exact research problem, target system or phenomenon,
scientific task, and outcome. Preserve distinctive technical terms from the title and
abstract. Do not broaden the query to the whole discipline or to generic terms such as
computer science, engineering, artificial intelligence, machine learning, optimization,
communication, or data analysis unless they are essential to the exact problem.
Do not use a DOI, author name, quotation marks, Boolean operators, or punctuation.
Use 5 to 12 precise search terms. Prefer the narrowest useful query; omit uncertain terms.

Return JSON only in this exact form:
{{
  "query": "precise plain-text Semantic Scholar query",
  "rationale": "one short explanation of why these terms identify the exact subfield"
}}"""
    response = completion(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        api_key=load_env_value("AZURE_API_KEY"),
        api_base=load_env_value("AZURE_API_BASE"),
        api_version=load_env_value("AZURE_API_VERSION"),
        temperature=0,
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    query = " ".join(str(data.get("query", "")).split())
    print(f"Generated query: {query}")
    if not query:
        raise ValueError("The LLM returned an empty Semantic Scholar query")
    return {"query": query, "rationale": str(data.get("rationale", ""))}


def s2_search(query: str) -> dict[str, Any] | None:
    global last_s2_request_time
    api_key = load_api_key()
    if not api_key:
        raise RuntimeError("S2_API_KEY is not configured in the environment or .env file")

    params = urlencode(
        {
            "query": query,
            "limit": RESULT_COUNT,
            "openAccessPdf": "",
            "fields": PAPER_FIELDS,
        }
    )
    request = Request(
        f"{S2_API_BASE_URL}/graph/v1/paper/search?{params}",
        headers={
            "Accept": "application/json",
            "User-Agent": "semantic-scholar-key-check/1.0",
            "x-api-key": api_key,
        },
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with s2_request_lock:
                elapsed = time.monotonic() - last_s2_request_time
                if elapsed < S2_MIN_REQUEST_INTERVAL_SECONDS:
                    time.sleep(S2_MIN_REQUEST_INTERVAL_SECONDS - elapsed)
                with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                    last_s2_request_time = time.monotonic()
                    return payload
        except HTTPError as error:
            if error.code == 404:
                return None
            if error.code in {401, 403}:
                raise RuntimeError(f"Semantic Scholar API rejected the configured key (HTTP {error.code})") from error
            if error.code == 429 and attempt < MAX_RETRIES:
                wait = max(float(error.headers.get("Retry-After", 0) or 0), attempt)
                print(f"[RATE] Semantic Scholar 429; retrying in {wait:.1f}s")
                time.sleep(wait)
                continue
            if attempt == MAX_RETRIES:
                print(f"[ERROR] Semantic Scholar search failed after {MAX_RETRIES} attempts: {error}")
                return None
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            if attempt == MAX_RETRIES:
                print(f"[ERROR] Semantic Scholar search failed after {MAX_RETRIES} attempts: {error}")
                return None
            wait = attempt
            print(f"[WARN] Semantic Scholar search failed: {error}; retrying in {wait:.1f}s")
            time.sleep(wait)
    return None


def has_downloadable_full_text(paper: dict[str, Any]) -> bool:
    pdf_info = paper.get("openAccessPdf") or {}
    pdf_url = pdf_info.get("url")
    if not pdf_url:
        return False
    pdf_url = str(pdf_url).strip()
    parts = urlsplit(pdf_url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        return False
    pdf_url = urlunsplit(
        (
            parts.scheme,
            parts.netloc,
            quote(parts.path, safe="/%:@!$&'()*+,;=-._~"),
            quote(parts.query, safe="=&/?%:@!$'()*+,;=-._~"),
            "",
        )
    )

    headers = {"Accept": "application/pdf", "User-Agent": "similar-paper-query/1.0"}
    try:
        head_request = Request(pdf_url, headers=headers, method="HEAD")
        with urlopen(head_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            if response.status == 200 and "application/pdf" in content_type:
                return True
    except (HTTPError, URLError, HTTPException, TimeoutError, ValueError, OSError):
        pass

    try:
        range_request = Request(
            pdf_url,
            headers={**headers, "Range": "bytes=0-4"},
        )
        with urlopen(range_request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            content_type = response.headers.get("Content-Type", "").lower()
            return "application/pdf" in content_type or response.read(5).startswith(b"%PDF-")
    except (HTTPError, URLError, HTTPException, TimeoutError, ValueError, OSError):
        return False


def paper_for_output(paper: dict[str, Any]) -> dict[str, Any]:
    return {
        "paper_id": paper.get("paperId"),
        "title": paper.get("title"),
        "abstract": paper.get("abstract"),
        "authors": paper.get("authors", []),
        "venue": paper.get("venue"),
        "year": paper.get("year"),
        "publication_date": paper.get("publicationDate"),
        "external_ids": paper.get("externalIds", {}),
        "url": paper.get("url"),
        "open_access_pdf": paper.get("openAccessPdf"),
        "publication_types": paper.get("publicationTypes"),
        "citation_count": paper.get("citationCount"),
        "similarity_score": paper.get("score"),
    }


def build_result(row: dict[str, str]) -> dict[str, Any]:
    generated_query = generate_query(row)
    response = s2_search(generated_query["query"])
    papers = [] if response is None else response.get("data", [])
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        downloadable = list(executor.map(has_downloadable_full_text, papers))
    papers = [paper for paper, is_downloadable in zip(papers, downloadable) if is_downloadable]
    return {
        "anchor_csv_data": row,
        "status": "ok",
        "search_query": generated_query["query"],
        "query_rationale": generated_query["rationale"],
        "similarity_method": "LLM-optimized narrow-subfield query + Semantic Scholar search + downloadable PDF validation",
        "similar_papers": [
            {"rank": rank, **paper_for_output(paper)}
            for rank, paper in enumerate(papers, start=1)
        ],
        "requested_count": RESULT_COUNT,
        "returned_count": len(papers),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a narrow LLM query per anchor and search Semantic Scholar."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.DictReader(input_file))
    selected_rows = rows[args.start : args.start + args.limit if args.limit is not None else None]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(build_result, row) for row in selected_rows]
        results = []
        for index, (row, future) in enumerate(zip(selected_rows, futures), start=args.start + 1):
            try:
                result = future.result()
            except (HTTPError, URLError, RuntimeError, TimeoutError, json.JSONDecodeError, ValueError) as error:
                result = {
                    "anchor_csv_data": row,
                    "status": "error",
                    "error": f"{type(error).__name__}: {error}",
                    "similar_papers": [],
                }
            results.append(result)
            print(f"Processed {index}/{len(rows)}: {row.get('Title', '')}")

    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(results, output_file, ensure_ascii=True, indent=2)
        output_file.write("\n")
    print(f"Saved {len(results)} anchor records to {args.output}")


if __name__ == "__main__":
    main()
