import csv
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen


INPUT_FILE = Path(__file__).with_name("filtered_output.csv")
OUTPUT_FILE = Path(__file__).with_name("experimental_dataset.csv")
TARGET_COUNT = 50
TARGET_SUBJECT = "Medicine"
SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper"
REQUEST_TIMEOUT_SECONDS = 30
MAX_RETRIES = 3


def has_target_subject(subjects: str) -> bool:
    """Return whether any label is the target subject or one of its subdomains."""
    for label in subjects.split(";"):
        # Strip the taxonomy prefix, for example ``(HSC)`` from
        # ``(HSC) Medicine - Surgery``.
        subject = label.strip().partition(")")[2].strip()
        if subject == TARGET_SUBJECT or subject.startswith(f"{TARGET_SUBJECT} - "):
            return True
    return False


def load_s2_api_key() -> str | None:
    api_key = os.getenv("S2_API_KEY") or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        return api_key.strip().strip('"')

    env_file = Path(__file__).with_name(".env")
    if not env_file.exists():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition("=")
        if separator and name.strip() in {"S2_API_KEY", "SEMANTIC_SCHOLAR_API_KEY"}:
            return value.strip().strip('"')
    return None


def semantic_scholar_pdf_url(doi: str) -> str | None:
    if not doi.strip():
        return None
    url = f"{SEMANTIC_SCHOLAR_URL}/DOI:{quote(doi.strip(), safe='')}?fields=openAccessPdf"
    headers = {"Accept": "application/json", "User-Agent": "dataset-filter/1.0"}
    api_key = load_s2_api_key()
    if api_key:
        headers["x-api-key"] = api_key
    request = Request(url, headers=headers)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                payload = json.loads(response.read().decode("utf-8"))
            pdf_info = payload.get("openAccessPdf") or {}
            pdf_url = str(pdf_info.get("url", "")).strip()
            return pdf_url or None
        except HTTPError as error:
            if error.code in {404, 429} and attempt < MAX_RETRIES:
                time.sleep(attempt * 2)
                continue
            return None
        except (URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            if attempt < MAX_RETRIES:
                time.sleep(attempt)
                continue
            return None
    return None


def is_downloadable_pdf(pdf_url: str | None) -> bool:
    if not pdf_url or not pdf_url.startswith(("http://", "https://")):
        return False
    headers = {"Accept": "application/pdf", "User-Agent": "dataset-filter/1.0"}
    try:
        request = Request(pdf_url, headers=headers, method="HEAD")
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            if response.status == 200 and "application/pdf" in response.headers.get("Content-Type", "").lower():
                return True
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        pass

    try:
        request = Request(pdf_url, headers={**headers, "Range": "bytes=0-4"})
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            return "application/pdf" in response.headers.get("Content-Type", "").lower() or response.read(5).startswith(b"%PDF-")
    except (HTTPError, URLError, TimeoutError, OSError, ValueError):
        return False


def has_downloadable_original_paper(row: dict[str, str]) -> bool:
    pdf_url = semantic_scholar_pdf_url(row.get("OriginalPaperDOI", ""))
    return is_downloadable_pdf(pdf_url)


def main() -> None:
    with INPUT_FILE.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if reader.fieldnames is None or "Subject" not in reader.fieldnames:
            raise ValueError("Input CSV must contain a 'Subject' column.")

        selected_rows = []
        for row in reader:
            if has_target_subject(row["Subject"]) and has_downloadable_original_paper(row):
                selected_rows.append(row)
                print(f"Selected row with DOI: {row.get('OriginalPaperDOI', 'N/A')}")
                if len(selected_rows) == TARGET_COUNT:
                    break

    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames)
        writer.writeheader()
        writer.writerows(selected_rows)

    print(f"Saved {len(selected_rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
