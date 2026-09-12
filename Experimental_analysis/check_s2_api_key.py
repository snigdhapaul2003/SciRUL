import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
TEST_QUERY = "computer network technology electronic information engineering application analysis"


def load_api_key() -> str | None:
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


def check_api_key(api_key: str) -> None:
    params = urlencode(
        {
            "query": TEST_QUERY,
            "limit": 50,
            "openAccessPdf": "",
            "fields": "paperId,title,openAccessPdf",
        }
    )
    request = Request(
        f"{API_URL}?{params}",
        headers={
            "Accept": "application/json",
            "User-Agent": "semantic-scholar-key-check/1.0",
            "x-api-key": api_key,
        },
    )

    with urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))

    papers = payload.get("data", [])
    print(papers)
    print("API key status: working")
    print(f"Academic Graph search status: {response.status}")
    print(f"Open-access papers returned: {len(papers)}")


def main() -> int:
    api_key = load_api_key()
    if not api_key:
        print("API key status: not found")
        print("Set S2_API_KEY in .env or the process environment.")
        return 1

    try:
        check_api_key(api_key)
    except HTTPError as error:
        if error.code in {401, 403}:
            print(f"API key status: rejected (HTTP {error.code})")
        elif error.code == 429:
            print("API key status: reached the API, but rate limited (HTTP 429)")
            print("The key may be valid; retry after the rate limit resets.")
        else:
            print(f"API request failed with HTTP {error.code}: {error.reason}")
        return 1
    except (URLError, TimeoutError, json.JSONDecodeError) as error:
        print(f"API request could not be completed: {type(error).__name__}: {error}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
