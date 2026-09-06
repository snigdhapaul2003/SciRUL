import json
from pathlib import Path


DATA_STORE = Path(__file__).resolve().parent / "data_store"
SUBJECTS = ("Computer_Science", "Medicine", "BioChemistry")
INPUT_NAME = "similar_papers_claims.json"
OUTPUT_FILE = DATA_STORE / "similar_papers_claims.json"


def merge_files() -> None:
    input_files = [DATA_STORE / subject / INPUT_NAME for subject in SUBJECTS]
    combined_records = []
    combined_statistics: dict[str, int | float] = {}

    for input_file in input_files:
        with input_file.open("r", encoding="utf-8") as file:
            document = json.load(file)

        combined_records.extend(document.get("records", []))

        for name, value in document.get("statistics", {}).items():
            combined_statistics[name] = combined_statistics.get(name, 0) + value

    combined_document = {
        "source_files": [str(path) for path in input_files],
        "records": combined_records,
        "statistics": combined_statistics,
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(combined_document, file, indent=2, ensure_ascii=False)

    print(f"Merged {len(combined_records)} records into {OUTPUT_FILE}")


if __name__ == "__main__":
    merge_files()
