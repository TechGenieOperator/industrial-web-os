import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs" / "latest"

sys.path.insert(0, str(PROJECT_ROOT))

from tools.save_company import create_company_record, save_company


def parse_json_file(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing file: {path.relative_to(PROJECT_ROOT)}"
        )

    raw_content = path.read_text(encoding="utf-8-sig").strip()

    if not raw_content:
        raise ValueError(
            f"{path.relative_to(PROJECT_ROOT)} is empty. "
            "Run crewai run again before importing."
        )

    # Remove Markdown code fences if CrewAI included them.
    raw_content = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        raw_content,
        flags=re.IGNORECASE,
    )

    raw_content = re.sub(
        r"\s*```\s*$",
        "",
        raw_content,
    ).strip()

    # Remove commentary before the first JSON character.
    object_start = raw_content.find("{")
    array_start = raw_content.find("[")

    valid_starts = [
        position
        for position in (object_start, array_start)
        if position >= 0
    ]

    if not valid_starts:
        raise ValueError(
            f"{path.relative_to(PROJECT_ROOT)} does not contain JSON."
        )

    raw_content = raw_content[min(valid_starts):]

    # Remove commentary after the final JSON character.
    if raw_content.startswith("{"):
        json_end = raw_content.rfind("}")
    else:
        json_end = raw_content.rfind("]")

    if json_end < 0:
        raise ValueError(
            f"{path.relative_to(PROJECT_ROOT)} contains incomplete JSON."
        )

    raw_content = raw_content[: json_end + 1]

    try:
        parsed = json.loads(raw_content)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in {path.relative_to(PROJECT_ROOT)} "
            f"at line {error.lineno}, column {error.colno}: {error.msg}"
        ) from error

    if not isinstance(parsed, dict):
        raise ValueError(
            f"{path.relative_to(PROJECT_ROOT)} must contain one JSON object."
        )

    return parsed


def find_company_name(research: dict) -> str:
    company_name = str(
        research.get("company_name")
        or research.get("metadata", {}).get("company_name")
        or ""
    ).strip()

    if not company_name:
        raise ValueError(
            "research.json does not contain company_name."
        )

    return company_name


def find_website_url(research: dict) -> str:
    return str(
        research.get("website_url")
        or research.get("metadata", {}).get("website_url")
        or ""
    ).strip()


def find_industry(research: dict) -> str:
    return str(
        research.get("industry")
        or research.get("metadata", {}).get("industry")
        or ""
    ).strip()


def normalize_research(research: dict) -> dict:
    if isinstance(research.get("research"), dict):
        normalized = dict(research["research"])

        for field in (
            "research_metrics",
            "website_analysis",
            "business_intelligence",
            "verification",
        ):
            if field in research:
                normalized[field] = research[field]

        return normalized

    return research


def import_latest_run() -> Path:
    research_path = RUNS_DIR / "research.json"
    architecture_path = RUNS_DIR / "architecture.json"

    print(f"Reading {research_path.relative_to(PROJECT_ROOT)}")
    research = parse_json_file(research_path)

    print(f"Reading {architecture_path.relative_to(PROJECT_ROOT)}")
    architecture = parse_json_file(architecture_path)

    company_name = find_company_name(research)
    website_url = find_website_url(research)
    industry = find_industry(research)

    normalized_research = normalize_research(research)

    combined_verification = {
        "confirmed_facts": [],
        "inferred_facts": [],
        "assumptions": [],
        "requires_client_confirmation": [],
        "contradictions": [],
        "research": research.get("verification", {}),
        "architecture": architecture.get("verification", {}),
    }

    record = create_company_record(
        company_name=company_name,
        website_url=website_url,
        industry=industry,
        research=normalized_research,
        architecture=architecture,
        verification=combined_verification,
    )

    # Preserve the new research metrics in the company memory.
    record["research_metrics"] = research.get(
        "research_metrics",
        normalized_research.get("research_metrics", {}),
    )

    record["website_analysis"] = research.get(
        "website_analysis",
        normalized_research.get("website_analysis", {}),
    )

    record["business_intelligence"] = research.get(
        "business_intelligence",
        normalized_research.get("business_intelligence", {}),
    )

    saved_path = Path(save_company(record))

    print(f"Saved {saved_path}")
    return saved_path


if __name__ == "__main__":
    try:
        import_latest_run()
    except Exception as error:
        print(f"Import failed: {error}")
        sys.exit(1)