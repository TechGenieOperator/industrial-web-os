import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDUSTRIES_DIR = PROJECT_ROOT / "knowledge" / "industries"
PATTERNS_DIR = PROJECT_ROOT / "knowledge" / "patterns"
OUTPUT_FILE = PATTERNS_DIR / "industrial_b2b.json"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def normalize(value: str) -> str:
    value = str(value).strip()
    return re.sub(r"\s+", " ", value)


def add_ranked_items(counter: Counter, items: list, weight: int = 1) -> None:
    for item in items:
        if isinstance(item, str):
            value = normalize(item)
            count = weight
        elif isinstance(item, dict):
            value = normalize(item.get("value", ""))
            count = int(item.get("company_count", weight) or weight)
        else:
            continue

        if value:
            counter[value] += count


def ranked(counter: Counter, minimum_count: int) -> list[dict]:
    return [
        {"value": value, "company_count": count}
        for value, count in counter.most_common()
        if count >= minimum_count
    ]


def extract_patterns() -> Path | None:
    INDUSTRIES_DIR.mkdir(parents=True, exist_ok=True)
    PATTERNS_DIR.mkdir(parents=True, exist_ok=True)

    industry_files = [
        path for path in sorted(INDUSTRIES_DIR.glob("*.json"))
        if path.name != "industry_template.json"
    ]

    if not industry_files:
        print("No industry records found.")
        return None

    pages = Counter()
    services = Counter()
    ctas = Counter()
    trust_elements = Counter()
    seo_recommendations = Counter()
    industries = []
    total_companies = 0

    for path in industry_files:
        try:
            industry = load_json(path)
        except (json.JSONDecodeError, OSError) as error:
            print(f"Skipped {path.name}: {error}")
            continue

        industry_name = normalize(industry.get("industry", ""))
        companies_analyzed = int(industry.get("companies_analyzed", 0) or 0)

        if industry_name:
            industries.append({
                "industry": industry_name,
                "companies_analyzed": companies_analyzed
            })

        total_companies += companies_analyzed
        add_ranked_items(services, industry.get("common_services", []))
        add_ranked_items(pages, industry.get("common_pages", []))
        add_ranked_items(ctas, industry.get("common_ctas", []))
        add_ranked_items(trust_elements, industry.get("common_trust_elements", []))
        add_ranked_items(seo_recommendations, industry.get("common_seo_recommendations", []))

    if not industries:
        print("No valid industry records found.")
        return None

    # Patterns need repetition. With fewer than two companies, keep observations
    # but clearly mark them as preliminary instead of learned patterns.
    minimum_count = 2 if total_companies >= 2 else 1
    confidence = "preliminary" if total_companies < 3 else "developing"
    now = datetime.now(timezone.utc).isoformat()

    record = {
        "pattern_id": "industrial_b2b",
        "scope": "cross_industry",
        "confidence": confidence,
        "minimum_company_count": minimum_count,
        "industries_analyzed": industries,
        "total_companies_analyzed": total_companies,
        "patterns": {
            "common_services": ranked(services, minimum_count),
            "common_pages": ranked(pages, minimum_count),
            "common_ctas": ranked(ctas, minimum_count),
            "common_trust_elements": ranked(trust_elements, minimum_count),
            "common_seo_recommendations": ranked(seo_recommendations, minimum_count)
        },
        "usage_note": (
            "Treat these as preliminary observations until more companies are analyzed."
            if confidence == "preliminary"
            else "Use these patterns as supporting context, not as guaranteed best practices."
        ),
        "memory": {
            "last_updated": now,
            "schema_version": "1.0"
        }
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(record, file, indent=2, ensure_ascii=False)

    print(f"Updated {OUTPUT_FILE.relative_to(PROJECT_ROOT)}")
    return OUTPUT_FILE


if __name__ == "__main__":
    try:
        extract_patterns()
    except Exception as error:
        print(f"Pattern extraction failed: {error}")
        sys.exit(1)
