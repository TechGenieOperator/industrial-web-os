import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMPANIES_DIR = PROJECT_ROOT / "knowledge" / "companies"
INDUSTRIES_DIR = PROJECT_ROOT / "knowledge" / "industries"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown_industry"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def unique_sorted(values):
    return sorted({str(value).strip() for value in values if str(value).strip()})


def build_industry_record(industry: str, companies: list[dict]) -> dict:
    service_counts = Counter()
    page_counts = Counter()
    cta_counts = Counter()
    trust_counts = Counter()
    seo_counts = Counter()
    company_names = []

    for company in companies:
        metadata = company.get("metadata", {})
        research = company.get("research", {})
        architecture = company.get("architecture", {})

        company_name = metadata.get("company_name")
        if company_name:
            company_names.append(company_name)

        for service in research.get("services", []):
            if isinstance(service, str):
                service_counts[service.strip()] += 1
            elif isinstance(service, dict):
                name = service.get("name") or service.get("title") or service.get("service")
                if name:
                    service_counts[str(name).strip()] += 1

        for page in architecture.get("sitemap", []):
            if isinstance(page, str):
                page_counts[page.strip()] += 1
            elif isinstance(page, dict):
                name = page.get("page") or page.get("label") or page.get("title")
                if name:
                    page_counts[str(name).strip()] += 1

        for cta in architecture.get("cta_strategy", []):
            if isinstance(cta, str):
                cta_counts[cta.strip()] += 1
            elif isinstance(cta, dict):
                text = cta.get("text") or cta.get("cta") or cta.get("label")
                if text:
                    cta_counts[str(text).strip()] += 1

        for trust in architecture.get("trust_elements", []):
            if isinstance(trust, str):
                trust_counts[trust.strip()] += 1
            elif isinstance(trust, dict):
                trust_type = trust.get("type") or trust.get("title")
                if trust_type:
                    trust_counts[str(trust_type).strip()] += 1

        for recommendation in architecture.get("seo_recommendations", []):
            if isinstance(recommendation, str):
                seo_counts[recommendation.strip()] += 1
            elif isinstance(recommendation, dict):
                action = recommendation.get("action") or recommendation.get("technique") or recommendation.get("issue")
                if action:
                    seo_counts[str(action).strip()] += 1

    now = datetime.now(timezone.utc).isoformat()

    def ranked(counter: Counter) -> list[dict]:
        return [
            {"value": value, "company_count": count}
            for value, count in counter.most_common()
        ]

    return {
        "industry": industry,
        "companies_analyzed": len(companies),
        "companies": unique_sorted(company_names),
        "common_services": ranked(service_counts),
        "common_pages": ranked(page_counts),
        "common_ctas": ranked(cta_counts),
        "common_trust_elements": ranked(trust_counts),
        "common_seo_recommendations": ranked(seo_counts),
        "memory": {
            "last_updated": now,
            "schema_version": "1.0"
        }
    }


def update_industries() -> list[Path]:
    COMPANIES_DIR.mkdir(parents=True, exist_ok=True)
    INDUSTRIES_DIR.mkdir(parents=True, exist_ok=True)

    grouped: dict[str, list[dict]] = {}

    for path in sorted(COMPANIES_DIR.glob("*.json")):
        if path.name == "example.json":
            continue

        try:
            company = load_json(path)
        except (json.JSONDecodeError, OSError) as error:
            print(f"Skipped {path.name}: {error}")
            continue

        metadata = company.get("metadata", {})
        industry = str(metadata.get("industry", "")).strip()
        if not industry:
            print(f"Skipped {path.name}: metadata.industry is missing")
            continue

        grouped.setdefault(industry, []).append(company)

    written = []

    for industry, companies in grouped.items():
        record = build_industry_record(industry, companies)
        output_path = INDUSTRIES_DIR / f"{slugify(industry)}.json"
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(record, file, indent=2, ensure_ascii=False)
        written.append(output_path)
        print(f"Updated {output_path.relative_to(PROJECT_ROOT)}")

    if not written:
        print("No valid company records found.")

    return written


if __name__ == "__main__":
    try:
        update_industries()
    except Exception as error:
        print(f"Industry update failed: {error}")
        sys.exit(1)
