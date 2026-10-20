import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
INDUSTRIES_DIR = PROJECT_ROOT / "knowledge" / "industries"
PATTERNS_FILE = PROJECT_ROOT / "knowledge" / "patterns" / "industrial_b2b.json"
RECOMMENDATIONS_DIR = PROJECT_ROOT / "knowledge" / "recommendations"


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "unknown"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ranked_values(items: list) -> list[dict]:
    results = []
    for item in items:
        if isinstance(item, str):
            value = item.strip()
            count = 1
        elif isinstance(item, dict):
            value = str(item.get("value", "")).strip()
            count = int(item.get("company_count", 1) or 1)
        else:
            continue

        if value:
            results.append({"value": value, "company_count": count})

    return results


def merge_ranked(primary: list, fallback: list) -> list[dict]:
    merged = {}

    for source, origin in ((fallback, "cross_industry"), (primary, "industry")):
        for item in ranked_values(source):
            value = item["value"]
            key = value.casefold()
            candidate = {
                "value": value,
                "company_count": item["company_count"],
                "source": origin,
            }

            if key not in merged or origin == "industry":
                merged[key] = candidate

    return sorted(
        merged.values(),
        key=lambda item: (
            0 if item["source"] == "industry" else 1,
            -item["company_count"],
            item["value"].casefold(),
        ),
    )


def confidence_label(company_count: int) -> str:
    if company_count < 3:
        return "preliminary"
    if company_count < 10:
        return "developing"
    return "established"


def recommend_architecture(industry: str) -> Path:
    industry_path = INDUSTRIES_DIR / f"{slugify(industry)}.json"

    if not industry_path.exists():
        raise FileNotFoundError(
            f"Industry memory not found: {industry_path.relative_to(PROJECT_ROOT)}. "
            "Run tools/update_industry.py after saving company records."
        )

    industry_data = load_json(industry_path)
    pattern_data = load_json(PATTERNS_FILE) if PATTERNS_FILE.exists() else {}
    global_patterns = pattern_data.get("patterns", {})

    companies_analyzed = int(industry_data.get("companies_analyzed", 0) or 0)
    confidence = confidence_label(companies_analyzed)

    pages = merge_ranked(
        industry_data.get("common_pages", []),
        global_patterns.get("common_pages", []),
    )
    ctas = merge_ranked(
        industry_data.get("common_ctas", []),
        global_patterns.get("common_ctas", []),
    )
    trust_elements = merge_ranked(
        industry_data.get("common_trust_elements", []),
        global_patterns.get("common_trust_elements", []),
    )
    seo = merge_ranked(
        industry_data.get("common_seo_recommendations", []),
        global_patterns.get("common_seo_recommendations", []),
    )
    services = merge_ranked(industry_data.get("common_services", []), [])

    now = datetime.now(timezone.utc).isoformat()

    recommendation = {
        "recommendation_id": f"{slugify(industry)}_architecture",
        "industry": industry_data.get("industry", industry),
        "confidence": confidence,
        "evidence": {
            "industry_companies_analyzed": companies_analyzed,
            "companies": industry_data.get("companies", []),
            "cross_industry_companies_analyzed": int(
                pattern_data.get("total_companies_analyzed", 0) or 0
            ),
        },
        "recommended_architecture": {
            "pages": pages,
            "services_to_feature": services,
            "ctas": ctas,
            "trust_elements": trust_elements,
            "seo_actions": seo,
        },
        "guardrails": {
            "treat_as_proven_best_practice": confidence == "established",
            "requires_company_specific_research": True,
            "note": (
                "Recommendations are preliminary because fewer than three companies "
                "have been analyzed in this industry."
                if confidence == "preliminary"
                else "Use these recommendations as evidence-backed starting points, "
                "then adapt them to the target company."
            ),
        },
        "memory": {
            "generated_at": now,
            "schema_version": "1.0",
        },
    }

    RECOMMENDATIONS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RECOMMENDATIONS_DIR / f"{slugify(industry)}_architecture.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(recommendation, file, indent=2, ensure_ascii=False)

    print(f"Updated {output_path.relative_to(PROJECT_ROOT)}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Recommend a website architecture using saved industry and pattern memory."
    )
    parser.add_argument(
        "industry",
        nargs="?",
        help='Industry name, for example: "Steel"',
    )
    args = parser.parse_args()

    industry = args.industry or input("Industry: ").strip()
    if not industry:
        print("Industry is required.")
        sys.exit(1)

    try:
        recommend_architecture(industry)
    except Exception as error:
        print(f"Architecture recommendation failed: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
