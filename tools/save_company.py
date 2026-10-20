import json
import os
from datetime import datetime


def create_company_record(
    company_name,
    website_url="",
    industry="",
    research=None,
    architecture=None,
    verification=None,
):
    timestamp = datetime.utcnow().isoformat()

    return {
        "metadata": {
            "company_id": company_name.lower().replace(" ", "_"),
            "company_name": company_name,
            "website_url": website_url,
            "industry": industry,
            "sub_industry": "",
            "date_analyzed": timestamp,
            "version": "1.0"
        },

        "research": research or {},

        "website_analysis": (
            research.get("website_analysis", {})
            if research else {}
        ),

        "architecture": architecture or {},

        "verification": verification or {},

        "patterns": {
            "services": [],
            "pages": [],
            "trust_elements": [],
            "ctas": [],
            "seo_strategies": []
        },

        "business_intelligence": {
            "lead_generation_opportunities": [],
            "content_opportunities": [],
            "recommended_next_steps": []
        },

        "outcomes": {
            "proposal_created": False,
            "website_created": False,
            "client_response": "",
            "won_project": None,
            "notes": []
        },

        "memory": {
            "saved_at": timestamp,
            "last_updated": timestamp
        }
    }


def save_company(record):
    company_name = (
        record["metadata"]["company_name"]
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    output_dir = "knowledge/companies"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(
        output_dir,
        f"{company_name}.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2)

    return output_path