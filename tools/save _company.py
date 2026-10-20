import json
import os
from datetime import datetime


def save_company(data):
    company_name = data.get("company_name", "unknown_company")

    safe_name = (
        company_name.lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )

    output_dir = "knowledge/companies"
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, f"{safe_name}.json")

    data["saved_at"] = datetime.utcnow().isoformat()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    return output_path