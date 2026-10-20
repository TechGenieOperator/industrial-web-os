import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = PROJECT_ROOT / "runs" / "latest"


def run_step(name: str, command: list[str]):
    print(f"\n🔄 {name}")
    print("-" * 50)

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\n❌ Pipeline failed during: {name}")
        sys.exit(result.returncode)

    print(f"✅ {name} completed")


def get_industry() -> str:
    research_file = RUNS_DIR / "research.json"

    if not research_file.exists():
        raise FileNotFoundError(
            f"Research file not found: {research_file}"
        )

    with open(research_file, "r", encoding="utf-8-sig") as file:
        research = json.load(file)

    industry = (
        research.get("industry")
        or research.get("metadata", {}).get("industry")
    )

    if not industry:
        raise ValueError(
            "Industry not found in research.json"
        )

    return industry


def main():
    print("\n🧠 Industrial Web OS Knowledge Pipeline")
    print("=" * 60)

    run_step(
        "Import Company Memory",
        ["python", "tools/import_latest_run.py"]
    )

    run_step(
        "Update Industry Memory",
        ["python", "tools/update_industry.py"]
    )

    run_step(
        "Extract Cross-Industry Patterns",
        ["python", "tools/extract_patterns.py"]
    )

    industry = get_industry()

    run_step(
        f"Generate {industry} Recommendations",
        [
            "python",
            "tools/recommend_architecture.py",
            industry
        ]
    )

    print("\n✅ Knowledge Pipeline Complete")
    print("=" * 60)
    print(f"Industry: {industry}")
    print("\nGenerated:")

    print("   • knowledge/companies/")
    print("   • knowledge/industries/")
    print("   • knowledge/patterns/")
    print("   • knowledge/recommendations/")


if __name__ == "__main__":
    main()