import subprocess
import sys


def run_step(name: str, command: list[str]):
    print(f"\n🚀 {name}")
    print("=" * 60)

    result = subprocess.run(command)

    if result.returncode != 0:
        print(f"\n❌ Failed: {name}")
        sys.exit(result.returncode)

    print(f"\n✅ Completed: {name}")


def main():
    print("\n🧠 Industrial Web OS")
    print("=" * 60)

    run_step(
        "CrewAI Research + Architecture",
        ["crewai", "run"]
    )

    run_step(
        "Knowledge Pipeline",
        ["python", "tools/post_run_pipeline.py"]
    )

    print("\n🎉 INDUSTRIAL WEB OS RUN COMPLETE")
    print("=" * 60)

    print("\nGenerated Assets:")

    print("✅ Research Output")
    print("✅ Architecture Output")
    print("✅ Company Memory")
    print("✅ Industry Memory")
    print("✅ Pattern Memory")
    print("✅ Architecture Recommendations")

    print("\nReady for the next company.")


if __name__ == "__main__":
    main()