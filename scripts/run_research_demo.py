"""Run the ML pipeline security research demo and print a report."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml_pipeline_security.agent_guardrails import evaluate_agent_plan, sample_agent_actions
from ml_pipeline_security.experiments import run_security_research_experiment


def main() -> None:
    report = run_security_research_experiment(seed=11)

    actions = sample_agent_actions()
    guardrail_report = evaluate_agent_plan(actions)

    output = {
        "ml_pipeline_security_report": report,
        "agent_guardrail_report": guardrail_report,
    }

    output_dir = ROOT / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "research_report.json"
    output_path.write_text(json.dumps(output, indent=2, default=str), encoding="utf-8")

    print("=== ML Pipeline Security Research Demo ===")
    unsecured = report["unsecured_pipeline"]
    secured = report["secured_pipeline"]
    print(
        "Summary: "
        f"ASR unsecured={unsecured['attack_success_rate']:.3f}, "
        f"secured={secured['attack_success_rate']:.3f} | "
        f"F1 unsecured={unsecured['metrics']['f1']:.3f}, "
        f"secured={secured['metrics']['f1']:.3f}"
    )
    print("\nFull report follows:\n")
    print(json.dumps(output, indent=2, default=str))
    print(f"\nReport written to: {output_path}")


if __name__ == "__main__":
    main()
