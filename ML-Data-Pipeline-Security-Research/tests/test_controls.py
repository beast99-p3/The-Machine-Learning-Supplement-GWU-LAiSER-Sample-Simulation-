"""Unit tests for key security control behavior."""

# pyright: reportMissingImports=false, reportMissingTypeStubs=false

from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ml_pipeline_security.agent_guardrails import AgentAction, evaluate_agent_plan
from ml_pipeline_security.data_generation import generate_base_dataset
from ml_pipeline_security.security_controls import (
    compute_dataset_fingerprint,
    sanitize_sensitive_columns,
    validate_schema,
)


class TestControls(unittest.TestCase):
    def test_schema_validation_passes_for_base_data(self) -> None:
        # Base synthetic data should satisfy required schema rules.
        df = generate_base_dataset(n_samples=500, seed=1)
        schema_ok, issues = validate_schema(df.drop(columns=["record_id"]))
        self.assertTrue(schema_ok)
        self.assertEqual(issues, [])

    def test_fingerprint_changes_when_data_changes(self) -> None:
        # Integrity fingerprint must change after content mutation.
        df = generate_base_dataset(n_samples=200, seed=1)
        fp_before = compute_dataset_fingerprint(df)
        df.loc[0, "query_length"] = 999.0
        fp_after = compute_dataset_fingerprint(df)
        self.assertNotEqual(fp_before, fp_after)

    def test_sensitive_column_is_sanitized(self) -> None:
        # Sensitive values should be transformed into fixed-length hashes.
        df = generate_base_dataset(n_samples=50, seed=2)
        df["email"] = [f"person{i}@sample.org" for i in range(len(df))]
        sanitized = sanitize_sensitive_columns(df)
        self.assertNotEqual(df["email"].iloc[0], sanitized["email"].iloc[0])
        self.assertEqual(len(str(sanitized["email"].iloc[0])), 16)

    def test_guardrail_blocks_unsafe_actions(self) -> None:
        # Known high-risk action should be blocked outright.
        report = evaluate_agent_plan(
            [
                AgentAction(
                    action="public_upload_raw_export",
                    target="public_gist",
                    rationale="share results quickly",
                )
            ]
        )
        self.assertEqual(report["summary"]["blocked"], 1)


if __name__ == "__main__":
    unittest.main()
