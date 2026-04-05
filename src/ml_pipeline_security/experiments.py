"""End-to-end empirical security experiments for ML pipelines."""

from __future__ import annotations

from typing import Any, Dict

from .data_generation import (
    generate_base_dataset,
    inject_label_poisoning,
    inject_sensitive_column,
    make_trigger_set,
    split_dataset,
)
from .modeling import attack_success_rate, evaluate_model, train_numpy_logreg, train_sklearn_logreg
from .security_controls import (
    build_control_audit_log,
    compute_dataset_fingerprint,
    sanitize_sensitive_columns,
    validate_schema,
    remove_anomalies,
    remove_suspicious_trigger_rows,
)

NON_FEATURE_COLUMNS = ["record_id", "email"]


def run_security_research_experiment(seed: int = 7) -> Dict[str, Any]:
    """Run unsecured vs secured pipeline experiment and return structured results."""
    # 1) Build and poison data to simulate a realistic attack scenario.
    base_df = generate_base_dataset(seed=seed)
    base_df = inject_sensitive_column(base_df, seed=seed)

    splits = split_dataset(base_df, seed=seed)
    poisoned_train = inject_label_poisoning(splits.train_df, poison_fraction=0.1, trigger_value=5.0, seed=seed)

    insecure_model = train_sklearn_logreg(poisoned_train)
    insecure_eval = evaluate_model(insecure_model, splits.test_df)
    trigger_test, _ = make_trigger_set(splits.test_df, trigger_value=5.0)
    insecure_asr = attack_success_rate(insecure_model, trigger_test, target_label=0)

    # 2) Apply defense-in-depth controls to the same poisoned training split.
    schema_ok, schema_issues = validate_schema(poisoned_train.drop(columns=NON_FEATURE_COLUMNS))
    fingerprint_before = compute_dataset_fingerprint(poisoned_train)
    sanitized_train = sanitize_sensitive_columns(poisoned_train)
    filtered_train, removed_rows_anomaly = remove_anomalies(sanitized_train, contamination=0.05, seed=seed)
    trigger_filtered_train, removed_rows_trigger = remove_suspicious_trigger_rows(filtered_train)
    fingerprint_after = compute_dataset_fingerprint(trigger_filtered_train)

    secure_model = train_sklearn_logreg(trigger_filtered_train, class_weight="balanced")
    secure_eval = evaluate_model(secure_model, splits.test_df)
    secure_asr = attack_success_rate(secure_model, trigger_test, target_label=0)

    # 3) Keep the NumPy model as "under-the-hood" evidence in the report.
    numpy_reference_model = train_numpy_logreg(trigger_filtered_train)
    numpy_reference_eval = evaluate_model(numpy_reference_model, splits.test_df)

    control_log = build_control_audit_log(
        schema_ok=schema_ok,
        schema_issues=schema_issues,
        removed_rows=removed_rows_anomaly + removed_rows_trigger,
        fingerprint=fingerprint_after,
    )

    return {
        "experiment": "poisoning_and_guardrail_validation",
        "unsecured_pipeline": {
            "model": insecure_model.model_name,
            "metrics": insecure_eval,
            "attack_success_rate": insecure_asr,
        },
        "secured_pipeline": {
            "model": secure_model.model_name,
            "metrics": secure_eval,
            "attack_success_rate": secure_asr,
        },
        "control_evidence": {
            "schema_ok": schema_ok,
            "schema_issues": schema_issues,
            "rows_removed_by_anomaly_control": removed_rows_anomaly,
            "rows_removed_by_trigger_filter": removed_rows_trigger,
            "fingerprint_before_controls": fingerprint_before,
            "fingerprint_after_controls": fingerprint_after,
            "audit_log": control_log,
            "numpy_reference_metrics": numpy_reference_eval,
        },
    }
