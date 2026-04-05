# ML-Data-Pipeline-Security-Research

A sanitized, portfolio-ready project that demonstrates empirical AI/ML security research across the end-to-end pipeline:

- Data collection and wrangling with Pandas/NumPy.
- Attack simulation (label poisoning and trigger manipulation).
- Security controls for schema, sensitive data handling, and anomaly filtering.
- Model training and evaluation, including a NumPy logistic regression implementation to show under-the-hood ML understanding.
- Guardrails for AI-agent behavior to reduce risky actions.

## Why this project is useful for AI security roles

This sample mirrors real research workflows where you must:

- Design experiments around ambiguous security questions.
- Validate controls with measurable outcomes.
- Handle sensitive data responsibly across collection, training, and inference.
- Investigate failure modes in ML systems and autonomous agent workflows.

## Repository structure

```text
ML-Data-Pipeline-Security-Research/
  README.md
  requirements.txt
  scripts/
    run_research_demo.py
  src/ml_pipeline_security/
    __init__.py
    data_generation.py
    security_controls.py
    modeling.py
    experiments.py
    agent_guardrails.py
  tests/
    test_controls.py
```

## Quick start

Run the commands below from the repository root:

```bash
cd ML-Data-Pipeline-Security-Research
```

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the demo:

```bash
python scripts/run_research_demo.py
```

If you are currently in the parent folder (for example `LAISER/`), run:

```bash
python ML-Data-Pipeline-Security-Research/scripts/run_research_demo.py
```

4. Run tests:

```bash
python -m unittest discover -s tests -p "test_*.py"
```

## Research outputs generated

Running the demo produces:

- Comparative metrics for unsecured vs secured pipelines.
- Attack success rate (ASR) under trigger manipulation.
- Evidence of control effects (rows filtered, schema checks, integrity fingerprint changes).
- AI-agent guardrail decision logs.

## Example findings from a demo run

In one representative run (seeded and reproducible):

- Unsecured pipeline attack success rate (ASR): 0.878
- Secured pipeline attack success rate (ASR): 0.000
- Unsecured F1: 0.449
- Secured F1: 0.648
- Rows removed by controls: 187

This demonstrates an empirical security workflow: attack simulation, control design, and measured robustness gains.

## Suggested resume/interview framing

- "Built a reproducible Python framework to evaluate ML data-pipeline security controls against poisoning and anomaly-based attacks."
- "Implemented end-to-end data protections (schema validation, sensitive-field sanitization, integrity checks) and measured impact on model robustness."
- "Developed and tested guardrail policies for autonomous agent actions, preventing unsafe operations such as data exfiltration and disabled audit controls."

## Notes on sanitization and scope

- This repository is intentionally synthetic and sanitized.
- No proprietary data, credentials, internal architecture diagrams, or confidential identifiers are included.
- The code is structured to showcase methods and experimental thinking rather than organization-specific implementation details.
