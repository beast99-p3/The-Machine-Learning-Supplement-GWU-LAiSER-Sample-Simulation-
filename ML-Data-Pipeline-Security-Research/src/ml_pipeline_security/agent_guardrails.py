"""Agent behavior simulation with security guardrail enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class AgentAction:
    """Single action proposed by an autonomous AI agent."""

    action: str
    target: str
    rationale: str


BLOCKED_ACTION_KEYWORDS = ("disable_audit", "public_upload", "raw_export", "share_token", "delete_logs")
RESTRICTED_TARGET_KEYWORDS = ("public", "external", "gist", "pastebin")


def guardrail_decision(action: AgentAction) -> Dict[str, Any]:
    """Evaluate whether an action should be blocked, escalated, or allowed."""
    low_action = action.action.lower()
    low_target = action.target.lower()

    # Hard deny-list for clearly unsafe operations.
    for keyword in BLOCKED_ACTION_KEYWORDS:
        if keyword in low_action:
            return {
                "decision": "blocked",
                "reason": f"action contains blocked keyword: {keyword}",
                "action": action,
            }

    # Soft restrictions route risky external actions to human approval.
    for keyword in RESTRICTED_TARGET_KEYWORDS:
        if keyword in low_target:
            return {
                "decision": "escalate_for_approval",
                "reason": f"target appears external: {keyword}",
                "action": action,
            }

    return {
        "decision": "allowed",
        "reason": "no policy violations detected",
        "action": action,
    }


def evaluate_agent_plan(actions: List[AgentAction]) -> Dict[str, Any]:
    """Run policy checks over a full agent plan."""
    decisions = [guardrail_decision(a) for a in actions]

    summary = {
        "total_actions": len(actions),
        "blocked": sum(1 for d in decisions if d["decision"] == "blocked"),
        "escalated": sum(1 for d in decisions if d["decision"] == "escalate_for_approval"),
        "allowed": sum(1 for d in decisions if d["decision"] == "allowed"),
    }

    return {"summary": summary, "decisions": decisions}


def sample_agent_actions() -> List[AgentAction]:
    """Provide representative safe/unsafe actions for demonstration."""
    return [
        AgentAction(
            action="aggregate_metrics",
            target="internal_dashboard",
            rationale="Track model drift and data quality signals.",
        ),
        AgentAction(
            action="public_upload_raw_export",
            target="public_gist",
            rationale="Share debugging output quickly.",
        ),
        AgentAction(
            action="disable_audit_logging",
            target="training_cluster",
            rationale="Reduce overhead during experiments.",
        ),
        AgentAction(
            action="request_model_retrain",
            target="internal_mlops_queue",
            rationale="Respond to detected drift.",
        ),
    ]
