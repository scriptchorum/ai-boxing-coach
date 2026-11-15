"""T14: Female usability study and 7-day retention follow-up aggregation (G3 gate)."""
from __future__ import annotations

from typing import Any, Sequence

USABILITY_PROTOCOL_VERSION = "g3-usability-0.1.0"
VALID_REUSE_TYPES = ("proactive", "reminded", "none")


def summarize_usability_study(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate T14 usability trial records with strict raw counts and 7-day maturity separation."""
    seen_ids: set[str] = set()
    mature_ids: list[str] = []
    immature_ids: list[str] = []
    independent_pass_count = 0
    understood_limits_count = 0
    proactive_reuse_count = 0
    reminded_reuse_count = 0
    no_reuse_count = 0
    improvement_backlog: list[dict[str, str]] = []

    for rec in records:
        pid = str(rec.get("participant_id", "")).strip()
        if not pid:
            raise ValueError("participant_id must be a non-empty string.")
        if pid in seen_ids:
            raise ValueError(f"Duplicate participant_id: {pid}")
        seen_ids.add(pid)

        days = int(rec.get("days_since_onboarding", 0))
        reuse = str(rec.get("reuse_type", "none")).lower()
        if reuse not in VALID_REUSE_TYPES:
            raise ValueError(f"Invalid reuse_type: {reuse}")

        if bool(rec.get("completed_independent_task", False)):
            independent_pass_count += 1
        if bool(rec.get("understood_limitations", False)):
            understood_limits_count += 1

        if days >= 7:
            mature_ids.append(pid)
            if reuse == "proactive":
                proactive_reuse_count += 1
            elif reuse == "reminded":
                reminded_reuse_count += 1
            else:
                no_reuse_count += 1
        else:
            immature_ids.append(pid)

        for issue in rec.get("issues", []):
            if isinstance(issue, str) and issue.strip():
                improvement_backlog.append({"participant_id": pid, "issue": issue.strip()})

    total = len(records)
    mature_count = len(mature_ids)
    immature_count = len(immature_ids)

    g3_ready = 5 <= total <= 8 and mature_count >= 5
    return {
        "protocol_version": USABILITY_PROTOCOL_VERSION,
        "g3_gate_status": "evaluated" if g3_ready else "pending_external_5_to_8_female_participants",
        "total_participants_raw": total,
        "mature_7_day_count_raw": mature_count,
        "immature_under_7_day_count_raw": immature_count,
        "mature_participant_ids": mature_ids,
        "immature_participant_ids": immature_ids,
        "independent_task_success_raw": f"{independent_pass_count}/{total}",
        "understood_limitations_raw": f"{understood_limits_count}/{total}",
        "mature_proactive_reuse_raw": f"{proactive_reuse_count}/{mature_count}",
        "mature_reminded_reuse_raw": f"{reminded_reuse_count}/{mature_count}",
        "mature_no_reuse_raw": f"{no_reuse_count}/{mature_count}",
        "proactive_reuse_count": proactive_reuse_count,
        "reminded_reuse_count": reminded_reuse_count,
        "no_reuse_count": no_reuse_count,
        "improvement_items": improvement_backlog,
        "claims_notice": (
            "本统计仅记录独立操作完成度、限制说明理解度及满 7 天后的主动/提醒复用原始人数；"
            "不宣称专业教学效果或动作改善率。"
        ),
    }
