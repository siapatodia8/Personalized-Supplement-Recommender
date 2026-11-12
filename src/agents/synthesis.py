"""
Synthesis Agent - Final Response Generator

Consumes all specialist outputs in the ConversationState and produces a single
patient-facing final answer.

Design goals:
- Grounded: use only the specialist result dicts, the normalizer's match results (to flag
  unrecognized names) and the cleaned patient lists (no external biomedical knowledge)
- Robust: if there is no API key, or the Claude call fails or returns nothing, generate a
  deterministic fallback answer
- Minimal state writes: sets final_answer and appends one evidence_chain entry; on the LLM
  paths it also sets error_message to None (it is never set to an error message here)
"""

import json  # STALE CODE: only used by the unused _safe_json below
import logging
import os
from typing import Any, Dict, List, Optional  # STALE CODE: Optional is never used in this file

from anthropic import Anthropic

from src.prompt_loader import load_prompt

logger = logging.getLogger(__name__)


# STALE CODE (unused): dumps any object to JSON text without raising. Nothing calls it; likely left
# over from an earlier version that put raw results in the prompt (the prompt now gets the text
# built by _build_findings_summary instead). Kept as is on purpose.
def _safe_json(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, indent=2, default=str)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)


def _truncate(text: str, max_chars: int) -> str:
    if not text:
        return ""
    text = str(text)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."

# =====================================================================
# PRE-PROCESSING — build structured findings summary for the prompt
# Groups safety interactions by supplement↔medication pair so the
# LLM doesn't repeat the same concern from multiple pathways.
# =====================================================================

def _build_evidence_path(ix: Dict[str, Any]) -> str:
    """
    Build a readable arrow chain evidence path from a single interaction row.

    The safety query (SAFETY_QUERY in src/tools/safety_check.py) returns 6 fields per row:
    supplement, target, description, severity, detail, pathway. This function uses only 4 of
    them: supplement, target, pathway and detail (detail may be null).

    Pathway names come from SAFETY_QUERY: "Direct Interaction", "Equivalent Active Ingredient",
    "Drug-Drug Interaction via Equivalent", "Reverse Drug-Drug Interaction via Equivalent" and
    "Similar Effect Category". The detail text is built inside the query with "->" arrows
    (e.g. "contains -> <ingredient> -> equivalent to -> <drug>"), while the arrows added here
    are "→", so one path can mix both styles.

    Args:
        ix: Single interaction row dict

    Returns:
        Arrow chain string e.g.
        "Fish Oil → [Equivalent Active Ingredient] → contains -> <ingredient> -> equivalent to -> <drug> → <medication>"
        "Fish Oil → [Direct Interaction] → directly interacts with -> Warfarin → Warfarin"
        When detail is null: "<supplement> → [<pathway>] → <medication>"
    """
    supplement = ix.get('supplement', '?')
    target = ix.get('target', '?')
    pathway = ix.get('pathway', 'Unknown Pathway')
    detail = ix.get('detail')

    if detail:
        return f"{supplement} → [{pathway}] → {detail} → {target}"
    else:
        return f"{supplement} → [{pathway}] → {target}"

def _build_findings_summary(state: Dict[str, Any]) -> str:
    """Build a human-readable findings summary from all specialist results."""
    lines = []

    # ── Unrecognized entities — detected from normalizer output ──  
    normalized_supps = state.get("normalized_supplements") or []
    unrecognized_supps = [
        s['user_input'] for s in normalized_supps
        if s.get('confidence') in ('NOT_FOUND', 'LOW')
    ]
    if unrecognized_supps:
        lines.append(
            f"SAFETY FINDINGS: The following supplement(s) were not recognized "
            f"in our database and could not be checked for interactions: "
            f"{', '.join(unrecognized_supps)}. Safety cannot be confirmed."
        )

    normalized_meds = state.get("normalized_medications") or []
    unrecognized_meds = [
        m['user_input'] for m in normalized_meds
        if m.get('confidence') in ('NOT_FOUND', 'LOW')
    ]
    if unrecognized_meds:
        lines.append(
            f"SAFETY FINDINGS: The following medication(s) were not recognized "
            f"in our database and could not be checked for interactions: "
            f"{', '.join(unrecognized_meds)}. Safety cannot be confirmed."
        )
    
    # ── No supplements in profile ──
    if not (state.get("supplements_list") or []) and not unrecognized_supps:
        lines.append(
            "SAFETY FINDINGS: No supplements were identified. "
            "There are no supplements to check for interactions."
        )
    
    # ── No medications in profile ──
    if not (state.get("medications_list") or []):
        lines.append(
            "SAFETY FINDINGS: No medications were identified. "
            "Supplement safety cannot be checked without knowing what medications are being taken."
        )

    # ── Safety (grouped by pair, with evidence paths) ──
    safety = state.get("safety_results") or {}
    interactions = safety.get("interactions") or []

    if safety.get("status") == "error":
        lines.append(
            "SAFETY FINDINGS: Safety check failed — interaction queries could not be executed. "
            "The absence of findings here does NOT mean this supplement is safe. "
            "Do not draw any conclusions about safety from this result."
        )

    elif interactions:
        # Group by supplement↔medication pair
        pairs: Dict[str, List[Dict]] = {}
        pair_severity: Dict[str, str] = {}

        for ix in interactions:
            key = f"{ix.get('supplement', '?')} and {ix.get('target', '?')}"
            pairs.setdefault(key, []).append(ix)

            # Track highest severity per pair
            sev = ix.get("severity", "MODERATE")
            if sev == "HIGH":
                pair_severity[key] = "HIGH"
            elif key not in pair_severity:
                pair_severity[key] = sev

        lines.append("SAFETY FINDINGS:")
        for pair, ix_list in pairs.items():
            sev = pair_severity.get(pair, "MODERATE")

            # Build description block from raw fields — LLM narrates from these
            finding_parts = []
            evidence_paths = []

            for ix in ix_list:
                description = ix.get('description', '')
                detail = ix.get('detail', '')
                pathway = ix.get('pathway', '')

                # Structured finding line
                finding_line = f"Pathway: {pathway}"
                if description:
                    finding_line += f" | Finding: {description}"
                if detail:
                    finding_line += f" | Detail: {detail}"
                finding_parts.append(finding_line)

                # Arrow chain evidence path
                evidence_paths.append(_build_evidence_path(ix))

            findings_block = " ; ".join(finding_parts)
            paths_block = " | EVIDENCE PATHS: " + " ; ".join(evidence_paths) if evidence_paths else ""

            lines.append(f"  [{sev}] {pair}: {findings_block}{paths_block}")

    elif safety:
        lines.append(f"SAFETY FINDINGS: {safety.get('summary', 'No interactions found.')}")

    # ── Deficiency ──
    deficiency = state.get("deficiency_results") or {}
    all_at_risk = deficiency.get("all_at_risk_details") or {}
    critical_nutrients = {o.get("nutrient") for o in (deficiency.get("critical_overlaps") or [])}
    if all_at_risk:
        lines.append("\nDEFICIENCY FINDINGS:")
        for nutrient, sources in all_at_risk.items():
            source_strs = []
            for s in sources:
                name, mech = s.get("source_name", ""), s.get("mechanism", "")
                if s.get("source_type") == "diet":
                    source_strs.append(f"your {name} diet")
                elif mech:
                    source_strs.append(f"{name} ({mech})")
                else:
                    source_strs.append(name)
            overlap = " [COMPOUNDED RISK — multiple sources]" if nutrient in critical_nutrients else ""
            lines.append(f"  {nutrient}: at risk due to {', '.join(source_strs)}{overlap}")
    elif deficiency:
        lines.append(f"\nDEFICIENCY FINDINGS: {deficiency.get('summary', 'No deficiencies found.')}")

    # ── Recommendations ──
    recs = state.get("recommendation_results") or {}
    candidates = recs.get("recommendations") or []
    if candidates:
        flagged = {ix.get("supplement", "").lower() for ix in interactions}
        lines.append("\nRECOMMENDATION FINDINGS:")
        for c in candidates:
            name = c.get("supplement_name", "Unknown")
            rating = c.get("safety_rating", "unknown")
            symptom = c.get("symptom_treated", "")
            flag_note = " [WARNING: also has safety concern above]" if name.lower() in flagged else ""
            symptom_note = f" for {symptom}" if symptom else ""
            lines.append(f"  {name}: safety rating '{rating}'{symptom_note}{flag_note}")
    elif recs:
        lines.append(f"\nRECOMMENDATION FINDINGS: {recs.get('summary', 'No candidates found.')}")

    return "\n".join(lines) if lines else "No specialist findings available."

# =====================================================================
# PROMPT BUILDER
# =====================================================================

def _build_prompt(state: Dict[str, Any]) -> str:
    """Build a grounded prompt with pre-processed findings."""
    meds = ", ".join(state.get("medications_list", [])) or "none"
    supps = ", ".join(state.get("supplements_list", [])) or "none"
    conditions = ", ".join(state.get("conditions_list", [])) or "none"
    restrictions = ", ".join(state.get("dietary_restrictions_list", [])) or "none"

    ran = [n for n, k in [("Safety", "safety_checked"), ("Deficiency", "deficiency_checked"),
                           ("Recommendations", "recommendations_checked")] if state.get(k)]
    skipped = [n for n, k in [("Safety", "safety_checked"), ("Deficiency", "deficiency_checked"),
                               ("Recommendations", "recommendations_checked")] if not state.get(k)]

    findings = _build_findings_summary(state)

    return load_prompt("synthesis")["synthesis"].format(
        meds=meds, supps=supps, conditions=conditions, restrictions=restrictions,
        question=state.get('user_question', ''),
        findings=findings,
        ran=', '.join(ran) or 'none',
        skipped=', '.join(skipped) or 'none',
    )


# =====================================================================
# DETERMINISTIC FALLBACK (used when no API key or LLM fails)
# =====================================================================

def _deterministic_fallback_answer(state: Dict[str, Any]) -> str:
    question = state.get("user_question", "")
    meds = state.get("medications_list", []) or []
    supps = state.get("supplements_list", []) or []
    conditions = state.get("conditions_list", []) or []
    restrictions = state.get("dietary_restrictions_list", []) or []

    safety = state.get("safety_results") or {}
    deficiency = state.get("deficiency_results") or {}
    recs = state.get("recommendation_results") or {}

    lines: List[str] = []
    lines.append("## Summary")
    lines.append(f"- **Question**: {_truncate(question, 400)}" if question else "- **Question**: (not provided)")
    lines.append(f"- **Medications**: {', '.join(meds) if meds else 'none'}")
    lines.append(f"- **Current supplements**: {', '.join(supps) if supps else 'none'}")
    lines.append(f"- **Conditions/goals**: {', '.join(conditions) if conditions else 'none'}")
    lines.append(f"- **Dietary restrictions**: {', '.join(restrictions) if restrictions else 'none'}")

    lines.append("")
    lines.append("## Safety check")
    if safety:
        lines.append(f"- **Result**: {safety.get('summary', '(no summary)')}")
        interactions = safety.get("interactions") or []
        if interactions:
            lines.append("- **Interactions found**:")
            for ix in interactions[:10]:
                pathway = ix.get("pathway", "UNKNOWN")
                sev = ix.get("severity", "UNKNOWN")
                supp = ix.get("supplement", "")
                target = ix.get("target", "")
                desc = ix.get("description", "")
                detail = ix.get("detail")
                detail_str = f" ({detail})" if detail else ""
                evidence = _build_evidence_path(ix)
                lines.append(
                    f"  - [{sev}] {supp} ↔ {target} — {pathway}: "
                    f"{_truncate(desc, 220)}{detail_str}"
                )
                lines.append(f"    Evidence: {evidence}")
            if len(interactions) > 10:
                lines.append(f"  - …and {len(interactions) - 10} more")
        else:
            lines.append("- **Interactions found**: none reported by the tool")
    else:
        lines.append("- Safety specialist did not run in this trace.")

    lines.append("")
    lines.append("## Nutrient deficiency risk check")
    if deficiency:
        lines.append(f"- **Result**: {deficiency.get('summary', '(no summary)')}")
        overlaps = deficiency.get("critical_overlaps") or []
        at_risk = deficiency.get("all_at_risk") or []
        if overlaps:
            lines.append("- **Critical overlaps (2+ sources affecting same nutrient)**:")
            for ov in overlaps[:10]:
                nutrient = ov.get("nutrient", "Unknown nutrient")
                warning = ov.get("warning", "")
                lines.append(f"  - {nutrient}: {_truncate(warning, 240)}")
            if len(overlaps) > 10:
                lines.append(f"  - …and {len(overlaps) - 10} more")
        elif at_risk:
            lines.append(f"- **Nutrients at risk**: {', '.join(at_risk[:25])}")
            if len(at_risk) > 25:
                lines.append(f"  - …and {len(at_risk) - 25} more")
        else:
            lines.append("- **Nutrients at risk**: none reported by the tool")
    else:
        lines.append("- Deficiency specialist did not run in this trace.")

    lines.append("")
    lines.append("## Candidate supplement recommendations")
    if recs:
        lines.append(f"- **Result**: {recs.get('summary', '(no summary)')}")
        candidates = recs.get("recommendations") or []
        if candidates:
            lines.append("- **Top candidates (from KG `TREATS` edges)**:")
            for c in candidates[:10]:
                name = c.get("supplement_name", "Unknown")
                rating = c.get("safety_rating", "unknown")
                treated = c.get("symptom_treated", "")
                treated_str = f" (treats: {treated})" if treated else ""
                lines.append(f"  - {name} — safety_rating: {rating}{treated_str}")
            if len(candidates) > 10:
                lines.append(f"  - …and {len(candidates) - 10} more")
        else:
            lines.append("- **Candidates**: none reported by the tool")
    else:
        lines.append("- Recommendation specialist did not run in this trace.")

    lines.append("")
    lines.append("## Notes")
    lines.append(
        "- This answer is synthesized from the project's specialist tool outputs. "
        "If something isn't present above, it wasn't found by the tools (or the tool didn't run)."
    )

    return "\n".join(lines).strip() + "\n"



# =====================================================================
# LANGGRAPH NODE
# =====================================================================

def synthesis_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: Generate the final answer from all specialist outputs.

    Reads from state:
      - user_question and the clean lists (medications_list, supplements_list,
        conditions_list, dietary_restrictions_list)
      - normalized_medications, normalized_supplements (to flag unrecognized names)
      - safety_checked, deficiency_checked, recommendations_checked
      - safety_results, deficiency_results, recommendation_results
      - evidence_chain (appended to, not used as content)

    Writes to state:
      - final_answer
      - evidence_chain (append one synthesis step)
      - error_message (set to None on the LLM path and when falling back after an LLM failure;
        never set to an error message here: an LLM failure is recorded only in evidence_chain.
        The no-API-key path does not set it at all)
    """
    print("\n" + "=" * 60)
    print("🧾 SYNTHESIS: Generating final answer...")
    print("=" * 60)

    prompt = _build_prompt(state)
    evidence_chain = state.get("evidence_chain", []) or []

    # If the API key isn't present, we still want the workflow to succeed.
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        answer = _deterministic_fallback_answer(state)
        print("   ⚠️  No ANTHROPIC_API_KEY found — using deterministic fallback synthesis")
        return {
            "final_answer": answer,
            "evidence_chain": evidence_chain + ["Synthesis: generated final answer (fallback, no API key)"],
        }

    client = Anthropic(api_key=api_key)

    try:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        answer_text = (response.content[0].text or "").strip()
        if not answer_text:
            raise ValueError("Empty synthesis response from LLM")

        print("   ✅ Final answer generated")
        print("=" * 60 + "\n")
        return {
            "final_answer": answer_text,
            "evidence_chain": evidence_chain + ["Synthesis: generated final answer (LLM)"],
            "error_message": None,
        }
    except Exception as e:
        logger.error(f"Synthesis LLM call failed: {e}")
        answer = _deterministic_fallback_answer(state)
        print("   ⚠️  Synthesis LLM failed — using deterministic fallback synthesis")
        print("=" * 60 + "\n")
        return {
            "final_answer": answer,
            "evidence_chain": evidence_chain + [f"Synthesis: LLM failed, used fallback ({type(e).__name__})"],
            "error_message": None,
        }
