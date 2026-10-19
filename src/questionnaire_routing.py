"""Applicability only: routing never adds weights to RT or financial capacity."""
from src.questionnaire_definitions import QUESTIONS


def applicable_questions(answers):
    """Return questions in document order; dormant answers remain stored but unused.

    Ambiguous advanced/alternative eligibility uses the user's explicit opt-in.
    Institutional questions require QA1=d; no inference from wealth/asset count.
    """
    active = set(QUESTIONS) - {
        "QB3", "QC4", "QF2", "QF3", "QF5", "QF6", "QH1", "QH2", "QH3",
        "QH4", "QH5", "QH6", "QI1", "QI2", "QJ1", "QJ2",
    }
    advanced = answers.get("advanced_enabled") is True
    if answers.get("QB1") in ("b", "c", "d", "e"):
        active.add("QB3")
    if answers.get("QC1") in ("c", "d", "e"):
        active.add("QC4")
    universe = answers.get("QF1")
    universe = universe if isinstance(universe, (list, tuple)) else ()
    if "h" in universe:
        active.add("QF2")
    if "g" in universe:
        active.add("QF3")
    if advanced:
        active.update(("QF5", "QF6", "QI2"))
    if answers.get("QA2") in ("a", "b"):
        active.add("QH1")
    if answers.get("QA2") == "f":
        active.add("QH2")
        if isinstance(answers.get("QH2"), str) and answers["QH2"] in QUESTIONS["QH2"].options:
            active.add("QH3")
    if answers.get("QA2") == "g" or advanced:
        active.add("QH4")
        if answers.get("QH4") in ("b", "c"):
            active.update(("QH5", "QH6"))
    if answers.get("QC1") in ("c", "d", "e") or advanced:
        active.add("QI1")
    if answers.get("QA1") == "d":
        active.update(("QJ1", "QJ2"))
    return tuple(code for code in QUESTIONS if code in active)


def portfolio_routes(answers, scores):
    """Return independent candidates, never dispatch an optimizer or force precedence."""
    routes = []
    if answers.get("QA1") in ("a", "b"):
        routes.append("Individual – beginner" if scores["EX"] < 40 or scores["KN"] < 40
                      else "Individual – experienced")
    if answers.get("QA2") == "f":
        routes.append("Benchmark-aware (tracking-error optimization unavailable)")
    active = applicable_questions(answers)
    if "QH5" in active and answers.get("QH5"):
        if scores["KN"] >= 60 and scores["CX"] >= 60:
            routes.append("Conviction-driven advanced (BL candidate; no automatic dispatch)")
        else:
            routes.append("Market views recorded (KN/CX advanced threshold not met)")
    if answers.get("QA1") == "d":
        routes.append("Institutional multi-asset (ERC available; custom budgeting/policy methods unavailable)")
    if not routes:
        routes.append("Constrained MVO candidate")
    return tuple(routes)
