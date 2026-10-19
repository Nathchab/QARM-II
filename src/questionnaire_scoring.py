"""Pure questionnaire validation, scoring and descriptive classification."""
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from statistics import mean

from src.questionnaire_definitions import DIMENSION_LABELS, QUESTIONS
from src.questionnaire_routing import applicable_questions


class QuestionnaireError(ValueError):
    def __init__(self, errors):
        self.errors = errors
        super().__init__("; ".join(f"{key}: {value}" for key, value in errors.items()))


@dataclass(frozen=True)
class ScoreResult:
    scores: dict[str, float | None]
    subscores: dict[str, dict[str, float]]
    applicable: tuple[str, ...]


def finite_number(value, minimum=None, maximum=None):
    return (isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)
            and (minimum is None or value >= minimum)
            and (maximum is None or value <= maximum))


def question_error(code, value):
    q = QUESTIONS[code]
    if q.multi:
        if not isinstance(value, (list, tuple)):
            return "Answer this question."
        if any(not isinstance(item, str) or item not in q.options for item in value):
            return "Unknown answer choice."
        if len(set(value)) != len(value):
            return "Duplicate choices are invalid."
        if "none" in value and len(value) != 1:
            return "None of these cannot be combined with other choices."
        if not value and code not in ("QG1", "QJ2"):
            return "Select an answer (or explicit None of these where offered)."
        if q.max_selections is not None and len(value) > q.max_selections:
            return f"Choose at most {q.max_selections}."
    elif not isinstance(value, str) or value not in q.options:
        return "Choose one answer."
    return None


def validate_views(views, view_type):
    if not isinstance(views, list) or not views:
        return {"QH5": "Enter at least one quantitative view.", "QH6": "Confidence is required for each entered view."}
    errors = {}
    for i, view in enumerate(views, 1):
        key = f"QH5[{i}]"
        if not isinstance(view, dict):
            errors[key] = "Invalid view."
            continue
        asset = view.get("asset")
        if not isinstance(asset, str) or not asset.strip():
            errors[key] = "Identify the asset or basket."
        if not finite_number(view.get("value")):
            errors[key] = "Enter a finite annual return/premium in percentage points."
        if view_type == "c":
            relative_to = view.get("relative_to")
            if not isinstance(relative_to, str) or not relative_to.strip():
                errors[key] = "Identify the comparison asset or basket."
            elif isinstance(asset, str) and relative_to.strip().casefold() == asset.strip().casefold():
                errors[key] = "Relative view needs distinct assets/baskets."
        elif view.get("measure") not in ("return", "premium"):
            errors[key] = "Specify expected return or premium."
        confidence = view.get("confidence")
        if not isinstance(confidence, str) or confidence not in QUESTIONS["QH6"].options:
            errors[f"QH6[{i}]"] = "Choose confidence for this view."
    return errors


def validate_answers(answers):
    if not isinstance(answers, dict):
        return {"questionnaire": "Answers must be a dictionary."}
    errors = {}
    if "advanced_enabled" in answers and not isinstance(answers["advanced_enabled"], bool):
        errors["advanced_enabled"] = "Advanced-section opt-in must be true or false."
    active = applicable_questions(answers)
    for code in active:
        if code in ("QH5", "QH6"):
            continue
        error = question_error(code, answers.get(code))
        if error:
            errors[code] = error
    if "QH5" in active:
        errors.update(validate_views(answers.get("QH5"), answers.get("QH4")))
    if "QF4" in active and answers.get("QF4") == "d":
        if not finite_number(answers.get("QF4_limit"), 0.000001, 100):
            errors["QF4_limit"] = "Specify a custom security limit above 0 and at most 100%."
    if "QH2" in active and answers.get("QH2") == "d":
        if not isinstance(answers.get("QH2_custom"), str) or not answers["QH2_custom"].strip():
            errors["QH2_custom"] = "Identify the custom benchmark."
    if "QH3" in active and answers.get("QH3") == "d":
        if not finite_number(answers.get("QH3_limit"), 15):
            errors["QH3_limit"] = "Specify a finite annualized tracking-error limit of at least 15%."
    # Preserve the answers and details, without translating them into optimizer rules.
    for code in ("QF5", "QF6"):
        custom = (code == "QF5" and answers.get(code) in ("b", "c")) or (code == "QF6" and answers.get(code) == "d")
        if code in active and custom:
            details = answers.get(f"{code}_details")
            if not isinstance(details, str) or not details.strip():
                errors[f"{code}_details"] = "Describe the requested limits; these are recorded only."
    return errors


def score_answers(answers):
    errors = validate_answers(answers)
    if errors:
        raise QuestionnaireError(errors)
    active = applicable_questions(answers)
    subscores = {dimension: {} for dimension in ("RT", "EX", "KN", "HZ", "CX", "CV")}
    for code in active:
        if code in ("QH5", "QH6"):
            continue
        q = QUESTIONS[code]
        values = answers[code] if q.multi else [answers[code]]
        for dimension in subscores:
            levels = [q.options[item].scores[dimension] for item in values
                      if dimension in q.options[item].scores]
            if levels:
                subscores[dimension][code] = max(levels)
    if "QH5" in active:
        for i, view in enumerate(answers["QH5"], 1):
            subscores["CV"][f"QH6[{i}]"] = QUESTIONS["QH6"].options[view["confidence"]].scores["CV"]
    scores = {dimension: mean(parts.values()) if parts else None for dimension, parts in subscores.items()}
    if any(score is not None and not finite_number(score, 0, 100) for score in scores.values()):
        raise ValueError("Question definitions produced a score outside [0, 100].")
    return ScoreResult(scores, subscores, active)


def score_band(value):
    if not finite_number(value, 0, 100):
        raise ValueError("Score must be finite and in [0, 100].")
    return 0 if value <= 25 else 1 if value <= 50 else 2 if value <= 75 else 3


def dimension_labels(scores):
    return {dimension: labels[score_band(scores[dimension])]
            for dimension, labels in DIMENSION_LABELS.items()}


def illustrative_risk_matches(scores):
    """Preserve overlapping illustrative rules without inventing priority or filling gaps."""
    rt, hz = scores["RT"], scores["HZ"]
    matches = []
    if rt <= 35 or hz <= 35:
        matches.append("Guided conservative")
    if 36 <= rt <= 65 and hz >= 40:
        matches.append("Balanced long-term")
    if rt >= 60 and hz >= 60:
        matches.append("Growth-oriented")
    return tuple(matches)
