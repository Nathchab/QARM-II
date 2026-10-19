"""Run offline: python -m unittest discover -s tests -v (no pytest dependency)."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import random
import re
import unittest
import xml.etree.ElementTree as ET
import zipfile

import numpy as np
import pandas as pd

from src.client_profile import ClientProfile
from src.questionnaire_definitions import QUESTIONS
from src.questionnaire_routing import applicable_questions, portfolio_routes
from src.questionnaire_scoring import (
    QuestionnaireError, dimension_labels, illustrative_risk_matches, score_answers,
    score_band, validate_answers,
)
from src.questionnaire_state import initialize_draft, publish_profile, save_widget


def completed_answers(**overrides):
    answers = {
        "QA1": "a", "QA2": "c", "QB1": "c", "QB2": ["a", "d"], "QB3": "c",
        "QC1": "b", "QC2": "b", "QC3": "a", "QD1": "d", "QD2": "d",
        "QE1": "c", "QE2": "c", "QE3": "c", "QE4": "c",
        "QF1": ["a", "b"], "QF4": "b", "QG1": ["a"], "QG2": "b",
        "advanced_enabled": False,
    }
    answers.update(overrides)
    for code in applicable_questions(answers):
        if code not in answers and code not in ("QH5", "QH6"):
            answers[code] = ["a"] if QUESTIONS[code].multi else "a"
    return answers


CAPACITY = dict(age="30-45", horizon="7-15 years", needs_money="No", income="Very stable", max_loss="-20%")


class DocumentParityTests(unittest.TestCase):
    def test_all_original_questions_options_and_scores_match_document(self):
        path = Path(__file__).resolve().parents[1] / "QARM_II_Questionnaire.docx"
        with zipfile.ZipFile(path) as document:
            root = ET.fromstring(document.read("word/document.xml"))
        ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
        paragraphs = ["".join(t.text or "" for t in p.findall(".//w:t", ns))
                      for p in root.findall(".//w:p", ns)]
        seen, current = set(), None
        seen_options = {code: set() for code in QUESTIONS}
        for text in paragraphs:
            match = re.match(r"(Q[A-J]\d)\. (.*)", text)
            if match:
                current, question_text = match.groups()
                seen.add(current)
                question_text = re.sub(r" \[.*\]$", "", question_text)
                self.assertEqual(QUESTIONS[current].text, question_text, current)
            elif current and (option := re.match(r"([a-i])\. (.*)", text)):
                code, content = option.groups()
                expected = {dimension: float(value) for dimension, value in
                            re.findall(r"\b(RT|EX|KN|HZ|CX|CV)(?: level)? (\d+)/100", content)}
                label = re.sub(r" \([^)]*\)$", "", content)
                self.assertEqual(QUESTIONS[current].options[code].text, label, (current, code))
                self.assertEqual(QUESTIONS[current].options[code].scores, expected, (current, code))
                seen_options[current].add(code)
        self.assertEqual(seen, set(QUESTIONS))
        for code, q in QUESTIONS.items():
            self.assertEqual(set(q.options) - {"none"}, seen_options[code])


class ScoringTests(unittest.TestCase):
    def test_exact_scores_and_subscore_audit(self):
        result = score_answers(completed_answers())
        self.assertEqual(result.scores["RT"], (55 + 55 + 65 + 50) / 4)
        self.assertEqual(result.scores["EX"], (45 + 40 + 50) / 3)
        self.assertEqual(result.scores["KN"], 75)
        self.assertEqual(result.scores["HZ"], 75)
        self.assertEqual(result.scores["CX"], (0 + 20 + 30 + 50) / 4)
        self.assertIsNone(result.scores["CV"])

    def test_all_rt_combinations_and_natural_extrema(self):
        scores = []
        for selected in product(*(QUESTIONS[q].options for q in ("QE1", "QE2", "QE3", "QE4"))):
            changes = dict(zip(("QE1", "QE2", "QE3", "QE4"), selected))
            result = score_answers(completed_answers(**changes))
            self.assertEqual(result.scores["RT"], sum(QUESTIONS[q].options[a].scores["RT"]
                                                    for q, a in changes.items()) / 4)
            scores.append(result.scores["RT"])
        self.assertEqual((min(scores), max(scores)), (5, 97.5))

    def test_multiselect_maxima_and_explicit_none(self):
        result = score_answers(completed_answers(QB2=["a", "h", "i"], QC1="c", QC4=["a", "f", "h"]))
        self.assertEqual(result.subscores["EX"]["QB2"], 100)
        self.assertEqual(result.subscores["KN"]["QC4"], 100)
        self.assertEqual(result.subscores["CX"]["QC4"], 100)
        result = score_answers(completed_answers(QB1="a", QB2=["none"], QC1="c", QC4=["none"]))
        self.assertEqual(result.scores["EX"], 0)
        self.assertEqual(result.subscores["KN"]["QC4"], 0)
        self.assertEqual(result.subscores["CX"]["QC4"], 0)
        for value in (None, [], ["none", "a"], ["a", "a"], ["unknown"], "a", [None], [["a"]]):
            with self.subTest(value=value), self.assertRaises(QuestionnaireError):
                score_answers(completed_answers(QB2=value))

    def test_hidden_scores_and_invalid_dormant_answers_are_excluded(self):
        answers = completed_answers(QB1="a", QB2=["none"], QB3="e", QC4=["h"],
                                    QH4="b", QH5=[{"value": float("nan")}], QF2="invalid", QI2="c")
        result = score_answers(answers)
        self.assertEqual(result.scores["EX"], 0)
        self.assertNotIn("QB3", result.subscores["CX"])
        self.assertNotIn("QC4", result.subscores["KN"])
        self.assertNotIn("QI2", result.subscores["CX"])
        self.assertIsNone(result.scores["CV"])

    def test_qb3_only_professional_answer_contributes_cx(self):
        for choice in "abcd":
            self.assertNotIn("QB3", score_answers(completed_answers(QB3=choice)).subscores["CX"])
        self.assertEqual(score_answers(completed_answers(QB3="e")).subscores["CX"]["QB3"], 100)

    def test_missing_invalid_single_choices_and_never_with_savings(self):
        for value in (None, [], {}, 3, True, "invalid"):
            with self.subTest(value=value), self.assertRaises(QuestionnaireError):
                score_answers(completed_answers(QE1=value))
        # The document does not prohibit savings for a person who never invested.
        self.assertEqual(score_answers(completed_answers(QB1="a", QB2=["a"])).scores["EX"], 5)
        for code in ("QE1", "QE2", "QE3", "QE4"):
            answers = completed_answers()
            del answers[code]
            with self.assertRaises(QuestionnaireError):
                score_answers(answers)
        self.assertIn("questionnaire", validate_answers(None))

    def test_multiselect_limits_and_explicit_empty_unscored_inputs(self):
        self.assertFalse(validate_answers(completed_answers(QG1=[])))
        self.assertIn("QG1", validate_answers(completed_answers(QG1=["a", "b", "c", "d"])))
        self.assertIn("QF1", validate_answers(completed_answers(QF1=[])))
        self.assertIn("QG1", validate_answers(completed_answers(QG1=None)))

    def test_cv_is_per_view_mean_and_only_when_applicable(self):
        views = [{"asset": "SPY", "value": 7, "measure": "return", "confidence": "a"},
                 {"asset": "EFA", "value": 3, "measure": "premium", "confidence": "d"}]
        answers = completed_answers(QA2="g", QH4="b", QH5=views)
        self.assertEqual(score_answers(answers).scores["CV"], 60)
        answers["QH4"] = "a"
        self.assertIsNone(score_answers(answers).scores["CV"])
        answers["QH4"] = "b"
        for bad in ([], [{"asset": "SPY", "value": 7}], [None],
                    [{"asset": "SPY", "value": float("inf"), "confidence": "a"}],
                    [{"asset": "", "value": 2, "confidence": "b"}]):
            answers["QH5"] = bad
            with self.subTest(bad=bad), self.assertRaises(QuestionnaireError):
                score_answers(answers)

    def test_relative_views_recorded_with_distinct_comparison(self):
        answers = completed_answers(QA2="g", QH4="c", QH5=[
            {"asset": "SPY", "relative_to": "EFA", "value": -2.5, "confidence": "c"}])
        self.assertEqual(score_answers(answers).scores["CV"], 80)
        answers["QH5"][0]["relative_to"] = "spy"
        self.assertTrue(validate_answers(answers))

    def test_custom_fields(self):
        answers = completed_answers(QA2="f", QH2="d", QH3="d", QF4="d")
        self.assertTrue({"QH2_custom", "QH3_limit", "QF4_limit"} <= validate_answers(answers).keys())
        answers.update(QH2_custom="Custom index", QH3_limit=17.5, QF4_limit=12.5)
        self.assertFalse(validate_answers(answers))
        answers["QH3_limit"] = float("nan")
        self.assertIn("QH3_limit", validate_answers(answers))

    def test_score_bands_fractional_boundaries(self):
        for value, band in [(0, 0), (25, 0), (25.01, 1), (50, 1), (50.1, 2), (75, 2), (75.01, 3), (100, 3)]:
            self.assertEqual(score_band(value), band)
        for invalid in (-1, 101, float("nan"), float("inf"), True, "50"):
            with self.assertRaises(ValueError):
                score_band(invalid)

    def test_all_dimension_scores_bounded_across_random_valid_branches(self):
        rng = random.Random(2026)
        for _ in range(300):
            answers = completed_answers()
            answers.update(QA1=rng.choice("abcd"), QA2=rng.choice("abcdefg"),
                           QB1=rng.choice("abcde"), QC1=rng.choice("abcde"), advanced_enabled=rng.choice([True, False]))
            for code in applicable_questions(answers):
                if code in ("QH5", "QH6"):
                    continue
                q = QUESTIONS[code]
                answers[code] = [rng.choice(list(q.options))] if q.multi else rng.choice(list(q.options))
            if answers["QB1"] == "a":
                answers["QB2"] = ["none"]
            answers.update(QF4_limit=10, QH2_custom="Test", QH3_limit=16,
                           QF5_details="Gross 120%, each short ≤ 10%", QF6_details="Gross 150%")
            # Recompute applicability after randomly changing parent choices.
            for _ in range(3):
                for code in applicable_questions(answers):
                    if code not in answers and code not in ("QH5", "QH6"):
                        answers[code] = ["a"] if QUESTIONS[code].multi else "a"
            if "QH5" in applicable_questions(answers):
                answers["QH5"] = [{"asset": "SPY", "relative_to": "EFA", "measure": "return",
                                   "value": 4.5, "confidence": rng.choice("abcd")}]
            for score in score_answers(answers).scores.values():
                self.assertTrue(score is None or 0 <= score <= 100)


class RoutingAndProfileTests(unittest.TestCase):
    def test_conditional_routes_and_opt_in(self):
        answers = completed_answers()
        active = applicable_questions(answers)
        self.assertNotIn("QC4", active)
        self.assertNotIn("QF5", active)
        self.assertNotIn("QJ1", active)
        answers.update(QC1="d", QF1=["g", "h"], QA2="f", QH2="a")
        active = applicable_questions(answers)
        self.assertTrue({"QC4", "QF2", "QF3", "QH2", "QH3", "QI1"} <= set(active))
        self.assertNotIn("QF5", active)  # knowledge never silently opts into advanced
        answers.update(advanced_enabled=True, QA1="d")
        self.assertTrue({"QF5", "QF6", "QH4", "QI2", "QJ1", "QJ2"} <= set(applicable_questions(answers)))

    def test_overlapping_illustrative_rules_preserved(self):
        self.assertEqual(illustrative_risk_matches({"RT": 62.5, "HZ": 75}),
                         ("Balanced long-term", "Growth-oriented"))
        self.assertEqual(illustrative_risk_matches({"RT": 70, "HZ": 50}), ())

    def test_new_profile_preserves_legacy_numeric_behavior(self):
        answers = completed_answers()
        legacy = ClientProfile(3, **CAPACITY, crypto="No")
        profile = ClientProfile.from_questionnaire(answers, CAPACITY, 3)
        for attribute in ("gamma", "gamma_willingness", "gamma_capacity", "capacity_score", "max_loss",
                          "recommended_assets", "recommended_constraints"):
            self.assertEqual(getattr(profile, attribute), getattr(legacy, attribute))
        self.assertTrue(profile.gamma_provisional)
        self.assertEqual(profile.optimization_gamma, profile.gamma)
        self.assertEqual(profile.risk_classification, "Growth-oriented")
        self.assertFalse(profile.portfolio_settings_applied)
        answers["QE1"] = "a"
        self.assertEqual(profile.questionnaire_answers["QE1"], "c")

    def test_scores_and_requested_limits_do_not_change_gamma_or_legacy_constraints(self):
        a = ClientProfile.from_questionnaire(completed_answers(), CAPACITY, 3)
        b = ClientProfile.from_questionnaire(completed_answers(
            QE1="e", QE2="e", QE3="d", QE4="e", QD1="a", QD2="a", QF4="a"), CAPACITY, 3)
        self.assertNotEqual(a.rt, b.rt)
        self.assertNotEqual(a.hz, b.hz)
        self.assertEqual(a.gamma, b.gamma)
        self.assertEqual(a.capacity_score, b.capacity_score)
        self.assertEqual(a.recommended_constraints, b.recommended_constraints)
        self.assertNotEqual(a.risk_classification, b.risk_classification)

    def test_legacy_capacity_floor_and_labels(self):
        for score, expected in [(0, 12), (4, 12), (5, 6), (8, 6), (9, 2), (13, 2)]:
            self.assertEqual(ClientProfile.gamma_from_capacity(score), expected)
        for gamma, label in [(4, "Aggressive"), (4.1, "Balanced"), (9, "Balanced"), (9.1, "Conservative")]:
            self.assertEqual(ClientProfile.profile_label(gamma), label)
        for gamma in (None, 0, -1, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError):
                ClientProfile.from_questionnaire(completed_answers(), CAPACITY, gamma)
        missing = dict(CAPACITY, income=None)
        with self.assertRaises(ValueError):
            ClientProfile.from_questionnaire(completed_answers(), missing, 3)
        with self.assertRaises(ValueError):
            ClientProfile.from_questionnaire(completed_answers(), None, 3)

    def test_risk_classification_independent_of_route(self):
        answers = completed_answers(QA2="f", QH2="b", QH3="a")
        profile = ClientProfile.from_questionnaire(answers, CAPACITY, 3)
        self.assertEqual(profile.risk_classification, "Growth-oriented")
        self.assertTrue(any("Benchmark-aware" in route for route in profile.portfolio_routes))
        self.assertEqual(dimension_labels(profile.scores)["RT"], profile.label)

    def test_draft_survives_widget_removal_and_profile_publish(self):
        state = {}
        initialize_draft(state)
        state["widget"] = ["h"]
        save_widget(state, "questionnaire_answers", "QC4", "widget")
        state["widget"].clear()
        del state["widget"]
        initialize_draft(state)
        self.assertEqual(state["questionnaire_answers"]["QC4"], ["h"])
        state.update(assets=["SPY"], constraints={}, pick_Equity=["SPY"], views={"SPY": (0.1, 0.5)})
        profile = ClientProfile.from_questionnaire(completed_answers(), CAPACITY, 3)
        publish_profile(state, profile)
        self.assertNotIn("pick_Equity", state)
        self.assertIn("views", state)
        self.assertEqual(state["questionnaire_answers"]["QC4"], ["h"])

    def test_all_existing_portfolio_builders_accept_new_profile(self):
        from src.portfolios import build_portfolios
        profile = ClientProfile.from_questionnaire(completed_answers(), CAPACITY, 3)
        tickers = profile.recommended_assets
        mu = pd.Series(np.linspace(0.02, 0.09, len(tickers)), index=tickers)
        cov = pd.DataFrame(np.diag(np.linspace(0.01, 0.04, len(tickers))), index=tickers, columns=tickers)
        weights = build_portfolios(mu, cov, 0.01, profile.gamma, profile.recommended_constraints, {}, n_sim=2)
        self.assertEqual(len(weights), 4)
        for w in weights.values():
            self.assertTrue(np.isfinite(w).all())
            self.assertAlmostEqual(w.sum(), 1)
            self.assertGreaterEqual(w.min(), -1e-8)
            self.assertLessEqual(w.max(), profile.recommended_constraints["max_weight_per_asset"] + 1e-6)


if __name__ == "__main__":
    unittest.main()
