"""Questionnaire -> gamma (tolerance) and allocation limits (capacity)."""
import itertools
import unittest

import numpy as np
import pandas as pd

from src.client_profile import (GAMMA_MAX, GAMMA_MIN, ClientProfile, gamma_from_rt, indifference_gamma,
                                qe3_distance, qe3_gamma_interval)
from src.optimizer import check_constraints, mean_variance
from src.questionnaire import CAPACITY, HORIZON, QE3_PROFILES, RISK_TOLERANCE, UNIVERSE, risk_tolerance_score

ALL_CLASSES = list(UNIVERSE[1])
BASE = dict(QE1="c", QE2="c", QE3="b", QE4="c", QD1="e", QD2="e",
            age="b", income="c", savings="c", commitments="a", QF1=ALL_CLASSES, QF2="b",
            QB1="a", QC1="a", QC3="d")


def profile(**changes):
    return ClientProfile.from_answers(dict(BASE, **changes))


class GammaTests(unittest.TestCase):
    def test_gamma_anchors_and_monotonicity(self):
        self.assertAlmostEqual(gamma_from_rt(0), GAMMA_MAX)
        self.assertAlmostEqual(gamma_from_rt(100), GAMMA_MIN)
        values = [gamma_from_rt(rt) for rt in range(101)]
        self.assertTrue(all(a > b for a, b in zip(values, values[1:])))
        with self.assertRaises(ValueError):
            gamma_from_rt(101)

    def test_qe3_thresholds_follow_mean_variance_utility(self):
        self.assertAlmostEqual(indifference_gamma("a", "b"), 8.0)
        self.assertAlmostEqual(indifference_gamma("b", "c"), 8 / 3)
        self.assertAlmostEqual(indifference_gamma("c", "d"), 1.6)
        for choice in QE3_PROFILES:
            low, high = qe3_gamma_interval(choice)
            gamma = (low + min(high, 20)) / 2
            utility = {k: m - gamma / 2 * s ** 2 for k, (m, s) in QE3_PROFILES.items()}
            self.assertEqual(max(utility, key=utility.get), choice)

    def test_calibration_puts_each_qe3_score_inside_its_revealed_interval(self):
        for choice, (_, score) in RISK_TOLERANCE["QE3"][1].items():
            low, high = qe3_gamma_interval(choice)
            self.assertTrue(low <= gamma_from_rt(score) <= high, choice)

    def test_consistency_note_only_for_clear_contradictions(self):
        note = lambda p: any("different directions" in n for n in p.notes)
        self.assertFalse(note(profile(QE1="d", QE2="d", QE3="c", QE4="d")))  # same QE3 interval
        self.assertFalse(note(profile(QE1="c", QE2="c", QE3="c", QE4="c")))  # one interval away
        self.assertTrue(note(profile(QE1="a", QE2="a", QE3="d", QE4="a")))   # boldest QE3, prudent elsewhere
        self.assertEqual(qe3_distance("a", 9.0), 0)
        self.assertEqual(qe3_distance("a", 1.2), 3)

    def test_consistency_note_is_rare_over_all_answer_patterns(self):
        patterns = list(itertools.product(*[list(RISK_TOLERANCE[k][1]) for k in RISK_TOLERANCE]))
        flagged = sum(qe3_distance(e[2], gamma_from_rt(risk_tolerance_score(dict(zip(RISK_TOLERANCE, e))))) >= 2
                      for e in patterns)
        self.assertLess(flagged / len(patterns), 0.2)

    def test_tolerance_always_changes_gamma_whatever_the_capacity(self):
        """The problem of the old version: a capacity floor erased the tolerance for most clients."""
        for capacity in [dict(age="d", income="a", savings="a", commitments="c"), {}]:
            prudent = profile(QE1="a", QE2="a", QE3="a", QE4="a", **capacity)
            bold = profile(QE1="e", QE2="e", QE3="d", QE4="e", **capacity)
            self.assertGreater(prudent.gamma, 3 * bold.gamma)

    def test_capacity_never_changes_gamma(self):
        rich = profile(age="a", income="c", savings="c", commitments="a")
        fragile = profile(age="d", income="a", savings="a", commitments="c", QD1="a", QD2="a")
        self.assertEqual(rich.gamma, fragile.gamma)


class CapacityTests(unittest.TestCase):
    def test_all_capacity_and_horizon_combinations_give_consistent_feasible_limits(self):
        keys = list(CAPACITY) + list(HORIZON)
        for combo in itertools.product(*[list(q[1]) for q in {**CAPACITY, **HORIZON}.values()]):
            p = profile(**dict(zip(keys, combo)))
            c = p.recommended_constraints
            self.assertTrue(0 <= c["min_bonds"] <= 1 - 0 and 0 <= c["max_equity"] <= 1)
            self.assertLessEqual(c["max_crypto"], 0.10)
            self.assertEqual(check_constraints(p.recommended_assets, c), [], combo)

    def test_most_restrictive_source_applies(self):
        p = profile(QD1="b")  # high capacity, but money needed in 1-3 years
        self.assertEqual(p.recommended_constraints["max_equity"], 0.4)
        self.assertEqual(p.recommended_constraints["min_bonds"], 0.4)
        self.assertEqual(p.limit_sources["max_equity"], ["your horizon (QD1)"])

    def test_low_savings_caps_the_tier_and_warns(self):
        p = profile(age="a", income="c", savings="a", commitments="a")  # 7 points, but no reserve
        self.assertEqual(p.capacity_score, 7)
        self.assertEqual(p.capacity_label, "Limited")
        self.assertTrue(p.safeguard)
        self.assertEqual(len(p.warnings), 1)

    def test_long_term_comfortable_client_has_no_warning(self):
        self.assertEqual(profile().warnings, [])

    def test_crypto_follows_choice_and_capacity(self):
        self.assertEqual(profile().recommended_constraints["max_crypto"], 0.10)
        self.assertEqual(profile(QF2="a").recommended_constraints["max_crypto"], 0.05)
        self.assertEqual(profile(QF1=["a", "b"]).recommended_constraints["max_crypto"], 0.0)
        low = profile(age="d", income="a", savings="b", commitments="b")
        self.assertEqual(low.recommended_constraints["max_crypto"], 0.0)
        self.assertNotIn("BTC-USD", low.recommended_assets)
        self.assertTrue(low.crypto_blocked)
        self.assertEqual(profile(QD1="c").recommended_constraints["max_crypto"], 0.0)  # money needed in 3-5 years

    def test_missing_products_are_added_when_limits_need_them(self):
        p = profile(QF1=["a"], QD1="c")  # only equities, but at least 20% bonds and at most 60% equities
        self.assertIn("AGG", p.added_assets)
        self.assertEqual(check_constraints(p.recommended_assets, p.recommended_constraints), [])

    def test_missing_answer_is_rejected(self):
        answers = dict(BASE)
        del answers["QE3"]
        with self.assertRaises(ValueError):
            ClientProfile.from_answers(answers)


class ExperienceTests(unittest.TestCase):
    EXPERT = dict(QB1="c", QC1="c", QC3="a")

    def test_all_three_conditions_are_needed(self):
        self.assertFalse(profile().experienced)
        p = profile(**self.EXPERT, QH4="b")
        self.assertTrue(p.experienced and p.wants_views)
        for code, wrong in [("QB1", "b"), ("QC1", "b"), ("QC3", "d")]:
            self.assertFalse(profile(**dict(self.EXPERT, **{code: wrong})).experienced, code)

    def test_views_question_only_required_and_used_for_experienced_investors(self):
        with self.assertRaises(ValueError):
            profile(**self.EXPERT)  # experienced: QH4 must be answered
        self.assertFalse(profile(**self.EXPERT, QH4="a").wants_views)
        self.assertFalse(profile(QH4="b").wants_views)  # not experienced: QH4 is ignored

    def test_overconfident_investor_fails_the_knowledge_check(self):
        p = profile(QB1="e", QC1="e", QC3="c")
        self.assertFalse(p.experienced)


class TolerantClientWithLowCapacityTests(unittest.TestCase):
    """The case that motivated the redesign: within the same limits, tolerance still matters."""

    def test_tolerant_client_gets_more_risk_inside_the_same_box(self):
        tickers = ["SPY", "EFA", "AGG", "SHY"]
        mu = pd.Series([.08, .075, .035, .025], index=tickers)
        vol = np.array([.18, .19, .055, .02])
        corr = np.array([[1, .85, .1, 0], [.85, 1, .1, 0], [.1, .1, 1, .5], [0, 0, .5, 1]])
        cov = pd.DataFrame(np.outer(vol, vol) * corr, index=tickers, columns=tickers)
        fragile = dict(age="d", income="a", savings="a", commitments="c")
        prudent = profile(QE1="a", QE2="a", QE3="a", QE4="a", **fragile)
        bold = profile(QE1="e", QE2="e", QE3="d", QE4="e", **fragile)
        self.assertEqual(prudent.recommended_constraints, bold.recommended_constraints)
        c = bold.recommended_constraints
        w_prudent = mean_variance(mu, cov, prudent.gamma, c)
        w_bold = mean_variance(mu, cov, bold.gamma, c)
        risk = lambda w: float(np.sqrt(w @ cov @ w))
        self.assertGreater(risk(w_bold), risk(w_prudent) + 0.01)
        self.assertLessEqual(w_bold[["SPY", "EFA"]].sum(), c["max_equity"] + 1e-9)


if __name__ == "__main__":
    unittest.main()
