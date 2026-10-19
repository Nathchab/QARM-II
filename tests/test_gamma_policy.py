"""Offline contracts for the limited automatic-gamma research prototype."""
from copy import deepcopy
from itertools import product
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.backtest import run_backtest
from src.client_profile import ClientProfile
from src.gamma_policy import ResearchGammaMapping, effective_constraints, gamma_binding_explanation
from src.portfolio_diagnostics import portfolio_diagnostics
from src.portfolios import build_portfolios
from src.questionnaire_definitions import QUESTIONS
from src.questionnaire_scoring import score_answers
from test_questionnaire import CAPACITY, completed_answers


HIGH = dict(age="Under 30", horizon="More than 15 years", needs_money="No",
            income="Very stable", max_loss="-30% or more")
MEDIUM = dict(age="30-45", horizon="3-7 years", needs_money="No",
              income="Fairly stable", max_loss="-10%")
LOW = dict(age="Over 60", horizon="Less than 3 years", needs_money="Yes",
           income="No regular income", max_loss="-5%")


class AutomaticGammaTests(unittest.TestCase):
    def test_illustrative_endpoints_and_midpoint(self):
        mapping = ResearchGammaMapping()
        self.assertAlmostEqual(mapping.willingness_gamma(5), 5.56716)
        self.assertAlmostEqual(mapping.willingness_gamma(97.5), 1.16315)
        self.assertAlmostEqual(mapping.willingness_gamma(51.25), np.sqrt(5.56716 * 1.16315))
        self.assertAlmostEqual(mapping.willingness_gamma(50), 2.599, places=3)

    def test_configurable_anchors_and_validation(self):
        mapping = ResearchGammaMapping(5, 97.5, 20, 1)
        profile = ClientProfile.from_questionnaire(completed_answers(), HIGH, gamma_mapping=mapping)
        self.assertEqual(profile.gamma_mapping_assumptions, mapping)
        self.assertAlmostEqual(profile.gamma_willingness, mapping.willingness_gamma(profile.rt))
        for args in [(5, 5, 6, 1), (-1, 100, 6, 1), (5, 101, 6, 1),
                     (5, 97.5, 0, 1), (5, 97.5, 1, 6), (5, 97.5, 6, float("inf")),
                     (True, 97.5, 6, 1), (5, 97.5, 6, "1")]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                ResearchGammaMapping(*args)
        for rt in [0, 4.99, 97.51, 100, None, True, "50", float("nan"), float("inf")]:
            with self.subTest(rt=rt), self.assertRaises(ValueError):
                ResearchGammaMapping().willingness_gamma(rt)

    def test_all_500_rt_patterns_preserve_all_scores_and_floor_rule(self):
        observed = {}
        for choices in product(*(QUESTIONS[q].options for q in ("QE1", "QE2", "QE3", "QE4"))):
            answers = completed_answers(**dict(zip(("QE1", "QE2", "QE3", "QE4"), choices)))
            before = deepcopy(answers)
            profile = ClientProfile.from_questionnaire(answers, HIGH)
            self.assertEqual(profile.scores, score_answers(answers).scores)
            self.assertEqual(answers, before)
            self.assertEqual(profile.gamma, max(profile.gamma_willingness, 2))
            self.assertTrue(profile.gamma_automatic)
            self.assertTrue(profile.gamma_provisional)
            self.assertEqual(profile.optimization_gamma, profile.gamma)
            self.assertEqual(profile.optimization_gamma_input, profile.gamma_willingness)
            observed[profile.rt] = profile.gamma_willingness
        self.assertEqual((min(observed), max(observed), len(observed)), (5, 97.5, 61))
        ordered = [observed[rt] for rt in sorted(observed)]
        self.assertTrue(all(a > b for a, b in zip(ordered, ordered[1:])))

    def test_capacity_overrides_and_willingness_binding(self):
        for capacity, floor in [(HIGH, 2), (MEDIUM, 6), (LOW, 12)]:
            for choices in [("a", "a", "a", "a"), ("e", "e", "d", "e")]:
                profile = ClientProfile.from_questionnaire(completed_answers(
                    **dict(zip(("QE1", "QE2", "QE3", "QE4"), choices))), capacity)
                self.assertEqual(profile.gamma_capacity, floor)
                self.assertEqual(profile.gamma, max(profile.gamma_willingness, floor))
                expected = "overrides" if floor > profile.gamma_willingness else "not binding"
                self.assertIn(expected, profile.gamma_binding_explanation)
        self.assertIn("both bind", gamma_binding_explanation(6, 6))

    def test_horizon_knowledge_and_optional_settings_never_adjust_gamma(self):
        base = ClientProfile.from_questionnaire(completed_answers(), HIGH)
        for d1, d2 in product(QUESTIONS["QD1"].options, QUESTIONS["QD2"].options):
            answers = completed_answers(QD1=d1, QD2=d2, QB1="a", QB2=["none"],
                                        QC1="a", QC2="a", QC3="c", QF4="a")
            profile = ClientProfile.from_questionnaire(answers, HIGH)
            self.assertEqual(profile.raw_horizon_answers, {"QD1": d1, "QD2": d2})
            self.assertEqual(profile.hz, score_answers(answers).scores["HZ"])
            self.assertEqual(profile.gamma, base.gamma)
            self.assertEqual(profile.capacity_score, base.capacity_score)
            self.assertEqual(profile.recommended_constraints, base.recommended_constraints)
        views = completed_answers(QA2="g", QH4="b", QH5=[
            dict(asset="SPY", value=7.5, confidence="d", measure="return")])
        profile = ClientProfile.from_questionnaire(views, HIGH)
        self.assertIsNotNone(profile.cv)
        self.assertEqual(profile.scores, score_answers(views).scores)
        self.assertEqual(profile.gamma, base.gamma)

    def test_legacy_construction_and_explicit_baseline_are_unchanged(self):
        for gamma in [1, 3, 20]:
            legacy = ClientProfile(gamma, **CAPACITY, crypto="No")
            explicit = ClientProfile.from_questionnaire(completed_answers(), CAPACITY, gamma)
            for attr in ["gamma", "gamma_willingness", "gamma_capacity", "capacity_score",
                         "max_loss", "recommended_constraints", "recommended_assets"]:
                self.assertEqual(getattr(legacy, attr), getattr(explicit, attr))
            self.assertFalse(explicit.gamma_automatic)
            self.assertEqual(explicit.mandatory_constraints, {})
        with self.assertRaises(ValueError):
            ClientProfile.from_questionnaire(completed_answers(), CAPACITY, None)

    def test_optional_preferences_cannot_relax_any_mandatory_limit(self):
        profile = ClientProfile.from_questionnaire(completed_answers(), LOW)
        mandatory = deepcopy(profile.mandatory_constraints)
        relaxed = dict(max_equity=1, min_bonds=0, max_other=1, max_crypto=1, max_weight_per_asset=1)
        self.assertEqual(effective_constraints(mandatory, relaxed), mandatory)
        tighter = dict(max_equity=.1, min_bonds=.5, max_other=.1, max_crypto=0, max_weight_per_asset=.3)
        self.assertEqual(effective_constraints(mandatory, tighter), tighter)
        self.assertEqual(profile.mandatory_constraints, mandatory)
        self.assertEqual(relaxed["max_equity"], 1)
        for invalid in [float("nan"), True, -1, 2]:
            with self.assertRaises(ValueError):
                effective_constraints(mandatory, dict(max_equity=invalid))


class PortfolioIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.c = dict(max_equity=1., min_bonds=0., max_other=0., max_crypto=0., max_weight_per_asset=1.)
        self.mu = pd.Series([.08, .03], index=["SPY", "AGG"])
        self.cov = pd.DataFrame(np.diag([.04, .01]), index=self.mu.index, columns=self.mu.index)

    def test_automatic_gamma_reaches_all_methods_and_erc_is_invariant(self):
        conservative = ClientProfile.from_questionnaire(completed_answers(QE1="a", QE2="a", QE3="a", QE4="a"), HIGH)
        adventurous = ClientProfile.from_questionnaire(completed_answers(QE1="e", QE2="e", QE3="d", QE4="e"), HIGH)
        builds = [build_portfolios(self.mu, self.cov, .02, p.gamma, self.c, {}, n_sim=8)
                  for p in (conservative, adventurous)]
        self.assertAlmostEqual(builds[0]["Mean-variance"].SPY, .2 + 1/conservative.gamma, delta=2e-7)
        self.assertAlmostEqual(builds[1]["Mean-variance"].SPY, .2 + 1/adventurous.gamma, delta=2e-7)
        for method in ("Mean-variance", "Resampled MV", "Black-Litterman"):
            self.assertFalse(np.allclose(builds[0][method], builds[1][method]), method)
        pd.testing.assert_series_equal(builds[0]["Risk parity (ERC)"], builds[1]["Risk parity (ERC)"])
        for weights in builds[0].values():
            diag = portfolio_diagnostics(weights, self.mu, self.cov, self.c)
            self.assertEqual(diag["Allocation limit violations"], [])

    def test_diagnostics_and_binding_limits_with_reordered_inputs(self):
        weights = pd.Series([.4, .6], index=self.mu.index)
        c = dict(self.c, max_equity=.4, min_bonds=.6, max_weight_per_asset=.6)
        result = portfolio_diagnostics(weights, self.mu.iloc[::-1], self.cov.iloc[::-1, ::-1], c)
        self.assertAlmostEqual(result["Expected return"], .05)
        self.assertAlmostEqual(result["Volatility"], .1)
        self.assertAlmostEqual(result["Equity exposure"], .4)
        self.assertAlmostEqual(result["Bond exposure"], .6)
        self.assertAlmostEqual(result["Largest holding"], .6)
        self.assertAlmostEqual(result["HHI"], .52)
        self.assertAlmostEqual(result["Effective holdings"], 1/.52)
        self.assertIn("max_equity", result["Binding allocation limits"])
        self.assertIn("min_bonds", result["Binding allocation limits"])
        self.assertIn("max_weight_per_asset (AGG)", result["Binding allocation limits"])
        violated = portfolio_diagnostics(weights, self.mu, self.cov, dict(c, max_equity=.2))
        self.assertIn("max_equity", violated["Allocation limit violations"])

    def test_backtest_uses_execution_gamma_in_initial_and_cost_aware_rebalancing(self):
        from src import backtest
        profile = ClientProfile.from_questionnaire(completed_answers(), LOW)
        rng = np.random.default_rng(321)
        dates = pd.date_range("2018-01-31", periods=65, freq="ME")
        returns = pd.DataFrame(rng.normal([.006, .002], [.04, .015], (65, 2)), index=dates, columns=self.mu.index)
        rf = pd.Series(.001, index=dates)
        benchmark = pd.Series([.6, .4], index=self.mu.index)
        with patch.object(backtest, "N_SIM", 2), patch.object(
                backtest, "build_portfolios", wraps=build_portfolios) as builder, patch.object(
                backtest, "mean_variance_with_costs", wraps=backtest.mean_variance_with_costs) as costly:
            results, history, turnover = run_backtest(
                returns, rf, list(self.mu.index), benchmark, profile.gamma, self.c, {},
                dates[60], dates[-1], rebalance=1)
        self.assertEqual(len(results), 5)
        self.assertTrue(np.isfinite(results).all().all())
        self.assertTrue(all(call.args[3] == 12 for call in builder.call_args_list))
        self.assertEqual(costly.call_count, 8)
        self.assertTrue(all(call.args[2] == 12 for call in costly.call_args_list))
        self.assertIn("Risk parity (ERC)", history)
        self.assertEqual(len(turnover), 5)


if __name__ == "__main__":
    unittest.main()
