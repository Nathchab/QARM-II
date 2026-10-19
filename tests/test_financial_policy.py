"""Approved v2 capacity, migration, numerical status and simulation contracts."""
from copy import deepcopy
from itertools import product
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.client_profile import ClientProfile
from src.financial_capacity import (AGE_POINTS, INCOME_POINTS, SAVINGS_POINTS, COMMITMENT_POINTS,
    FINANCIAL_FIELDS, CAPACITY_LIMITS, HORIZON_LIMITS, WITHDRAWAL_LIMITS, POLICY_VERSION,
    assess_capacity, allocation_limits, volatility_allowance)
from src.gamma_policy import DEFAULT_RESEARCH_MAPPING
from src.optimizer import mean_variance
from src.portfolio_policy import (SCENARIO_NAMES, allocation_feasibility, plan_allocation,
    assess_portfolios, volatility_screen, load_covariance_scenarios, assess_backtest_weights)
from src.portfolios import METHODS, build_portfolios
from src.questionnaire_scoring import score_answers
from src.questionnaire_state import publish_profile
from test_questionnaire import CAPACITY, completed_answers

FINANCIAL_HIGH = dict(age="Under 30", income="Very stable", emergency_savings="More than 6 months", commitments="None significant")
FINANCIAL_MID = dict(age="30-45", income="Very stable", emergency_savings="3-6 months", commitments="Manageable")
FINANCIAL_LOW = dict(age="Over 60", income="No regular income", emergency_savings="Less than 3 months",
                     commitments="Significant obligations or financial difficulties")


def new_profile(capacity=None, **overrides):
    return ClientProfile.from_financial_questionnaire(completed_answers(QD1="e", QD2="e", **overrides), capacity or FINANCIAL_HIGH)


def inputs():
    root = Path(__file__).resolve().parents[1] / "analysis/gamma_calibration"
    mu = pd.read_csv(root / "inputs_base.csv", index_col=0).expected_return
    cov = pd.read_csv(root / "covariance_base.csv", index_col=0)
    return mu, cov


class FinancialCapacityTests(unittest.TestCase):
    def test_original_factory_accepts_new_schema_without_reinterpreting_legacy_inputs(self):
        q = completed_answers()
        p = ClientProfile.from_questionnaire(q, FINANCIAL_HIGH)
        self.assertEqual(p.capacity_policy_version, POLICY_VERSION)
        self.assertEqual(p.gamma_willingness, DEFAULT_RESEARCH_MAPPING.willingness_gamma(p.rt))
        old = ClientProfile.from_questionnaire(q, CAPACITY, 3.)
        self.assertFalse(hasattr(old, "capacity_policy_version"))
        self.assertEqual(old.gamma, 3.)
        with self.assertRaisesRegex(ValueError, "automatic"):
            ClientProfile.from_questionnaire(q, FINANCIAL_HIGH, 3.)

    def test_all_108_financial_patterns_keep_approved_scores_floors_and_safeguard(self):
        seen = []
        for age, income, savings, commitments in product(AGE_POINTS, INCOME_POINTS, SAVINGS_POINTS, COMMITMENT_POINTS):
            answers = dict(age=age, income=income, emergency_savings=savings, commitments=commitments)
            saved = deepcopy(answers)
            p = new_profile(answers)
            expected = AGE_POINTS[age] + INCOME_POINTS[income] + SAVINGS_POINTS[savings] + COMMITMENT_POINTS[commitments]
            self.assertEqual(p.capacity_score, expected)
            base = 12 if expected <= 3 else 6 if expected <= 6 else 2
            guard = savings == "Less than 3 months" or commitments == "Significant obligations or financial difficulties"
            self.assertEqual(p.gamma_capacity_base, base)
            self.assertEqual(p.gamma_capacity, max(base, 6) if guard else base)
            self.assertEqual(p.gamma, max(p.gamma_willingness, p.gamma_capacity))
            self.assertEqual(p.capacity_answers, saved)
            self.assertEqual(answers, saved)
            self.assertIsNone(p.max_loss)
            seen.append(expected)
        self.assertEqual((len(seen), min(seen), max(seen)), (108, 0, 9))

    def test_missing_unknown_and_wrong_type_answers_are_not_scored_as_zero(self):
        for key in FINANCIAL_FIELDS:
            for value in [None, "", "Unknown", True, [], float("nan")]:
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    assess_capacity(dict(FINANCIAL_HIGH, **{key: value}))
        with self.assertRaises(ValueError):
            assess_capacity(None)

    def test_every_original_rt_answer_pattern_and_all_dimensions_unchanged(self):
        for answers in product("abcde", "abcde", "abcd", "abcde"):
            q = completed_answers(**dict(zip(("QE1", "QE2", "QE3", "QE4"), answers)))
            profile = ClientProfile.from_financial_questionnaire(q, FINANCIAL_HIGH)
            self.assertEqual(profile.scores, score_answers(q).scores)
            self.assertEqual(profile.gamma_willingness, DEFAULT_RESEARCH_MAPPING.willingness_gamma(profile.rt))
            self.assertEqual(profile.optimization_gamma, profile.gamma)

    def test_all_capacity_qd_combinations_and_crypto_limits_are_feasible(self):
        distinct = set()
        for age, income, savings, commitments, qd1, qd2, crypto in product(
                AGE_POINTS, INCOME_POINTS, SAVINGS_POINTS, COMMITMENT_POINTS, "abcde", "abcde", (False, True)):
            capacity = dict(age=age, income=income, emergency_savings=savings, commitments=commitments)
            q = completed_answers(QD1=qd1, QD2=qd2, QF1=["a", "b", "h"] if crypto else ["a", "b"])
            p = ClientProfile.from_financial_questionnaire(q, capacity)
            limits = p.recommended_constraints
            self.assertEqual(limits["max_equity"], min(CAPACITY_LIMITS[p.gamma_capacity][0], HORIZON_LIMITS[qd1][0], WITHDRAWAL_LIMITS[qd2][0]))
            self.assertEqual(limits["min_bonds"], max(CAPACITY_LIMITS[p.gamma_capacity][1], HORIZON_LIMITS[qd1][1], WITHDRAWAL_LIMITS[qd2][1]))
            self.assertEqual(limits["max_crypto"], .1 if crypto else 0.)
            tag = tuple(sorted(limits.items()))
            if tag not in distinct:
                self.assertTrue(allocation_feasibility(p.recommended_assets, limits).feasible)
                distinct.add(tag)

    def test_low_capacity_high_rt_and_raw_liquidity_flags(self):
        p = new_profile(FINANCIAL_LOW, QE1="e", QE2="e", QE3="d", QE4="e")
        self.assertEqual(p.rt, 97.5)
        self.assertEqual(p.gamma, 12)
        self.assertEqual(p.recommended_constraints["max_equity"], .2)
        self.assertEqual(p.recommended_constraints["min_bonds"], .6)
        self.assertIn("significant_commitments_or_difficulties", p.financial_review_flags)
        near = ClientProfile.from_financial_questionnaire(completed_answers(QD1="b", QD2="a"), FINANCIAL_HIGH)
        self.assertIn("unquantified_near_term_need", near.financial_review_flags)
        self.assertIn("unquantified_unexpected_withdrawal", near.financial_review_flags)
        self.assertEqual(near.raw_horizon_answers, {"QD1": "b", "QD2": "a"})

    def test_allowance_is_continuous_without_changing_gamma(self):
        self.assertEqual(volatility_allowance(75, 2, "e", "e"), .05)
        self.assertAlmostEqual(volatility_allowance(75.00001, 2, "e", "e"), .05, places=6)
        self.assertEqual(volatility_allowance(97.5, 2, "e", "e"), .1)
        for floor, qd1, qd2, flags in [(6, "e", "e", ()), (2, "a", "e", ()),
                                      (2, "e", "d", ()), (2, "e", "e", ("difficulty",))]:
            self.assertEqual(volatility_allowance(97.5, floor, qd1, qd2, flags), .05)

    def test_migration_is_explicit_archives_old_limits_and_leaves_old_object_unchanged(self):
        old = ClientProfile.from_questionnaire(completed_answers(), CAPACITY, 3)
        before = old.policy_snapshot()
        new = new_profile(FINANCIAL_MID)
        state = dict(profile=old, assets=["SPY"], constraints={"max_equity": .1})
        with self.assertRaisesRegex(ValueError, "migration"):
            publish_profile(state, new)
        self.assertIs(state["profile"], old)
        publish_profile(state, new, migrate=True)
        self.assertIs(state["profile"], new)
        self.assertEqual(old.policy_snapshot(), before)
        self.assertEqual(new.migration_history[0]["recommended_constraints"], before["recommended_constraints"])
        self.assertEqual(new.migration_history[0]["saved_optional_constraints"], {"max_equity": .1})
        self.assertEqual(state["previous_profile"].gamma, old.gamma)
        self.assertNotIn("constraints", state)
        self.assertEqual(new.capacity_policy_version, POLICY_VERSION)


class PolicyStatusTests(unittest.TestCase):
    def setUp(self):
        self.mu, self.cov = inputs()
        self.profile = new_profile()
        self.c = self.profile.mandatory_constraints
        self.ref = mean_variance(self.mu, self.cov, self.profile.gamma, self.c)
        self.raw = {method: self.ref.copy() for method in METHODS}

    def test_valid_allocation_has_normal_or_information_state_without_a_screen(self):
        results = assess_portfolios(self.mu, self.cov, .02, self.profile, self.c, raw_portfolios=self.raw)
        self.assertTrue(all(result.status in ("normal", "information") for result in results.values()))
        self.assertTrue(all(result.scenarios.empty and result.allowance is None for result in results.values()))
        unbound = pd.Series([.2, .15, .1, .1, .2, .15, .05, .05], index=self.mu.index)
        result = assess_portfolios(self.mu, self.cov, .02, self.profile, self.c,
                                    raw_portfolios={m: unbound for m in METHODS})["Mean-variance"]
        self.assertEqual(result.status, "normal")

    def test_capacity_floor_binding_is_information_not_a_warning(self):
        p = new_profile(QE1="e", QE2="e", QE3="d", QE4="e")
        ref = mean_variance(self.mu, self.cov, p.gamma, p.mandatory_constraints)
        results = assess_portfolios(self.mu, self.cov, .02, p, p.mandatory_constraints,
                                    raw_portfolios={m: ref for m in METHODS})
        self.assertTrue(all(r.status == "information" for r in results.values()))
        self.assertTrue(any("willingness" in m for m in results["Mean-variance"].information))
        self.assertTrue(all(not r.profile_warnings and not r.messages for r in results.values()))

    def test_higher_volatility_is_not_a_failure_and_weights_are_unmodified(self):
        risky = pd.Series(0., index=self.mu.index)
        risky["SPY"], risky["EEM"], risky["AGG"] = .4, .4, .2
        before = risky.copy()
        raw = dict(self.raw, **{"Black-Litterman": risky})
        result = assess_portfolios(self.mu, self.cov, .02, self.profile, self.c, raw_portfolios=raw)["Black-Litterman"]
        self.assertIn(result.status, ("normal", "information"))
        pd.testing.assert_series_equal(result.weights, before)
        pd.testing.assert_series_equal(risky, before)
        self.assertFalse(result.messages)
        self.assertGreater(result.diagnostics["Volatility"], (self.ref @ self.cov @ self.ref)**.5)
        self.assertTrue(result.scenarios.empty)

    def test_constraints_violation_is_never_called_compliant_even_if_physically_valid(self):
        low = new_profile(FINANCIAL_LOW)
        ref = mean_variance(self.mu, self.cov, low.gamma, low.mandatory_constraints)
        raw = {m: self.ref for m in METHODS}
        raw["Mean-variance"] = ref
        result = assess_portfolios(self.mu, self.cov, .02, low, low.mandatory_constraints, raw_portfolios=raw)["Risk parity (ERC)"]
        self.assertEqual(result.status, "warning")
        self.assertIsNone(result.weights)
        self.assertTrue(any("allocation limits" in m for m in result.messages))
        self.assertTrue(any("financial review" in m for m in result.profile_warnings))

    def test_invalid_weights_are_not_normalized_or_fabricated(self):
        invalid = self.ref * .5
        result = assess_portfolios(self.mu, self.cov, .02, self.profile, self.c,
                                    raw_portfolios=dict(self.raw, **{"Resampled MV": invalid}))["Resampled MV"]
        self.assertEqual(result.status, "warning")
        self.assertIsNone(result.weights)
        self.assertAlmostEqual(invalid.sum(), .5)
        self.assertIn("No normalization", result.messages[0])

    def test_runtime_needs_no_historical_scenarios_or_mean_variance_reference(self):
        raw = dict(self.raw)
        raw.pop("Mean-variance")
        with patch("src.portfolio_policy.load_covariance_scenarios", side_effect=AssertionError("retired screen")), \
             patch("src.portfolio_policy.volatility_screen", side_effect=AssertionError("retired screen")):
            results = assess_portfolios(self.mu, self.cov, .02, self.profile, self.c, raw_portfolios=raw,
                                        covariance_scenarios={}, scenario_issues=["historical only"], reference_weights=None)
        self.assertIsNone(results["Mean-variance"].weights)
        self.assertTrue(all(results[m].status in ("normal", "information") for m in METHODS if m != "Mean-variance"))

    def test_live_covariance_for_products_outside_old_panel_does_not_create_a_warning(self):
        mu, cov = self.mu.rename(index={"SPY": "QQQ"}), self.cov.rename(index={"SPY": "QQQ"}, columns={"SPY": "QQQ"})
        raw = {m: w.rename(index={"SPY": "QQQ"}) for m, w in self.raw.items()}
        results = assess_portfolios(mu, cov, .02, self.profile, self.c, raw_portfolios=raw)
        self.assertTrue(all(r.weights is not None and r.status != "warning" for r in results.values()))

    def test_financial_flags_and_missing_critical_data_still_warn(self):
        p = new_profile(FINANCIAL_LOW)
        raw = build_portfolios(self.mu, self.cov, .02, p.gamma, p.mandatory_constraints, {}, n_sim=2)
        results = assess_portfolios(self.mu, self.cov, .02, p, p.mandatory_constraints, raw_portfolios=raw)
        self.assertTrue(all(r.status in ("normal", "information") and r.weights is not None for r in results.values()))
        self.assertTrue(all(len(r.profile_warnings) == 2 for r in results.values()))
        results = assess_portfolios(self.mu, self.cov, .02, self.profile, self.c, raw_portfolios=self.raw,
                                    data_issues=["Insufficient overlapping data"])
        self.assertTrue(all(r.status == "warning" for r in results.values()))

    def test_invalid_estimated_covariance_is_unavailable_not_a_zero_risk_simulation(self):
        for invalid in (float("nan"), -1.):
            bad = self.cov.copy()
            bad.iloc[0, 0] = invalid
            results = assess_portfolios(self.mu, bad, .02, self.profile, self.c, raw_portfolios=self.raw)
            self.assertTrue(all(result.status == "warning" and result.weights is None for result in results.values()))
            self.assertTrue(all("covariance" in result.messages[0] for result in results.values()))
        results = assess_portfolios(self.mu, self.cov, float("nan"), self.profile, self.c, raw_portfolios=self.raw)
        self.assertTrue(all(r.status == "warning" and r.weights is None for r in results.values()))

    def test_one_solver_failure_keeps_other_methods(self):
        with patch("src.portfolio_policy.resampled", side_effect=RuntimeError("sample solve failed")):
            results = assess_portfolios(self.mu, self.cov, .02, self.profile, self.c, n_sim=2)
        self.assertEqual(results["Resampled MV"].status, "warning")
        self.assertTrue(all(results[m].weights is not None for m in METHODS if m != "Resampled MV"))

    def test_model_specific_ceiling_uses_fixed_weights_in_stress(self):
        index = ["SPY", "AGG"]
        w = pd.Series([1., 0.], index=index)
        ref = pd.Series([.5, .5], index=index)
        scenarios = {"base": pd.DataFrame(np.diag([.01, .09]), index=index, columns=index),
                     "stress": pd.DataFrame(np.diag([.16, .01]), index=index, columns=index)}
        result = volatility_screen(w, ref, scenarios, .1)
        self.assertTrue(result[result.Scenario.eq("base")].Pass.iloc[0])
        self.assertFalse(result[result.Scenario.eq("stress")].Pass.iloc[0])

    def test_all_optimizers_keep_constraints_and_erc_gamma_independence(self):
        low = build_portfolios(self.mu, self.cov, .02, 2., self.c, {}, n_sim=4)
        high = build_portfolios(self.mu, self.cov, .02, 12., self.c, {}, n_sim=4)
        pd.testing.assert_series_equal(low["Risk parity (ERC)"], high["Risk parity (ERC)"])
        results = assess_portfolios(self.mu, self.cov, .02, self.profile, self.c, raw_portfolios=low)
        for r in results.values():
            self.assertEqual(r.diagnostics["Allocation limit violations"], [])


class FallbackAndHistoryTests(unittest.TestCase):
    def test_infeasible_selected_products_get_disclosed_default_universe_not_relaxed_limits(self):
        p = new_profile(FINANCIAL_LOW)
        c = deepcopy(p.mandatory_constraints)
        plan = plan_allocation(["SPY", "EFA"], c, c, p.recommended_assets, p.recommended_assets)
        self.assertTrue(plan.feasible)
        self.assertTrue(plan.illustrative)
        self.assertEqual(plan.constraints, c)
        self.assertIn("AGG", plan.tickers)
        self.assertTrue(any("bonds" in message for message in plan.problems))
        mu, cov = inputs()
        ref = mean_variance(mu, cov, p.gamma, c)
        result = assess_portfolios(mu, cov, .02, p, c, raw_portfolios={m: ref for m in METHODS},
                                  illustrative=plan.illustrative, assumptions=plan.assumptions)["Mean-variance"]
        self.assertEqual(result.status, "warning")
        self.assertTrue(result.illustrative)
        self.assertIsNotNone(result.weights)
        self.assertEqual(result.diagnostics["Allocation limit violations"], [])

    def test_infeasible_optional_requests_can_be_omitted_only_in_labelled_illustration(self):
        p = new_profile()
        requested = dict(p.mandatory_constraints, max_weight_per_asset=.05)
        original = deepcopy(requested)
        plan = plan_allocation(p.recommended_assets, requested, p.mandatory_constraints, p.recommended_assets, p.recommended_assets)
        self.assertTrue(plan.feasible)
        self.assertTrue(plan.illustrative)
        self.assertEqual(plan.constraints, p.mandatory_constraints)
        self.assertTrue(any("optional" in message for message in plan.assumptions))
        self.assertEqual(requested, original)

    def test_no_available_bonds_or_invalid_limits_produces_transparent_diagnostic(self):
        p = new_profile(FINANCIAL_LOW)
        plan = plan_allocation(["SPY", "EFA"], p.mandatory_constraints, p.mandatory_constraints,
                               ["SPY", "EFA"], p.recommended_assets)
        self.assertFalse(plan.feasible)
        self.assertTrue(plan.problems)
        for value in [float("nan"), True, -1, 2]:
            self.assertEqual(allocation_feasibility(["SPY", "AGG"], dict(p.mandatory_constraints, min_bonds=value)).kind, "invalid")

    def test_backtest_audit_uses_only_prior_observations_and_checks_actual_mv_holdings(self):
        p = new_profile()
        rng = np.random.default_rng(12)
        mu, cov = inputs()
        dates = pd.date_range("2010-01-31", periods=65, freq="ME")
        returns = pd.DataFrame(rng.normal(.004, .025, (65, len(mu))), index=dates, columns=mu.index)
        dates_of_trades = [dates[60], dates[63]]
        w = pd.Series(.0, index=mu.index)
        w["SPY"], w["EEM"], w["AGG"] = .4, .4, .2
        histories = {method: pd.DataFrame([w, w], index=dates_of_trades) for method in METHODS}
        before = deepcopy(histories)
        with patch("src.portfolio_policy.mean_variance", side_effect=AssertionError("No MV screen in a backtest")):
            first = assess_backtest_weights(returns, histories, p, p.mandatory_constraints)
        changed = returns.copy()
        changed.loc[dates[60]:] = .5
        second = assess_backtest_weights(changed, histories, p, p.mandatory_constraints)
        pd.testing.assert_frame_equal(first[first.Date.eq(dates[60])].reset_index(drop=True),
                                      second[second.Date.eq(dates[60])].reset_index(drop=True))
        self.assertEqual(len(first), 8)
        self.assertFalse(first.Code.eq("warning").any())
        self.assertNotIn("Maximum excess (pp)", first.columns)
        self.assertTrue(first["Allocation violations"].eq("").all())
        for method in METHODS:
            pd.testing.assert_frame_equal(histories[method], before[method])


if __name__ == "__main__":
    unittest.main()
