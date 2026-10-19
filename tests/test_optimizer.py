"""Numerical contracts for the existing constrained mean-variance problem."""
from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint

from src import optimizer
from src.portfolios import build_portfolios


class MeanVarianceTests(unittest.TestCase):
    def setUp(self):
        self.mu = pd.Series([.08, .03], index=["SPY", "AGG"])
        self.cov = pd.DataFrame(np.diag([.04, .01]), index=self.mu.index, columns=self.mu.index)
        self.c = dict(max_equity=1., min_bonds=0., max_other=0., max_crypto=0., max_weight_per_asset=1.)

    def assert_feasible(self, w, c=None):
        c = c or self.c
        rules = [LinearConstraint(np.ones(len(w)), 1, 1)] + optimizer.client_constraints(list(w.index), c)
        self.assertLessEqual(optimizer._mv_violation(w, rules, optimizer.weight_bounds(w.index, c)), 1e-9)

    def test_two_asset_closed_form_and_risk_aversion_direction(self):
        previous = 1.
        for g in [.05, .25, 1., 1.25, 2., 6., 12., 100., 1000.]:
            with self.subTest(gamma=g):
                w = optimizer.mean_variance(self.mu, self.cov, g, self.c)
                self.assertIsInstance(w, pd.Series)
                self.assertTrue(w.index.equals(self.mu.index))
                self.assertAlmostEqual(w.SPY, np.clip(.2 + 1/g, 0, 1), delta=2e-7)
                self.assertLessEqual(w.SPY, previous+1e-8)
                self.assert_feasible(w)
                previous = w.SPY

    def test_common_positive_scaling_preserves_weights(self):
        for g in [.05, 2., 6., 12., 1000.]:
            expected = optimizer.mean_variance(self.mu, self.cov, g, self.c)
            for factor in [1e-250, 1e-12, 1e-6, 1/12, 12., 1e6, 1e12, 1e250]:
                with self.subTest(gamma=g, factor=factor):
                    actual = optimizer.mean_variance(self.mu*factor, self.cov*factor, g, self.c)
                    np.testing.assert_allclose(actual, expected, atol=2e-7, rtol=0)
                    self.assert_feasible(actual)

    def test_small_gamma_normalization_preserves_an_interior_analytical_solution(self):
        mu = pd.Series([.035, .03], index=self.mu.index)
        for g in [.25, .5, .75]:
            with self.subTest(gamma=g):
                w = optimizer.mean_variance(mu, self.cov, g, self.c)
                self.assertAlmostEqual(w.SPY, .2 + .1/g, delta=2e-7)
                self.assert_feasible(w)

    def test_multi_asset_scaling_and_binding_constraints(self):
        tickers = ["SPY", "AGG", "TLT", "GLD"]
        mu = pd.Series([.12, .02, .025, .04], index=tickers)
        cov = pd.DataFrame(np.diag([.04, .003, .02, .03]), index=tickers, columns=tickers)
        c = dict(self.c, max_equity=.2, min_bonds=.7, max_other=.1, max_weight_per_asset=.4)
        w = optimizer.mean_variance(mu, cov, .2, c)
        self.assert_feasible(w, c)
        np.testing.assert_allclose(w, [.2, .3, .4, .1], atol=2e-7, rtol=0)
        for factor in [1e-12, 1/12, 12., 1e12]:
            np.testing.assert_allclose(optimizer.mean_variance(mu*factor, cov*factor, .2, c), w, atol=2e-7, rtol=0)

    def test_initialization_is_feasible_deterministic_and_independent_of_returns(self):
        real_minimize = optimizer.minimize
        starts = []
        def inspect_start(fun, x0, **kwargs):
            starts.append(x0.copy())
            self.assertLessEqual(optimizer._mv_violation(x0, kwargs["constraints"], kwargs["bounds"]), 1e-9)
            return real_minimize(fun, x0, **kwargs)
        c = dict(self.c, max_equity=.2)
        with patch.object(optimizer, "minimize", inspect_start):
            optimizer.mean_variance(self.mu, self.cov, 2., c)
            optimizer.mean_variance(self.mu*3, self.cov/2, 20., c)
        np.testing.assert_array_equal(starts[0], starts[1])

    def test_infeasible_capacity_and_missing_asset_class_raise(self):
        for c in [dict(self.c, max_weight_per_asset=.4), dict(self.c, min_bonds=1.1),
                  dict(self.c, max_equity=-.1), dict(self.c, max_weight_per_asset=-.1)]:
            with self.subTest(c=c), self.assertRaisesRegex(ValueError, "infeasible"):
                optimizer.mean_variance(self.mu, self.cov, 2., c)
        mu = self.mu.iloc[:1]
        with self.assertRaisesRegex(ValueError, "infeasible"):
            optimizer.mean_variance(mu, self.cov.loc[mu.index, mu.index], 2., dict(self.c, min_bonds=.2))

    def test_solver_failure_is_not_hidden_by_cleanup(self):
        result = SimpleNamespace(success=False, status=9, message="Iteration limit reached", x=np.array([.4, .6]))
        with patch.object(optimizer, "minimize", return_value=result):
            with self.assertRaisesRegex(RuntimeError, "SLSQP failed.*Iteration limit"):
                optimizer.mean_variance(self.mu, self.cov, 2., self.c)

    def test_success_flag_does_not_allow_invalid_weights(self):
        for weights in [[1.2, -.2], [.4, .5], [np.nan, .5], [.5], [np.inf, 0]]:
            result = SimpleNamespace(success=True, x=np.array(weights))
            with self.subTest(weights=weights), patch.object(optimizer, "minimize", return_value=result):
                with self.assertRaisesRegex(RuntimeError, "invalid weights"):
                    optimizer.mean_variance(self.mu, self.cov, 2., self.c)

    def test_validated_weights_are_returned_without_clipping_or_normalizing(self):
        raw = np.array([.4, .6000000005])  # accepted floating-point budget residual
        with patch.object(optimizer, "minimize", return_value=SimpleNamespace(success=True, x=raw)):
            actual = optimizer.mean_variance(self.mu, self.cov, 2., self.c)
        np.testing.assert_array_equal(actual.values, raw)

    def test_feasibility_solver_failure_is_descriptive(self):
        c = dict(self.c, max_equity=.2)
        with patch.object(optimizer, "linprog", return_value=SimpleNamespace(success=False, status=4, message="Numerical issue")):
            with self.assertRaisesRegex(RuntimeError, "initialization failed.*Numerical issue"):
                optimizer.mean_variance(self.mu, self.cov, 2., c)

    def test_invalid_inputs_are_rejected(self):
        for g in [0, -1, np.nan, np.inf, True, "2"]:
            with self.subTest(gamma=g), self.assertRaisesRegex(ValueError, "risk_aversion"):
                optimizer.mean_variance(self.mu, self.cov, g, self.c)
        bad_cov = self.cov.copy()
        bad_cov.iloc[0, 0] = -.04
        with self.assertRaisesRegex(ValueError, "positive semidefinite"):
            optimizer.mean_variance(self.mu, bad_cov, 2., self.c)
        bad_cov.iloc[0, 0], bad_cov.iloc[0, 1] = .04, .01
        with self.assertRaisesRegex(ValueError, "symmetric"):
            optimizer.mean_variance(self.mu, bad_cov, 2., self.c)
        with self.assertRaisesRegex(ValueError, "match"):
            optimizer.mean_variance(self.mu, self.cov.iloc[::-1, ::-1], 2., self.c)
        with self.assertRaisesRegex(ValueError, "finite"):
            optimizer.mean_variance(self.mu*np.nan, self.cov, 2., self.c)
        with self.assertRaisesRegex(ValueError, "max_equity"):
            optimizer.mean_variance(self.mu, self.cov, 2., dict(self.c, max_equity=np.nan))

    def test_zero_and_singular_covariance_are_not_regularized(self):
        zero = self.cov*0
        w = optimizer.mean_variance(self.mu, zero, 2., self.c)
        self.assertAlmostEqual(w.SPY, 1., delta=1e-8)
        tied = optimizer.mean_variance(self.mu*0, zero, 2., dict(self.c, max_equity=.2))
        self.assert_feasible(tied, dict(self.c, max_equity=.2))
        rank_one = pd.DataFrame([[.04, .02], [.02, .01]], index=self.mu.index, columns=self.mu.index)
        self.assert_feasible(optimizer.mean_variance(self.mu, rank_one, 2., self.c))

    def test_inputs_are_not_mutated(self):
        mu, cov, c = self.mu.copy(), self.cov.copy(), deepcopy(self.c)
        optimizer.mean_variance(mu, cov, 2., c)
        pd.testing.assert_series_equal(mu, self.mu)
        pd.testing.assert_frame_equal(cov, self.cov)
        self.assertEqual(c, self.c)

    def test_resampling_and_black_litterman_keep_existing_interfaces(self):
        c = dict(self.c, max_equity=.7)
        for views in [{}, {"SPY": (.07, .5)}]:
            portfolios = build_portfolios(self.mu, self.cov, .02, 3., c, views, n_sim=8)
            self.assertEqual(len(portfolios), 4)
            for w in portfolios.values():
                self.assert_feasible(w, c)
            repeated = optimizer.resampled(self.mu, self.cov, 3., c, n_sim=8)
            np.testing.assert_allclose(repeated, portfolios["Resampled MV"], atol=1e-12, rtol=0)


if __name__ == "__main__":
    unittest.main()
