"""Focused regression checks against the saved USD demonstration covariance."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize

from src.optimizer import risk_parity, risk_contributions, client_constraints, weight_bounds
from src.portfolio_diagnostics import portfolio_diagnostics

HERE = Path(__file__).resolve().parents[1] / "analysis/risk_parity_validation"
LIMITS = dict(max_equity=.2, min_bonds=.6, max_weight_per_asset=.4, max_other=.2, max_crypto=0.)


class RiskParityValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cov = pd.read_csv(HERE / "covariance_fresh_USD.csv", index_col=0)
        cls.zero = pd.Series(0., index=cls.cov.index)

    def test_risk_shares_are_normalized_euler_volatility_contributions(self):
        w = risk_parity(self.cov, LIMITS)
        sigma = self.cov.to_numpy()
        vol = float(np.sqrt(w @ self.cov @ w))
        absolute = w.to_numpy() * (sigma @ w.to_numpy()) / vol
        shares = risk_contributions(w, self.cov)
        np.testing.assert_allclose(shares, absolute / vol, atol=1e-13)
        self.assertAlmostEqual(absolute.sum(), vol, places=12)
        self.assertAlmostEqual(shares.sum(), 1., places=12)
        np.testing.assert_allclose(risk_contributions(w, self.cov / 12), shares, atol=1e-13)

    def test_current_constrained_solve_converges_and_reproduces_observed_risk_shares(self):
        captured = []
        def tracked(*args, **kwargs):
            result = minimize(*args, **kwargs)
            captured.append(result)
            return result
        with patch("src.optimizer.minimize", side_effect=tracked):
            w = risk_parity(self.cov, LIMITS)
        result = captured[-1]
        self.assertTrue(result.success, result.message)
        self.assertEqual(result.status, 0)
        self.assertAlmostEqual(1000 * ((risk_contributions(w, self.cov) - .125)**2).sum(), result.fun, places=4)
        self.assertAlmostEqual(result.fun, 66.27947, places=3)
        np.testing.assert_allclose(w.values, [.05, .05, .05, .05, .4, .2, .06, .14], atol=.006)
        np.testing.assert_allclose(risk_contributions(w,self.cov), [.06,.08,.09,.09,.27,.29,.06,.06], atol=.005)
        d = portfolio_diagnostics(w, self.zero, self.cov, LIMITS)
        self.assertEqual(d["Allocation limit violations"], [])
        self.assertTrue({"max_equity", "min_bonds", "max_other", "max_weight_per_asset (AGG)"}
                        .issubset(set(d["Binding allocation limits"])))

    def test_less_constrained_approved_profile_is_more_balanced_but_other_cap_still_binds(self):
        constrained = risk_parity(self.cov, LIMITS)
        capable_limits = dict(LIMITS, max_equity=.8, min_bonds=0.)
        capable = risk_parity(self.cov, capable_limits)
        shares = risk_contributions(capable, self.cov)
        current_sse = ((risk_contributions(constrained, self.cov) - .125)**2).sum()
        self.assertLess(((shares - .125)**2).sum(), current_sse / 4)
        d = portfolio_diagnostics(capable, self.zero, self.cov, capable_limits)
        self.assertEqual(d["Allocation limit violations"], [])
        self.assertIn("max_other", d["Binding allocation limits"])
        self.assertGreater(np.max(abs(shares - .125)), .01)

    def test_alternative_initialization_does_not_materially_improve_feasible_erc(self):
        production = risk_parity(self.cov, LIMITS)
        objective = lambda w: 1000 * ((risk_contributions(w, self.cov) - .125)**2).sum()
        starting = np.array([.05,.05,.05,.05,.4,.2,.06,.14])
        rules = [LinearConstraint(np.ones(8),1,1)] + client_constraints(self.cov.index,LIMITS)
        result = minimize(objective, starting, bounds=weight_bounds(self.cov.index,LIMITS), constraints=rules,
                          method="SLSQP", options=dict(ftol=1e-8,maxiter=1000))
        d = portfolio_diagnostics(pd.Series(result.x,index=self.cov.index),self.zero,self.cov,LIMITS)
        self.assertEqual(d["Allocation limit violations"], [])
        self.assertLessEqual(objective(production) - objective(result.x), 1e-3)
        self.assertLess(np.max(abs(production.values - result.x)), .0002)

    def test_equal_risk_reference_violates_the_existing_other_limit(self):
        diagnostic_limits = dict(LIMITS, max_equity=1., min_bonds=0., max_other=1., max_weight_per_asset=1.)
        equal = risk_parity(self.cov, diagnostic_limits)
        np.testing.assert_allclose(risk_contributions(equal,self.cov), .125, atol=1e-5)
        capable = dict(LIMITS,max_equity=.8,min_bonds=0.)
        d = portfolio_diagnostics(equal,self.zero,self.cov,capable)
        self.assertEqual(d["Allocation limit violations"], ["max_other"])
        self.assertGreater(equal.GLD + equal.DBC, .30)


if __name__ == "__main__":
    unittest.main()
