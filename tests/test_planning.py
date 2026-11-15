"""Projection, savings goal, order list and PDF report."""
import unittest

import numpy as np
import pandas as pd

from src.planning import order_list, simulate_wealth, worst_drawdowns


class ProjectionTests(unittest.TestCase):
    def test_without_risk_the_projection_is_the_savings_formula(self):
        """vol = 0: every future is the same, W_T = W_0 (1+r)^n + C ((1+r)^n - 1) / r with r = mu/12."""
        wealth, growth = simulate_wealth(mu=.06, vol=0., years=10, initial=10_000, monthly=100, n_paths=3)
        r, n = .06 / 12, 120
        expected = 10_000 * (1 + r) ** n + 100 * ((1 + r) ** n - 1) / r
        np.testing.assert_allclose(wealth[:, -1], expected)
        np.testing.assert_allclose(growth[:, -1], (1 + r) ** n)

    def test_more_risk_gives_a_wider_range_of_futures(self):
        calm, _ = simulate_wealth(.05, .05, 10, 10_000, 100)
        wild, _ = simulate_wealth(.05, .20, 10, 10_000, 100)
        width = lambda w: np.percentile(w[:, -1], 95) - np.percentile(w[:, -1], 5)
        self.assertGreater(width(wild), 2 * width(calm))

    def test_the_simulation_gives_the_same_numbers_every_time(self):
        np.testing.assert_array_equal(simulate_wealth(.05, .1, 5, 1_000, 10)[0],
                                      simulate_wealth(.05, .1, 5, 1_000, 10)[0])

    def test_worst_drawdown(self):
        growth = np.array([[1, 1.2, 0.9, 1.3, 1.17]])  # peak 1.2, then 0.9: -25%
        np.testing.assert_allclose(worst_drawdowns(growth), [-.25])


class OrderListTests(unittest.TestCase):
    def test_whole_etf_shares_crypto_in_fractions_and_cash_left(self):
        weights = pd.Series({"SPY": .5, "AGG": .3, "BTC-USD": .2, "GLD": 0.})
        prices = pd.Series({"SPY": 600., "AGG": 100., "BTC-USD": 90_000., "GLD": 300.})
        orders, cash = order_list(weights, prices, 10_000)
        self.assertEqual(list(orders["Quantity"].round(4)), [8, 30, .0222])  # GLD (weight 0) is not bought
        self.assertAlmostEqual(cash, 10_000 - 8 * 600 - 30 * 100 - .0222 * 90_000, places=6)
        self.assertGreaterEqual(cash, 0)


class StressTestTests(unittest.TestCase):
    def test_a_young_product_is_replaced_by_an_older_fund_or_kept_as_cash(self):
        from src.stress_test import replay, too_young
        days = pd.date_range("2007-10-09", periods=4)
        prices = pd.DataFrame({"ACWI": [np.nan, np.nan, 50., 55.], "SPY": [100., 90., 80., 88.],
                               "CHCORP.SW": [np.nan, np.nan, np.nan, 100.]}, index=days)
        weights = pd.Series({"ACWI": .5, "CHCORP.SW": .5})
        self.assertEqual(too_young(prices, weights, days[0]), {"ACWI": "SPY", "CHCORP.SW": None})
        value = replay(prices, weights, days[0])
        np.testing.assert_allclose(value, [1, .95, .9, .94])  # ACWI follows SPY, CHCORP.SW stays in cash


class ReportTests(unittest.TestCase):
    def test_the_report_is_a_pdf(self):
        from src.client_profile import ClientProfile
        from src.report import client_report
        profile = ClientProfile(dict(QE1="c", QE2="c", QE3="b", QE4="c", QD1="b", QD2="c", age="b", income="b",
                                     savings="a", commitments="b", QF1=["a", "b"], QF2="b", QF3="c", QB1="a", QC1="a",
                                     QC3="d"))
        weights = pd.Series({"SPY": .5, "AGG": .3, "SHY": .2})
        orders, cash = order_list(weights, pd.Series({"SPY": 600., "AGG": 100., "SHY": 80.}), 10_000)
        plan = dict(initial=10_000, monthly=100, years=10, paid_in=22_000, goal=30_000, p5=18_000, p50=28_000,
                    p95=40_000, prob_goal=.45, loss_limit=.2, prob_loss=.1)
        pdf = client_report(profile, profile.recommended_constraints, "Market equilibrium (mean-variance)", weights,
                            .05, .09, "CHF", plan, orders, cash)
        self.assertTrue(pdf.startswith(b"%PDF"))


if __name__ == "__main__":
    unittest.main()
