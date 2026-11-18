"""The optimisers: known solutions, client limits, transaction costs and risk parity."""
import unittest

import numpy as np
import pandas as pd

from src.optimizer import (binding_limits, lacks_home_equity, check_constraints, frontier, gamma_for_max_volatility, mean_variance, mean_variance_with_costs,
                           portfolio_stats, resampled, respects_limits, risk_contributions, risk_parity, weight_bounds)

NO_LIMIT = dict(max_equity=1., min_bonds=0., max_other=1., max_crypto=0., max_weight_per_asset=1.)


def diagonal_cov(tickers, variances):
    return pd.DataFrame(np.diag(variances), index=tickers, columns=tickers)


class MeanVarianceTests(unittest.TestCase):
    def test_two_assets_match_the_formula_by_hand(self):
        """Two uncorrelated assets: x_SPY = 0.2 + 1/gamma (between 0 and 1), found with the Lagrangian."""
        mu = pd.Series([.08, .03], index=["SPY", "AGG"])
        cov = diagonal_cov(mu.index, [.04, .01])
        for gamma in [.25, 1., 2., 6., 12., 100.]:
            w = mean_variance(mu, cov, gamma, NO_LIMIT)
            self.assertAlmostEqual(w.SPY, min(.2 + 1 / gamma, 1), places=6)
            self.assertAlmostEqual(w.sum(), 1, places=9)

    def test_limits_are_respected_and_can_be_binding(self):
        tickers = ["SPY", "AGG", "TLT", "GLD"]
        mu = pd.Series([.12, .02, .025, .04], index=tickers)
        cov = diagonal_cov(tickers, [.04, .003, .02, .03])
        c = dict(max_equity=.2, min_bonds=.7, max_other=.1, max_crypto=0., max_weight_per_asset=.4)
        w = mean_variance(mu, cov, .2, c)  # a very tolerant client: every limit binds
        np.testing.assert_allclose(w, [.2, .3, .4, .1], atol=1e-6)
        self.assertTrue(respects_limits(w.values, tickers, c))
        self.assertEqual(binding_limits(w, c), ["maximum in equities", "minimum in bonds", "maximum in gold and commodities"])

    def test_minimum_in_short_term_treasuries(self):
        tickers = ["SPY", "TLT", "SHY"]
        mu = pd.Series([.08, .04, .01], index=tickers)
        cov = diagonal_cov(tickers, [.04, .02, .0004])
        c = dict(NO_LIMIT, min_bonds=.5, min_short_term=.3)
        w = mean_variance(mu, cov, 1., c)
        self.assertAlmostEqual(w.SHY, .3, places=6)  # SHY has the lowest return: exactly the minimum
        self.assertIn("SHY", check_constraints(["SPY", "TLT"], c)[0])  # SHY missing: explained

    def test_at_least_half_of_the_equities_in_the_region(self):
        tickers = ["CSSMI.SW", "ACWI", "CSBGC7.SW"]
        mu = pd.Series([.05, .09, .02], index=tickers)  # the world looks better, but the client prefers Switzerland
        cov = diagonal_cov(tickers, [.03, .03, .002])
        c = dict(NO_LIMIT, min_home_equity=.5, region="Switzerland")
        w = mean_variance(mu, cov, 1., c)
        swiss = w["CSSMI.SW"] + .02 * w["ACWI"]  # ACWI is about 2% Swiss
        self.assertGreaterEqual(swiss, .5 * (w["CSSMI.SW"] + w["ACWI"]) - 1e-6)
        self.assertLess(mean_variance(mu, cov, 1., NO_LIMIT)["CSSMI.SW"], w["CSSMI.SW"])  # without the rule: less
        self.assertFalse(lacks_home_equity(tickers, c))
        self.assertTrue(lacks_home_equity(["ACWI", "CSBGC7.SW"], c))  # only the world fund: no Swiss equities

    def test_same_units_for_mu_and_cov_do_not_matter(self):
        mu = pd.Series([.08, .03, .05], index=["SPY", "AGG", "GLD"])
        cov = diagonal_cov(mu.index, [.04, .01, .03])
        yearly = mean_variance(mu, cov, 4., NO_LIMIT)
        monthly = mean_variance(mu / 12, cov / 12, 4., NO_LIMIT)
        np.testing.assert_allclose(monthly, yearly, atol=1e-6)

    def test_impossible_limits_are_explained_and_refused(self):
        tickers = ["SPY", "AGG"]
        c = dict(NO_LIMIT, max_weight_per_asset=.4)  # 2 products x 40% = 80%: cannot invest 100%
        self.assertIn("80%", check_constraints(tickers, c)[0])
        self.assertEqual(check_constraints(tickers, NO_LIMIT), [])
        mu = pd.Series([.08, .03], index=tickers)
        with self.assertRaises(ValueError):
            mean_variance(mu, diagonal_cov(tickers, [.04, .01]), 2., c)

    def test_more_risk_aversion_means_less_volatility_on_the_frontier(self):
        mu = pd.Series([.08, .03, .05], index=["SPY", "AGG", "GLD"])
        cov = diagonal_cov(mu.index, [.04, .01, .03])
        volatility = frontier(mu, cov, NO_LIMIT)["Volatility"]
        self.assertTrue((volatility.diff().dropna() <= 1e-9).all())

    def test_target_volatility_raises_gamma_only_when_needed(self):
        """Lab LN1 'targeting a volatility': the smallest gamma whose portfolio has at most the target volatility."""
        mu = pd.Series([.08, .03, .05], index=["SPY", "AGG", "GLD"])
        cov = diagonal_cov(mu.index, [.04, .01, .03])
        vol = lambda g: portfolio_stats(mean_variance(mu, cov, g, NO_LIMIT), mu, cov)[1]
        self.assertEqual(gamma_for_max_volatility(mu, cov, 2., NO_LIMIT, max_vol=.50), 2.)  # limit not binding
        g = gamma_for_max_volatility(mu, cov, 2., NO_LIMIT, max_vol=.09)
        self.assertGreater(g, 2.)
        self.assertLessEqual(vol(g), .09 + 1e-6)
        self.assertGreater(vol(g * .95), .09)  # a slightly smaller gamma would be too risky

    def test_resampling_gives_the_same_valid_portfolio_every_time(self):
        mu = pd.Series([.08, .03, .05], index=["SPY", "AGG", "GLD"])
        cov = diagonal_cov(mu.index, [.04, .01, .03])
        first, second = resampled(mu, cov, 3., NO_LIMIT, n_sim=8), resampled(mu, cov, 3., NO_LIMIT, n_sim=8)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(respects_limits(first.values, list(mu.index), NO_LIMIT))


class TransactionCostTests(unittest.TestCase):
    def setUp(self):
        self.mu = pd.Series([.08, .03, .05], index=["SPY", "AGG", "GLD"])
        self.cov = diagonal_cov(self.mu.index, [.04, .01, .03])
        self.x0 = np.array([.2, .6, .2])  # current portfolio
        self.target = mean_variance(self.mu, self.cov, 3., NO_LIMIT)

    def turnover(self, w):
        return np.abs(w.values - self.x0).sum()

    def test_without_costs_we_go_to_the_mean_variance_portfolio(self):
        w = mean_variance_with_costs(self.mu, self.cov, 3., NO_LIMIT, self.x0, cost=0.)
        np.testing.assert_allclose(w, self.target, atol=1e-3)

    def test_costs_reduce_trading_and_very_high_costs_stop_it(self):
        cheap = mean_variance_with_costs(self.mu, self.cov, 3., NO_LIMIT, self.x0, cost=.01)
        expensive = mean_variance_with_costs(self.mu, self.cov, 3., NO_LIMIT, self.x0, cost=.50)
        self.assertLess(self.turnover(cheap), self.turnover(self.target))
        self.assertLess(self.turnover(expensive), 1e-3)

    def test_turnover_limit_is_respected(self):
        w = mean_variance_with_costs(self.mu, self.cov, 3., NO_LIMIT, self.x0, cost=.001, max_turnover=.10)
        self.assertLessEqual(self.turnover(w), .10 + 1e-4)


class RiskParityTests(unittest.TestCase):
    def setUp(self):
        self.tickers = ["SPY", "EFA", "AGG", "TLT", "GLD"]
        vol = np.array([.16, .18, .05, .14, .15])
        corr = np.full((5, 5), .2) + np.eye(5) * .8
        self.cov = pd.DataFrame(np.outer(vol, vol) * corr, index=self.tickers, columns=self.tickers)

    def test_without_limits_every_asset_brings_the_same_risk(self):
        w = risk_parity(self.cov, NO_LIMIT)
        np.testing.assert_allclose(risk_contributions(w, self.cov), .2, atol=1e-4)
        self.assertGreater(w.AGG, w.SPY)  # the less volatile asset gets the larger weight

    def test_risk_contributions_sum_to_one(self):
        w = pd.Series([.1, .2, .3, .2, .2], index=self.tickers)
        self.assertAlmostEqual(risk_contributions(w, self.cov).sum(), 1)

    def test_with_limits_the_portfolio_stays_inside_them(self):
        c = dict(max_equity=.2, min_bonds=.6, max_other=.1, max_crypto=0., max_weight_per_asset=.4)
        w = risk_parity(self.cov, c)
        self.assertTrue(respects_limits(w.values, self.tickers, c, 1e-4))
        self.assertGreater(np.abs(risk_contributions(w, self.cov) - .2).max(), .01)  # equal shares are impossible

    def test_limits_that_leave_a_single_portfolio(self):
        """Equities at most 20% and two bond products at their 40% maximum: only 20/40/40 is possible."""
        tickers = ["SPY", "AGG", "TLT"]
        cov = diagonal_cov(tickers, [.03, .003, .02])
        c = dict(max_equity=.2, min_bonds=.6, max_other=.1, max_crypto=0., max_weight_per_asset=.4)
        np.testing.assert_allclose(risk_parity(cov, c), [.2, .4, .4], atol=1e-4)

    def test_cash_and_satellite_products_have_their_own_maximum(self):
        bounds = dict(zip(["SPY", "SHY", "VNQ"], weight_bounds(["SPY", "SHY", "VNQ"], NO_LIMIT | {"max_weight_per_asset": .4})))
        self.assertEqual(bounds, {"SPY": (0, .4), "SHY": (0, 1.0), "VNQ": (0, .15)})


if __name__ == "__main__":
    unittest.main()
