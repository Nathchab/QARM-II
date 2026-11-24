"""Backtest: transaction costs and fee are taken from the return, at the right time."""
import unittest

import numpy as np
import pandas as pd

from src.backtest import performance, run_backtest
from src.black_litterman import MARKET_PORTFOLIO
from src.planning import MANAGEMENT_FEE

LIMITS = dict(max_equity=.6, min_bonds=.3, min_short_term=0., max_other=.15, max_crypto=0., max_weight_per_asset=.4)
TICKERS = ["SPY", "EFA", "AGG", "SHY", "GLD"]


class BacktestCostTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(1)
        months = pd.date_range("2010-01-31", periods=100, freq="ME")
        columns = list(dict.fromkeys(list(MARKET_PORTFOLIO) + TICKERS))
        cls.returns = pd.DataFrame(rng.normal(.005, .03, (100, len(columns))), index=months, columns=columns)
        cls.rf = pd.Series(.001, index=months)
        cls.benchmark = pd.Series({"SPY": .5, "AGG": .5})

    def run_with(self, cost):
        """Without cost-aware optimisation the trades do not depend on the cost: only what is paid changes."""
        return run_backtest(self.returns, self.rf, TICKERS, self.benchmark, 4., LIMITS, (), self.returns.index[60],
                            self.returns.index[-1], 3, cost, False)

    def test_costs_lower_the_return_by_what_is_paid(self):
        free, _, _, no_costs = self.run_with(0.)
        paid, _, turnover, costs = self.run_with(.005)
        self.assertTrue((no_costs == 0).all().all())
        np.testing.assert_allclose(paid, free - costs)  # same portfolios, minus the costs of each month
        np.testing.assert_allclose(costs.iloc[0], .005)  # first month: the whole portfolio is bought
        np.testing.assert_allclose(costs, .005 * turnover.reindex(costs.index).fillna(0))  # only when trading
        self.assertTrue((paid.mean() < free.mean()).all())

    def test_return_before_costs_and_fee_adds_them_back(self):
        free, _, turnover, _ = self.run_with(0.)
        paid, _, _, costs = self.run_with(.005)
        mine = performance(paid["Equilibrium"], self.rf, paid["Benchmark"], turnover["Equilibrium"],
                           costs["Equilibrium"], MANAGEMENT_FEE)
        gross = performance(free["Equilibrium"] + MANAGEMENT_FEE / 12, self.rf, paid["Benchmark"],
                            turnover["Equilibrium"], 0 * costs["Equilibrium"], 0.)
        self.assertAlmostEqual(mine["Annual return before costs and fee"], gross["Annual return after costs and fee"])
        self.assertLess(mine["Annual return after costs and fee"], mine["Annual return before costs and fee"])
        self.assertAlmostEqual(mine["Transaction costs per year"], costs["Equilibrium"].sum() / (len(paid) / 12))

    def test_black_litterman_keeps_its_2_percent_tilt_at_every_rebalancing(self):
        """The backtest holds the same Black-Litterman portfolio as the portfolio page: at most MAX_ACTIVE_RISK
        (tracking error) away from the equilibrium portfolio."""
        from src.portfolios import MAX_ACTIVE_RISK, WINDOW, inputs
        views = (("GLD", None, .15, .95),)
        _, weights, _, _ = run_backtest(self.returns, self.rf, TICKERS, self.benchmark, 4., LIMITS, views,
                                        self.returns.index[60], self.returns.index[-1], 3, 0., True)
        for date in weights["Black-Litterman"].index:
            t = self.returns.index.get_loc(date)
            cov = inputs(self.returns.iloc[t - WINDOW:t], TICKERS)[1]
            active = weights["Black-Litterman"].loc[date] - weights["Equilibrium"].loc[date]
            self.assertLessEqual(np.sqrt(active @ cov @ active), MAX_ACTIVE_RISK + 1e-4)


if __name__ == "__main__":
    unittest.main()
