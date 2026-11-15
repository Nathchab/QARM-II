"""The recommended portfolio: which method, and efficiency with our forecast."""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.black_litterman import MARKET_PORTFOLIO, posterior_returns
from src.portfolios import (BENCHMARKS, bad_year_loss, benchmark_weights, build_portfolios, dominated_by, estimate,
                            gamma_for_loss_limit, recommended_method)

TICKERS = ["SPY", "EFA", "AGG", "TLT", "SHY", "GLD"]
VOL = np.array([.16, .17, .05, .14, .02, .15])
CORR = np.full((6, 6), .3) + np.eye(6) * .7
COV = pd.DataFrame(np.outer(VOL, VOL) * CORR, index=TICKERS, columns=TICKERS)
PAST = pd.Series([.11, .07, -.02, -.10, .01, .14], index=TICKERS)  # noisy "last 5 years"
PI = .01 + 2.5 * COV @ pd.Series([.3, .15, .4, .05, .05, .05], index=TICKERS)  # market equilibrium returns
LIMITS = dict(max_equity=.5, min_bonds=.3, min_short_term=.1, max_other=.15, max_crypto=0., max_weight_per_asset=.4)
VIEW = (("GLD", None, .08, .5),)
client = lambda preserve=False, inconsistent=False, max_loss=-.2: SimpleNamespace(
    preserve_capital=preserve, inconsistent=inconsistent, max_loss=max_loss)


class RecommendationTests(unittest.TestCase):
    def test_black_litterman_only_for_clients_with_views(self):
        without, _ = build_portfolios(PAST, COV, PI, 3., LIMITS, (), n_sim=5)
        with_views, _ = build_portfolios(PAST, COV, PI, 3., LIMITS, VIEW, n_sim=5)
        self.assertNotIn("Black-Litterman", without)
        self.assertIn("Black-Litterman", with_views)
        pi = posterior_returns(PI, COV, ())
        self.assertEqual(recommended_method(client(), without, pi, COV), "Equilibrium")
        self.assertEqual(recommended_method(client(preserve=True), without, pi, COV), "Minimum variance")
        self.assertEqual(recommended_method(client(), with_views, pi, COV), "Black-Litterman")
        self.assertEqual(recommended_method(client(preserve=True), with_views, pi, COV), "Minimum variance")  # safety
        self.assertEqual(recommended_method(client(inconsistent=True), without, pi, COV), "Risk parity (ERC)")
        # risk parity is too risky for a client who accepts only 1% of loss: back to the market equilibrium
        self.assertEqual(recommended_method(client(inconsistent=True, max_loss=-.01), without, pi, COV), "Equilibrium")

    def test_recommended_portfolio_is_never_dominated_with_our_forecast(self):
        for views, preserve in [((), False), ((), True), (VIEW, False)]:  # (risk parity is not mean-variance)
            forecast = posterior_returns(PI, COV, views)
            for gamma in [1.06, 2.7, 6., 10.6]:
                portfolios, errors = build_portfolios(PAST, COV, PI, gamma, LIMITS, views, n_sim=10)
                self.assertEqual(errors, {})
                best = recommended_method(client(preserve), portfolios, forecast, COV)
                self.assertEqual(dominated_by(portfolios, forecast, COV, best), [], (gamma, views, best))
                self.assertGreaterEqual(portfolios[best]["SHY"], .1 - 1e-6)  # money needed soon

    def test_the_loss_limit_of_qe2_is_respected(self):
        forecast = posterior_returns(PI, COV, ())
        for max_loss in [-.05, -.10, -.20, -.30]:
            gamma = gamma_for_loss_limit(forecast, COV, 1.06, LIMITS, max_loss)  # very tolerant answers
            portfolios, _ = build_portfolios(PAST, COV, PI, gamma, LIMITS, (), n_sim=5)
            loss = bad_year_loss(portfolios["Equilibrium"], forecast, COV)
            self.assertGreaterEqual(loss, max_loss - 1e-4, max_loss)
        self.assertEqual(gamma_for_loss_limit(forecast, COV, 1.06, LIMITS, None), 1.06)  # "more than 30%": no limit

    def test_minimum_variance_has_the_lowest_risk(self):
        portfolios, _ = build_portfolios(PAST, COV, PI, 3., LIMITS, VIEW, n_sim=5)
        risk = {m: float(w @ COV @ w) for m, w in portfolios.items()}
        self.assertEqual(min(risk, key=risk.get), "Minimum variance")

    def test_dominated_by_finds_more_return_with_less_risk(self):
        """AGG has a higher return than SPY with the same risk: 60/40 beats 70/30 (more return AND less risk)."""
        forecast = pd.Series([.03, .06], index=["SPY", "AGG"])
        cov = pd.DataFrame(np.diag([.04, .04]), index=forecast.index, columns=forecast.index)
        portfolios = {"Equilibrium": pd.Series([.7, .3], index=forecast.index),
                      "Mean-variance": pd.Series([.6, .4], index=forecast.index),
                      "Risk parity (ERC)": pd.Series([.9, .1], index=forecast.index)}
        self.assertEqual(dominated_by(portfolios, forecast, cov, "Equilibrium"), ["Mean-variance"])

    def test_equilibrium_inputs_come_from_one_market_portfolio(self):
        rng = np.random.default_rng(0)
        months = pd.date_range("2015-01-31", periods=80, freq="ME")
        returns = pd.DataFrame(rng.normal(.005, .04, (80, 9)), index=months, columns=list(MARKET_PORTFOLIO) + ["TLT", "SHY"])
        few = estimate(returns, ["SPY", "SHY"])[2]
        many = estimate(returns, ["SPY", "TLT", "AGG", "GLD", "SHY"])[2]
        self.assertAlmostEqual(few.SPY, many.SPY)  # does not depend on the other products the client chose
        mu, cov, market_cov = estimate(returns, list(MARKET_PORTFOLIO))
        np.testing.assert_allclose(market_cov, cov @ pd.Series(MARKET_PORTFOLIO))  # = Sigma w for the market itself

    def test_default_benchmark_has_bonds_in_the_currency_of_the_client(self):
        self.assertEqual(dict(benchmark_weights(BENCHMARKS[0], TICKERS, .6, "USD")), {"SPY": .6, "AGG": .4})
        self.assertEqual(dict(benchmark_weights(BENCHMARKS[0], TICKERS, .6, "CHF")), {"SPY": .6, "CSBGC7.SW": .4})


if __name__ == "__main__":
    unittest.main()
