"""The recommended portfolio: which method, and efficiency with our forecast."""
import unittest

import numpy as np
import pandas as pd

from src.black_litterman import posterior_returns
from src.portfolios import bad_year_loss, build_portfolios, dominated_by, gamma_for_loss_limit, recommended_method

TICKERS = ["SPY", "EFA", "AGG", "TLT", "SHY", "GLD"]
VOL = np.array([.16, .17, .05, .14, .02, .15])
CORR = np.full((6, 6), .3) + np.eye(6) * .7
COV = pd.DataFrame(np.outer(VOL, VOL) * CORR, index=TICKERS, columns=TICKERS)
PAST = pd.Series([.11, .07, -.02, -.10, .01, .14], index=TICKERS)  # noisy "last 5 years"
LIMITS = dict(max_equity=.5, min_bonds=.3, min_short_term=.1, max_other=.15, max_crypto=0., max_weight_per_asset=.4)
VIEW = (("GLD", None, .08, .5),)


class RecommendationTests(unittest.TestCase):
    def test_black_litterman_only_for_clients_with_views(self):
        self.assertEqual(recommended_method(()), "Equilibrium")
        self.assertEqual(recommended_method(VIEW), "Black-Litterman")
        without, _ = build_portfolios(PAST, COV, .01, 3., LIMITS, (), n_sim=5)
        with_views, _ = build_portfolios(PAST, COV, .01, 3., LIMITS, VIEW, n_sim=5)
        self.assertNotIn("Black-Litterman", without)
        self.assertIn("Black-Litterman", with_views)

    def test_recommended_portfolio_is_never_dominated_with_our_forecast(self):
        for views in [(), VIEW]:
            forecast = posterior_returns(COV, .01, views)
            for gamma in [1.06, 2.7, 6., 10.6]:
                portfolios, errors = build_portfolios(PAST, COV, .01, gamma, LIMITS, views, n_sim=10)
                self.assertEqual(errors, {})
                best = recommended_method(views)
                self.assertEqual(dominated_by(portfolios, forecast, COV, best), [], (gamma, views))
                self.assertGreaterEqual(portfolios[best]["SHY"], .1 - 1e-6)  # money needed soon

    def test_the_loss_limit_of_qe2_is_respected(self):
        forecast = posterior_returns(COV, .01, ())
        for max_loss in [-.05, -.10, -.20, -.30]:
            gamma = gamma_for_loss_limit(forecast, COV, 1.06, LIMITS, max_loss)  # very tolerant answers
            portfolios, _ = build_portfolios(PAST, COV, .01, gamma, LIMITS, (), n_sim=5)
            loss = bad_year_loss(portfolios["Equilibrium"], forecast, COV)
            self.assertGreaterEqual(loss, max_loss - 1e-4, max_loss)
        self.assertEqual(gamma_for_loss_limit(forecast, COV, 1.06, LIMITS, None), 1.06)  # "more than 30%": no limit

    def test_dominated_by_finds_more_return_with_less_risk(self):
        """AGG has a higher return than SPY with the same risk: 60/40 beats 70/30 (more return AND less risk)."""
        forecast = pd.Series([.03, .06], index=["SPY", "AGG"])
        cov = pd.DataFrame(np.diag([.04, .04]), index=forecast.index, columns=forecast.index)
        portfolios = {"Equilibrium": pd.Series([.7, .3], index=forecast.index),
                      "Mean-variance": pd.Series([.6, .4], index=forecast.index),
                      "Risk parity (ERC)": pd.Series([.9, .1], index=forecast.index)}
        self.assertEqual(dominated_by(portfolios, forecast, cov, "Equilibrium"), ["Mean-variance"])


if __name__ == "__main__":
    unittest.main()
