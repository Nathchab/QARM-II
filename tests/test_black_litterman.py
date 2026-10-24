"""Black-Litterman with absolute and relative views."""
import unittest

import numpy as np
import pandas as pd

from src.black_litterman import (CONFIDENCE_LEVELS, TAU, normalize_views, posterior_returns, prior_returns,
                                 restrict_views, view_check, view_problems)

TICKERS = ["SPY", "EEM", "AGG", "GLD"]
VOL = np.array([.16, .22, .05, .15])
CORR = np.array([[1, .7, .1, .05], [.7, 1, .1, .1], [.1, .1, 1, .2], [.05, .1, .2, 1]])
COV = pd.DataFrame(np.outer(VOL, VOL) * CORR, index=TICKERS, columns=TICKERS)
RF = 0.02


class BlackLittermanTests(unittest.TestCase):
    def setUp(self):
        self.pi = prior_returns(COV, RF)

    def test_no_view_gives_the_equilibrium(self):
        pd.testing.assert_series_equal(posterior_returns(COV, RF, ()), self.pi)
        pd.testing.assert_series_equal(posterior_returns(COV, RF, {}), self.pi)

    def test_old_absolute_format_is_the_same_as_the_new_one(self):
        old = posterior_returns(COV, RF, {"GLD": (.08, .5)})
        new = posterior_returns(COV, RF, [("GLD", None, .08, .5)])
        pd.testing.assert_series_equal(old, new)

    def test_absolute_view_moves_the_asset_towards_the_view_more_with_more_confidence(self):
        q = self.pi.GLD + .04
        moves = [posterior_returns(COV, RF, [("GLD", None, q, c)]).GLD - self.pi.GLD
                 for c in CONFIDENCE_LEVELS.values()]
        self.assertTrue(all(0 < a < b < .04 for a, b in zip(moves, moves[1:])))

    def test_relative_view_moves_the_spread_not_the_level(self):
        spread = self.pi.EEM - self.pi.SPY
        post = posterior_returns(COV, RF, [("EEM", "SPY", spread + .03, .8)])
        self.assertGreater(post.EEM - post.SPY, spread + .01)
        self.assertLess(post.EEM - post.SPY, spread + .03)

    def test_view_equal_to_equilibrium_changes_nothing(self):
        view = ("EEM", "SPY", self.pi.EEM - self.pi.SPY, .9)
        np.testing.assert_allclose(posterior_returns(COV, RF, [view]), self.pi, atol=1e-12)

    def test_omega_uses_the_variance_of_the_view_portfolio(self):
        """One relative view with confidence c: closed form for the posterior spread."""
        c, q = .5, self.pi.EEM - self.pi.SPY + .02
        p = np.array([-1, 1, 0, 0])
        var = p @ COV.values @ p
        omega = TAU * var * (1 - c) / c
        # posterior of P mu = P pi + tau P S P' / (tau P S P' + omega) * (q - P pi)
        expected = p @ self.pi.values + TAU * var / (TAU * var + omega) * (q - p @ self.pi.values)
        post = posterior_returns(COV, RF, [("EEM", "SPY", q, c)])
        self.assertAlmostEqual(p @ post.values, expected, places=12)

    def test_view_check_and_extreme_views(self):
        check = view_check(COV, self.pi, ("EEM", "SPY", self.pi.EEM - self.pi.SPY, .5))
        self.assertAlmostEqual(check["gap"], 0)
        sd = np.sqrt(VOL[0] ** 2 + VOL[1] ** 2 - 2 * CORR[0, 1] * VOL[0] * VOL[1])
        self.assertAlmostEqual(check["volatility"], sd)
        self.assertFalse(check["extreme"])
        self.assertTrue(view_check(COV, self.pi, ("GLD", None, self.pi.GLD + .5, .5))["extreme"])

    def test_invalid_views_are_rejected(self):
        self.assertTrue(view_problems([("SPY", "SPY", .01, .5)]))
        self.assertTrue(view_problems([("SPY", None, .08, .5), ("SPY", None, .09, .8)]))
        self.assertTrue(view_problems([("EEM", "SPY", .02, .5), ("SPY", "EEM", -.02, .5)]))
        self.assertTrue(view_problems([("SPY", None, .08, 1.0)]))
        self.assertEqual(view_problems([("SPY", None, .08, .5), ("EEM", "SPY", .02, .8)]), [])
        with self.assertRaises(ValueError):
            posterior_returns(COV, RF, [("SPY", "SPY", .01, .5)])

    def test_views_on_products_not_in_the_portfolio_are_dropped(self):
        views = [("SPY", None, .08, .5), ("EEM", "QQQ", .02, .5)]
        self.assertEqual(restrict_views(views, TICKERS), (("SPY", None, .08, .5),))
        self.assertEqual(normalize_views(None), ())


if __name__ == "__main__":
    unittest.main()
