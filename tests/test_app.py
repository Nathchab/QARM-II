"""Offline test of every page with Streamlit's AppTest (invented prices, no download).
The pages are opened through app.py, exactly as a visitor does."""
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # pragma: no cover
    AppTest = None

from src.client_profile import ClientProfile
from src.data import ASSETS, FX, RF

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["app_pages/welcome.py", "app_pages/profile.py", "app_pages/choices.py", "app_pages/portfolio.py",
         "app_pages/plan.py", "app_pages/backtest.py", "app_pages/stress_test.py", "app_pages/how_it_works.py"]
ANSWERS = dict(QE1="e", QE2="e", QE3="d", QE4="e", QD1="e", QD2="e",
               age="d", income="a", savings="a", commitments="c", QF1=list("abcdefgh"), QF2="b",
               QB1="c", QC1="c", QC3="a", QH4="b")


def synthetic_prices():
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2005-01-03", "2026-10-16")
    factor = rng.normal(0, 1, len(days))
    columns = {}
    for ticker, (_, kind) in ASSETS.items():
        beta = {"Equity": .8, "Bond": -.1, "Other": .2, "Crypto": .3}[kind]
        vol = {"Equity": .18, "Bond": .06, "Other": .18, "Crypto": .7}[kind]
        noise = beta * factor + np.sqrt(1 - beta ** 2) * rng.normal(0, 1, len(days))
        series = pd.Series(100 * np.exp(np.cumsum(.0003 + vol / np.sqrt(252) * noise)), index=days)
        if kind == "Crypto":
            series[series.index < "2015-01-01"] = np.nan
        columns[ticker] = series
    columns[FX] = pd.Series(0.9, index=days)
    columns[RF] = pd.Series(2.0, index=days)
    return pd.DataFrame(columns)


@unittest.skipIf(AppTest is None, "Streamlit is not installed")
class AppSmokeTests(unittest.TestCase):
    def setUp(self):
        patcher = patch("src.data.download_prices", return_value=synthetic_prices())
        patcher.start()
        self.addCleanup(patcher.stop)

    def open_page(self, page, profile=None):
        app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120)
        if profile is not None:
            app.session_state["profile"] = profile
        app.switch_page(page)
        return app.run()

    def test_every_page_runs_for_a_tolerant_client_with_low_capacity(self):
        profile = ClientProfile(ANSWERS)
        for page in PAGES:
            with self.subTest(page=page):
                app = self.open_page(page, profile)
                self.assertEqual(len(app.exception), 0, [e.value for e in app.exception])
                self.assertEqual(len(app.error), 0, [e.value for e in app.error])

    def test_profile_page_without_profile_shows_the_questionnaire(self):
        app = self.open_page("app_pages/profile.py")
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.radio), 10)

    def test_start_over_forgets_the_client(self):
        app = self.open_page("app_pages/portfolio.py", ClientProfile(ANSWERS))
        self.assertIn("profile", app.session_state)
        next(b for b in app.button if "Start over" in b.label).click().run()
        self.assertNotIn("profile", app.session_state)
        self.assertEqual(app.title[0].value, "Welcome to your robo-advisor")


class LossLimitNoteTests(unittest.TestCase):
    def test_note_explains_raised_gamma_and_impossible_limit(self):
        from src.app_helpers import loss_limit_note
        prudent = ClientProfile(dict(ANSWERS, QE1="a", QE2="a", QE3="a", QE4="a"))  # tolerates at most -5%
        self.assertIsNone(loss_limit_note(prudent, prudent.gamma, -0.04, "USD"))
        self.assertIn("closest risk aversion", loss_limit_note(prudent, prudent.gamma * 2, -0.05, "USD"))
        self.assertIn("currency risk", loss_limit_note(prudent, 1000.0, -0.09, "CHF"))
        self.assertIsNone(loss_limit_note(ClientProfile(ANSWERS), 3.0, -0.40, "USD"))  # "more than 30%": no limit


if __name__ == "__main__":
    unittest.main()
