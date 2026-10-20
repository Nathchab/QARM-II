"""Offline smoke test of every page with Streamlit's AppTest (synthetic prices, no download).
Skipped when Streamlit is not installed."""
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
PAGES = ["app_pages/profile.py", "app_pages/choices.py", "app_pages/portfolio.py",
         "app_pages/backtest.py", "app_pages/time_machine.py", "app_pages/how_it_works.py"]
ANSWERS = dict(QE1="e", QE2="e", QE3="d", QE4="e", QD1="e", QD2="e",
               age="d", income="a", savings="a", commitments="c", QF1=list("abcdefgh"), QF2="b")


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

    def test_every_page_runs_for_a_tolerant_client_with_low_capacity(self):
        profile = ClientProfile.from_answers(ANSWERS)
        for page in PAGES:
            with self.subTest(page=page):
                app = AppTest.from_file(str(ROOT / page), default_timeout=120)
                app.session_state["profile"] = profile
                app.run()
                self.assertEqual(len(app.exception), 0, [e.value for e in app.exception])
                self.assertEqual(len(app.error), 0, [e.value for e in app.error])

    def test_profile_page_without_profile_shows_the_questionnaire(self):
        app = AppTest.from_file(str(ROOT / "app_pages/profile.py"), default_timeout=60)
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.radio), 10)


if __name__ == "__main__":
    unittest.main()
