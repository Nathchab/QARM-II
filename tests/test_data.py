"""Download from Yahoo Finance: a ticker that fails is tried again; essential series must be there."""
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src import data
from src.data import ASSETS, FX, RF


def fake_yahoo(failing):
    """A fake yf.download: random prices, except for the tickers in `failing` (empty columns)."""
    calls = []

    def download(tickers, **kwargs):
        calls.append(list(tickers))
        days = pd.bdate_range("2005-01-03", "2010-12-31")
        prices = pd.DataFrame(100 * np.exp(np.random.default_rng(0).normal(0, .01, (len(days), len(tickers))).cumsum(0)),
                              index=days, columns=tickers)
        prices[[t for t in tickers if t in failing]] = np.nan
        return pd.concat({"Close": prices}, axis=1)
    return download, calls


class DownloadTests(unittest.TestCase):
    def setUp(self):
        data.download_prices.clear()

    def test_a_failed_ticker_is_tried_again(self):
        download, calls = fake_yahoo(failing=set())
        first = {"done": False}

        def flaky(tickers, **kwargs):  # EFA fails the first time only
            result = download(tickers, **kwargs)
            if not first["done"]:
                first["done"] = True
                result[("Close", "EFA")] = np.nan
            return result
        with patch("src.data.yf.download", flaky):
            prices = data.download_prices()
        self.assertTrue(prices["EFA"].notna().all())
        self.assertEqual(calls[1], ["EFA"])

    def test_a_missing_essential_series_stops_the_app(self):
        for ticker in ["AGG", "SHY", FX, RF]:
            data.download_prices.clear()
            with patch("src.data.yf.download", fake_yahoo(failing={ticker})[0]):
                with self.assertRaises(ConnectionError):
                    data.download_prices()

    def test_a_missing_product_is_not_offered(self):
        with patch("src.data.yf.download", fake_yahoo(failing={"QQQ"})[0]):
            self.assertNotIn("QQQ", data.available_assets())
            self.assertEqual(len(data.available_assets()), len(ASSETS) - 1)


if __name__ == "__main__":
    unittest.main()
