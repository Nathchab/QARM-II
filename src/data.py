"""
Data: the app downloads all the prices directly from Yahoo Finance (yfinance) when it starts.
Nothing has to be run before: no file, no script. The prices stay in memory for one day
(Streamlit cache), and the user can download them again with the "Update prices" button.
"""
import pandas as pd
import streamlit as st
import yfinance as yf

from src.black_litterman import MARKET_PORTFOLIO

# Products the client can choose from: ticker -> (name, asset class)
# The ".SW" products trade in CHF on the Swiss exchange (SIX); all the others trade in USD.
ASSETS = {
    # Equities
    "SPY": ("US large caps (S&P 500)", "Equity"),
    "QQQ": ("US technology (Nasdaq 100)", "Equity"),
    "IWM": ("US small caps (Russell 2000)", "Equity"),
    "ACWI": ("World equities (MSCI ACWI)", "Equity"),
    "EFA": ("Developed markets ex-US", "Equity"),
    "VGK": ("European equities", "Equity"),
    "EZU": ("Eurozone equities", "Equity"),
    "EWU": ("UK equities", "Equity"),
    "EWG": ("German equities", "Equity"),
    "CSSMI.SW": ("Swiss large caps (SMI, in CHF)", "Equity"),
    "CSSMIM.SW": ("Swiss mid caps (SMIM, in CHF)", "Equity"),
    "EWL": ("Swiss equities (in USD)", "Equity"),
    "EWJ": ("Japanese equities", "Equity"),
    "EEM": ("Emerging markets equities", "Equity"),
    "VNQ": ("US real estate (REITs)", "Equity"),
    # Bonds
    "AGG": ("US aggregate bonds", "Bond"),
    "SHY": ("US Treasuries 1-3 years", "Bond"),
    "TLT": ("US Treasuries 20+ years", "Bond"),
    "TIP": ("US inflation-linked bonds", "Bond"),
    "LQD": ("US corporate bonds (investment grade)", "Bond"),
    "HYG": ("US high yield bonds", "Bond"),
    "EMB": ("Emerging markets bonds", "Bond"),
    "BNDX": ("International bonds (hedged to USD)", "Bond"),
    "BWX": ("International government bonds", "Bond"),
    "CSBGC3.SW": ("Swiss government bonds 0-3 years (CHF)", "Bond"),
    "CSBGC7.SW": ("Swiss government bonds 3-7 years (CHF)", "Bond"),
    "CSBGC0.SW": ("Swiss government bonds 7-15 years (CHF)", "Bond"),
    "CHCORP.SW": ("Swiss franc corporate bonds (CHF)", "Bond"),
    # Others
    "GLD": ("Gold", "Other"),
    "SLV": ("Silver", "Other"),
    "DBC": ("Commodities", "Other"),
    # Crypto
    "BTC-USD": ("Bitcoin", "Crypto"),
    "ETH-USD": ("Ethereum", "Crypto"),
}

CHF_LISTED = [t for t in ASSETS if t.endswith(".SW")]
# The "cash" of the robo-advisor: short-term government bonds, almost no risk, no maximum weight.
# SHY for a portfolio in USD, Swiss government bonds 0-3 years for a portfolio in CHF (no currency risk).
SHORT_TERM = {"USD": "SHY", "CHF": "CSBGC3.SW"}
# Narrow "satellite" products (one sector, one country, one risk): at most 15% each, so that the portfolio stays
# built around broad "core" products (core-satellite approach). Without this, the most tolerant clients got
# 35% in real estate (VNQ) and 21% in emerging markets. The Swiss large caps (CSSMI.SW) are not a satellite:
# they are the core of the clients who want to invest in Switzerland.
SATELLITES = ["QQQ", "IWM", "EWL", "EWJ", "EEM", "VNQ", "HYG", "EMB", "SLV", "EWU", "EWG", "CSSMIM.SW"]

# Regional exposure of each product: approximate shares of the index by region (rounded, for the charts and for
# the "at least half of your equities in your region" rule). Gold, commodities and crypto have no region.
REGIONS = ["Switzerland", "Europe", "North America", "Japan & Pacific", "Emerging markets", "Gold, commodities, crypto"]
HOME_REGIONS = {"Switzerland": ["Switzerland"], "Europe": ["Europe", "Switzerland"], "USA": ["North America"]}
_US = {"North America": 1}
REGION_SHARES = {
    "SPY": _US, "QQQ": _US, "IWM": _US, "VNQ": _US, "AGG": _US, "SHY": _US, "TLT": _US, "TIP": _US, "LQD": _US,
    "HYG": _US,
    "ACWI": {"North America": .67, "Europe": .11, "Switzerland": .02, "Japan & Pacific": .08, "Emerging markets": .12},
    "EFA": {"Europe": .55, "Switzerland": .10, "Japan & Pacific": .35},
    "VGK": {"Europe": .86, "Switzerland": .14},
    "EZU": {"Europe": 1}, "EWU": {"Europe": 1}, "EWG": {"Europe": 1},
    "CSSMI.SW": {"Switzerland": 1}, "CSSMIM.SW": {"Switzerland": 1}, "EWL": {"Switzerland": 1},
    "CSBGC3.SW": {"Switzerland": 1}, "CSBGC7.SW": {"Switzerland": 1}, "CSBGC0.SW": {"Switzerland": 1},
    "CHCORP.SW": {"Switzerland": 1},
    "EWJ": {"Japan & Pacific": 1}, "EEM": {"Emerging markets": 1}, "EMB": {"Emerging markets": 1},
    "BNDX": {"Europe": .58, "Japan & Pacific": .22, "North America": .07, "Emerging markets": .13},
    "BWX": {"Europe": .50, "Japan & Pacific": .30, "North America": .08, "Emerging markets": .12},
    **{t: {"Gold, commodities, crypto": 1} for t in ["GLD", "SLV", "DBC", "BTC-USD", "ETH-USD"]},
}
SATELLITE_MAX = 0.15

FX = "CHF=X"   # number of CHF for 1 USD
RF = "^IRX"    # 13-week US Treasury bill yield, in % per year

# CHF risk-free rate: approximate yearly average of the SNB policy rate, in % (source: SNB)
SNB_RATE = {2005: 0.75, 2006: 1.5, 2007: 2.5, 2008: 2.0, 2009: 0.25, 2010: 0.0, 2011: 0.0,
            2012: 0.0, 2013: 0.0, 2014: 0.0, 2015: -0.75, 2016: -0.75, 2017: -0.75, 2018: -0.75,
            2019: -0.75, 2020: -0.75, 2021: -0.75, 2022: 0.25, 2023: 1.5, 2024: 1.25, 2025: 0.1,
            2026: 0.0}


def _download(tickers):
    data = yf.download(tickers, start="2005-01-01", auto_adjust=True, progress=False)["Close"]
    return data.reindex(columns=tickers)  # a product that failed to download stays empty


@st.cache_data(ttl="1d", show_spinner="Downloading the latest prices from Yahoo Finance (about 10 seconds)...")
def download_prices():
    """Daily prices since 2005 of all products, the USD/CHF rate and the US T-bill rate."""
    tickers = list(ASSETS) + [FX, RF]
    data = _download(tickers)
    missing = [t for t in tickers if data[t].count() < 250]
    if missing:  # Yahoo sometimes refuses a few tickers when it gets many requests at once: we try them once more
        data[missing] = _download(missing).reindex(data.index)
    # Without these series no portfolio can be computed: the market portfolio (equilibrium returns), the "cash"
    # of the portfolios, the USD/CHF rate and the riskless rate. A missing product is simply not offered.
    if data[list(MARKET_PORTFOLIO) + list(SHORT_TERM.values()) + [FX, RF]].count().min() < 1000:
        raise ConnectionError("The download from Yahoo Finance failed")
    data = data[data["SPY"].notna()]  # keep only trading days (crypto also trades on weekends)
    return data.ffill()


def available_assets():
    """Products for which we have data."""
    data = download_prices()
    return [t for t in ASSETS if data[t].notna().any()]


def first_date(ticker):
    return download_prices()[ticker].first_valid_index()


def load_prices(currency="USD"):
    """Daily prices of all products, in USD or in CHF (price in CHF = price in USD x CHF per USD)."""
    data = download_prices()
    prices = data[list(ASSETS)].copy()
    usd_listed = [t for t in ASSETS if t not in CHF_LISTED]
    if currency == "CHF":
        prices[usd_listed] = prices[usd_listed].mul(data[FX], axis=0)
    else:
        prices[CHF_LISTED] = prices[CHF_LISTED].div(data[FX], axis=0)
    return prices


def monthly_returns(prices):
    """Monthly simple returns (NaN before a product exists)."""
    return prices.resample("ME").last().pct_change(fill_method=None)


def risk_free(currency="USD"):
    """Monthly risk-free rate (as a monthly return)."""
    data = download_prices()
    if currency == "USD":
        yearly = data[RF].resample("ME").mean() / 100
    else:
        months = data[RF].resample("ME").mean().index
        yearly = pd.Series([SNB_RATE.get(year, 0.0) for year in months.year], index=months) / 100
    return yearly / 12
