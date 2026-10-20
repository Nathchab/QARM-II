"""
Data: the app downloads all the prices directly from Yahoo Finance (yfinance) when it starts.
Nothing has to be run before: no file, no script. The prices stay in memory for one day
(Streamlit cache), and the user can download them again with the "Update data now" button.
"""
import pandas as pd
import streamlit as st
import yfinance as yf

# Products the client can choose from: ticker -> (name, asset class)
ASSETS = {
    # Equities
    "SPY": ("US large caps (S&P 500)", "Equity"),
    "QQQ": ("US technology (Nasdaq 100)", "Equity"),
    "IWM": ("US small caps (Russell 2000)", "Equity"),
    "EFA": ("Developed markets ex-US", "Equity"),
    "VGK": ("European equities", "Equity"),
    "EWL": ("Swiss equities", "Equity"),
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
    # Others
    "GLD": ("Gold", "Other"),
    "SLV": ("Silver", "Other"),
    "DBC": ("Commodities", "Other"),
    # Crypto
    "BTC-USD": ("Bitcoin", "Crypto"),
    "ETH-USD": ("Ethereum", "Crypto"),
}

FX = "CHF=X"   # number of CHF for 1 USD
RF = "^IRX"    # 13-week US Treasury bill yield, in % per year

# CHF risk-free rate: approximate yearly average of the SNB policy rate, in % (source: SNB)
SNB_RATE = {2005: 0.75, 2006: 1.5, 2007: 2.5, 2008: 2.0, 2009: 0.25, 2010: 0.0, 2011: 0.0,
            2012: 0.0, 2013: 0.0, 2014: 0.0, 2015: -0.75, 2016: -0.75, 2017: -0.75, 2018: -0.75,
            2019: -0.75, 2020: -0.75, 2021: -0.75, 2022: 0.25, 2023: 1.5, 2024: 1.25, 2025: 0.1,
            2026: 0.0}


@st.cache_data(ttl="1d", show_spinner="Downloading the latest prices from Yahoo Finance (about 10 seconds)...")
def download_prices():
    """Daily prices since 2005 of all products, the USD/CHF rate and the US T-bill rate."""
    tickers = list(ASSETS) + [FX, RF]
    data = yf.download(tickers, start="2005-01-01", auto_adjust=True, progress=False)["Close"]
    data = data.reindex(columns=tickers)  # a product that failed to download stays empty
    if data["SPY"].count() < 1000:
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
    """Daily prices of all products, in USD or in CHF."""
    data = download_prices()
    prices = data[list(ASSETS)]
    if currency == "CHF":
        prices = prices.mul(data[FX], axis=0)  # price in CHF = price in USD x CHF per USD
    return prices


def monthly_returns(prices):
    """Monthly simple returns (NaN before a product exists)."""
    return prices.resample("ME").last().pct_change()


def risk_free(currency="USD"):
    """Monthly risk-free rate (as a monthly return)."""
    data = download_prices()
    if currency == "USD":
        yearly = data[RF].resample("ME").mean() / 100
    else:
        months = data[RF].resample("ME").mean().index
        yearly = pd.Series([SNB_RATE.get(year, 0.0) for year in months.year], index=months) / 100
    return yearly / 12
