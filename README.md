# Robo-advisor – QARM II project (HEC Lausanne)

A Streamlit web app that works like a robo-advisor:

1. **Your profile**: an adaptive questionnaire describes risk tolerance and investor preferences.
   Four financial questions assess capacity. Gamma is calculated automatically from risk tolerance,
   with a financial-capacity floor.
2. **Your products & constraints**: start from suggested products and allocation limits, then choose
   ETFs and cryptocurrencies. Optional preferences can tighten the profile's equity, bond, crypto
   and per-product limits.
3. **Your portfolio**: Mean-Variance, Resampled Mean-Variance (Michaud), Black-Litterman with optional
   return views entered on the portfolio page, and Risk Parity (ERC). Weights, expected return,
   volatility, Sharpe ratio, risk contributions, efficient frontier, concentration, beta and tracking
   error. Risk Parity does not use gamma.
4. **Backtest**: out-of-sample comparisons of the four methods. Choose the period, benchmark,
   rebalancing frequency, transaction cost and turnover limit, with optional cost-aware optimisation.
5. **Time machine**: buy-and-hold replays of the 2008 financial crisis, 2020 COVID crash and 2022
   inflation shock, showing losses, recovery times and comparison with a benchmark.

Everything can be shown in USD or CHF. Prices are downloaded automatically from Yahoo Finance,
cached for up to one day, and can be refreshed with the "Update data now" button.

## Run the app

Use Python 3.12, then run from the project folder:

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

On macOS/Linux, activate with `source .venv/bin/activate` instead.

## Code structure

```
app.py                    Streamlit menu and sidebar
app_pages/                one file per page
src/questionnaire_*.py    adaptive questions, scoring and saved answers
src/client_profile.py     investor profile, gamma and allocation limits
src/financial_capacity.py financial-capacity scoring and limits
src/gamma_policy.py       automatic risk-tolerance-to-gamma mapping
src/data.py               products, prices, returns and risk-free rates
src/optimizer.py          portfolio optimizers and efficient frontier
src/black_litterman.py    equilibrium returns and posterior estimates
src/portfolios.py         four portfolio methods and benchmarks
src/backtest.py           out-of-sample backtesting and performance measures
src/time_machine.py       historical crisis replay
tests/                    automated tests
```

## Data

19 ETFs covering equities, bonds, precious metals and commodities, plus Bitcoin and Ethereum.
Yahoo Finance supplies adjusted prices, the USD/CHF rate (`CHF=X`) and the 13-week US T-bill
yield (`^IRX`). The CHF risk-free rate uses stored annual approximations of the SNB policy rate.
