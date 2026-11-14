"""
Stress test: what would have happened to the client's portfolio during past crises?
We buy the portfolio on the first day of the crisis and hold it (no rebalancing).
"""

# name: (start = market peak, end = market bottom, short story)
CRISES = {
    "2008 Global Financial Crisis": ("2007-10-09", "2009-03-09",
                                     "US housing bubble bursts, Lehman Brothers fails, stocks lose more than half."),
    "2020 COVID crash": ("2020-02-19", "2020-03-23",
                         "Lockdowns all over the world: the fastest stock market crash in history."),
    "2022 Inflation shock": ("2022-01-03", "2022-10-12",
                             "Inflation and fast rate hikes: stocks AND bonds fall together."),
}


def replay(prices, weights, start, end=None):
    """Value over time of 1 unit invested at `start` (buy and hold)."""
    p = prices.loc[start:end, weights.index]
    growth = (p / p.iloc[0]).fillna(1)  # an asset that did not exist yet is kept as cash
    return growth @ weights


def crisis_summary(prices, weights, start, end):
    """Worst loss during the crisis and number of months to get the money back."""
    value = replay(prices, weights, start)  # from the start of the crisis until today
    worst_loss = value.loc[:end].min() - 1
    bottom = value.loc[:end].idxmin()
    after = value.loc[bottom:]
    recovered = after[after >= 1]
    if worst_loss >= 0:
        months = 0
    elif len(recovered) == 0:
        months = None  # not recovered yet
    else:
        months = round((recovered.index[0] - value.index[0]).days / 30.4)
    return worst_loss, months
