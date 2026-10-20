"""
Investor questionnaire (taken from QARM_II_Questionnaire.docx, sections D, E and F).

Only the questions that change the portfolio are asked:
  - E. Risk tolerance (QE1-QE4)      -> score RT (0-100)     -> risk aversion gamma
  - D. Horizon and liquidity (QD1-QD2) -> allocation limits
  - Financial situation (4 questions)  -> capacity score (0-9) -> allocation limits
  - F. Asset universe (QF1-QF2)        -> recommended products and crypto limit
Black-Litterman views and their confidence (QH5-QH6) are entered on the portfolio page.

Each answer is stored as a letter "a", "b", ... (or a list of letters for QF1).
"""
from statistics import mean


def options(*rows):
    """[(text, value), ...] -> {"a": (text, value), "b": ...}"""
    return {chr(97 + i): row for i, row in enumerate(rows)}


# ---------- E. Risk tolerance: (text, RT subscore 0-100) ----------
RISK_TOLERANCE = {
    "QE1": ("Your portfolio loses 15% over three months. What do you do?", options(
        ("Sell my entire risky allocation", 0),
        ("Reduce risky assets significantly", 20),
        ("Make no change", 55),
        ("Buy a little more", 80),
        ("Increase my risky allocation materially", 100))),
    "QE2": ("What is the largest temporary loss you could tolerate without abandoning the strategy?", options(
        ("About 5%", 10), ("About 10%", 30), ("About 20%", 55), ("About 30%", 75), ("More than 30%", 100))),
    "QE3": ("Which risk/return profile would you prefer for one year?", options(
        ("Expected return 5%, volatility 5%", 10),
        ("Expected return 8%, volatility 10%", 35),
        ("Expected return 12%, volatility 20%", 65),
        ("Expected return 16%, volatility 30%", 90))),
    "QE4": ("“I accept substantial short-term fluctuations for a higher long-term expected return.”", options(
        ("Strongly disagree", 0), ("Disagree", 25), ("Neutral", 50), ("Agree", 75), ("Strongly agree", 100))),
}

# QE3 options as (expected return, volatility): used to check the gamma calibration
QE3_PROFILES = {"a": (0.05, 0.05), "b": (0.08, 0.10), "c": (0.12, 0.20), "d": (0.16, 0.30)}

# QE2 as a number, for the time machine (None = the client accepts more than 30%)
QE2_MAX_LOSS = {"a": -0.05, "b": -0.10, "c": -0.20, "d": -0.30, "e": None}

# ---------- D. Horizon and liquidity: (text, (max equity, min bonds)) ----------
HORIZON = {
    "QD1": ("When do you expect to need a substantial part of this capital?", options(
        ("Within 1 year", (0.2, 0.6)),
        ("In 1–3 years", (0.4, 0.4)),
        ("In 3–5 years", (0.6, 0.2)),
        ("In 5–10 years", (0.8, 0.0)),
        ("In more than 10 years", (1.0, 0.0)))),
    "QD2": ("What share might you need to withdraw unexpectedly in the next 12 months?", options(
        ("More than 50%", (0.2, 0.6)),
        ("25–50%", (0.4, 0.4)),
        ("10–25%", (0.6, 0.2)),
        ("Less than 10%", (0.8, 0.0)),
        ("None, this money is fully long term", (1.0, 0.0)))),
}

# ---------- Financial situation: (text, capacity points) ----------
CAPACITY = {
    "age": ("Your age", options(("Under 30", 3), ("30–45", 2), ("46–60", 1), ("Over 60", 0))),
    "income": ("How stable is your income?", options(
        ("No regular income", 0), ("Fairly stable", 1), ("Very stable", 2))),
    "savings": ("How many months of essential expenses could your emergency savings cover, "
                "outside this investment?", options(
        ("Less than 3 months", 0), ("3–6 months", 1), ("More than 6 months", 2))),
    "commitments": ("How would you describe your financial commitments (debts, dependants, obligations)?", options(
        ("None significant", 2), ("Manageable", 1), ("Significant obligations or financial difficulties", 0))),
}

# ---------- F. Asset universe: (text, products) ----------
UNIVERSE = ("Which asset classes are you willing to include?", options(
    ("Global equities", ["SPY", "EFA"]),
    ("Government bonds", ["AGG", "TLT"]),
    ("Corporate bonds", ["LQD"]),
    ("Gold and commodities", ["GLD", "DBC"]),
    ("Listed real estate (REITs)", ["VNQ"]),
    ("Cash / short-term Treasuries", ["SHY"]),
    ("Emerging-market equities", ["EEM"]),
    ("Cryptoassets", ["BTC-USD"])))
CRYPTO_CHOICE = "h"
CRYPTO_CAP = ("What is the maximum you want in crypto?", options(
    ("5%", 0.05), ("10%", 0.10), ("20%", 0.20), ("30%", 0.30), ("No explicit cap", 1.0)))

RT_LABELS = ["Conservative", "Moderate-conservative", "Growth-oriented", "Aggressive"]


def single_choice_questions():
    """All single-choice questions in display order: code -> (text, options)."""
    return {**RISK_TOLERANCE, **HORIZON, **CAPACITY}


def missing_answers(answers):
    """Codes of the questions that still need an answer."""
    missing = [code for code, (_, opts) in single_choice_questions().items() if answers.get(code) not in opts]
    universe = answers.get("QF1")
    if not isinstance(universe, list) or not universe or any(u not in UNIVERSE[1] for u in universe):
        missing.append("QF1")
    elif CRYPTO_CHOICE in universe and answers.get("QF2") not in CRYPTO_CAP[1]:
        missing.append("QF2")
    return missing


def risk_tolerance_score(answers):
    """RT = mean of the QE1-QE4 subscores (as in the questionnaire document), between 5 and 97.5."""
    return mean(RISK_TOLERANCE[code][1][answers[code]][1] for code in RISK_TOLERANCE)


def capacity_score(answers):
    """Sum of the points of the four financial questions, between 0 and 9."""
    return sum(CAPACITY[code][1][answers[code]][1] for code in CAPACITY)


def rt_label(rt):
    return RT_LABELS[0 if rt <= 25 else 1 if rt <= 50 else 2 if rt <= 75 else 3]
