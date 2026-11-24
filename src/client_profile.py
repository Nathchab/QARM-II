"""
Investor profile: risk aversion gamma and allocation limits.

Two separate dimensions, each with its own job:

1. Risk TOLERANCE (willingness, QE1-QE4) -> gamma, the risk aversion used by the optimisers.
       gamma(RT) = GAMMA_MAX * (GAMMA_MIN / GAMMA_MAX) ** (RT / 100)
   The anchors are calibrated on QE3: with the mean-variance utility U = mu - gamma/2 * sigma^2
   of our optimiser, each QE3 option reveals an interval of gamma (see qe3_gamma_interval), and
   gamma(RT) evaluated at the RT subscore of each option lies inside the interval of that option.

2. Risk CAPACITY (ability: financial situation + horizon + liquidity) -> allocation limits
   (maximum in risky assets, minimum in bonds and short-term bonds, maximum in gold/commodities and crypto)
   and the largest loss he can afford in a bad year.
   The most restrictive source applies. Capacity does NOT change gamma: it would be counted
   twice (gamma floor + limits), and the client's tolerance would no longer matter.

Result: a tolerant client with a low capacity gets the most dynamic portfolio *inside* the
limits that his situation allows. The limits are a "box", gamma chooses the point in the box.
"""
from src.data import ASSETS, SHORT_TERM
from src.questionnaire import (CRYPTO_CAP, CRYPTO_CHOICE, HOME_BIAS, HORIZON, QE2_MAX_LOSS, QE3_PROFILES,
                               REGION_QUESTION, VIEWS_QUESTION, capacity_score, is_experienced, missing_answers,
                               products_for, risk_tolerance_score, rt_label)

GAMMA_MAX = 12.0  # gamma at RT = 0 (very risk averse)
GAMMA_MIN = 1.0   # gamma at RT = 100 (very risk tolerant)

MAX_WEIGHT_PER_ASSET = 0.40
# Capacity tiers: highest score of the tier -> (name, max in risky assets, min bonds, max gold/commodities,
# max crypto, largest loss the client can afford in a bad year). Risky assets = equities, gold and commodities,
# crypto: the horizon and capacity caps apply to all of them, not only to equities (otherwise a client who needs his
# money within 3 years could hold 20% commodities on top of his equities). Gold and commodities stay a small
# diversifier (5-15%), as in professional portfolios. The loss is needed because "minimum in bonds" alone does not
# limit the risk: the optimiser can choose 20-year bonds, which are as risky as equities.
CAPACITY_TIERS = {
    2: ("Low", 0.30, 0.50, 0.05, 0.00, -0.10),
    4: ("Limited", 0.50, 0.30, 0.10, 0.00, -0.15),
    6: ("Moderate", 0.70, 0.15, 0.10, 0.05, -0.20),
    9: ("High", 1.00, 0.00, 0.15, 0.10, None),
}
# Long-term bonds (20-year Treasuries, Swiss 7-15 years) only for money invested for at least 5 years: they can lose
# a lot when interest rates rise (TLT lost about 30% in 2022), which money needed sooner cannot afford.
LONG_BONDS = ["TLT", "CSBGC0.SW"]


# ---------- Risk tolerance -> gamma ----------
def gamma_from_rt(rt):
    """Exponential interpolation between GAMMA_MAX (RT = 0) and GAMMA_MIN (RT = 100)."""
    return GAMMA_MAX * (GAMMA_MIN / GAMMA_MAX) ** (rt / 100)


def indifference_gamma(option_1, option_2):
    """Gamma at which mu1 - gamma/2 sigma1^2 = mu2 - gamma/2 sigma2^2."""
    (m1, s1), (m2, s2) = QE3_PROFILES[option_1], QE3_PROFILES[option_2]
    return 2 * (m2 - m1) / (s2 ** 2 - s1 ** 2)


# Gamma at which the client is indifferent between two neighbouring QE3 options: 8 (a/b), 2.67 (b/c), 1.6 (c/d)
QE3_THRESHOLDS = [indifference_gamma(a, b) for a, b in zip("abc", "bcd")]


def qe3_gamma_interval(choice):
    """Interval of gamma for which `choice` is the best QE3 option (mean-variance utility)."""
    edges = [float("inf")] + QE3_THRESHOLDS + [0.0]
    i = "abcd".index(choice)
    return edges[i + 1], edges[i]


def qe3_distance(choice, gamma):
    """How many QE3 intervals separate the QE3 answer from the interval of the final gamma (0 = consistent)."""
    interval_of_gamma = sum(gamma < threshold for threshold in QE3_THRESHOLDS)  # 0 = "a", ..., 3 = "d"
    return abs(interval_of_gamma - "abcd".index(choice))


# ---------- Risk capacity -> allocation limits ----------
def capacity_tier(score, answers):
    """Tier from the capacity score; low savings or heavy commitments cap the tier at 'Limited'.
    Returns (tier, True if this safeguard lowered the tier)."""
    tier = next(values for top, values in CAPACITY_TIERS.items() if score <= top)
    fragile = answers["savings"] == "a" or answers["commitments"] == "c"
    if fragile and tier[0] in ("Moderate", "High"):
        return CAPACITY_TIERS[4], True
    return tier, False


class ClientProfile:
    """Everything the app knows about the client, computed from the answers of the questionnaire."""

    def __init__(self, answers, currency="USD"):
        missing = missing_answers(answers)
        if missing:
            raise ValueError("Please answer: " + ", ".join(missing))
        a = self.answers = dict(answers)
        self.currency = currency  # the products depend on it: bonds and cash in the currency of the portfolio
        self.region = REGION_QUESTION[1][a["QF3"]][1]

        # Risk tolerance -> gamma
        self.rt = risk_tolerance_score(a)
        self.gamma = gamma_from_rt(self.rt)
        self.label = rt_label(self.rt)
        self.qe2_loss = QE2_MAX_LOSS[a["QE2"]]  # the loss he says he can tolerate (None = more than 30%)
        # Money needed within a year, or more than half may be withdrawn within 12 months: the goal is to keep the
        # capital, not to make it grow -> we recommend the minimum variance portfolio
        self.preserve_capital = a["QD1"] == "a" or a["QD2"] == "a"
        # Contradictory answers: the final gamma is at least two QE3 intervals away from the QE3 answer
        self.inconsistent = qe3_distance(a["QE3"], self.gamma) >= 2

        # Experience -> can the client enter Black-Litterman views?
        self.experienced = is_experienced(a)
        self.wants_views = self.experienced and VIEWS_QUESTION[1][a["QH4"]][1]

        # Risk capacity -> allocation limits: the most restrictive of capacity, horizon and withdrawals
        self.capacity_score = capacity_score(a)
        (self.capacity_label, cap_equity, cap_bonds, max_other, cap_crypto, cap_loss), self.safeguard = \
            capacity_tier(self.capacity_score, a)
        # One sentence for the client: his score, its tier, and why it was lowered (if it was)
        score_tier = capacity_tier(self.capacity_score, {"savings": "b", "commitments": "a"})[0][0]
        reasons = [why for why, on in [("your emergency savings cover less than 3 months", a["savings"] == "a"),
                                       ("you have significant obligations", a["commitments"] == "c")] if on]
        tiers = "Low 0–2, Limited 3–4, Moderate 5–6, High 7–9"
        self.capacity_explanation = (f"Score {self.capacity_score}/9, which gives {score_tier} ({tiers}); lowered to "
                                     f"Limited because {' and '.join(reasons)}." if self.safeguard
                                     else f"Score {self.capacity_score}/9 ({tiers}).")
        # Loss limit of the portfolio in a bad year: what he can tolerate (QE2) or what he can afford (capacity),
        # the stricter of the two. loss_reason completes the sentence "... the 10% <loss_reason>".
        losses = {"you can tolerate (QE2)": self.qe2_loss,
                  f"your financial capacity ({self.capacity_label}) allows": cap_loss}
        losses = {reason: loss for reason, loss in losses.items() if loss is not None}
        self.loss_reason = max(losses, key=losses.get) if losses else None
        self.max_loss = losses.get(self.loss_reason)
        sources = {  # name of the source -> (maximum in risky assets, minimum in bonds, minimum in short-term)
            f"financial capacity ({self.capacity_label})": (cap_equity, cap_bonds, 0.0),
            "your horizon (QD1)": HORIZON["QD1"][1][a["QD1"]][1],
            "possible withdrawals (QD2)": HORIZON["QD2"][1][a["QD2"]][1],
        }
        max_equity = min(v[0] for v in sources.values())
        min_bonds = max(v[1] for v in sources.values())
        min_short_term = max(v[2] for v in sources.values())
        wants_crypto = CRYPTO_CHOICE in a["QF1"]
        if a["QD1"] not in ("d", "e") or a["QD2"] not in ("d", "e"):
            cap_crypto = 0.0  # no crypto for money needed within 5 years or likely to be withdrawn
        max_crypto = min(cap_crypto, CRYPTO_CAP[1][a["QF2"]][1]) if wants_crypto else 0.0
        self.recommended_constraints = {
            "max_equity": max_equity, "min_bonds": min_bonds, "min_short_term": min_short_term,
            "max_other": max_other, "max_crypto": max_crypto, "max_weight_per_asset": MAX_WEIGHT_PER_ASSET,
            "min_home_equity": 0.0 if self.region == "World" else HOME_BIAS, "region": self.region}
        self.limit_sources = {  # which answer sets each limit (shown on the profile page)
            "max_equity": [name for name, v in sources.items() if v[0] == max_equity],
            "min_bonds": [name for name, v in sources.items() if v[1] == min_bonds] if min_bonds > 0 else [],
            "min_short_term": [name for name, v in sources.items() if v[2] == min_short_term] if min_short_term > 0 else [],
        }
        self.crypto_blocked = wants_crypto and cap_crypto == 0

        # Recommended products: the asset classes chosen in QF1 (equities of the QF3 region, bonds in the currency
        # of the portfolio), plus the short-term government bonds (our "cash": without it the optimiser cannot
        # lower the risk to the level of a prudent client). With it, the limits can always be respected.
        tickers = list(dict.fromkeys(t for choice in a["QF1"] for t in products_for(choice, self.region, currency)))
        if max_crypto == 0:
            tickers = [t for t in tickers if ASSETS[t][1] != "Crypto"]
        self.short_horizon = a["QD1"] in ("a", "b", "c") or min_short_term > 0  # money needed within 5 years
        self.removed_long_bonds = [t for t in tickers if t in LONG_BONDS] if self.short_horizon else []
        tickers = [t for t in tickers if t not in self.removed_long_bonds]
        self.added_assets = []
        short = SHORT_TERM[currency]
        if short not in tickers:
            tickers.append(short)
            self.added_assets.append(short)
        if len(tickers) < 2:  # only cash was chosen: a portfolio needs two products, we add a government bond fund
            tickers.append(products_for("b", self.region, currency)[0])
            self.added_assets.append(tickers[-1])
        self.recommended_assets = tickers

        # Messages: warnings = material financial facts, notes = neutral explanations
        self.warnings, self.notes = [], []
        if a["savings"] == "a":
            self.warnings.append("Your emergency savings cover less than 3 months of expenses: "
                                 "build a cash reserve before taking market risk.")
        if a["commitments"] == "c":
            self.warnings.append("You have significant obligations or financial difficulties: "
                                 "losses on this portfolio could be hard to absorb.")
        if a["QD1"] == "a":
            self.warnings.append("You need a large part of this money within 1 year: "
                                 "risky assets are not suitable for money needed so soon.")
        if a["QD2"] in ("a", "b"):
            self.warnings.append("You may withdraw more than 25% within 12 months: "
                                 "keep that amount in cash outside the portfolio.")
        if min_short_term > 0:
            self.notes.append(f"Because of {' and '.join(self.limit_sources['min_short_term'])}, you may need part of "
                              f"this money soon: at least {min_short_term:.0%} is kept in short-term government bonds "
                              f"({short}), which almost never lose value in {currency}, instead of long bonds or equities.")
        if self.region != "World":
            self.notes.append(f"You prefer {self.region} (QF3): at least {HOME_BIAS:.0%} of your equities are invested "
                              "there, the rest stays worldwide, because one region alone is less diversified.")
        if self.safeguard:
            self.notes.append(f"Your capacity score ({self.capacity_score}/9) gives {score_tier}, but we lower it to "
                              f"Limited because {' and '.join(reasons)}: a loss would be harder to absorb.")
        if self.removed_long_bonds:
            self.notes.append("You may need all or part of this money within 5 years: we leave out long-term bonds ("
                              + ", ".join(self.removed_long_bonds) + "), which can lose a lot when interest rates "
                              "rise (about 30% in 2022). You can still add them on the next page.")
        if self.rt > 50 and max_equity <= 0.5:
            self.notes.append(f"Your risk tolerance ({self.label}) is higher than what your situation allows: "
                              f"risky assets (equities, gold and commodities, crypto) are limited to {max_equity:.0%} by "
                              f"{' and '.join(self.limit_sources['max_equity'])}. Within this limit, the "
                              "portfolio still follows your tolerance.")
        if self.crypto_blocked:
            self.notes.append("You selected crypto, but your financial capacity or your horizon does not allow it: "
                              "crypto is set to 0%.")
        if self.added_assets:
            self.notes.append("To respect your limits and to adjust the risk to your profile we added: " + ", ".join(
                f"{t} ({ASSETS[t][0]})" for t in self.added_assets) + ".")
        if self.inconsistent:  # only clear contradictions
            low, high = qe3_gamma_interval(a["QE3"])
            self.notes.append(f"Your answers point in different directions: your choice in QE3 alone implies "
                              f"γ between {low:.1f} and {'∞' if high == float('inf') else f'{high:.1f}'}, but your "
                              f"four answers together give γ = {self.gamma:.1f}. You may want to review part 1."
                              + ("" if self.preserve_capital else " Until then we recommend risk parity, which does "
                                 "not need your risk aversion (if it respects your loss limit)."))
