"""
Investor profile: risk aversion gamma and allocation limits.

Two separate dimensions, each with its own job:

1. Risk TOLERANCE (willingness, QE1-QE4) -> gamma, the risk aversion used by the optimisers.
       gamma(RT) = GAMMA_MAX * (GAMMA_MIN / GAMMA_MAX) ** (RT / 100)
   The anchors are calibrated on QE3: with the mean-variance utility U = mu - gamma/2 * sigma^2
   of our optimiser, each QE3 option reveals an interval of gamma (see qe3_gamma_interval), and
   gamma(RT) evaluated at the RT subscore of each option lies inside the interval of that option.

2. Risk CAPACITY (ability: financial situation + horizon + liquidity) -> allocation limits
   (maximum in equities, minimum in bonds, maximum in gold/commodities and crypto).
   The most restrictive source applies. Capacity does NOT change gamma: it would be counted
   twice (gamma floor + limits), and the client's tolerance would no longer matter.

Result: a tolerant client with a low capacity gets the most dynamic portfolio *inside* the
limits that his situation allows. The limits are a "box", gamma chooses the point in the box.
"""
from dataclasses import dataclass, field

from src.data import ASSETS
from src.optimizer import check_constraints
from src.questionnaire import (CRYPTO_CAP, CRYPTO_CHOICE, HORIZON, QE2_MAX_LOSS, QE3_PROFILES, UNIVERSE,
                               VIEWS_QUESTION, capacity_score, is_experienced, missing_answers,
                               risk_tolerance_score, rt_label)

GAMMA_MAX = 12.0  # gamma at RT = 0 (very risk averse)
GAMMA_MIN = 1.0   # gamma at RT = 100 (very risk tolerant)

MAX_WEIGHT_PER_ASSET = 0.40
# Capacity tiers: (name, max equity, min bonds, max gold/commodities, max crypto)
CAPACITY_TIERS = [
    (2, ("Low", 0.30, 0.50, 0.10, 0.00)),
    (4, ("Limited", 0.50, 0.30, 0.15, 0.00)),
    (6, ("Moderate", 0.70, 0.15, 0.20, 0.05)),
    (9, ("High", 1.00, 0.00, 0.20, 0.10)),
]
# Products added (in this order) when the selected asset classes cannot satisfy the limits
FILLERS = ["AGG", "SHY", "SPY", "EFA", "GLD"]


def gamma_from_rt(rt):
    """Exponential interpolation between GAMMA_MAX (RT = 0) and GAMMA_MIN (RT = 100)."""
    if not 0 <= rt <= 100:
        raise ValueError("RT must be between 0 and 100.")
    return GAMMA_MAX * (GAMMA_MIN / GAMMA_MAX) ** (rt / 100)


def indifference_gamma(option_1, option_2):
    """Gamma at which mu1 - gamma/2 sigma1^2 = mu2 - gamma/2 sigma2^2."""
    (m1, s1), (m2, s2) = QE3_PROFILES[option_1], QE3_PROFILES[option_2]
    return 2 * (m2 - m1) / (s2 ** 2 - s1 ** 2)


def qe3_gamma_interval(choice):
    """Interval of gamma for which `choice` is the best QE3 option (mean-variance utility)."""
    letters = sorted(QE3_PROFILES)
    i = letters.index(choice)
    upper = indifference_gamma(letters[i - 1], choice) if i > 0 else float("inf")
    lower = indifference_gamma(choice, letters[i + 1]) if i < len(letters) - 1 else 0.0
    return lower, upper


def qe3_bracket(gamma):
    """QE3 answer whose revealed gamma interval contains gamma."""
    return next(letter for letter in sorted(QE3_PROFILES)
                if qe3_gamma_interval(letter)[0] <= gamma <= qe3_gamma_interval(letter)[1])


def qe3_distance(choice, gamma):
    """How many QE3 intervals separate the QE3 answer from the interval of the final gamma (0 = consistent)."""
    letters = sorted(QE3_PROFILES)
    return abs(letters.index(qe3_bracket(gamma)) - letters.index(choice))


def capacity_tier(score, answers):
    """Tier from the capacity score; low savings or heavy commitments cap the tier at 'Limited'.
    Returns (tier, True if this safeguard lowered the tier)."""
    tier = next(values for top, values in CAPACITY_TIERS if score <= top)
    fragile = answers["savings"] == "a" or answers["commitments"] == "c"
    if fragile and tier[0] in ("Moderate", "High"):
        return CAPACITY_TIERS[1][1], True
    return tier, False


@dataclass
class ClientProfile:
    answers: dict
    rt: float = 0.0
    gamma: float = 0.0
    label: str = ""
    capacity_score: int = 0
    capacity_label: str = ""
    recommended_constraints: dict = field(default_factory=dict)
    recommended_assets: list = field(default_factory=list)
    limit_sources: dict = field(default_factory=dict)  # which answer sets each limit
    warnings: list = field(default_factory=list)        # material financial facts
    notes: list = field(default_factory=list)           # neutral explanations
    max_loss: float | None = None                        # QE2, used by the time machine
    safeguard: bool = False                              # tier lowered by low savings / commitments
    crypto_blocked: bool = False                         # crypto wanted but not allowed
    added_assets: list = field(default_factory=list)     # products added to make the limits feasible
    experienced: bool = False                            # QB1, QC1 and QC3 passed
    wants_views: bool = False                            # experienced and QH4 = yes

    @classmethod
    def from_answers(cls, answers):
        missing = missing_answers(answers)
        if missing:
            raise ValueError("Please answer: " + ", ".join(missing))
        p = cls(dict(answers))
        p.rt = risk_tolerance_score(answers)
        p.gamma = gamma_from_rt(p.rt)
        p.label = rt_label(p.rt)
        p.max_loss = QE2_MAX_LOSS[answers["QE2"]]
        p.experienced = is_experienced(answers)
        p.wants_views = p.experienced and VIEWS_QUESTION[1][answers["QH4"]][1]
        p._set_limits()
        p._set_assets()
        p._set_messages()
        return p

    # ---------- allocation limits ----------
    def _set_limits(self):
        a = self.answers
        self.capacity_score = capacity_score(a)
        (self.capacity_label, cap_equity, cap_bonds, max_other, cap_crypto), self.safeguard = \
            capacity_tier(self.capacity_score, a)
        sources = {
            f"financial capacity ({self.capacity_label})": (cap_equity, cap_bonds),
            "your horizon (QD1)": HORIZON["QD1"][1][a["QD1"]][1],
            "possible withdrawals (QD2)": HORIZON["QD2"][1][a["QD2"]][1],
        }
        max_equity = min(v[0] for v in sources.values())
        min_bonds = max(v[1] for v in sources.values())
        wants_crypto = CRYPTO_CHOICE in a["QF1"]
        if a["QD1"] not in ("d", "e") or a["QD2"] not in ("d", "e"):
            cap_crypto = 0.0  # no crypto for money needed within 5 years or likely to be withdrawn
        max_crypto = min(cap_crypto, CRYPTO_CAP[1][a["QF2"]][1]) if wants_crypto else 0.0
        self.recommended_constraints = {"max_equity": max_equity, "min_bonds": min_bonds,
                                        "max_other": max_other, "max_crypto": max_crypto,
                                        "max_weight_per_asset": MAX_WEIGHT_PER_ASSET}
        self.limit_sources = {
            "max_equity": [name for name, v in sources.items() if v[0] == max_equity],
            "min_bonds": [name for name, v in sources.items() if v[1] == min_bonds] if min_bonds > 0 else [],
        }
        self.crypto_blocked = wants_crypto and cap_crypto == 0

    # ---------- products ----------
    def _set_assets(self):
        tickers = [t for choice in self.answers["QF1"] for t in UNIVERSE[1][choice][1] if t in ASSETS]
        if self.recommended_constraints["max_crypto"] == 0:
            tickers = [t for t in tickers if ASSETS[t][1] != "Crypto"]
        for filler in FILLERS:
            if not check_constraints(tickers, self.recommended_constraints):
                break
            if filler not in tickers:
                tickers.append(filler)
                self.added_assets.append(filler)
        self.recommended_assets = tickers

    # ---------- messages ----------
    def _set_messages(self):
        a, c = self.answers, self.recommended_constraints
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
        if self.safeguard:
            self.notes.append("Because of low savings or heavy commitments, your capacity is treated as "
                              "'Limited' at most.")
        if self.rt > 50 and c["max_equity"] <= 0.5:
            self.notes.append(f"Your risk tolerance ({self.label}) is higher than what your situation allows: "
                              f"equities are limited to {c['max_equity']:.0%} by "
                              f"{' and '.join(self.limit_sources['max_equity'])}. Within this limit, the "
                              "portfolio still follows your tolerance.")
        if self.crypto_blocked:
            self.notes.append("You selected crypto, but your financial capacity or your horizon does not allow it: "
                              "crypto is set to 0%.")
        if self.added_assets:
            self.notes.append("To respect your limits we added: " + ", ".join(
                f"{t} ({ASSETS[t][0]})" for t in self.added_assets) + ".")
        # Only clear contradictions: the final gamma is at least two QE3 intervals away from the QE3 answer
        if qe3_distance(a["QE3"], self.gamma) >= 2:
            low, high = qe3_gamma_interval(a["QE3"])
            self.notes.append(f"Your answers point in different directions: your choice in QE3 alone implies "
                              f"γ between {low:.1f} and {'∞' if high == float('inf') else f'{high:.1f}'}, but your "
                              f"four answers together give γ = {self.gamma:.1f}. You may want to review part 1.")
