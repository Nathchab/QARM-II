"""Question text and scores transcribed from QARM_II_Questionnaire.docx.

The only added choices are explicitly approved ``none`` answers on QB2/QC4.
Option identifiers are stable and independent of translated/displayed labels.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Option:
    text: str
    scores: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Question:
    code: str
    text: str
    options: dict[str, Option]
    multi: bool = False
    max_selections: int | None = None
    note: str = ""


def question(code, text, rows, *, multi=False, max_selections=None, note=""):
    return Question(code, text, {
        chr(97 + i): Option(label, scores) for i, (label, scores) in enumerate(rows)
    }, multi, max_selections, note)


QUESTIONS = {q.code: q for q in [
    question("QA1", "What best describes you as an investor?", [
        ("Individual investor", {"CX": 0}),
        ("Experienced / sophisticated individual investor", {"CX": 40}),
        ("Professional investor / asset manager", {"CX": 80}),
        ("Institutional investor / pension fund / family office", {"CX": 100}),
    ]),
    question("QA2", "What is your primary investment objective?", [
        ("Preserve capital", {"CX": 0}), ("Generate stable income", {"CX": 0}),
        ("Long-term capital growth", {"CX": 20}), ("Maximize long-term return", {"CX": 30}),
        ("Diversify an existing portfolio", {"CX": 40}),
        ("Beat or track a benchmark", {"CX": 70}),
        ("Express market views / convictions", {"CX": 90}),
    ]),
    question("QB1", "How long have you been investing?", [
        ("Never", {"EX": 0}), ("< 2 years", {"EX": 20}), ("2–5 years", {"EX": 45}),
        ("5–10 years", {"EX": 70}), ("> 10 years", {"EX": 100}),
    ]),
    question("QB2", "Which asset classes have you personally invested in?", [
        ("Cash / savings", {"EX": 10}), ("Bonds", {"EX": 30}), ("Equities", {"EX": 40}),
        ("ETFs / index funds", {"EX": 40}), ("Real estate / REITs", {"EX": 50}),
        ("Commodities", {"EX": 60}), ("Cryptoassets", {"EX": 60}),
        ("Options / futures / derivatives", {"EX": 90}), ("Structured products", {"EX": 100}),
    ], multi=True, note="Subscore = highest EX level among selected asset classes."),
    question("QB3", "How frequently do you review/manage investments?", [
        ("< once/year", {"EX": 10}), ("A few times/year", {"EX": 30}),
        ("Monthly", {"EX": 50}), ("Weekly or more", {"EX": 75}),
        ("Professionally", {"EX": 100, "CX": 100}),
    ], note="If QB1 ≠ Never. Only the professional answer has a specified CX subscore."),
    question("QC1", "How would you rate your knowledge of financial markets?", [
        ("Beginner", {"KN": 0}), ("Basic", {"KN": 25}), ("Intermediate", {"KN": 50}),
        ("Advanced", {"KN": 75}), ("Professional", {"KN": 100}),
    ]),
    question("QC2", "Which statement best describes diversification?", [
        ("Many securities always eliminate risk", {"KN": 0}),
        ("Imperfectly correlated assets can reduce asset-specific risk", {"KN": 100}),
        ("Diversification means selecting highest-return assets", {"KN": 0}),
        ("I do not know", {"KN": 0}),
    ]),
    question("QC3", "Two individually volatile assets can produce lower portfolio volatility when…", [
        ("Their returns are not perfectly positively correlated", {"KN": 100}),
        ("They have the same expected return", {"KN": 0}),
        ("They are from the same sector", {"KN": 0}), ("I do not know", {"KN": 0}),
    ]),
    question("QC4", "Which concepts are you comfortable interpreting?", [
        ("Volatility", {"KN": 20, "CX": 0}), ("Sharpe ratio", {"KN": 30, "CX": 10}),
        ("Maximum drawdown", {"KN": 40, "CX": 10}),
        ("Correlation / covariance", {"KN": 50, "CX": 20}),
        ("Efficient frontier", {"KN": 65, "CX": 40}),
        ("Tracking error", {"KN": 75, "CX": 60}),
        ("Risk contribution / budgeting", {"KN": 85, "CX": 80}),
        ("Black-Litterman / market views", {"KN": 100, "CX": 100}),
    ], multi=True, note="If QC1 ≥ Intermediate. KN and CX each use the highest selected corresponding level."),
    question("QD1", "When do you expect to need a substantial part of this capital?", [
        ("Within 1 year", {"HZ": 0}), ("1–3 years", {"HZ": 20}),
        ("3–5 years", {"HZ": 40}), ("5–10 years", {"HZ": 70}), ("> 10 years", {"HZ": 100}),
    ]),
    question("QD2", "What share might you need to withdraw unexpectedly in the next 12 months?", [
        ("> 50%", {"HZ": 0}), ("25–50%", {"HZ": 20}), ("10–25%", {"HZ": 50}),
        ("< 10%", {"HZ": 80}), ("None / fully long term", {"HZ": 100}),
    ]),
    question("QE1", "Your portfolio loses 15% over three months. What do you do?", [
        ("Sell entire risky allocation", {"RT": 0}),
        ("Reduce risky assets significantly", {"RT": 20}), ("Make no change", {"RT": 55}),
        ("Buy a little more", {"RT": 80}), ("Increase risky allocation materially", {"RT": 100}),
    ]),
    question("QE2", "Largest temporary loss you could tolerate without abandoning the strategy?", [
        ("~5%", {"RT": 10}), ("~10%", {"RT": 30}), ("~20%", {"RT": 55}),
        ("~30%", {"RT": 75}), (">30%", {"RT": 100}),
    ]),
    question("QE3", "Which simplified risk/return profile would you prefer?", [
        ("Return 5% / vol 5%", {"RT": 10}), ("Return 8% / vol 10%", {"RT": 35}),
        ("Return 12% / vol 20%", {"RT": 65}), ("Return 16% / vol 30%", {"RT": 90}),
    ]),
    question("QE4", "“I accept substantial short-term fluctuations for higher long-term expected return.”", [
        ("Strongly disagree", {"RT": 0}), ("Disagree", {"RT": 25}),
        ("Neutral", {"RT": 50}), ("Agree", {"RT": 75}), ("Strongly agree", {"RT": 100}),
    ]),
    question("QF1", "Which asset classes are you willing to include?", [
        ("Global equities", {}), ("Government bonds", {}), ("Corporate bonds", {}),
        ("Commodities", {}), ("Listed real estate / REITs", {}), ("Cash / money market", {}),
        ("Emerging-market equities", {}), ("Cryptoassets", {}),
    ], multi=True, note="Universe only. Selections are recorded, not applied to optimization in this stage."),
    question("QF2", "Maximum crypto allocation?", [
        ("5%", {}), ("10%", {}), ("20%", {}), ("30%", {}),
        ("No explicit cap beyond general limits", {}),
    ], note="If crypto selected."),
    question("QF3", "Maximum EM share of equity exposure?", [
        ("10%", {}), ("20%", {}), ("1/3", {}), ("50%", {}), ("No specific cap", {}),
    ], note="If EM selected. This is a fraction of equity exposure, not total portfolio value."),
    question("QF4", "Maximum weight per individual security?", [
        ("5%", {"CX": 20}), ("10%", {"CX": 30}), ("15%", {"CX": 40}),
        ("Custom limit", {"CX": 80}), ("No security-specific cap", {"CX": 50}),
    ]),
    question("QF5", "Are short positions allowed?", [
        ("No", {"CX": 30}), ("Yes, limited gross exposure", {"CX": 70}),
        ("Yes, custom long/short limits", {"CX": 100}),
    ], note="Advanced/professional only; explicit advanced-section opt-in in this stage."),
    question("QF6", "Is leverage allowed?", [
        ("No", {"CX": 30}), ("Up to 120%", {"CX": 60}),
        ("Up to 150%", {"CX": 80}), ("Custom institutional limit", {"CX": 100}),
    ], note="Advanced/professional only; explicit advanced-section opt-in in this stage."),
    question("QG1", "Which characteristics matter most when selecting equities?", [
        ("Momentum", {}), ("Value", {}), ("Quality", {}), ("Low volatility", {}),
        ("Growth", {}), ("Market sentiment", {}), ("Diversification / low correlation", {}),
        ("Sustainability / ESG", {}),
    ], multi=True, max_selections=3, note="Choose up to 3; factor inputs, recorded only."),
    question("QG2", "How strongly should these characteristics influence the portfolio?", [
        ("Very little / near neutral", {"CX": 20}), ("Moderately", {"CX": 50}),
        ("Strongly", {"CX": 80}),
    ]),
    question("QH1", "Minimum defensive allocation?", [
        ("No minimum", {}), ("≥25% bonds + cash", {}),
        ("≥50% bonds + cash", {}), ("≥70% bonds + cash", {}),
    ], note="If preserve capital / stable income."),
    question("QH2", "Which benchmark?", [
        ("MSCI World", {"CX": 60}), ("S&P 500", {"CX": 60}),
        ("STOXX Europe 600", {"CX": 60}), ("Custom benchmark", {"CX": 100}),
    ], note="If beat/track benchmark."),
    question("QH3", "Maximum annualized tracking error?", [
        ("2%", {}), ("5%", {}), ("10%", {}), ("≥15%", {}),
    ], note="After QH2. Recorded only: tracking-error optimization is unavailable."),
    question("QH4", "Explicit market views?", [
        ("No", {"CX": 30}), ("Yes – absolute views", {"CX": 80}),
        ("Yes – relative views", {"CX": 100}),
    ], note="If market-view objective or explicit advanced option."),
    question("QH5", "Enter the view quantitatively.", [
        ("Absolute expected return / premium", {}), ("Relative out/underperformance", {}),
    ], note="If QH4 = yes. Each view is stored independently; no optimizer connection in this stage."),
    question("QH6", "Confidence in this view?", [
        ("Low", {"CV": 20}), ("Medium", {"CV": 50}), ("High", {"CV": 80}),
        ("Very high / research-backed", {"CV": 100}),
    ], note="For each entered view."),
    question("QI1", "Show how optimal weights change as target risk changes?", [
        ("No", {"CX": 20}), ("Yes", {"CX": 70}),
    ], note="If alternatives selected or intermediate/advanced. Uncertain alternative-asset eligibility uses opt-in."),
    question("QI2", "Sensitivity analysis to expected-return / covariance changes?", [
        ("No", {"CX": 20}), ("Basic sensitivity", {"CX": 60}),
        ("Extensive scenarios / perturbations", {"CX": 100}),
    ], note="Explicit optional advanced section; scenario methods are not implemented in this stage."),
    question("QJ1", "How should risk be distributed?", [
        ("Equal risk contribution", {"CX": 80}), ("Custom risk budgets", {"CX": 100}),
        ("Return-seeking with policy constraints", {"CX": 90}),
        ("Benchmark-aware active portfolio", {"CX": 100}),
    ], note="Institutional identity only in this stage."),
    question("QJ2", "Which policy constraints are required?", [
        ("Minimum government-bond allocation", {}), ("Maximum equity allocation", {}),
        ("Maximum illiquid / alternative allocation", {}), ("Maximum currency exposure", {}),
        ("Maximum active weight vs benchmark", {}), ("Other custom constraint", {}),
    ], multi=True, note="Constraints are recorded only, not enforced by the optimizer."),
]}

# User-approved extension, not part of the original Word document.
QUESTIONS["QB2"].options["none"] = Option("None of these", {"EX": 0})
QUESTIONS["QC4"].options["none"] = Option("None of these", {"KN": 0, "CX": 0})

SECTIONS = {
    "A": "Investor identity and objective", "B": "Investment experience",
    "C": "Market knowledge", "D": "Investment horizon and risk capacity",
    "E": "Risk tolerance", "F": "Asset universe and constraints",
    "G": "Investment-style preferences / stock scoring", "H": "Conditional objective branches",
    "I": "Alternative-assets / sensitivity branch", "J": "Institutional / multi-asset branch",
}

DIMENSION_LABELS = {
    "RT": ("Conservative", "Moderate-conservative", "Growth-oriented", "Aggressive"),
    "EX": ("New investor", "Limited experience", "Experienced", "Highly experienced"),
    "KN": ("Beginner", "Basic/intermediate", "Advanced", "Professional-level"),
    "HZ": ("Short / low capacity", "Medium-short", "Long", "Very long / high capacity"),
    "CX": ("Guided", "Standard", "Advanced", "Professional / institutional"),
}
