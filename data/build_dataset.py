"""
Builds the model-ready IPL auction dataset from real, independently-sourced data.

Source of truth: data/external/cricsheet_training_dataset.csv — real IPL auction
prices (from published auction records) joined to real per-season player stats
computed from Cricsheet ball-by-ball data. Both were produced by a data pipeline
built for a related project (CricLens); this script re-derives a fresh
model-ready CSV from those already-audited, real numbers rather than
re-scraping. Nothing in this file is synthetic or price-derived — see
docs/DATA_QUALITY.md for exactly what's real, what's missing, and why.

Nationality is joined in separately from data/external/ipl_auction_raw.csv
(same provenance) because the training-dataset export doesn't carry it.

Missing stats are NEVER filled. A player who has never bowled gets NaN for
economy/bowling average, not a league-average substitute — filling would
invent information the auction market never had. Each stat that can be
missing gets an explicit `<stat>_is_missing` flag so the model can learn
"this player has no bowling record" as a real, distinct signal.
"""

from pathlib import Path
import pandas as pd
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
EXTERNAL_DIR = BASE_DIR / "data" / "external"
TRAINING_SOURCE = EXTERNAL_DIR / "cricsheet_training_dataset.csv"
AUCTION_RAW = EXTERNAL_DIR / "ipl_auction_raw.csv"
OUTPUT_PATH = BASE_DIR / "data" / "ipl_auction_data.csv"

ROLE_MAP = {
    "Batter": "Batsman",
    "Bowler": "Bowler",
    "All-Rounder": "All-Rounder",
    "Wicketkeeper": "Wicketkeeper-Batsman",
}

# `franchise` below stays the name in force at the time (historically
# accurate for a year-by-year trend line). `current_franchise` collapses
# post-rename identities so franchise-level totals aren't split across a
# team's old and new name — e.g. Delhi Daredevils/Delhi Capitals are the
# same ownership group, just renamed in 2019.
CURRENT_FRANCHISE_MAP = {
    "Delhi Daredevils": "Delhi Capitals",
    "Kings XI Punjab": "Punjab Kings",
    "Royal Challengers Bangalore": "Royal Challengers Bengaluru",
    "Rising Pune Supergiants": "Rising Pune Supergiants",
}


def minmax(series: pd.Series) -> pd.Series:
    """0-1 rescale that leaves NaN as NaN so an undefined rate never scores."""
    lo, hi = series.min(skipna=True), series.max(skipna=True)
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.nan, index=series.index)
    return (series - lo) / (hi - lo)


def compute_performance_score(df: pd.DataFrame) -> pd.Series:
    """
    Blend of normalised career stats, weighted by role. Every sub-score is
    NaN-safe: a player with no bowling record contributes only their (real,
    measured) wicket count, not a guessed economy rate. This mirrors the
    same honest-blending approach used in CricLens's build_training_dataset.py,
    reimplemented here against this project's own column names.
    """
    norm_avg = minmax(df["battingavg"])
    norm_sr = minmax(df["battingsr"])
    norm_wkts = minmax(df["wickets"])
    norm_econ_inv = minmax(-df["economyrate"])
    norm_stump = minmax(df["stumpings"])

    def blend(*parts):
        return pd.concat(parts, axis=1).mean(axis=1, skipna=True)

    batting = blend(norm_avg, norm_sr)
    bowling = blend(norm_wkts, norm_econ_inv)

    score = pd.Series(index=df.index, dtype=float)
    is_bat = df["role"] == "Batsman"
    is_bowl = df["role"] == "Bowler"
    is_all = df["role"] == "All-Rounder"
    is_keeper = df["role"] == "Wicketkeeper-Batsman"

    score[is_bat] = batting[is_bat]
    score[is_bowl] = bowling[is_bowl]
    score[is_all] = 0.5 * batting[is_all] + 0.5 * bowling[is_all]
    score[is_keeper] = 0.8 * batting[is_keeper] + 0.2 * norm_stump[is_keeper]

    return (score * 100).round(2)


def build_dataset() -> pd.DataFrame:
    td = pd.read_csv(TRAINING_SOURCE)
    td = td[td["competition"] == "IPL"].copy()

    raw = pd.read_csv(AUCTION_RAW)
    country_lookup = (
        raw[["player", "season", "country"]]
        .drop_duplicates(subset=["player", "season"])
    )
    td = td.merge(country_lookup, on=["player", "season"], how="left")

    def nationality(country):
        if pd.isna(country):
            return "Unknown"
        return "Indian" if country == "India" else "Overseas"

    td["nationality"] = td["country"].map(nationality)
    td["role"] = td["role"].map(ROLE_MAP)
    td = td.dropna(subset=["role"])  # 4 rows with no resolvable role at all

    out = pd.DataFrame({
        "year": td["season"].astype(int),
        "playername": td["player"],
        "role": td["role"],
        "nationality": td["nationality"],
        "career_matches": td["career_matches"],
        "battingavg": td["career_batting_average"],
        "battingsr": td["career_strike_rate"],
        "runsscored": td["career_runs"],
        "fours": td["career_fours"],
        "sixes": td["career_sixes"],
        "wickets": td["career_wickets"],
        "economyrate": td["career_economy"],
        "bowlingavg": td["career_bowling_average"],
        "bowlingsr": td["career_bowling_strike_rate"],
        "catches": td["career_catches"],
        "stumpings": td["career_stumpings"],
        "basepricecr": (td["base_price_lacs"] / 100).round(3),
        "soldpricecr": td["sold_price_cr"],
        "franchise": td["team"],
    })
    out["current_franchise"] = out["franchise"].map(
        lambda f: CURRENT_FRANCHISE_MAP.get(f, f)
    )

    # missing-indicator flags — real absence, encoded as a signal, not filled
    for col in ["battingavg", "battingsr", "economyrate", "bowlingavg"]:
        out[f"{col}_missing"] = out[col].isna().astype(int)

    out["performancescore"] = compute_performance_score(out)
    out["valueindex"] = (
        out["performancescore"] / out["soldpricecr"].replace(0, np.nan)
    ).round(2)

    out = out.sort_values(["year", "soldpricecr"], ascending=[True, False]).reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_PATH, index=False)
    return out


if __name__ == "__main__":
    print("Building IPL auction dataset from real, independently-sourced stats...")
    df = build_dataset()
    print(f"Dataset saved to: {OUTPUT_PATH}")
    print(f"Total records: {len(df)}")
    print(f"Years covered: {df['year'].min()} to {df['year'].max()}")
    print(f"Unique players: {df['playername'].nunique()}")
    print(f"Price range: {df['soldpricecr'].min():.2f} Cr to {df['soldpricecr'].max():.2f} Cr")
    print(f"Role breakdown:\n{df['role'].value_counts()}")
    print(f"Missing career_batting_average: {df['battingavg'].isna().sum()} / {len(df)}")
    print(f"Missing career_economy: {df['economyrate'].isna().sum()} / {len(df)}")
    print(df.head(5).to_string(index=False))
