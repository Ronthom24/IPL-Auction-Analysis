# Data quality

**Read this before quoting a number from this project.**

The data is real — nothing in `data/ipl_auction_data.csv` is synthetic — but
it is not complete or perfectly clean. This file is the single place issues
are tracked.

Last rebuilt: 2026-09-28.

---

## What's real

- **Auction prices** (`soldpricecr`, `basepricecr`) come from published IPL
  auction records for every year 2009–2026.
- **Player performance stats** (`battingavg`, `battingsr`, `wickets`,
  `economyrate`, `bowlingavg`, `bowlingsr`, `runsscored`, `fours`, `sixes`,
  `catches`, `stumpings`, `career_matches`) are computed from real Cricsheet
  ball-by-ball delivery data — a player's career figures **as of the season
  they were auctioned**, not backfilled from later seasons.
- Both were sourced via a data pipeline built for a related project, then
  re-joined and re-scored by `data/build_dataset.py` in this repo. See the
  README's "Data provenance" section.

## Fixed in this rebuild (2026-09-28)

| Was | Now |
|---|---|
| Performance stats (`battingavg`, `battingsr`, `wickets`, `economyrate`) were generated as a mathematical function of the player's own historical auction price — the model was partly predicting price from a noisy transform of itself. Reported R² = 0.66. | Stats are real, computed independently from ball-by-ball data. No feature is derived from the target. Honest R² = 0.33 (grouped CV). |
| `sold_classifier.pkl` was trained on an old dataset that had unsold rows, then silently reused against a newer dataset that had none — the app showed a "Sale Probability" from a stale, mismatched model. | Removed entirely. No classifier is trained or shown; `models/classifier_metrics.json` records why. |
| Missing bowling/batting stats for specialist players weren't distinguished from zero performance. | Explicit `_missing` flags (`battingavg_missing`, `battingsr_missing`, `economyrate_missing`, `bowlingavg_missing`); the value itself stays `NaN` rather than being filled. |
| R² was reported from a single random 80/20 split, which lets a player's other auction years leak between train and test. | Headline metric is 5-fold CV grouped by player (`GroupKFold`); the random-split number is kept separately, labelled, and never called the headline. |
| Renamed franchises (Delhi Daredevils/Capitals, Kings XI Punjab/Punjab Kings, RCB Bangalore/Bengaluru) were two separate rows in franchise-level aggregates, understating both. | `current_franchise` column merges them for aggregation; the raw historical name is kept per-record. |

## Known gaps — still true after this rebuild

- **No real unsold-player data exists anywhere checked.** Published auction
  result tables only list who was bought. A real sold/unsold classifier would
  need a sourced, complete per-year list of unsold players (e.g. scraping
  year-by-year recap articles) that doesn't currently exist in this project's
  data. Tracked as a real follow-up, not attempted here.
- **~11% of rows have no recorded batting average, ~24% have no recorded
  economy rate** — real absences (specialist bowlers who rarely bat,
  specialist batters who never bowl), not missing data errors. Encoded as
  `NaN` + a `_missing` flag, never filled.
- **No age/date-of-birth field.** Cricsheet ball-by-ball data doesn't carry
  player birth dates, so "age" isn't a feature in this version — earlier
  versions of this project fabricated an age curve; this one simply doesn't
  claim a feature it can't back with real data.
- **Country/nationality resolves for ~77% of rows** (729/945 before role
  filtering); the rest are labelled `"Unknown"` rather than guessed.
- **`role` has 4 categories** (Batsman, Bowler, All-Rounder,
  Wicketkeeper-Batsman), not 3 — a wicketkeeper's primary value driver is
  batting, but a small scoring bump is given for stumpings. This is a
  modelling choice, not a ground-truth label from the source data.

## What "Value Index" is and isn't

`value_index = performance_score / sold_price_cr`. It is a ranking heuristic
built entirely from this project's own scoring formula. It is **not**
validated against any outcome — there's no check for whether a high-value
pick actually went on to perform well, get retained, or justify the "bargain"
label. Read the Value Finder and Franchise Insights pages as directional, not
as a proven undervaluation detector. This is stated in the app itself, not
just here.
