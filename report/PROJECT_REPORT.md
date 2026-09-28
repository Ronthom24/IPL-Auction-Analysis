# Cric Auction IQ — Business Intelligence Project Report

**Course:** Business Intelligence
**Tools:** Python · scikit-learn · SHAP · Streamlit · Power BI
**Dataset:** Real IPL auction records + real Cricsheet-derived player stats, 2009–2026 (941 entries, 463 players, 15 franchises)

---

## 1. Executive Summary

The IPL auction is one of the most high-stakes, data-driven events in
professional sport: ten franchises bid for players within a fixed budget,
and outcomes are notoriously inconsistent — teams overpay for marquee names
while value sits elsewhere on the board.

This project builds an end-to-end BI solution on **real** data — real
auction prices, real per-season player performance computed from ball-by-ball
match data — that:

- Predicts a player's auction price from independently-measured performance
  stats using a Random Forest model (headline R² = 0.33, 5-fold CV grouped
  by player)
- Explains predictions with SHAP, identifying exactly which features drove
  a given price
- Ranks players by a performance-per-crore Value Index
- Reports franchise-level spend efficiency
- Delivers everything through an interactive Streamlit dashboard and a
  Power BI dashboard guide

**Business problem:** how can franchises make more data-informed auction
decisions — maximising performance per crore spent?

This report is deliberately specific about what the data can and can't
support. A sold/unsold classifier was scoped out because no real unsold-player
data exists in any source checked (see Section 8). An earlier version of
this project reported a higher R² (0.66) that turned out to be inflated by a
leakage bug — this version fixes that and reports the honest number.

---

## 2. Domain Background

### 2.1 How the IPL auction works
- Each player enters with a **base price** (₹20 lakh – ₹2 crore)
- Franchises bid against each other; the highest bid wins
- Each franchise has a **salary cap** (~₹90–120 crore depending on season)
- A maximum of overseas players per squad is capped (creates scarcity pricing)
- Players can go **unsold** if no franchise bids at base price — but unsold
  results aren't published in a structured, complete way (see Section 8)

### 2.2 Why this is a BI problem
Franchise management teams make multi-crore decisions in seconds, under
competitive pressure, often on incomplete information. BI tooling that
aggregates historical patterns, models player value, and surfaces
model-explained insight gives a systematic edge — that's the scope of this
project.

---

## 3. Data Description

### 3.1 Dataset overview

| Attribute | Detail |
|---|---|
| Records | 941 auction entries |
| Years | 2009–2026 (18 seasons) |
| Unique players | 463 |
| Franchises | 15 (post-rename identities merged for aggregation) |
| Features | 25 columns |
| Source | Real published auction records + real Cricsheet ball-by-ball stats |

### 3.2 Key features

| Feature | Description |
|---|---|
| `soldpricecr` | Hammer price at auction (₹ Crore) — **regression target** |
| `battingavg`, `battingsr` | Career batting average / strike rate, as of that auction |
| `wickets`, `economyrate`, `bowlingavg`, `bowlingsr` | Career bowling figures, as of that auction |
| `career_matches` | Real career match count at auction time (used in place of an "age" feature — no birth-date data exists in the source) |
| `performancescore` | Engineered composite metric (Section 4) |
| `valueindex` | `performancescore ÷ soldpricecr` — the efficiency metric |
| `*_missing` flags | Explicit indicators for players with no batting or bowling record — the value itself stays `NaN`, never filled |

### 3.3 Data quality

Real, but not uniform: ~11% of rows have no recorded batting average, ~24%
have no recorded economy rate (real absences, not gaps to fill), and
nationality resolves for ~77% of records. Full detail, including what was
fixed in this project's most recent rebuild, is in `docs/DATA_QUALITY.md` —
that file is the source of truth over any number in this report if the two
ever disagree.

---

## 4. Methodology

### 4.1 Feature engineering

**Performance Score** — a 0–1 blend of min-max-normalised career stats,
weighted by role, with missing sub-scores excluded rather than guessed:

- **Batsman:** mean of normalised batting average and strike rate
- **Bowler:** mean of normalised wickets and normalised (inverse) economy
- **All-Rounder:** 0.5 × batting component + 0.5 × bowling component
- **Wicketkeeper-Batsman:** 0.8 × batting component + 0.2 × normalised stumpings

**Value Index:** `performancescore ÷ soldpricecr`. Higher = more measured
performance per crore spent. This is a ranking heuristic built from this
project's own formula — not validated against any outcome (see Section 8).

### 4.2 Exploratory data analysis

Findings from the real data (see `notebooks/plots/eda_overview.png`):

1. Auction prices show a rising trend across seasons with high year-to-year
   volatility — early/late years have small samples, so treat the slope as
   directional, not a precise growth rate.
2. Bowlers make up the largest share of auction entries in this dataset
   (426/941), followed by Batsmen (224), All-Rounders (215), and
   Wicketkeeper-Batsmen (76).
3. Career match count correlates with price but noisily — some short-career
   players command very high prices (a hype/potential premium the model
   partially captures via other features, not fully).
4. A `basepricecr` (BCCI-set tier) is the single strongest SHAP driver of
   final price — the market rarely moves a player far from their assigned
   tier.

### 4.3 Machine learning model

**Algorithm:** Random Forest Regressor
**Tuning:** 5-fold `GridSearchCV` over `n_estimators` (100/200/400),
`max_depth` (8/12/None), `min_samples_leaf` (2/4)
**Best parameters:** `n_estimators=400, max_depth=12, min_samples_leaf=4`

**Headline result — 5-fold CV, grouped by player (the number to quote):**

| Metric | Value |
|---|---|
| R² | **0.3295** (± 0.0632) |

**Holdout result — single 20% split (noisier, reference only):**

| Metric | Value |
|---|---|
| R² | 0.3855 |
| RMSE | ₹3.03 Cr |
| MAE | ₹1.94 Cr |

Grouping by player matters here specifically because 463 unique players
generated 941 rows — a substantial share of players appear more than once.
An ungrouped split lets a player's other season appear in training while
their test-year is held out, which partly turns the task into "recognise
this player" rather than "predict from stats," inflating R².

### 4.4 Why R² = 0.33, not higher

This is the honest headline, and it's lower than the earlier version of this
project reported (0.66) — deliberately. That 0.66 came from a leakage bug:
performance stats were generated as a function of the player's own
historical price, so the model was partly predicting price from a noisy
transform of itself. This version's stats are real and price-independent, so
0.33 reflects genuine signal: real performance stats explain roughly a third
of price variance. The rest is bidding-war dynamics, franchise-specific
squad needs, and reputation effects that no stats table captures.

### 4.5 Explainable AI — SHAP

`shap.TreeExplainer` runs against the trained Random Forest for:
1. **Global feature importance** — which features drive price across all
   players (`notebooks/plots/shap_global.png`)
2. **Per-player waterfall values** — exactly how much each stat contributed
   to one prediction, shown live in the Streamlit Price Predictor page

Base price is consistently the dominant SHAP driver in this dataset,
followed by batting strike rate, auction year, and career match count.

---

## 5. Key Insights

### Insight 1 — Base price anchors the outcome
The BCCI-assigned base price tier is the single strongest predictor of final
price. Franchises rarely move a player far outside their assigned tier,
regardless of career stats — a real, measurable anchoring effect.

### Insight 2 — Value Index is dominated by base-price-floor players
The highest value-index players in this dataset are almost all bought at the
₹0.1–0.2 Cr floor. This is a property of the ratio (any decent performance
divided by a tiny price spikes the index), not evidence the model found
hidden gems — stated plainly on the Value Finder page rather than oversold.

### Insight 3 — Career match count is a noisy but real signal
More career experience correlates with price, but the relationship is loose
— short-career players with high potential are sometimes bid up sharply,
which the model only partially captures.

### Insight 4 — Franchise efficiency varies meaningfully
Grouping by current franchise identity (merging renamed teams), average
value index varies noticeably across the 15 franchises — see the Franchise
Insights page for the live, current ranking rather than a static number here
(it will shift as the model/data are rebuilt).

---

## 6. System Architecture

```
data/external/ (real source data)
     |
     v
data/build_dataset.py  -->  data/ipl_auction_data.csv (941 rows)
     |
     +-- EDA (notebooks/train_models.py)  -->  4 EDA plots (PNG)
     |
     +-- Feature engineering (performance_score, value_index, missing flags)
     |
     +-- Random Forest Regressor  -->  models/price_predictor.pkl
     |        (grouped 5-fold CV headline R² = 0.33)     models/shap_explainer.pkl
     |
     +-- Sold/unsold classifier: deliberately not built
     |        (models/classifier_metrics.json explains why)
     |
     +-- Streamlit App (5 pages)
     |     Overview | Player Analysis | Value Finder | Price Predictor | Franchise Insights
     |
     +-- Power BI dashboard guide (powerbi_guide/POWERBI_SETUP.md)
```

---

## 7. Tools & Technologies

| Tool | Purpose |
|---|---|
| Python 3.12 | Core language |
| pandas, NumPy | Data manipulation |
| scikit-learn | RandomForest, GridSearchCV, GroupKFold, metrics |
| SHAP | Explainable AI |
| Plotly | Interactive charts in Streamlit |
| Streamlit | Web application framework |
| Power BI | Dashboard guide (DAX measures, no live `.pbix` shipped yet) |
| joblib | Model serialization |
| matplotlib | Static EDA plots |

---

## 8. Limitations & Future Work

### Current limitations
- **No sold/unsold classification.** No source checked (including a more
  rigorous sibling project's own audit) has a complete, structured list of
  unsold players per year — auction result tables only publish who was
  bought. Building this would require fabricating a negative class, which
  this project doesn't do.
- **R² = 0.33 is a moderate, honest ceiling** given the auction market's
  real noise (bidding wars, franchise-specific needs, reputation premiums).
- **Value Index is unvalidated** as an "undervalued player" signal — it's a
  ranking heuristic, not a backtested one.
- **No age feature** — Cricsheet data carries no birth dates; an earlier
  version fabricated an age curve, this one doesn't.
- **No live deployment** — runs locally via `streamlit run`; no Streamlit
  Cloud URL exists yet.

### Future enhancements
1. Source a real, structured unsold-players list (e.g. year-by-year
   ESPNcricinfo/IPLT20 recap pages) to finally make sold/unsold
   classification honest.
2. Backtest Value Index against next-season performance or retention to
   validate — or falsify — it as an actual undervaluation signal.
3. Replace the greedy budget optimizer with an ILP/knapsack solver for a
   genuinely optimal squad-selection result.
4. Live CricAPI/Cricbuzz integration for in-season stat updates.
5. Deploy the Streamlit app and Power BI dashboard for real, and link them
   here.

---

## 9. Conclusion

This project is a complete, honestly-reported BI pipeline on real sports
data: real auction records, real independently-sourced performance stats, a
grouped-CV-validated regression model, SHAP explainability, and an
interactive dashboard — with its limitations documented rather than hidden.
The headline R² of 0.33 is lower than a flashier number this project used to
report, and that's the point: it's the number that survives scrutiny.
