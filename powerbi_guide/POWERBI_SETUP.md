# Power BI Setup Guide — Cric Auction IQ

Follow these steps after running `python data/build_dataset.py` and
`python notebooks/train_models.py`. Column names below match what those
scripts actually write to `models/*.csv` — check `models/processed_data.csv`
yourself before building if you've changed the pipeline.

**Status: this is a build guide, not a shipped dashboard.** No `.pbix` file
exists in this repo yet. If you build this, save your `.pbix` and update this
line with where it lives (and a published link, if you share it) — don't let
a resume bullet claim a dashboard that isn't actually in the repo.

## STEP 1 — Import Data into Power BI
File → Get Data → Text/CSV. Import from `/models/`:
- `processed_data.csv` — main dataset, 941 rows (all real auction records; there is no `is_sold` column — every row is a sold record, see `docs/DATA_QUALITY.md` for why there's no unsold data)
- `franchise_efficiency.csv` — franchise-level summary (15 franchises, post-rename identities merged)
- `top_value_players.csv` — top 20 value-index picks

## STEP 2 — Data Types (Power Query Editor)
`processed_data`:
| Column | Type |
|---|---|
| `year` | Whole Number |
| `soldpricecr`, `basepricecr` | Decimal Number |
| `performancescore`, `valueindex` | Decimal Number |
| `career_matches`, `wickets`, `catches`, `stumpings` | Whole Number |
| `battingavg_missing`, `battingsr_missing`, `economyrate_missing`, `bowlingavg_missing` | Whole Number (0/1) |

## STEP 3 — DAX Measures

### Basic KPIs
```
Total Auction Records = COUNTROWS(processed_data)

Avg Sale Price (Cr) = AVERAGE(processed_data[soldpricecr])

Max Sale Price (Cr) = MAX(processed_data[soldpricecr])

Avg Value Index = AVERAGE(processed_data[valueindex])
```

There is deliberately no "Unsold Rate %" measure — see
`docs/DATA_QUALITY.md` for why no unsold data exists in this project.

### Advanced measures
```
Price Growth YoY % =
VAR CurrentYear = MAX(processed_data[year])
VAR CurrentAvg = CALCULATE(AVERAGE(processed_data[soldpricecr]),
                            processed_data[year] = CurrentYear)
VAR PrevAvg = CALCULATE(AVERAGE(processed_data[soldpricecr]),
                         processed_data[year] = CurrentYear - 1)
RETURN DIVIDE(CurrentAvg - PrevAvg, PrevAvg, 0) * 100

Top Value Player =
CALCULATE(
    FIRSTNONBLANK(processed_data[playername], 1),
    TOPN(1, processed_data, processed_data[valueindex], DESC)
)

-- group by CURRENT_FRANCHISE, not the historical `franchise` column,
-- so renamed teams (Delhi Daredevils/Capitals etc.) aren't split in two
Franchise Efficiency Rank =
RANKX(
    ALL(processed_data[current_franchise]),
    CALCULATE(AVERAGE(processed_data[valueindex])),
    ,
    DESC
)

Overseas Premium % =
VAR OverseasAvg = CALCULATE(AVERAGE(processed_data[soldpricecr]),
                             processed_data[nationality] = "Overseas")
VAR IndianAvg   = CALCULATE(AVERAGE(processed_data[soldpricecr]),
                             processed_data[nationality] = "Indian")
RETURN DIVIDE(OverseasAvg - IndianAvg, IndianAvg, 0) * 100
```

## STEP 4 — Page Layout (4 pages)

**Page 1 — Executive Overview**
- KPI cards: Total Auction Records | Avg Sale Price | Max Sale Price | Avg Value Index
- Line chart: X=`year`, Y=Avg Sale Price (add a median measure as a second line)
- Donut: role breakdown (Legend=`role`, Values=count)
- Stacked bar: X=`year`, Y=sum(`soldpricecr`), Legend=`current_franchise` (top 6 by total spend)
- Slicers: Year range, Role, Nationality

**Page 2 — Player Deep Dive**
- Scatter: X=`performancescore`, Y=`soldpricecr`, Size=`career_matches`, Color=`role`, Tooltip=`playername`, `year`, `franchise`
- Table: `playername`, `role`, `nationality`, `year`, `battingavg`, `battingsr`, `wickets`, `economyrate`, `soldpricecr`, `valueindex` — sorted by `valueindex` descending. Note: blank batting/bowling cells are real absences, not errors (see `docs/DATA_QUALITY.md`) — don't apply "replace null with 0" in Power Query, it would fabricate data.

**Page 3 — Value Intelligence**
- Bubble: X=`soldpricecr`, Y=`performancescore`, Bubble size=`valueindex`, Color=`role`. Add a text box calling out that Value Index is a heuristic, not a validated undervaluation signal — same caveat the Streamlit app states.
- Bar: top 15 by `valueindex`, colored by role
- Table: import of `top_value_players.csv`, all 20 rows

**Page 4 — Franchise Strategy**
- Scatter: X=`total_spend`, Y=`avg_performance` (from `franchise_efficiency.csv`), Size=`players_bought`, Color=`avg_value_index`, Labels=`franchise`
- Bar: Franchise Efficiency Rank, sorted descending, conditional color (green top half / red bottom half)
- Area chart: yearly total spend by `current_franchise`
- Table: `franchise_efficiency.csv`, all columns, conditional formatting on `avg_value_index`

## STEP 5 — Formatting

Match the Streamlit app's palette for consistency across both deliverables:
- Primary: `#3A5CE0` (blue) · Secondary: `#0FB5AE` (teal) · Positive: `#1E9E6B` · Negative: `#D6455B` · Amber: `#E0982A`
- Font: Segoe UI throughout
- Chart titles: 14pt, bold
- Gridlines: light gray (`#E4E7F1`)
- Data labels on horizontal bar charts
- Conditional formatting on `valueindex`: above-average = green background, below = red

## STEP 6 — Publish & Share

File → Publish to Power BI Service. Get the shareable link and put it in
this file and the README — don't reference a dashboard link that doesn't
exist yet.
