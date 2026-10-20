# Data Notes

Everything the team needs to know about the cleaned dataset before building features and models.
Cleaning steps and the reasoning behind each one are documented in `notebooks/01_eda.ipynb`.

## 1. Dataset

- **Source:** Rossmann Store Sales (Kaggle), a drug store chain in Germany.
- **Files used:** `train.csv` (daily sales per store) and `store.csv` (store characteristics), merged on `Store`. `test.csv` was not used because it has no `Sales` column.
- **Period:** 2013-01-01 to 2015-07-31 (942 days), 1,115 stores.
- **Target:** `Sales` (daily turnover per store).
- **Level:** store level only. The dataset has no product dimension.

## 2. Cleaned file

`data/processed/cleaned_data.csv`

- 844,338 rows x 19 columns, sorted by `Store`, then `Date`.
- `Sales` is always positive (minimum 46), so MAPE is well defined.
- Read it with `parse_dates`, otherwise `Date` is loaded as text:

```python
df = pd.read_csv("data/processed/cleaned_data.csv", parse_dates=["Date"])
```

## 3. Columns

| Column | Type | Notes |
|---|---|---|
| Store | int | Store ID (1 to 1115) |
| DayOfWeek | int | 1 = Monday, 7 = Sunday |
| Date | datetime | Date of the record |
| **Sales** | int | **Target** |
| Customers | int | **Do not use as a feature** (see section 4) |
| Promo | 0/1 | Store runs a promotion that day |
| StateHoliday | text | `0` (none), `a`, `b`, `c`. Needs encoding |
| SchoolHoliday | 0/1 | Affected by public school closure |
| StoreType | text | `a`, `b`, `c`, `d`. Needs encoding |
| Assortment | text | `a`, `b`, `c`. Needs encoding |
| CompetitionDistance | float | Meters to the nearest competitor |
| CompetitionOpenSinceMonth / Year | float | **0 means unknown**, not a real value |
| HasCompetitionDate | 0/1 | 1 if the competitor open date is known (added during cleaning) |
| Promo2 | 0/1 | Store takes part in the continuing promotion |
| Promo2SinceWeek / Year | float | **0 means the store is not in Promo2** |
| PromoInterval | text | Months when Promo2 restarts. `"None"` for stores not in Promo2. Needs encoding |
| IsOutlier | 0/1 | **Do not use as a feature** (see section 4) |

## 4. Columns that must NOT be used as features

- **`Customers`**: the number of customers is only known at the end of the day it describes. At prediction time it is not available, so using it causes data leakage. The model would look very accurate in testing and fail in real use. A lagged version (for example customers 7 days ago) is safe, because it is known at prediction time.
- **`IsOutlier`**: it is computed from `Sales`, which is the target. It exists for analysis and reporting only.
- **`Open`**: removed from the data (it is always 1 after filtering).

General rule: any column that would not be known at the moment of prediction must not enter the model.

## 5. Cleaning decisions

| Step | Decision | Reason |
|---|---|---|
| Closed days | Removed 172,817 rows with `Open == 0` and 54 open days with zero sales | A closed store has zero sales because it is closed, not because of low demand. Zero sales also break MAPE |
| CompetitionDistance | Filled with the median (3 stores) | Median is robust to very large distances |
| CompetitionOpenSince* | Filled with 0 and flagged with `HasCompetitionDate` | Date unknown, not an error |
| Promo2Since*, PromoInterval | Filled with 0 / `"None"` | Store is not in Promo2 |
| Outliers | Detected with IQR per store, **kept** and flagged in `IsOutlier` | Real sales (pre-holiday and promo days). Removing them would make the model unable to predict high-demand days. 16,008 rows (1.90%) are flagged |

## 6. Lag and rolling features

A lag feature is a value from the past placed next to the current day. For example, `Sales_lag_7` is the store's sales 7 days earlier. The plan uses lags of 7, 14 and 28 days, plus rolling mean, std, min and max.

Because closed days were removed, **dates have gaps inside each store**. A plain `shift(7)` moves 7 rows, not 7 days, which silently gives wrong values.

Recommended approach:

1. For each store, reindex to the full daily calendar so that missing days become `NaN` in `Sales`.
2. Build lag and rolling features on that complete calendar.
3. Drop the rows where `Sales` is `NaN` (the closed days) before training.

Always build lags inside each store, never across stores:

```python
df["Sales_lag_7"] = df.groupby("Store")["Sales"].shift(7)
```

The first 7 days of each store have no `lag_7`, and the first 28 days have no `lag_28`. These rows will contain `NaN` and must be dropped or handled.

## 7. Train / validation / test split

The split must be **time-based, never random**. Proposed dates (all members must use the same ones):

| Set | From | To |
|---|---|---|
| Test | 2015-06-20 | 2015-07-31 |
| Validation | 2015-05-09 | 2015-06-19 |
| Train | 2013-01-01 | 2015-05-08 |

Each held-out set is the last 6 weeks (42 days) before the next one.

To avoid leakage, every scaler and encoder must be fitted on the train set only, then applied to validation and test.

## 8. Known caveats

- `CompetitionOpenSinceYear` has a minimum of 1900, which looks unusual. It was left unchanged, so be careful if it is used to compute how long a competitor has existed.
- Zero in `CompetitionOpenSince*` and `Promo2Since*` is a placeholder, not a real month, week or year. Do not use it in arithmetic without handling it first.
- Average sales differ strongly between stores, so errors should also be reviewed per store and per `StoreType` during evaluation.
