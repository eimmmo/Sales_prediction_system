"""
Feature engineering for Rossmann store-sales forecasting.

Typical use (from a notebook or a training script):

    from features import (load_data, clean_data, build_features, time_split,
                          get_feature_columns, build_preprocessor, TARGET)

    df = build_features(clean_data(load_data("data/processed/cleaned_data.csv")))
    train, valid, test = time_split(df)
    numeric_cols, categorical_cols = get_feature_columns(df)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

Design rules
------------
* No leakage: lag / rolling features only use sales from earlier dates.
* The split is chronological, never random.
* Fit the preprocessor on the training split only.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Column names (change here if the team agrees on different names).
STORE = "Store"
DATE = "Date"
TARGET = "Sales"

LAGS = (7, 14, 28)
ROLLING_WINDOW_DAYS = 28

# Columns that must never be used as model inputs.
NEVER_FEATURES = {
    TARGET,
    DATE,
    "Customers",  # not known before the day happens
    "Open",       # constant once closed days are removed
    "IsOutlier",  # derived from Sales -> leakage
    "Year",       # test years are unseen in training
}

# Known categorical columns in Rossmann (others are detected by dtype).
KNOWN_CATEGORICAL = [STORE, "StoreType", "Assortment", "StateHoliday", "PromoInterval"]


# --------------------------------------------------------------------------
# Loading and cleaning
# --------------------------------------------------------------------------
def load_data(data_path, store_path=None):
    """Read the CSV. If store columns are missing, merge them from store.csv."""
    df = pd.read_csv(data_path, low_memory=False)
    if "StoreType" not in df.columns and store_path and Path(store_path).exists():
        df = df.merge(pd.read_csv(store_path), on=STORE, how="left")
    return df


def clean_data(df):
    """Validate columns, fix dtypes, drop incomplete rows, sort, check duplicates."""
    missing = {STORE, DATE, TARGET} - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df.copy()
    out[DATE] = pd.to_datetime(out[DATE], errors="coerce")
    out[TARGET] = pd.to_numeric(out[TARGET], errors="coerce")
    out = out.dropna(subset=[STORE, DATE, TARGET])

    # StateHoliday mixes ints and strings (0, "0", "a"); unify as text.
    if "StateHoliday" in out.columns:
        out["StateHoliday"] = out["StateHoliday"].astype(str)

    out = out.sort_values([STORE, DATE]).reset_index(drop=True)

    n_duplicates = out.duplicated([STORE, DATE]).sum()
    if n_duplicates:
        raise ValueError(f"{n_duplicates} duplicate Store/Date rows. Fix them before building lags.")
    return out


# --------------------------------------------------------------------------
# Feature builders
# --------------------------------------------------------------------------
def add_lag_features(data, lags=LAGS):
    """Add Sales_Lag_<n>: sales of the same store exactly <n> calendar days earlier.

    Uses a merge on the date (not shift) so missing days give NaN instead of
    silently picking the wrong row.
    """
    result = data.copy()
    base = result[[STORE, DATE, TARGET]]
    for lag in lags:
        past = base.copy()
        past[DATE] = past[DATE] + pd.Timedelta(days=lag)
        past = past.rename(columns={TARGET: f"Sales_Lag_{lag}"})
        result = result.merge(past, on=[STORE, DATE], how="left", validate="one_to_one")
    return result


def add_rolling_features(data, window_days=ROLLING_WINDOW_DAYS):
    """Add rolling mean/std/min/max of the previous `window_days` calendar days per store.

    closed="left" excludes the current day, so today's sales are never an input.
    """
    window = f"{window_days}D"
    columns = {stat: f"Sales_Rolling_{stat.capitalize()}_{window}"
               for stat in ("mean", "std", "min", "max")}

    result = data.copy()
    for col in columns.values():
        result[col] = np.nan

    for _, group in result.groupby(STORE, sort=False):
        group = group.sort_values(DATE)
        rolling = group.set_index(DATE)[TARGET].rolling(window, closed="left")
        for stat, col in columns.items():
            result.loc[group.index, col] = getattr(rolling, stat)().to_numpy()
    return result


def add_calendar_features(data):
    """Add Day, Month, Year, DayOfWeek (Mon=0) and IsWeekend (Sat/Sun).

    Holiday flags (StateHoliday, SchoolHoliday) already exist in the data
    and are kept as they are.
    """
    result = data.copy()
    result["Day"] = result[DATE].dt.day
    result["Month"] = result[DATE].dt.month
    result["Year"] = result[DATE].dt.year
    result["DayOfWeek"] = result[DATE].dt.dayofweek
    result["IsWeekend"] = result["DayOfWeek"].isin([5, 6]).astype(int)
    return result


def build_features(df):
    """Run all feature builders on a cleaned dataframe."""
    return add_calendar_features(add_rolling_features(add_lag_features(df)))


# --------------------------------------------------------------------------
# Feature selection, split, preprocessing
# --------------------------------------------------------------------------
def get_feature_columns(df):
    """Return (numeric_cols, categorical_cols) to use as model inputs."""
    candidates = [c for c in df.columns if c not in NEVER_FEATURES]
    categorical = [
        c for c in candidates
        if c in KNOWN_CATEGORICAL
        or pd.api.types.is_bool_dtype(df[c])
        or not pd.api.types.is_numeric_dtype(df[c])
    ]
    numeric = [c for c in candidates if c not in categorical]
    return numeric, categorical


def time_split(df, train_frac=0.70, valid_frac=0.15):
    """Chronological train / validation / test split on unique dates."""
    dates = np.sort(df[DATE].unique())
    if len(dates) < 10:
        raise ValueError("Too few dates for a reliable time-based split.")

    train_end = dates[int(len(dates) * train_frac)]
    valid_end = dates[int(len(dates) * (train_frac + valid_frac))]

    train = df[df[DATE] < train_end].copy()
    valid = df[(df[DATE] >= train_end) & (df[DATE] < valid_end)].copy()
    test = df[df[DATE] >= valid_end].copy()

    assert train[DATE].max() < valid[DATE].min() <= valid[DATE].max() < test[DATE].min()
    return train, valid, test


def build_preprocessor(numeric_cols, categorical_cols):
    """Impute + scale numeric columns, impute + one-hot encode categorical ones.

    Output is sparse when mostly zeros (one-hot of Store). Fit on train only.
    """
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer(
        [("num", numeric, numeric_cols), ("cat", categorical, categorical_cols)],
        remainder="drop",
    )
