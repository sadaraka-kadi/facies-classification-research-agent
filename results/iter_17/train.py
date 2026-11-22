"""
IGS 2026 Agentic AI Lab Competition - Well Log Facies Classification

This is the ONLY code file you and your agent edit. evaluate.py imports it and calls:

    fit(train_df)       -> model      train_df has all columns including "Facies"
    predict(model, df)  -> labels     df has the same columns without "Facies";
                                      return one integer facies (1-9) per row, in row order

Rules of the interface:
  - Use only the DataFrames passed in. Do not read any file from this module.
  - Everything learned from data (scalers, encoders, statistics, weights) must be
    learned inside fit() and stored in the returned model.
  - One call to fit() trains ONE fixed configuration: no grid / random / Bayesian
    search, no AutoML, no loop over candidate configurations.
  - Allowed model families: Random Forest, XGBoost, LightGBM, SVM, MLP, 1D-CNN, LSTM
    (ensembles of these are allowed and count as one configuration).
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

MODEL_FAMILY = "LightGBM"  # keep this in sync with what fit() actually trains
SEED = 42

LOGS = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS"]
CURVES = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE"]
WINDOW = 3  # samples, centred: one sample above and below


def context_features(well: pd.DataFrame, z_stats=None) -> pd.DataFrame:
    """Rolling and neighbour features for ONE well, already sorted by Depth.

    z_stats: optional {curve: (mean, std)} used for the z-score instead of the well's own
    statistics (only for single-facies training wells, see fit()).
    """
    out = {}
    for c in CURVES:
        s = well[c]
        roll = s.rolling(WINDOW, center=True, min_periods=1)
        out[f"{c}_mean{WINDOW}"] = roll.mean()
        out[f"{c}_std{WINDOW}"] = roll.std().fillna(0.0)
        out[f"{c}_lag1"] = s.shift(1).fillna(s)
        out[f"{c}_lead1"] = s.shift(-1).fillna(s)
        # Well-relative level, ADDED alongside the absolute value (iter 3 replaced it and lost
        # WELL_02's high PE). Uses only this well's own rows, also at prediction time.
        mu, sd = z_stats[c] if z_stats is not None else (s.mean(), s.std())
        out[f"{c}_z"] = (s - mu) / (sd if sd > 0 else 1.0)
    return pd.DataFrame(out, index=well.index)


def build_features(df: pd.DataFrame, pooled_z=None, pooled_wells=()) -> pd.DataFrame:
    """Turn raw rows into model inputs. Must keep the row order and row count.

    Context is computed within each well, sorted by Depth, from that well's rows only.
    """
    ordered = df.sort_values(["Well Name", "Depth"], kind="stable")
    parts = [
        context_features(well, pooled_z if name in pooled_wells else None)
        for name, well in ordered.groupby("Well Name", sort=False)
    ]
    context = pd.concat(parts).reindex(df.index)
    return pd.concat([df[LOGS], context], axis=1)


def fit(train_df: pd.DataFrame):
    # A training well that holds a single facies (the synthetic WELL_02, all F9) has no rock
    # mix to normalise against: z-scored against itself, all its rows sit at z = 0, while real
    # F9 has PE_z of about +1.5. Such wells are z-scored with the pooled statistics of the
    # other training wells instead. Prediction is unchanged: every well uses its own rows.
    n_classes = train_df.groupby("Well Name")["Facies"].nunique()
    pooled_wells = set(n_classes[n_classes == 1].index)
    ref = train_df[~train_df["Well Name"].isin(pooled_wells)]
    pooled_z = {c: (ref[c].mean(), ref[c].std()) for c in CURVES}
    X = build_features(train_df, pooled_z, pooled_wells)
    y = train_df["Facies"].to_numpy()
    # Iter 7 LightGBM settings unchanged, now on the iter 8 features (absolute + per-well z).
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X, y)
    return model


def predict(model, df: pd.DataFrame) -> np.ndarray:
    return model.predict(build_features(df))
