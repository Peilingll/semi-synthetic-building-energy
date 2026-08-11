"""Stage 3 routes: train one M1 LightGBM on dev GT, apply to hold-out for
M1 (GT features) and each M3 instance (vision features). M0 = dev majority.
"""

import logging
from pathlib import Path

import pandas as pd
from lightgbm import LGBMClassifier

from src.stage2.features import (
    BINARY_POSITIVE,
    CATEGORICAL,
    build_master_table,
    feature_matrix,
    target_col,
)
from src.stage2.train_eval import params_for
from src.stage3.features import to_X

logger = logging.getLogger(__name__)


def train_m1_model(dev_path: Path | None = None,
                   task: str = "7class") -> tuple[LGBMClassifier, dict, str]:
    """Train the shared LightGBM on the full dev set (GT S_full features).

    dev_path overrides the pooled dev_fold_indices.parquet (e.g. LOCO splits).
    Returns (model, category dtypes, dev majority class).
    """
    master = build_master_table(dev_path=dev_path)
    X, cat = feature_matrix(master, "S_full")
    y = master[target_col(task)]
    cat_dtypes = {c: X[c].cat.categories for c in CATEGORICAL if c in X.columns}
    clf = LGBMClassifier(**params_for(task))
    clf.fit(X, y, categorical_feature=cat)
    majority = y.value_counts().idxmax()
    logger.info("M1[%s] trained on %d dev buildings; dev majority=%s", task, len(X), majority)
    return clf, cat_dtypes, majority


def predict_route(clf: LGBMClassifier, df: pd.DataFrame, cat_dtypes: dict) -> pd.Series:
    """Predict energy class for a route's hold-out feature frame."""
    X = to_X(df, cat_dtypes)
    return pd.Series(clf.predict(X), index=df.index)


def predict_proba_route(clf: LGBMClassifier, df: pd.DataFrame, cat_dtypes: dict) -> pd.Series:
    """P(D-G) for a route's hold-out feature frame (binary task only)."""
    X = to_X(df, cat_dtypes)
    pos = list(clf.classes_).index(BINARY_POSITIVE)
    return pd.Series(clf.predict_proba(X)[:, pos], index=df.index)
