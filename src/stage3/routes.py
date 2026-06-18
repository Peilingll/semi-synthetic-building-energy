"""Stage 3 routes: train one M1 LightGBM on dev GT, apply to hold-out for
M1 (GT features) and each M3 instance (vision features). M0 = dev majority.
"""

import logging

import pandas as pd
from lightgbm import LGBMClassifier

from src.stage2.features import CATEGORICAL, build_master_table, feature_matrix
from src.stage2.train_eval import FIXED_PARAMS
from src.stage3.features import to_X

logger = logging.getLogger(__name__)


def train_m1_model() -> tuple[LGBMClassifier, dict, str]:
    """Train the shared LightGBM on the full dev set (GT S_full features).

    Returns (model, category dtypes, dev majority class).
    """
    master = build_master_table()
    X, cat = feature_matrix(master, "S_full")
    y = master["energy_class"]
    cat_dtypes = {c: X[c].cat.categories for c in CATEGORICAL if c in X.columns}
    clf = LGBMClassifier(**FIXED_PARAMS)
    clf.fit(X, y, categorical_feature=cat)
    majority = y.value_counts().idxmax()
    logger.info("M1 model trained on %d dev buildings; dev majority=%s", len(X), majority)
    return clf, cat_dtypes, majority


def predict_route(clf: LGBMClassifier, df: pd.DataFrame, cat_dtypes: dict) -> pd.Series:
    """Predict energy class for a route's hold-out feature frame."""
    X = to_X(df, cat_dtypes)
    return pd.Series(clf.predict(X), index=df.index)
