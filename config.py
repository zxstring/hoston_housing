"""Parameter grids for Boston housing price prediction models."""

import os

from scipy.stats import randint, uniform, loguniform

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

N_ITER_FAST = 100   # Ridge, DT (instant)
N_ITER_MED  = 30    # RandomForest (each fit trains many trees)
N_ITER_BOOST = 200  # XGBoost, LightGBM (larger search budget)
N_ITER_SVR = 60     # SVR (needs broader search)
N_ITER_ET = 40      # ExtraTrees (moderate search)
N_ITER_KRR = 60     # KernelRidge (rbf)
N_ITER_CAT = 50     # CatBoost

# Use a single shuffle split for boosting search to emphasize holdout
BOOST_CV_MODE = "shuffle"
BOOST_SHUFFLE_SPLITS = 1
BOOST_SHUFFLE_TEST_SIZE = 0.2
BOOST_CV = {"n_splits": 3, "n_repeats": 1}
REGULAR_CV = 5

# Per-model timeout (seconds) — boosting models get more headroom
TIMEOUT_FAST = 60
TIMEOUT_MED = 120
TIMEOUT_BOOST = 900
TIMEOUT_SVR = 300
TIMEOUT_ET = 240
TIMEOUT_KRR = 240
TIMEOUT_CAT = 360

# Optional GPU acceleration (set USE_GPU=1 in env to enable)
USE_GPU = os.environ.get("USE_GPU", "0") == "1"

# ── RandomizedSearchCV distributions ──────────────────────────────────────

RIDGE_PARAM_DIST = {
    "ridge__alpha": uniform(0.01, 100),
}

DT_PARAM_DIST = {
    "max_depth": randint(2, 20),
    "min_samples_split": randint(2, 20),
    "min_samples_leaf": randint(1, 10),
}

RF_PARAM_DIST = {
    "n_estimators": randint(50, 250),
    "max_depth": randint(3, 25),
    "min_samples_split": randint(2, 15),
    "min_samples_leaf": randint(1, 8),
    "max_features": uniform(0.3, 0.7),
}

XGB_PARAM_DIST = {
    "n_estimators": [600, 800, 1000, 1200, 1600, 2000],
    "max_depth": [4, 5, 6, 7, 8],
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
    "subsample": [0.8, 0.9, 1.0],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "min_child_weight": [1, 2, 3],
    "gamma": [0, 0.05, 0.1, 0.2],
    "reg_lambda": [0.5, 1.0, 2.0, 3.0],
    "reg_alpha": [0, 0.1, 0.3, 0.5],
}

LGB_PARAM_DIST = {
    "n_estimators": [600, 800, 1000, 1200, 1600, 2000],
    "max_depth": [-1, 4, 5, 6, 7, 8],
    "learning_rate": [0.02, 0.03, 0.05, 0.08, 0.1],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0, 0.5, 1.0, 2.0],
    "reg_alpha": [0, 0.1, 0.5],
    "num_leaves": [31, 63, 127, 255],
    "min_child_samples": [2, 5, 10, 20],
    "min_split_gain": [0, 0.1, 0.2],
}

ET_PARAM_DIST = {
    "n_estimators": randint(200, 1200),
    "max_depth": randint(3, 30),
    "min_samples_split": randint(2, 12),
    "min_samples_leaf": randint(1, 6),
    "max_features": uniform(0.4, 0.6),
    "bootstrap": [True, False],
}

SVR_PARAM_DIST = {
    "regressor__svr__C": loguniform(1e0, 1e3),
    "regressor__svr__gamma": loguniform(1e-4, 1e-1),
    "regressor__svr__epsilon": uniform(0.01, 0.5),
}

KRR_PARAM_DIST = {
    "regressor__krr__alpha": loguniform(1e-3, 1e1),
    "regressor__krr__gamma": loguniform(1e-4, 1e-1),
}

CAT_PARAM_DIST = {
    "iterations": randint(500, 2000),
    "depth": randint(4, 10),
    "learning_rate": loguniform(0.02, 0.2),
    "l2_leaf_reg": uniform(1, 9),
    "bagging_temperature": uniform(0, 2),
    "random_strength": uniform(0, 2),
}

MODEL_PARAMS = {
    "BaselineLR":       ({"lr__fit_intercept": [True]}, 1, TIMEOUT_FAST),
    "LinearRegression": ({},             N_ITER_FAST,   TIMEOUT_FAST),
    "Ridge":            (RIDGE_PARAM_DIST, N_ITER_FAST,   TIMEOUT_FAST),
    "DecisionTree":     (DT_PARAM_DIST,   N_ITER_FAST,   TIMEOUT_FAST),
    "RandomForest":     (RF_PARAM_DIST,   N_ITER_MED,    TIMEOUT_MED),
    "XGBoost":          (XGB_PARAM_DIST,  N_ITER_BOOST,  TIMEOUT_BOOST),
    "LightGBM":         (LGB_PARAM_DIST,  N_ITER_BOOST,  TIMEOUT_BOOST),
    "ExtraTrees":       (ET_PARAM_DIST,   N_ITER_ET,     TIMEOUT_ET),
    "SVR":              (SVR_PARAM_DIST,  N_ITER_SVR,    TIMEOUT_SVR),
    "KernelRidge":      (KRR_PARAM_DIST,  N_ITER_KRR,    TIMEOUT_KRR),
    "CatBoost":         (CAT_PARAM_DIST,  N_ITER_CAT,    TIMEOUT_CAT),
}
