"""Parameter grids for Boston housing price prediction models."""

from scipy.stats import randint, uniform

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

N_ITER_FAST = 100   # Ridge, DT (instant)
N_ITER_MED  = 30    # RandomForest (each fit trains many trees)
N_ITER_BOOST = 30   # XGBoost, LightGBM (RepeatedKFold 5×2 = 10 folds, 30×10=300 fits)

# Use RepeatedKFold for boosting models to reduce CV variance
BOOST_CV = {"n_splits": 5, "n_repeats": 2}
REGULAR_CV = 5

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
    "n_estimators": randint(100, 1000),
    "max_depth": randint(2, 10),
    "learning_rate": uniform(0.005, 0.25),
    "subsample": uniform(0.5, 0.5),
    "colsample_bytree": uniform(0.5, 0.5),
    "reg_lambda": uniform(0, 15),
    "reg_alpha": uniform(0, 15),
}

LGB_PARAM_DIST = {
    "n_estimators": randint(100, 1000),
    "max_depth": randint(2, 10),
    "learning_rate": uniform(0.005, 0.25),
    "subsample": uniform(0.5, 0.5),
    "colsample_bytree": uniform(0.5, 0.5),
    "reg_lambda": uniform(0, 15),
    "reg_alpha": uniform(0, 15),
    "num_leaves": randint(15, 200),
}

MODEL_PARAMS = {
    "LinearRegression": ({},             N_ITER_FAST),
    "Ridge":        (RIDGE_PARAM_DIST, N_ITER_FAST),
    "DecisionTree": (DT_PARAM_DIST,   N_ITER_FAST),
    "RandomForest": (RF_PARAM_DIST,   N_ITER_MED),
    "XGBoost":      (XGB_PARAM_DIST,  N_ITER_BOOST),
    "LightGBM":     (LGB_PARAM_DIST,  N_ITER_BOOST),
}
