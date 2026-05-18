"""Parameter grids for Boston housing price prediction models."""

from scipy.stats import randint, uniform, loguniform

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

N_ITER_FAST = 100   # Ridge, DT (instant)
N_ITER_MED  = 30    # RandomForest (each fit trains many trees)
N_ITER_BOOST = 60   # XGBoost, LightGBM (higher search budget for better R2)
N_ITER_SVR = 60     # SVR (needs broader search)
N_ITER_ET = 40      # ExtraTrees (moderate search)
N_ITER_STACK = 1    # Stacking (fixed architecture)
N_ITER_KRR = 60     # KernelRidge (rbf)

# Use RepeatedKFold for boosting models to reduce CV variance
BOOST_CV = {"n_splits": 5, "n_repeats": 2}
REGULAR_CV = 5

# Per-model timeout (seconds) — boosting models get more headroom
TIMEOUT_FAST = 60
TIMEOUT_MED = 120
TIMEOUT_BOOST = 420
TIMEOUT_SVR = 300
TIMEOUT_ET = 240
TIMEOUT_STACK = 600
TIMEOUT_KRR = 240
TIMEOUT_XGB_FIXED = 240

# Boosting-specific validation split + early stopping
BOOST_VALID_SIZE = 0.2
BOOST_EARLY_STOP = 50

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
    "n_estimators": randint(200, 2000),
    "max_depth": randint(2, 10),
    "learning_rate": loguniform(0.01, 0.3),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "min_child_weight": randint(1, 12),
    "gamma": uniform(0, 5),
    "reg_lambda": uniform(0, 20),
    "reg_alpha": uniform(0, 20),
}

LGB_PARAM_DIST = {
    "n_estimators": randint(200, 2000),
    "max_depth": randint(2, 12),
    "learning_rate": loguniform(0.01, 0.3),
    "subsample": uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.6, 0.4),
    "reg_lambda": uniform(0, 20),
    "reg_alpha": uniform(0, 20),
    "num_leaves": randint(20, 256),
    "min_child_samples": randint(5, 60),
    "min_split_gain": uniform(0, 0.5),
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

MODEL_PARAMS = {
    "LinearRegression": ({},             N_ITER_FAST,   TIMEOUT_FAST),
    "Ridge":            (RIDGE_PARAM_DIST, N_ITER_FAST,   TIMEOUT_FAST),
    "DecisionTree":     (DT_PARAM_DIST,   N_ITER_FAST,   TIMEOUT_FAST),
    "RandomForest":     (RF_PARAM_DIST,   N_ITER_MED,    TIMEOUT_MED),
    "XGBoost":          (XGB_PARAM_DIST,  N_ITER_BOOST,  TIMEOUT_BOOST),
    "LightGBM":         (LGB_PARAM_DIST,  N_ITER_BOOST,  TIMEOUT_BOOST),
    "ExtraTrees":       (ET_PARAM_DIST,   N_ITER_ET,     TIMEOUT_ET),
    "SVR":              (SVR_PARAM_DIST,  N_ITER_SVR,    TIMEOUT_SVR),
    "Stacking":         ({},              N_ITER_STACK, TIMEOUT_STACK),
    "KernelRidge":      (KRR_PARAM_DIST,  N_ITER_KRR,    TIMEOUT_KRR),
    "XGBoostFixed":      ({},              1,            TIMEOUT_XGB_FIXED),
}
