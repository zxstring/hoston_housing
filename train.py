"""Boston housing — single fast run. Called by run.sh with $SEED env var."""

import json, multiprocessing as mp, os, sys, time, warnings
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from scipy.stats import randint, uniform

warnings.filterwarnings("ignore")

SEED = int(os.environ.get("SEED", 42))
N_ITER = 30
CV = 5
TIMEOUT_PER_MODEL = 120  # seconds

OUTPUT_DIR = Path(__file__).resolve().parent / f"output_s{SEED}"
OUTPUT_DIR.mkdir(exist_ok=True)

PARAM_DISTS = {
    "LinearRegression": {},
    "Ridge":        {"alpha": uniform(0.01, 100)},
    "DecisionTree": {"max_depth": randint(2, 20), "min_samples_split": randint(2, 20), "min_samples_leaf": randint(1, 10)},
    "RandomForest": {"n_estimators": randint(50, 300), "max_depth": randint(3, 20), "min_samples_split": randint(2, 15), "min_samples_leaf": randint(1, 8), "max_features": uniform(0.3, 0.7)},
    "XGBoost":      {"n_estimators": randint(100, 800), "max_depth": randint(3, 12), "learning_rate": uniform(0.01, 0.3), "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4), "reg_lambda": uniform(0, 10), "reg_alpha": uniform(0, 10)},
    "LightGBM":     {"n_estimators": randint(100, 800), "max_depth": randint(3, 12), "learning_rate": uniform(0.01, 0.3), "subsample": uniform(0.6, 0.4), "colsample_bytree": uniform(0.6, 0.4), "reg_lambda": uniform(0, 10), "reg_alpha": uniform(0, 10), "num_leaves": randint(15, 127)},
}

def load_and_prep():
    bunch = fetch_openml(name="boston", version=1, as_frame=True, parser="auto")
    df = bunch.frame; df.columns = [c.upper() for c in df.columns]
    for c in df.columns:
        if df[c].dtype.name in ("category", "object"): df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(df.median())
    def w(s, lim=1.5):
        q1, q3 = s.quantile(0.25), s.quantile(0.75); return s.clip(q1 - lim * (q3 - q1), q3 + lim * (q3 - q1))
    for c in ["CRIM", "RM", "B", "ZN", "DIS"]:
        if c in df.columns: df[c] = w(df[c])
    y = df["MEDV"].values; X = df.drop(columns=["MEDV"])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED)
    sc = StandardScaler(); X_tr = sc.fit_transform(X_tr); X_te = sc.transform(X_te)
    return X_tr, X_te, y_tr, y_te

def build(name):
    if name == "LinearRegression": return LinearRegression()
    if name == "Ridge":        return Ridge(random_state=SEED)
    if name == "DecisionTree": return DecisionTreeRegressor(random_state=SEED)
    if name == "RandomForest": return RandomForestRegressor(random_state=SEED)
    if name == "XGBoost":      return XGBRegressor(random_state=SEED, tree_method="hist", device="cpu", verbosity=0, nthread=4)
    if name == "LightGBM":     return LGBMRegressor(random_state=SEED, device="cpu", verbose=-1, num_threads=4)

def _fit_model(name, Xt, yt, Xe, ye):
    """Actual tuning + evaluation. Runs in a subprocess so a hang doesn't block main."""
    t0 = time.perf_counter()
    s = RandomizedSearchCV(build(name), PARAM_DISTS[name], n_iter=N_ITER, cv=CV,
                           scoring="neg_mean_squared_error", n_jobs=1, random_state=SEED)
    s.fit(Xt, yt)
    est = s.best_estimator_
    dt = time.perf_counter() - t0
    yp = est.predict(Xe)
    mse = mean_squared_error(ye, yp); mae = mean_absolute_error(ye, yp); r2 = r2_score(ye, yp)
    cv_s = cross_val_score(est, Xt, yt, cv=CV, scoring="neg_mean_squared_error")
    return {"model": name, "mse": round(mse,4), "mae": round(mae,4), "r2": round(r2,4),
            "cv_mse": round(-cv_s.mean(),4), "seed": SEED, "time_s": round(dt,1),
            "best_params": json.dumps(s.best_params_, default=str)}

def _fit_subprocess(q, name, Xt, yt, Xe, ye):
    """Entry point for the subprocess — calls _fit_model and puts result on queue."""
    try:
        result = _fit_model(name, Xt, yt, Xe, ye)
        q.put(result)
    except Exception as e:
        q.put({"error": str(e)})

def tune(name, Xt, yt, Xe, ye, timeout=TIMEOUT_PER_MODEL):
    """Tune one model with process-level timeout. Fall back to default-param fit on timeout."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_fit_subprocess, args=(q, name, Xt, yt, Xe, ye))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join()
        t0 = time.perf_counter()
        est = build(name).fit(Xt, yt)
        dt = time.perf_counter() - t0
        yp = est.predict(Xe)
        mse = mean_squared_error(ye, yp); mae = mean_absolute_error(ye, yp); r2 = r2_score(ye, yp)
        cv_s = cross_val_score(est, Xt, yt, cv=CV, scoring="neg_mean_squared_error")
        return {"model": name, "mse": round(mse,4), "mae": round(mae,4), "r2": round(r2,4),
                "cv_mse": round(-cv_s.mean(),4), "seed": SEED, "time_s": round(dt,1),
                "best_params": "{}", "note": "timed_out"}
    result = q.get()
    if "error" in result:
        raise RuntimeError(f"Tuning failed for {name}: {result['error']}")
    return result

def main():
    print(f"SEED={SEED}  N_ITER={N_ITER}  CV={CV}")
    Xt, Xe, yt, ye = load_and_prep()
    print(f"Train={Xt.shape[0]}  Test={Xe.shape[0]}  Features={Xt.shape[1]}")
    results = []
    for m in ["LinearRegression", "Ridge", "DecisionTree", "RandomForest", "XGBoost", "LightGBM"]:
        r = tune(m, Xt, yt, Xe, ye)
        results.append(r)
        print(f"  {r['model']:>14s}  MSE={r['mse']:.4f}  R²={r['r2']:.4f}  {r['time_s']:.0f}s")
    pd.DataFrame(results).drop(columns=["best_params"]).sort_values("mse").to_csv(OUTPUT_DIR / "results.csv", index=False)
    best = min(results, key=lambda r: r["mse"])
    print(f"Best: {best['model']}  MSE={best['mse']:.4f}  R²={best['r2']:.4f}")
    return best

if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
