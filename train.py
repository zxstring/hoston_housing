"""Boston housing — single-seed run. Called by run.sh with $SEED env var."""

import json, multiprocessing as mp, os, time, warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, RepeatedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

import config

warnings.filterwarnings("ignore")

SEED = int(os.environ.get("SEED", 42))

OUTPUT_DIR = Path(__file__).resolve().parent / f"output_s{SEED}"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Data loading ───────────────────────────────────────────────────────────

def load_and_prep():
    bunch = fetch_openml(name="boston", version=1, as_frame=True, parser="auto")
    df = bunch.frame
    df.columns = [c.upper() for c in df.columns]

    for c in df.columns:
        if df[c].dtype.name in ("category", "object"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.fillna(df.median())

    def winsorize(s, lim=1.5):
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        return s.clip(q1 - lim * (q3 - q1), q3 + lim * (q3 - q1))

    for c in ["CRIM", "RM", "B", "ZN", "DIS"]:
        if c in df.columns:
            df[c] = winsorize(df[c])

    df["RM2"] = df["RM"] ** 2
    df["RM_LSTAT"] = df["RM"] * df["LSTAT"]
    df["DIS_NOX"] = df["DIS"] * df["NOX"]

    y = df["MEDV"].values
    X = df.drop(columns=["MEDV"])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=SEED)
    return X_tr, X_te, y_tr, y_te


# ── Model builders ──────────────────────────────────────────────────────────

def build(name):
    if name == "LinearRegression":
        return Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("lr", LinearRegression()),
        ])
    elif name == "Ridge":
        return Pipeline([
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
            ("ridge", Ridge(random_state=SEED)),
        ])
    elif name == "DecisionTree":
        return DecisionTreeRegressor(random_state=SEED)
    elif name == "RandomForest":
        return RandomForestRegressor(random_state=SEED)
    elif name == "XGBoost":
        return XGBRegressor(random_state=SEED, tree_method="hist",
                            device="cpu", verbosity=0, nthread=1)
    elif name == "LightGBM":
        return LGBMRegressor(random_state=SEED, device="cpu",
                             verbose=-1, num_threads=1)
    raise ValueError(f"Unknown model: {name}")


# ── Tuning / evaluation ────────────────────────────────────────────────────

def _fit_model(name, Xt, yt, Xe, ye):
    """Actual tuning + evaluation. Runs in a subprocess for timeout safety."""
    param_dist, n_iter, _ = config.MODEL_PARAMS[name]
    cv = RepeatedKFold(**config.BOOST_CV, random_state=SEED) \
        if name in ("XGBoost", "LightGBM") else config.REGULAR_CV

    t0 = time.perf_counter()
    search = RandomizedSearchCV(
        build(name), param_distributions=param_dist, n_iter=n_iter,
        cv=cv, scoring="neg_mean_squared_error", n_jobs=1, random_state=SEED)
    search.fit(Xt, yt)
    est = search.best_estimator_
    dt = time.perf_counter() - t0

    yp = est.predict(Xe)
    mse = mean_squared_error(ye, yp)
    mae = mean_absolute_error(ye, yp)
    r2 = r2_score(ye, yp)

    eval_cv = RepeatedKFold(**config.BOOST_CV, random_state=SEED) \
        if name in ("XGBoost", "LightGBM") else config.REGULAR_CV
    cv_s = cross_val_score(est, Xt, yt, cv=eval_cv,
                           scoring="neg_mean_squared_error")

    return {
        "model": name, "mse": round(mse, 4), "mae": round(mae, 4),
        "r2": round(r2, 4), "cv_mse": round(-cv_s.mean(), 4),
        "seed": SEED, "time_s": round(dt, 1),
        "best_params": json.dumps(search.best_params_, default=str),
    }


def _fit_subprocess(q, name, Xt, yt, Xe, ye):
    try:
        result = _fit_model(name, Xt, yt, Xe, ye)
        q.put(result)
    except Exception as e:
        q.put({"error": str(e)})


def tune(name, Xt, yt, Xe, ye):
    _, _, timeout = config.MODEL_PARAMS[name]
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
        mse = mean_squared_error(ye, yp)
        mae = mean_absolute_error(ye, yp)
        r2 = r2_score(ye, yp)
        cv = RepeatedKFold(**config.BOOST_CV, random_state=SEED) \
            if name in ("XGBoost", "LightGBM") else config.REGULAR_CV
        cv_s = cross_val_score(est, Xt, yt, cv=cv,
                               scoring="neg_mean_squared_error")
        return {
            "model": name, "mse": round(mse, 4), "mae": round(mae, 4),
            "r2": round(r2, 4), "cv_mse": round(-cv_s.mean(), 4),
            "seed": SEED, "time_s": round(dt, 1),
            "best_params": "{}", "note": "timed_out",
        }
    result = q.get()
    if "error" in result:
        raise RuntimeError(f"Tuning failed for {name}: {result['error']}")
    return result


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    t_total = time.perf_counter()
    print(f"SEED={SEED}")
    Xt, Xe, yt, ye = load_and_prep()
    print(f"Train={Xt.shape[0]}  Test={Xe.shape[0]}  Features={Xt.shape[1]}")

    model_order = ["LinearRegression", "Ridge", "DecisionTree",
                   "RandomForest", "XGBoost", "LightGBM"]

    # Show config for each model
    for m in model_order:
        _, n_iter, timeout = config.MODEL_PARAMS[m]
        print(f"  [{m}] n_iter={n_iter} timeout={timeout}s")

    # Run all models in parallel
    print(f"\nTraining {len(model_order)} models in parallel ...\n")
    results = []
    with ThreadPoolExecutor(max_workers=len(model_order)) as executor:
        future_map = {executor.submit(tune, m, Xt, yt, Xe, ye): m
                      for m in model_order}
        for future in as_completed(future_map):
            m = future_map[future]
            r = future.result()
            results.append(r)
            extra = f" ({r['note']})" if r.get("note") else ""
            print(f"  [{r['model']:>14s}] MSE={r['mse']:.4f}  R²={r['r2']:.4f}  "
                  f"{r['time_s']:.0f}s{extra}")

    results.sort(key=lambda r: r["mse"])
    pd.DataFrame(results).drop(columns=["best_params"]).to_csv(
        OUTPUT_DIR / "results.csv", index=False)
    best = results[0]
    total = time.perf_counter() - t_total
    print(f"\nBest: {best['model']}  MSE={best['mse']:.4f}  R²={best['r2']:.4f}")
    print(f"Total time: {total:.0f}s")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    main()
