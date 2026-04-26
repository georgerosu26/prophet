import json
import os
import pickle
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import cmdstanpy
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from prophet import Prophet


API_TOKEN = os.getenv("PROPHET_API_TOKEN", "")
MODEL_DIR = os.getenv("MODEL_DIR", "/app/models")
DEFAULT_HORIZON_DAYS = int(os.getenv("DEFAULT_HORIZON_DAYS", "91"))
MAX_HORIZON_DAYS = int(os.getenv("MAX_HORIZON_DAYS", "180"))
MIN_TRAIN_ROWS = int(os.getenv("MIN_TRAIN_ROWS", "60"))
APP_VERSION = os.getenv("APP_VERSION", "1.0.1")

app = FastAPI(title="Prophet Forecast API", version="1.0.0")


def _configure_cmdstan() -> None:
    # Prefer explicitly-installed CmdStan locations in the container.
    candidates = [
        os.getenv("CMDSTAN", ""),
        "/opt/cmdstan/cmdstan-2.38.0",
        "/opt/cmdstan",
    ]
    for path in candidates:
        if not path:
            continue
        makefile = os.path.join(path, "makefile")
        if os.path.exists(makefile):
            cmdstanpy.set_cmdstan_path(path)
            return

    # Fallback: search /opt for installed cmdstan-* directories.
    try:
        for name in sorted(os.listdir("/opt"), reverse=True):
            if not name.startswith("cmdstan-"):
                continue
            candidate = os.path.join("/opt", name)
            if os.path.exists(os.path.join(candidate, "makefile")):
                cmdstanpy.set_cmdstan_path(candidate)
                return
    except FileNotFoundError:
        pass


class TrainRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(default_factory=list)
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    seasonality_mode: str = "multiplicative"


class PredictRequest(BaseModel):
    rows: List[Dict[str, Any]] = Field(default_factory=list)


class TrainPredictRequest(BaseModel):
    train_rows: List[Dict[str, Any]] = Field(default_factory=list)
    future_rows: List[Dict[str, Any]] = Field(default_factory=list)
    horizon_days: int = DEFAULT_HORIZON_DAYS


def _auth(x_api_token: Optional[str]):
    if API_TOKEN and x_api_token != API_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _model_paths(store_id: str):
    safe = "".join(ch for ch in store_id if ch.isalnum() or ch in ("-", "_"))
    return (
        os.path.join(MODEL_DIR, f"{safe}.pkl"),
        os.path.join(MODEL_DIR, f"{safe}.meta.json"),
    )


def _prepare_df(rows: List[Dict[str, Any]], require_y: bool) -> pd.DataFrame:
    if not rows:
        raise HTTPException(status_code=400, detail="rows is required")
    df = pd.DataFrame(rows)
    if "ds" not in df.columns:
        raise HTTPException(status_code=400, detail="rows must include ds")
    df["ds"] = pd.to_datetime(df["ds"]).dt.date
    df = df.sort_values("ds").drop_duplicates(subset=["ds"], keep="last")

    if require_y:
        if "y" not in df.columns:
            raise HTTPException(status_code=400, detail="train rows must include y")
        df["y"] = pd.to_numeric(df["y"], errors="coerce")
        df = df.dropna(subset=["y"])
        if len(df) < MIN_TRAIN_ROWS:
            raise HTTPException(
                status_code=400,
                detail=f"At least {MIN_TRAIN_ROWS} valid rows are required for training",
            )

    # Cast all feature columns to numeric (except ds/y).
    for col in [c for c in df.columns if c not in ("ds", "y")]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    return df


def _ensure_columns(df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in feature_columns:
        if col not in out.columns:
            out[col] = 0.0
    keep_cols = ["ds"] + feature_columns
    if "y" in out.columns:
        keep_cols += ["y"]
    for col in keep_cols:
        if col in ("ds", "y"):
            continue
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out[keep_cols]


def _build_fallback_future(days: int) -> pd.DataFrame:
    days = max(1, min(MAX_HORIZON_DAYS, int(days)))
    start = date.today() + timedelta(days=1)
    return pd.DataFrame({"ds": [start + timedelta(days=i) for i in range(days)]})


def _fit_model(
    train_df: pd.DataFrame,
    changepoint_prior_scale: float,
    seasonality_prior_scale: float,
    seasonality_mode: str,
):
    _configure_cmdstan()
    model = Prophet(
        stan_backend="CMDSTANPY",
        growth="linear",
        weekly_seasonality=True,
        yearly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale,
        seasonality_mode=seasonality_mode,
    )

    feature_columns = [c for c in train_df.columns if c not in ("ds", "y")]
    for col in feature_columns:
        model.add_regressor(col)

    model.fit(train_df)
    return model, feature_columns


def _save_model(store_id: str, model: Prophet, feature_columns: List[str], train_rows: int):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path, meta_path = _model_paths(store_id)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "store_id": store_id,
                "feature_columns": feature_columns,
                "train_rows": train_rows,
                "trained_at": datetime.utcnow().isoformat() + "Z",
            },
            f,
        )


def _load_model(store_id: str):
    model_path, meta_path = _model_paths(store_id)
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        raise HTTPException(status_code=404, detail=f"No model found for store {store_id}")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


@app.get("/health")
def health():
    return {"ok": True, "time": datetime.utcnow().isoformat() + "Z"}


@app.get("/version")
def version():
    return {
        "app_version": APP_VERSION,
        "defaults": {
            "default_horizon_days": DEFAULT_HORIZON_DAYS,
            "max_horizon_days": MAX_HORIZON_DAYS,
            "min_train_rows": MIN_TRAIN_ROWS,
        },
        "time": datetime.utcnow().isoformat() + "Z",
    }


@app.post("/train/{store_id}")
def train(store_id: str, payload: TrainRequest, x_api_token: Optional[str] = Header(default=None)):
    _auth(x_api_token)
    train_df = _prepare_df(payload.rows, require_y=True)
    model, feature_columns = _fit_model(
        train_df=train_df,
        changepoint_prior_scale=payload.changepoint_prior_scale,
        seasonality_prior_scale=payload.seasonality_prior_scale,
        seasonality_mode=payload.seasonality_mode,
    )
    _save_model(store_id, model, feature_columns, len(train_df))
    return {
        "store_id": store_id,
        "trained_rows": len(train_df),
        "feature_columns": feature_columns,
    }


@app.post("/predict/{store_id}")
def predict(store_id: str, payload: PredictRequest, x_api_token: Optional[str] = Header(default=None)):
    _auth(x_api_token)
    model, meta = _load_model(store_id)
    feature_columns = meta.get("feature_columns", [])
    future_df = _prepare_df(payload.rows, require_y=False)
    future_df = _ensure_columns(future_df, feature_columns)
    forecast = model.predict(future_df)
    out = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    out["ds"] = out["ds"].dt.strftime("%Y-%m-%d")
    return {
        "store_id": store_id,
        "predictions": out.to_dict(orient="records"),
        "feature_columns": feature_columns,
    }


@app.post("/train-predict/{store_id}")
def train_predict(
    store_id: str,
    payload: TrainPredictRequest,
    x_api_token: Optional[str] = Header(default=None),
):
    _auth(x_api_token)
    train_df = _prepare_df(payload.train_rows, require_y=True)
    model, feature_columns = _fit_model(
        train_df=train_df,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10.0,
        seasonality_mode="multiplicative",
    )
    _save_model(store_id, model, feature_columns, len(train_df))

    if payload.future_rows:
        future_df = _prepare_df(payload.future_rows, require_y=False)
    else:
        future_df = _build_fallback_future(payload.horizon_days)
    future_df = _ensure_columns(future_df, feature_columns)

    forecast = model.predict(future_df)
    pred = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    pred["ds"] = pred["ds"].dt.strftime("%Y-%m-%d")
    return {
        "store_id": store_id,
        "trained_rows": len(train_df),
        "feature_columns": feature_columns,
        "predictions": pred.to_dict(orient="records"),
    }


@app.post("/backtest/{store_id}")
def backtest(store_id: str, payload: TrainRequest, x_api_token: Optional[str] = Header(default=None)):
    _auth(x_api_token)
    df = _prepare_df(payload.rows, require_y=True)
    split_idx = max(int(len(df) * 0.8), MIN_TRAIN_ROWS)
    split_idx = min(split_idx, len(df) - 14)
    if split_idx <= 0 or split_idx >= len(df):
        raise HTTPException(status_code=400, detail="Not enough data for backtest split")

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    model, feature_columns = _fit_model(
        train_df=train_df,
        changepoint_prior_scale=payload.changepoint_prior_scale,
        seasonality_prior_scale=payload.seasonality_prior_scale,
        seasonality_mode=payload.seasonality_mode,
    )
    future = _ensure_columns(test_df.drop(columns=["y"]), feature_columns)
    pred = model.predict(future)
    y_true = test_df["y"].to_numpy(dtype=float)
    y_hat = pred["yhat"].to_numpy(dtype=float)
    mae = float(np.mean(np.abs(y_true - y_hat)))
    mape = float(np.mean(np.abs((y_true - y_hat) / np.maximum(y_true, 1e-6))) * 100)
    rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
    return {
        "store_id": store_id,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "metrics": {"mae": mae, "mape": mape, "rmse": rmse},
    }


@app.get("/model-status/{store_id}")
def model_status(store_id: str, x_api_token: Optional[str] = Header(default=None)):
    _auth(x_api_token)
    _, meta = _load_model(store_id)
    return meta
