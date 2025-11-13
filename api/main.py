from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from .db import Base, engine, get_db, SessionLocal
from . import crud
from .ml import (
    load_default_dataset,
    train_and_evaluate,
    dump_artifact,
    load_artifact,
)
from .schemas import PredictRequest, PredictResponse, MetricsHistoryResponse, RetrainResponse, ModelInfo


app = FastAPI(title="Bank Marketing - Logistic Regression API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    # Create tables
    Base.metadata.create_all(bind=engine)
    # Ensure at least one model exists
    db = SessionLocal()
    try:
        latest = crud.get_latest_model(db)
        if latest is None:
            base_df = load_default_dataset()
            pipe, metrics, _schema = train_and_evaluate(base_df)
            version = crud.next_version(db)
            artifact = dump_artifact(pipe)
            crud.create_model_version(db, version=version, artifact=artifact, metrics=metrics)
    finally:
        db.close()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model/latest", response_model=ModelInfo)
def model_latest(db: Session = Depends(get_db)):
    mv = crud.get_latest_model(db)
    if mv is None:
        raise HTTPException(status_code=500, detail="Model not available")
    return {
        "version": mv.version,
        "created_at": mv.created_at,
        "metrics": mv.metrics,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, db: Session = Depends(get_db)):
    mv = crud.get_latest_model(db)
    if mv is None:
        raise HTTPException(status_code=500, detail="Model not available")
    pipe = load_artifact(mv.artifact)

    # Build single-row DataFrame with potential missing columns
    X_in = pd.DataFrame([req.features])

    # Ensure all expected columns exist (based on stored schema)
    schema = (mv.metrics or {}).get("schema", {})
    cat_cols = schema.get("categorical", [])
    num_cols = schema.get("numerical", [])
    for col in cat_cols:
        if col not in X_in.columns:
            X_in[col] = "unknown"
    for col in num_cols:
        if col not in X_in.columns:
            X_in[col] = 0

    # Reorder columns to match training order (optional but safer)
    all_cols = schema.get("all")
    if all_cols:
        X_in = X_in.reindex(columns=all_cols, fill_value=0)

    # Predict
    try:
        proba = float(pipe.predict_proba(X_in)[:, 1][0])
        pred = int(1 if proba >= 0.5 else 0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")

    # Save prediction
    p = crud.save_prediction(
        db,
        features=req.features,
        predicted=pred,
        probability=proba,
        model=mv,
    )

    return PredictResponse(
        predicted=pred,
        probability=proba,
        timestamp=p.created_at,
        model_version=mv.version,
    )


@app.get("/metrics", response_model=MetricsHistoryResponse)
def metrics(limit: int = 5, db: Session = Depends(get_db)):
    rows = crud.get_metrics_history(db, limit=limit)
    history = [
        {
            "version": r.version,
            "created_at": r.created_at,
            "metrics": r.metrics,
        }
        for r in rows
    ]
    return {"history": history}


def _read_csv_auto(content: bytes) -> pd.DataFrame:
    # Detectar separador automáticamente (',' o ';')
    import csv
    sample = content[:2048].decode("utf-8", errors="ignore")
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;")
        sep = dialect.delimiter
    except Exception:
        sep = ","
    return pd.read_csv(io.BytesIO(content), sep=sep)


@app.post("/retrain", response_model=RetrainResponse)
def retrain(
    labeled_csv: Optional[UploadFile] = File(default=None, description="Optional CSV with same schema including 'y' target"),
    db: Session = Depends(get_db),
):
    base_df = load_default_dataset()
    add_df = None
    if labeled_csv is not None:
        try:
            content = labeled_csv.file.read()
            add_df = _read_csv_auto(content)
            if "y" not in add_df.columns:
                raise ValueError("Provided CSV must include 'y' target")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid CSV: {e}")

    pipe, metrics, _schema = train_and_evaluate(base_df, additional_df=add_df)
    version = crud.next_version(db)
    artifact = dump_artifact(pipe)
    mv = crud.create_model_version(db, version=version, artifact=artifact, metrics=metrics)

    return RetrainResponse(version=mv.version, created_at=mv.created_at, metrics=mv.metrics)