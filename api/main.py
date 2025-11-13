from __future__ import annotations

import io
import os
from datetime import datetime
from typing import Optional, Dict, Any

import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, BackgroundTasks
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
from .data_mining import collect_training_data
from .schemas import (
    PredictRequest,
    PredictResponse,
    MetricsHistoryResponse,
    RetrainResponse,
    ModelInfo,
    FeedbackRequest,
    FeedbackResponse,
)


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
            # Minado de datos: construir dataset desde fuente externa (SQL/HTTP)
            base_df = collect_training_data()
            pipe, metrics, _schema = train_and_evaluate(base_df)
            version = crud.next_version(db)
            artifact = dump_artifact(pipe)
            crud.create_model_version(db, version=version, artifact=artifact, metrics=metrics)
    finally:
        db.close()


# Auto-retrain settings (disabled by default for a simpler experience)
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "false").lower() in {"1", "true", "yes", "y", "on"}
# Nuevo: reentrenar después de cada predicción (controlado por variable de entorno)
AUTO_RETRAIN_AFTER_PREDICTION = os.getenv("AUTO_RETRAIN_AFTER_PREDICTION", "true").lower() in {"1", "true", "yes", "y", "on"}
try:
    RETRAIN_MIN_FEEDBACK = int(os.getenv("RETRAIN_MIN_FEEDBACK", "1"))
except Exception:
    RETRAIN_MIN_FEEDBACK = 1


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
def predict(req: PredictRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
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

    # Disparar reentrenamiento en background si está habilitado
    if AUTO_RETRAIN_AFTER_PREDICTION:
        try:
            background_tasks.add_task(_retrain_from_feedback_background)
        except Exception:
            pass

    return PredictResponse(
        predicted=pred,
        probability=proba,
        timestamp=p.created_at,
        model_version=mv.version,
    )


def _labeled_examples_to_df(examples: list[Dict[str, Any]]) -> pd.DataFrame:
    if not examples:
        return pd.DataFrame()
    rows = []
    for ex in examples:
        # Expect keys: features (dict), y (int)
        row = dict(ex["features"])  # shallow copy
        row["y"] = int(ex["y"])
        rows.append(row)
    return pd.DataFrame(rows)


def _retrain_from_feedback_background():
    db = SessionLocal()
    try:
        # Reentrenar a partir del dataset minado + ejemplos etiquetados (si existen)
        base_df = collect_training_data()
        # Fetch labeled examples and transform to DataFrame
        labeled = crud.get_all_labeled_examples(db)
        labeled_dicts = [{"features": r.features, "y": r.y} for r in labeled]
        add_df = _labeled_examples_to_df(labeled_dicts)

        pipe, metrics, _schema = train_and_evaluate(base_df, additional_df=add_df if not add_df.empty else None)
        version = crud.next_version(db)
        artifact = dump_artifact(pipe)
        crud.create_model_version(db, version=version, artifact=artifact, metrics=metrics)
    finally:
        db.close()


@app.post("/feedback", response_model=FeedbackResponse)
def submit_feedback(payload: FeedbackRequest, background_tasks: BackgroundTasks, db: Session = Depends(get_db)):
    # Guardar ejemplo etiquetado
    crud.add_labeled_example(db, features=payload.features, y=int(payload.y))

    retrain_started = False
    if AUTO_RETRAIN:
        try:
            total = crud.count_labeled_examples(db)
            if RETRAIN_MIN_FEEDBACK <= 1 or (total % max(1, RETRAIN_MIN_FEEDBACK) == 0):
                background_tasks.add_task(_retrain_from_feedback_background)
                retrain_started = True
        except Exception:
            retrain_started = False

    return FeedbackResponse(accepted=True, retrain_started=retrain_started)


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