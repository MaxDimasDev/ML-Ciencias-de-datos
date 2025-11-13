from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # Features are flexible; missing values are allowed and will be imputed
    features: Dict[str, Any]


class PredictResponse(BaseModel):
    predicted: int
    probability: float
    timestamp: datetime
    model_version: str


class Metrics(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float | None = None
    confusion_matrix: List[List[int]]
    y_distribution: Dict[str, int]
    roc_curve: Dict[str, List[float]]
    pr_curve: Dict[str, List[float]]


class ModelInfo(BaseModel):
    version: str
    created_at: datetime
    metrics: Metrics


class MetricsHistoryResponse(BaseModel):
    history: List[ModelInfo]


class RetrainResponse(BaseModel):
    version: str
    created_at: datetime
    metrics: Metrics