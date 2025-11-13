from datetime import datetime
from sqlalchemy import Column, Integer, String, DateTime, Float, ForeignKey, LargeBinary
from sqlalchemy.types import JSON
from sqlalchemy.orm import relationship

from .db import Base


def _json_type():
    # Use portable JSON type for cross-dialect compatibility
    return JSON


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, index=True)
    version = Column(String, index=True, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    # Serialized joblib pipeline
    artifact = Column(LargeBinary, nullable=False)
    # Metrics summary and curves
    metrics = Column(_json_type(), nullable=False)

    predictions = relationship("Prediction", back_populates="model", cascade="all, delete")


class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    features = Column(_json_type(), nullable=False)
    predicted = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)

    model_id = Column(Integer, ForeignKey("model_versions.id"), nullable=False)
    model = relationship("ModelVersion", back_populates="predictions")