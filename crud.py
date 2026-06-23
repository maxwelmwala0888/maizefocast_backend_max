"""
crud.py — Database CRUD operations
MaizeForecast Backend
"""

import pickle
from typing import Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc
from database import Dataset, TrainedModel, PredictionHistory
from datetime import datetime


# ══════════════════════════════════════════════════════════════════════════════
# DATASET CRUD
# ══════════════════════════════════════════════════════════════════════════════

def save_dataset(
    db: Session,
    filename: str,
    rows: int,
    columns: int,
    column_names: list,
    data: list,
    avg_price: float = None,
    min_price: float = None,
    max_price: float = None,
) -> Dataset:
    """Save a new dataset and deactivate any previous ones."""
    # Deactivate old active datasets
    db.query(Dataset).filter(Dataset.is_active == True).update({"is_active": False})

    dataset = Dataset(
        filename=filename,
        rows=rows,
        columns=columns,
        column_names=column_names,
        data=data,
        avg_price=avg_price,
        min_price=min_price,
        max_price=max_price,
        is_active=True,
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


def get_active_dataset(db: Session) -> Optional[Dataset]:
    """Return the most recently uploaded active dataset."""
    return (
        db.query(Dataset)
        .filter(Dataset.is_active == True)
        .order_by(desc(Dataset.uploaded_at))
        .first()
    )


def list_datasets(db: Session, limit: int = 10) -> list:
    """List recent dataset uploads (summary only, no data blob)."""
    rows = (
        db.query(
            Dataset.id, Dataset.filename, Dataset.rows,
            Dataset.columns, Dataset.avg_price, Dataset.uploaded_at, Dataset.is_active
        )
        .order_by(desc(Dataset.uploaded_at))
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "filename": r.filename,
            "rows": r.rows,
            "columns": r.columns,
            "avg_price": r.avg_price,
            "uploaded_at": r.uploaded_at.isoformat() if r.uploaded_at else None,
            "is_active": r.is_active,
        }
        for r in rows
    ]


# ══════════════════════════════════════════════════════════════════════════════
# TRAINED MODEL CRUD
# ══════════════════════════════════════════════════════════════════════════════

def save_model(
    db: Session,
    model,           # sklearn model object
    scaler,          # StandardScaler object
    encoders: dict,  # dict of LabelEncoders
    feature_cols: list,
    metrics: dict,
    dataset_id: int = None,
) -> TrainedModel:
    """Serialize and save a trained model. Deactivates previous active model."""
    # Deactivate old active models
    db.query(TrainedModel).filter(TrainedModel.is_active == True).update({"is_active": False})

    record = TrainedModel(
        dataset_id=dataset_id,
        model_blob=pickle.dumps(model),
        scaler_blob=pickle.dumps(scaler),
        encoders_blob=pickle.dumps(encoders),
        feature_cols=feature_cols,
        mae=metrics.get("mae"),
        rmse=metrics.get("rmse"),
        r2=metrics.get("r2"),
        accuracy=metrics.get("accuracy"),
        samples_trained=metrics.get("samples_trained"),
        samples_tested=metrics.get("samples_tested"),
        feature_importances=metrics.get("feature_importances"),
        is_active=True,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def load_active_model(db: Session) -> Optional[TrainedModel]:
    """Load the most recently trained active model record."""
    return (
        db.query(TrainedModel)
        .filter(TrainedModel.is_active == True)
        .order_by(desc(TrainedModel.trained_at))
        .first()
    )


def deserialize_model(record: TrainedModel) -> tuple:
    """Unpickle model, scaler, and encoders from a TrainedModel record."""
    model    = pickle.loads(record.model_blob)
    scaler   = pickle.loads(record.scaler_blob)
    encoders = pickle.loads(record.encoders_blob)
    return model, scaler, encoders, record.feature_cols


def list_models(db: Session, limit: int = 10) -> list:
    """List recent trained models (no blobs)."""
    rows = (
        db.query(
            TrainedModel.id, TrainedModel.dataset_id,
            TrainedModel.mae, TrainedModel.rmse, TrainedModel.r2,
            TrainedModel.accuracy, TrainedModel.samples_trained,
            TrainedModel.trained_at, TrainedModel.is_active
        )
        .order_by(desc(TrainedModel.trained_at))
        .limit(limit)
        .all()
    )
    return [
        {
            "id": r.id,
            "dataset_id": r.dataset_id,
            "mae": r.mae,
            "rmse": r.rmse,
            "r2": r.r2,
            "accuracy": r.accuracy,
            "samples_trained": r.samples_trained,
            "trained_at": r.trained_at.isoformat() if r.trained_at else None,
            "is_active": r.is_active,
        }
        for r in rows
    ]


# ══════════════════════════════════════════════════════════════════════════════
# PREDICTION HISTORY CRUD
# ══════════════════════════════════════════════════════════════════════════════

def save_prediction(
    db: Session,
    input_data: dict,
    predicted_price: float,
    prediction_mode: str,
    model_r2: float = None,
    model_id: int = None,
) -> PredictionHistory:
    """Log a prediction to the database."""
    record = PredictionHistory(
        model_id=model_id,
        region=input_data.get("region"),
        market=input_data.get("market"),
        year=input_data.get("year"),
        month=input_data.get("month"),
        inflation_rate=input_data.get("inflation_rate"),
        open_price=input_data.get("open_price"),
        high_price=input_data.get("high_price"),
        low_price=input_data.get("low_price"),
        confidence_score=input_data.get("confidence_score"),
        trust_score=input_data.get("trust_score"),
        maize_inflation=input_data.get("maize_inflation"),
        predicted_price=predicted_price,
        prediction_mode=prediction_mode,
        model_r2=model_r2,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def get_prediction_history(
    db: Session,
    region: str = None,
    limit: int = 50,
    offset: int = 0,
) -> list:
    """Fetch prediction history, optionally filtered by region."""
    query = db.query(PredictionHistory)
    if region:
        query = query.filter(PredictionHistory.region.ilike(f"%{region}%"))
    rows = query.order_by(desc(PredictionHistory.created_at)).offset(offset).limit(limit).all()
    return [
        {
            "id": r.id,
            "region": r.region,
            "market": r.market,
            "year": r.year,
            "month": r.month,
            "predicted_price": r.predicted_price,
            "prediction_mode": r.prediction_mode,
            "model_r2": r.model_r2,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


def get_prediction_stats(db: Session) -> dict:
    """Aggregate stats across all predictions."""
    from sqlalchemy import func
    result = db.query(
        func.count(PredictionHistory.id).label("total"),
        func.avg(PredictionHistory.predicted_price).label("avg_price"),
        func.min(PredictionHistory.predicted_price).label("min_price"),
        func.max(PredictionHistory.predicted_price).label("max_price"),
    ).first()

    return {
        "total_predictions": result.total or 0,
        "avg_predicted_price": round(float(result.avg_price), 2) if result.avg_price else None,
        "min_predicted_price": round(float(result.min_price), 2) if result.min_price else None,
        "max_predicted_price": round(float(result.max_price), 2) if result.max_price else None,
    }
