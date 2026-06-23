"""
database.py — SQLite connection & SQLAlchemy ORM models
MaizeForecast Backend
"""

from sqlalchemy import (
    create_engine, Column, Integer, Float, String,
    DateTime, Text, LargeBinary, Boolean, JSON
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

# ─── Connection URL ────────────────────────────────────────────────────────────
# Use SQLite for development (no external database needed)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./maizeforecast.db"
)

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ══════════════════════════════════════════════════════════════════════════════
# ORM MODELS
# ══════════════════════════════════════════════════════════════════════════════

class Dataset(Base):
    """Stores uploaded CSV/Excel datasets as JSON rows."""
    __tablename__ = "datasets"

    id          = Column(Integer, primary_key=True, index=True)
    filename    = Column(String(255), nullable=False)
    rows        = Column(Integer)
    columns     = Column(Integer)
    column_names = Column(JSON)          # list of column name strings
    data        = Column(JSON)           # list of row dicts (the full dataset)
    avg_price   = Column(Float, nullable=True)
    min_price   = Column(Float, nullable=True)
    max_price   = Column(Float, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    is_active   = Column(Boolean, default=True)  # latest active dataset flag

    def __repr__(self):
        return f"<Dataset id={self.id} file='{self.filename}' rows={self.rows}>"


class TrainedModel(Base):
    """Stores serialized trained ML models (pickle bytes) and their metrics."""
    __tablename__ = "trained_models"

    id              = Column(Integer, primary_key=True, index=True)
    dataset_id      = Column(Integer, nullable=True)   # FK to datasets.id
    model_blob      = Column(LargeBinary, nullable=False)  # pickled model
    scaler_blob     = Column(LargeBinary, nullable=False)  # pickled scaler
    encoders_blob   = Column(LargeBinary, nullable=False)  # pickled encoders dict
    feature_cols    = Column(JSON)        # list of feature column names
    mae             = Column(Float)
    rmse            = Column(Float)
    r2              = Column(Float)
    accuracy        = Column(Float)
    samples_trained = Column(Integer)
    samples_tested  = Column(Integer)
    feature_importances = Column(JSON)   # dict {feature: importance}
    trained_at      = Column(DateTime, default=datetime.utcnow)
    is_active       = Column(Boolean, default=True)  # latest active model flag

    def __repr__(self):
        return f"<TrainedModel id={self.id} r2={self.r2} trained_at={self.trained_at}>"


class PredictionHistory(Base):
    """Logs every prediction request and its result."""
    __tablename__ = "prediction_history"

    id               = Column(Integer, primary_key=True, index=True)
    model_id         = Column(Integer, nullable=True)   # FK to trained_models.id
    region           = Column(String(100))
    market           = Column(String(100), nullable=True)
    year             = Column(Integer)
    month            = Column(Integer)
    inflation_rate   = Column(Float, nullable=True)
    open_price       = Column(Float, nullable=True)
    high_price       = Column(Float, nullable=True)
    low_price        = Column(Float, nullable=True)
    confidence_score = Column(Float, nullable=True)
    trust_score      = Column(Float, nullable=True)
    maize_inflation  = Column(Float, nullable=True)
    predicted_price  = Column(Float)
    prediction_mode  = Column(String(20))   # "ml" or "demo"
    model_r2         = Column(Float, nullable=True)
    created_at       = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Prediction id={self.id} region={self.region} price={self.predicted_price}>"


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def get_db():
    """FastAPI dependency — yields a DB session and closes it after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """Create all tables if they don't exist yet."""
    Base.metadata.create_all(bind=engine)
    print("✅ PostgreSQL tables created / verified.")
