"""
main.py — FastAPI + Pre‑trained RandomForest (9 features)
"""
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib
import io
import os
from datetime import datetime, date
import traceback

from database import get_db, init_db
import crud

from fastapi import FastAPI

# ... other imports ...

# 1. CREATE THE APP INSTANCE FIRST
app = FastAPI(title="MaizeForecast API", version="6.0.0")

# 2. ADD CORS MIDDLEWARE (fix the typo in the URL!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://maizefocastappppp.netlify.app"],  # no extra 'p's
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. THEN DEFINE YOUR ROUTES
@app.get("/api/health")
def health():
    return {"status": "ok"}

# ... rest of your routes ...
# ─── (Optional) Serve static files if needed later ──────────────────────────
# app.mount("/static", StaticFiles(directory="static"), name="static")

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    error_detail = traceback.format_exc()
    print("🔥 UNHANDLED ERROR:")
    print(error_detail)
    return JSONResponse(status_code=500, content={"detail": str(exc), "trace": error_detail})

# ─── MODEL LOADER ───────────────────────────────────────────────────────────
MODEL_PATH = "random_forest_c_maize_model.joblib"
PRE_TRAINED_MODEL = None   # the RandomForest model object

def load_model():
    global PRE_TRAINED_MODEL
    if os.path.exists(MODEL_PATH):
        try:
            PRE_TRAINED_MODEL = joblib.load(MODEL_PATH)
            print(f"✓ Loaded RandomForest model from {MODEL_PATH}")
        except Exception as e:
            print(f"⚠ Error loading model: {e}")
            PRE_TRAINED_MODEL = None
    else:
        print("⚠ Model file not found. Predictions will use demo data.")

@app.on_event("startup")
def on_startup():
    init_db()
    load_model()
    print("✓ Backend initialized successfully")

# ─── ENCODINGS (exactly as during training) ─────────────────────────────────
MKT_ENCODING = {
    "Area 23": 0, "Balaka Boma": 1, "Bangula": 2, "Bembeke turn off": 3, "Bolero": 4,
    "Bowe": 5, "Bvumbwe": 6, "Chamama": 7, "Chatoloma": 8, "Chikhwawa": 9,
    "Chikuli": 10, "Chikweo": 11, "Chilinga": 12, "Chilumba": 13, "Chimbiya": 14,
    "Chinakanaka": 15, "Chinamwali": 16, "Chintheche": 17, "Chiponde": 18,
    "Chiradzulu Boma": 19, "Chitakale": 20, "Chitipa Boma": 21, "Dowa Boma": 22,
    "Dwangwa": 23, "Dyelatu": 24, "Dzaleka": 25, "Dzaleka (inside Camp)": 26,
    "Embangweni": 27, "Euthini": 28, "Golomoti": 29, "Hewe": 30, "Jali": 31,
    "Jenda": 32, "Kambilonje": 33, "Kameme": 34, "Kamsonga": 35, "Kamuzu Road": 36,
    "Kamwendo": 37, "Karonga Boma": 38, "Kasiya": 39, "Kasungu Boma": 40,
    "Khuwi": 41, "Lilongwe": 42, "Limbe": 43, "Limbuli": 44, "Lirangwe": 45,
    "Liwonde": 46, "Lizulu": 47, "Luchenza": 48, "Luncheza": 49, "Lunzu": 50,
    "Madisi": 51, "Makanjila": 52, "Makanjira": 53, "Makhanga": 54, "Malomo": 55,
    "Mangamba": 56, "Mangochi Boma": 57, "Mangochi Turn Off": 58, "Manyamula": 59,
    "Marka": 60, "Market Average": 61, "Mayaka": 62, "Mbela": 63, "Mbonechela": 64,
    "Mchinji Boma": 65, "Migowi": 66, "Misuku": 67, "Mitundu": 68, "Mkanda": 69,
    "Monkey Bay": 70, "Mpamba": 71, "Mpita": 72, "Mponela": 73, "Mtakataka": 74,
    "Mtowe": 75, "Mulanje Boma": 76, "Mulomba": 77, "Muloza": 78, "Mwansambo": 79,
    "Mwanza Boma": 80, "Mzimba": 81, "Mzuzu": 82, "Nambuma": 83, "Namwera": 84,
    "Nanjiri": 85, "Nayuchi": 86, "Nchalo": 87, "Neno Boma": 88, "Ngabu": 89,
    "Nkhamenya": 90, "Nkhatabay Boma": 91, "Nkhate": 92, "Nkhoma": 93,
    "Nkhotakota Boma": 94, "Nsanama": 95, "Nsanje Boma": 96, "Nserema": 97,
    "Nsikawanjala": 98, "Nsundwe": 99, "Nsungwi": 100, "Ntaja": 101, "Ntakataka": 102,
    "Ntcheu Boma": 103, "Ntchisi Boma": 104, "Nthalire": 105, "Ntonda": 106,
    "Ntowe": 107, "Phalombe Boma": 108, "Phaloni": 109, "Phalula": 110,
    "Rumphi Boma": 111, "Salima": 112, "Santhe": 113, "Sharpevaley": 114,
    "Songani": 115, "Songwe": 116, "Sorgin": 117, "Thavite": 118, "Thekerani": 119,
    "Thete": 120, "Thondwe": 121, "Thyolo Boma": 122, "Tomali": 123,
    "Tsangano turnoff": 124, "Uliwa": 125, "Ulongwe": 126, "Vigwagwa": 127,
    "Waliranji": 128, "Zomba Boma": 129
}

ADM1_ENCODING = {
    "Central Region": 0, "Market Average": 1, "Northern Region": 2, "Southern Region": 3
}

ADM2_ENCODING = {
    "Balaka": 0, "Blantyre": 1, "Blantyre City": 2, "Chikwawa": 3, "Chiradzulu": 4,
    "Chitipa": 5, "Dedza": 6, "Dowa": 7, "Karonga": 8, "Kasungu": 9, "Lilongwe": 10,
    "Lilongwe City": 11, "Machinga": 12, "Mangochi": 13, "Market Average": 14,
    "Mchinji": 15, "Mulanje": 16, "Mwanza": 17, "Mzimba": 18, "Mzuzu City": 19,
    "Neno": 20, "Nkhata Bay": 21, "Nkhotakota": 22, "Nsanje": 23, "Ntcheu": 24,
    "Ntchisi": 25, "Phalombe": 26, "Rumphi": 27, "Salima": 28, "Thyolo": 29,
    "Zomba": 30, "Zomba City": 31
}

# ─── SCHEMAS ─────────────────────────────────────────────────────────────────
class LoginRequest(BaseModel):
    username: str
    password: str
    role: str

class PredictionInput(BaseModel):
    region: Optional[str] = "Central"
    market: Optional[str] = "General"
    year: Optional[int] = datetime.now().year
    month: Optional[int] = datetime.now().month
    inflation_rate: Optional[float] = 8.5
    open_price: Optional[float] = 340.0
    high_price: Optional[float] = 355.0
    low_price: Optional[float] = 335.0
    trust_score: Optional[float] = 0.9
    maize_inflation: Optional[float] = 8.5
    class Config:
        extra = "ignore"

class SarimaxInput(BaseModel):
    year: int = datetime.now().year
    month: int = datetime.now().month
    avg_maize_price: Optional[float] = None
    infl: Optional[float] = None
    inflation_maize: Optional[float] = None
    trust_maize: Optional[float] = None
    c_rice: Optional[float] = None
    mkt: Optional[str] = None
    adm1: Optional[str] = None
    adm2: Optional[str] = None
    class Config:
        extra = "ignore"

# ─── FEATURES (exactly the 9 columns used during training) ──────────────────
FEATURE_ORDER = [
    'year', 'month', 'infl', 'inflation_maize', 'trust_maize',
    'c_rice', 'mkt_name_encoded', 'adm1_name_encoded', 'adm2_name_encoded'
]

def make_prediction(inp: SarimaxInput) -> float:
    if PRE_TRAINED_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    mkt_enc = MKT_ENCODING.get(inp.mkt, 0) if inp.mkt else 0
    adm1_enc = ADM1_ENCODING.get(inp.adm1, 0) if inp.adm1 else 0
    adm2_enc = ADM2_ENCODING.get(inp.adm2, 0) if inp.adm2 else 0

    row = {
        'year': inp.year,
        'month': inp.month,
        'infl': inp.infl if inp.infl is not None else 8.5,
        'inflation_maize': inp.inflation_maize if inp.inflation_maize is not None else 8.5,
        'trust_maize': inp.trust_maize if inp.trust_maize is not None else 0.9,
        'c_rice': inp.c_rice if inp.c_rice is not None else 400,
        'mkt_name_encoded': mkt_enc,
        'adm1_name_encoded': adm1_enc,
        'adm2_name_encoded': adm2_enc,
    }

    input_df = pd.DataFrame([row], columns=FEATURE_ORDER)
    return float(PRE_TRAINED_MODEL.predict(input_df)[0])

# ─── ROUTES ──────────────────────────────────────────────────────────────────
@app.post("/api/login")
def login(req: LoginRequest):
    if req.username and req.password and req.role in ["farmer", "analyst"]:
        return {"status": "success", "user": req.username, "role": req.role,
                "token": f"token_{req.username}_{datetime.now().timestamp()}",
                "message": f"Welcome {req.username} ({req.role})"}
    return {"status": "failed", "message": "Invalid credentials", "code": "AUTH_FAILED"}

@app.get("/api/health")
def api_health(db = Depends(get_db)):
    m = crud.load_active_model(db)
    d = crud.get_active_dataset(db)
    s = crud.get_prediction_stats(db)
    return {"status": "ok", "model_trained": m is not None or PRE_TRAINED_MODEL is not None,
            "dataset_loaded": d is not None,
            "total_predictions": s.get("total_predictions", 0), "timestamp": datetime.now().isoformat()}

@app.get("/api/dashboard")
def api_dashboard(db = Depends(get_db)):
    dataset = crud.get_active_dataset(db)
    model = crud.load_active_model(db)
    stats = crud.get_prediction_stats(db)
    if dataset is None:
        return {"current_price": 350, "price_change": 2.3, "forecast_price": 362,
                "model_confidence": 0.89, "total_markets": 12, "regions": 3,
                "total_predictions": stats.get("total_predictions", 0), "mode": "demo"}
    df = pd.DataFrame(dataset.data)
    price_col = next((c for c in df.columns if any(k in c.lower() for k in ["close", "c_maize", "price"])), None)
    current = float(df[price_col].iloc[-1]) if price_col else 350
    avg = float(df[price_col].mean()) if price_col else 350
    return {"current_price": round(current, 2), "price_change": round((current - avg) / avg * 100, 2),
            "forecast_price": round(current * 1.023, 2), "model_confidence": model.accuracy if model else 0.89,
            "total_markets": int(df["market"].nunique()) if "market" in df.columns else 12,
            "regions": int(df["region"].nunique()) if "region" in df.columns else 3,
            "total_predictions": stats.get("total_predictions", 0), "mode": "live" if model else "demo"}

@app.get("/api/farmer-dashboard")
def api_farmer_dashboard(db = Depends(get_db)):
    return api_dashboard(db=db)

@app.get("/api/regional-prices")
def api_regional_prices(db = Depends(get_db)):
    dataset = crud.get_active_dataset(db)
    if dataset is None:
        return {"regions": ["Northern", "Central", "Southern"], "prices": [345, 350, 360],
                "rec_scores": [75, 82, 68], "mode": "demo"}
    df = pd.DataFrame(dataset.data)
    price_col = next((c for c in df.columns if any(k in c.lower() for k in ["close", "c_maize", "price"])), None)
    if "region" in df.columns:
        grouped = df.groupby("region")[price_col].mean().round(2)
        prices = grouped.values.tolist()
        max_p = max(prices) if prices else 1
        return {"regions": grouped.index.tolist(), "prices": prices,
                "rec_scores": [round(p / max_p * 100, 1) for p in prices], "mode": "live"}
    return {"regions": ["Central"], "prices": [350], "rec_scores": [100], "mode": "live"}

@app.get("/api/forecast")
def api_forecast(region: str = "Central", months: int = 6):
    if PRE_TRAINED_MODEL is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    prices, labels = [], []
    today = datetime.now()
    for i in range(1, months + 1):
        fm = (today.month + i - 1) % 12 + 1
        fy = today.year + (today.month + i - 1) // 12
        inp = SarimaxInput(year=fy, month=fm, infl=8.5, inflation_maize=8.5,
                           trust_maize=0.9, c_rice=400, mkt="Lilongwe",
                           adm1="Central Region", adm2="Lilongwe")
        price = make_prediction(inp)
        prices.append(round(price, 2))
        labels.append(date(fy, fm, 1).strftime("%b %Y"))
    return {"region": region, "labels": labels, "prices": prices}

@app.get("/api/historical-trends")
def api_historical_trends(months: int = 12, region: str = None, db = Depends(get_db)):
    dataset = crud.get_active_dataset(db)
    if dataset is None:
        prices = [round(340 + 5 * np.sin(i * np.pi / 6) + np.random.normal(0, 3), 2) for i in range(months)]
        return {"labels": [f"Month {i+1}" for i in range(months)], "prices": prices, "mode": "demo"}
    df = pd.DataFrame(dataset.data)
    price_col = next((c for c in df.columns if any(k in c.lower() for k in ["close", "c_maize", "price"])), None)
    if not price_col:
        raise HTTPException(status_code=422, detail="No price column found.")
    if region and "region" in df.columns:
        df = df[df["region"].str.lower() == region.lower()]
    df_tail = df.tail(months)
    date_col = next((c for c in df.columns if "date" in c.lower()), None)
    labels = df_tail[date_col].astype(str).tolist() if date_col else [f"Period {i+1}" for i in range(len(df_tail))]
    return {"labels": labels, "prices": df_tail[price_col].round(2).tolist(),
            "region": region or "All", "mode": "live"}

@app.post("/api/predict")
def api_predict(inp: PredictionInput, db = Depends(get_db)):
    if PRE_TRAINED_MODEL:
        sarinp = SarimaxInput(year=inp.year, month=inp.month, infl=inp.inflation_rate,
                              inflation_maize=inp.maize_inflation, trust_maize=inp.trust_score,
                              c_rice=400, mkt=inp.market, adm1=inp.region, adm2="Lilongwe")
        price = make_prediction(sarinp)
        return {"input": inp.dict(), "predicted_price": price, "timestamp": datetime.now().isoformat()}
    return {"predicted_price": 350, "note": "demo"}

@app.post("/api/sarimax-predict")
def sarimax_predict(inp: SarimaxInput):
    """Main prediction endpoint used by the frontend."""
    try:
        price = make_prediction(inp)
        return {
            "predicted_price": round(price, 2),
            "input": inp.dict(),
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recommendations")
def api_recommendations(region: str = "Central", db = Depends(get_db)):
    inp = SarimaxInput(year=datetime.now().year, month=datetime.now().month,
                       infl=8.5, inflation_maize=8.5, trust_maize=0.9, c_rice=400,
                       mkt="Lilongwe", adm1=region, adm2="Lilongwe")
    curr = make_prediction(inp)
    next_inp = SarimaxInput(year=datetime.now().year, month=datetime.now().month % 12 + 1,
                            infl=8.5, inflation_maize=8.5, trust_maize=0.9, c_rice=400,
                            mkt="Lilongwe", adm1=region, adm2="Lilongwe")
    nxt = make_prediction(next_inp)
    chg = (nxt - curr) / curr * 100 if curr else 0
    if chg > 3: action, reason, color = "HOLD", f"Price rising {chg:.1f}% next month.", "keep"
    elif chg < -3: action, reason, color = "SELL", f"Price dropping {abs(chg):.1f}%.", "sell"
    else: action, reason, color = "PLANT", "Prices stable.", "plant"
    return {"region": region, "current_price": curr, "next_month_price": nxt,
            "price_change_pct": round(chg, 2), "recommendation": action, "reason": reason, "color": color}

@app.get("/api/features")
def api_features():
    if PRE_TRAINED_MODEL is None:
        return {"features": FEATURE_ORDER, "importances": [0]*len(FEATURE_ORDER), "mode": "demo"}
    importances = PRE_TRAINED_MODEL.feature_importances_.tolist()
    return {"features": FEATURE_ORDER, "importances": importances, "mode": "live"}

@app.get("/api/feature-importance")
def api_feature_importance():
    return api_features()

@app.get("/api/model-performance")
def api_model_performance():
    return {
        "rmse": 14.665,
        "mae": 9.163,
        "r2": 0.977,
        "mode": "live" if PRE_TRAINED_MODEL else "demo"
    }

# ─── DATASET / UPLOAD (kept for historical data) ─────────────────────────────
@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...), db = Depends(get_db)):
    fname = file.filename.lower()
    if not any(fname.endswith(ext) for ext in [".csv", ".xlsx", ".xls"]):
        raise HTTPException(400, "Only CSV/Excel supported.")
    contents = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(contents)) if fname.endswith(".csv") else pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(422, f"File error: {e}")
    price_col = next((c for c in df.columns if any(k in c.lower() for k in ["close","c_maize","price"])), None)
    dataset_record = crud.save_dataset(db, file.filename, len(df), len(df.columns),
                                       list(df.columns), df.fillna("").to_dict(orient="records"),
                                       avg_price=float(df[price_col].mean()) if price_col else None,
                                       min_price=float(df[price_col].min()) if price_col else None,
                                       max_price=float(df[price_col].max()) if price_col else None)
    return {"filename": file.filename, "dataset_id": dataset_record.id, "rows": len(df),
            "columns": list(df.columns), "preview": df.head(5).fillna("").to_dict(orient="records")}

@app.post("/api/train")
def api_train(db = Depends(get_db)):
    dataset = crud.get_active_dataset(db)
    if dataset is None:
        raise HTTPException(400, detail="No dataset uploaded.")
    return {"status": "skipped", "message": "Pre‑trained model is already in use."}

@app.get("/api/dataset/info")
def api_dataset_info(db = Depends(get_db)):
    dataset = crud.get_active_dataset(db)
    if dataset is None: return {"loaded": False}
    return {"loaded": True, "id": dataset.id, "filename": dataset.filename,
            "rows": dataset.rows, "columns": dataset.column_names}

@app.get("/api/predictions/history")
def api_prediction_history(region: str = None, limit: int = Query(default=50, le=500), offset: int = 0,
                           db = Depends(get_db)):
    rows = crud.get_prediction_history(db, region=region, limit=limit, offset=offset)
    stats = crud.get_prediction_stats(db)
    return {"stats": stats, "history": rows, "count": len(rows)}

@app.get("/api/models/history")
def api_models_history(db = Depends(get_db)):
    return {"models": crud.list_models(db)}

@app.get("/api/datasets/history")
def api_datasets_history(db = Depends(get_db)):
    return {"datasets": crud.list_datasets(db)}

# ─── ROOT REDIRECT & STATIC ──────────────────────────────────────────────────
@app.get("/")
def root():
    return RedirectResponse(url="/loginpage.html")

if os.path.exists("."):
    app.mount("/", StaticFiles(directory=".", html=True), name="static")
