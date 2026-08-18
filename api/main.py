from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

MODEL_DIR = Path(__file__).parent.parent / "model"

xgb_high     = joblib.load(MODEL_DIR / "xgb_high.pkl")
xgb_low      = joblib.load(MODEL_DIR / "xgb_low.pkl")
feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")

app = FastAPI(title="Delivery Promise API", version="1.0.0")


class OrderFeatures(BaseModel):
    """Features disponibles en el momento del checkout."""
    checkout_timestamp:          datetime
    distance_km:                 float  = Field(..., gt=0) # Greater Than
    merchant_avg_prep_min:       float  = Field(..., ge=0) # Greater or Equal
    merchant_std_prep_min:       float  = Field(..., ge=0)
    zone_avg_delivery_min:       float  = Field(..., ge=0)
    zone_avg_rider_dispatch_min: float  = Field(..., ge=0)
    merchant_orders_ongoing:     int    = Field(..., ge=0)
    order_category:              str    
    day_of_week:                 int    = Field(..., ge=0, le=6) # Less or Equal

    @field_validator("order_category")
    @classmethod
    def validate_category(cls, v: str) -> str:
        valid = {"food", "pharmacy", "supermarket"}
        if v not in valid:
            raise ValueError(f"order_category debe ser uno de {valid}")
        return v


class DeliveryPromise(BaseModel):
    promise_start:          datetime
    promise_end:            datetime
    estimated_minutes_low:  float
    estimated_minutes_high: float
    notify_merchant_at:     datetime

def build_feature_vector(order: OrderFeatures) -> pd.DataFrame:
    """Construye el vector de features replicando el preprocesamiento de feature_engineering.ipynb."""
    hour = order.checkout_timestamp.hour

    row: dict = {
        "sin_hour":                    np.sin(2 * np.pi * hour / 24),
        "cos_hour":                    np.cos(2 * np.pi * hour / 24),
        "distance_km":                 order.distance_km,
        "merchant_avg_prep_min":       order.merchant_avg_prep_min,
        "merchant_std_prep_min":       order.merchant_std_prep_min,
        "zone_avg_delivery_min":       order.zone_avg_delivery_min,
        "zone_avg_rider_dispatch_min": order.zone_avg_rider_dispatch_min,
        "merchant_orders_ongoing":     order.merchant_orders_ongoing,
    }

    # OHE order_category drop_first=True elimina "food" (first)
    for cat in ["pharmacy", "supermarket"]:
        row[f"order_category_{cat}"] = int(order.order_category == cat)

    # OHE day_of_week drop_first=True elimina día 0 (lunes, first)
    for d in range(1, 7):
        row[f"day_of_week_{d}"] = int(order.day_of_week == d)

    return pd.DataFrame([row])[feature_cols]

# Endpoint
@app.post("/delivery-promise", response_model=DeliveryPromise)
def delivery_promise(order: OrderFeatures) -> DeliveryPromise:
    try:
        X = build_feature_vector(order)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error en feature engineering: {e}")
    
    pred_low  = float(xgb_low.predict(X)[0])
    pred_high = float(xgb_high.predict(X)[0])

    # Fuerzo que la ventana tenga siempre al menos 5 min
    if pred_high - pred_low < 5:
        pred_low = pred_high - 5

    # Notificar al merchant cuando el rider esté a merchant_avg_prep_min de llegar.
    # Si el prep time supera el dispatch time, notificar de inmediato.
    notify_delay_min = max(
        0.0,
        order.zone_avg_rider_dispatch_min - order.merchant_avg_prep_min
    )
    notify_merchant_at = order.checkout_timestamp + timedelta(minutes=notify_delay_min)

    return DeliveryPromise(
        promise_start          = order.checkout_timestamp + timedelta(minutes=pred_low),
        promise_end            = order.checkout_timestamp + timedelta(minutes=pred_high),
        estimated_minutes_low  = round(pred_low, 1),
        estimated_minutes_high = round(pred_high, 1),
        notify_merchant_at     = notify_merchant_at
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_features": len(feature_cols)}