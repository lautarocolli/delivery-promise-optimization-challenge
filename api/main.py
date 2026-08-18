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


class OrderRequest(BaseModel):
    """Lo que el cliente conoce al momento del checkout."""
    order_id:           str
    merchant_id:        str
    zone_id:            str
    checkout_timestamp: datetime
    order_category:     str
    distance_km:        float = Field(..., gt=0)

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


# ---------------------------------------------------------------------------
# SQL de referencia: lo que esta función ejecutaría en producción
# ---------------------------------------------------------------------------
_FEATURE_QUERY = """
-- Ejecutar en BigQuery con parámetros:
--   @merchant_id, @zone_id, @order_category, @checkout_timestamp

WITH merchant_stats AS (
    SELECT
        AVG(obs_prep_min)  AS merchant_avg_prep_min,
        STDDEV(obs_prep_min) AS merchant_std_prep_min
    FROM orders
    WHERE merchant_id        = @merchant_id
      AND rider_wait_min     > 2                       
      AND checkout_timestamp < @checkout_timestamp     
      AND checkout_timestamp >= TIMESTAMP_SUB(@checkout_timestamp, INTERVAL 30 DAY)
),
zone_stats AS (
    SELECT
        AVG(total_delivery_min)     AS zone_avg_delivery_min,
        AVG(rider_dispatch_min)     AS zone_avg_rider_dispatch_min
    FROM orders
    WHERE zone_id            = @zone_id
      AND checkout_timestamp < @checkout_timestamp
      AND checkout_timestamp >= TIMESTAMP_SUB(@checkout_timestamp, INTERVAL 30 DAY)
      AND EXTRACT(HOUR FROM checkout_timestamp) = EXTRACT(HOUR FROM @checkout_timestamp)
),
concurrent AS (
    SELECT COUNT(*) AS merchant_orders_ongoing
    FROM orders
    WHERE merchant_id        = @merchant_id
      AND checkout_timestamp <= @checkout_timestamp
      AND delivered_timestamp > @checkout_timestamp
)
SELECT
    COALESCE(m.merchant_avg_prep_min,  cat.fallback_avg)  AS merchant_avg_prep_min,
    COALESCE(m.merchant_std_prep_min,  cat.fallback_std)  AS merchant_std_prep_min,
    z.zone_avg_delivery_min,
    z.zone_avg_rider_dispatch_min,
    c.merchant_orders_ongoing
FROM merchant_stats  m
CROSS JOIN zone_stats     z
CROSS JOIN concurrent     c
-- fallback cold-start: promedio de la categoría si el merchant no tiene historial
LEFT JOIN (
    SELECT
        AVG(obs_prep_min)    AS fallback_avg,
        STDDEV(obs_prep_min) AS fallback_std
    FROM orders
    WHERE order_category     = @order_category
      AND rider_wait_min     > 2
      AND checkout_timestamp < @checkout_timestamp
) cat ON m.merchant_avg_prep_min IS NULL
"""


def _compute_features(req: OrderRequest) -> dict:
    """
    TODO: reemplazar por una query real a BigQuery.

    En producción ejecutar _FEATURE_QUERY con los parámetros de la request
    y devolver el primer (y único) row como dict.

    Por ahora devuelve valores dummy para que el endpoint sea testeable
    sin conexión a base de datos.
    """
    # --- PLACEHOLDER: borrar y reemplazar por resultado de la query ---
    return {
        "merchant_avg_prep_min":       20.0,
        "merchant_std_prep_min":        5.0,
        "zone_avg_delivery_min":       35.0,
        "zone_avg_rider_dispatch_min": 12.0,
        "merchant_orders_ongoing":      3,
        "day_of_week":                 req.checkout_timestamp.weekday(),
    }
    # ------------------------------------------------------------------


def build_feature_vector(req: OrderRequest, computed: dict) -> pd.DataFrame:
    hour = req.checkout_timestamp.hour
    row: dict = {
        "sin_hour":                    np.sin(2 * np.pi * hour / 24),
        "cos_hour":                    np.cos(2 * np.pi * hour / 24),
        "distance_km":                 req.distance_km,
        "merchant_avg_prep_min":       computed["merchant_avg_prep_min"],
        "merchant_std_prep_min":       computed["merchant_std_prep_min"],
        "zone_avg_delivery_min":       computed["zone_avg_delivery_min"],
        "zone_avg_rider_dispatch_min": computed["zone_avg_rider_dispatch_min"],
        "merchant_orders_ongoing":     computed["merchant_orders_ongoing"],
    }
    for cat in ["pharmacy", "supermarket"]:
        row[f"order_category_{cat}"] = int(req.order_category == cat)
    for d in range(1, 7):
        row[f"day_of_week_{d}"] = int(computed["day_of_week"] == d)
    return pd.DataFrame([row])[feature_cols]


@app.post("/delivery-promise", response_model=DeliveryPromise)
def delivery_promise(req: OrderRequest) -> DeliveryPromise:
    try:
        computed = _compute_features(req)
        X        = build_feature_vector(req, computed)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Error en feature engineering: {e}")

    pred_low  = float(xgb_low.predict(X)[0])
    pred_high = float(xgb_high.predict(X)[0])

    if pred_high - pred_low < 5:
        pred_low = pred_high - 5

    notify_delay_min = max(
        0.0,
        computed["zone_avg_rider_dispatch_min"] - computed["merchant_avg_prep_min"]
    )
    checkout = req.checkout_timestamp

    return DeliveryPromise(
        promise_start          = checkout + timedelta(minutes=pred_low),
        promise_end            = checkout + timedelta(minutes=pred_high),
        estimated_minutes_low  = round(pred_low, 1),
        estimated_minutes_high = round(pred_high, 1),
        notify_merchant_at     = checkout + timedelta(minutes=notify_delay_min),
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_features": len(feature_cols)}