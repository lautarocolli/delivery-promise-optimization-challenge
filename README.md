# Delivery Promise Optimization Challenge

## Problema

En el checkout, el producto Proximity de MercadoLibre muestra al usuario un intervalo
de entrega estimado (ej: *Tu pedido llega entre las 20:15 y las 20:45*). Este intervalo debe ser:

- **Confiable**: tasa de entregas tardías <= 3% (órdenes que llegan después del `promise_end`)
- **Útil**: el intervalo más angosto posible sin violar la restricción de late rate y controlando el early rate (utilizamos target de early rate de ~15% a lo largo del proyecto).

El objetivo es predecir este intervalo usando features disponibles al momento del checkout.

---

## Solución

Dos modelos independientes de regresión cuantílica con XGBoost:

| Modelo | Cuantil | Rol |
|--------|---------|-----|
| `xgb_low` | TAU = 0.15 | Cota inferior (`promise_start`) |
| `xgb_high` | TAU = 0.975 | Cota superior (`promise_end`) |

La variable target es `total_delivery_min`: minutos transcurridos desde el checkout
hasta que el pedido llega al cliente.

El endpoint también calcula `notify_merchant_at`: el momento para notificar al merchant
de forma que el pedido esté listo al llegar el rider, y no antes. La lógica es
`checkout + max(0, zone_avg_rider_dispatch_min - merchant_avg_prep_min)`.
Si el prep time supera el dispatch time, se notifica de inmediato.

---

## Decisiones de diseño

### 1. Variable latente: `t_pedido_listo` no es observable

El momento en que el pedido está listo nunca se registra. Lo que se registra es
`out_for_delivery_time = max(t_rider_arrival, t_pedido_listo)`, por lo que el
prep time observado `obs_prep_min = out_for_delivery - merchant_notif` sobreestima
el prep time real cuando el rider llegó antes de que el pedido estuviera listo (Escenario A).

**Solución**: usar `obs_prep_min` solo en órdenes donde `rider_wait_min > 2 min`
(Escenario B), donde el pedido estaba listo antes de que llegara el rider y
`obs_prep_min ~= prep time real`. Los merchants sin historial en Escenario B
usan el promedio de su categoría como fallback.

### 2. TAU_HIGH = 0.975, no 0.97

Todos los modelos entrenados en TAU = 0.97 produjeron ~3.5-3.7% de late rate en
validación independientemente del learning rate o la profundidad — un gap de
calibración consistente de ~0.6 puntos percentilares. Entrenar en TAU = 0.975
compensa este gap y alcanza 3.0% en validación y 2.7% en el test set.

### 3. `merchant_pct_rider_waits` eliminado

Incluido inicialmente como proxy de variabilidad del merchant. El análisis SHAP
lo rankeó tercero en importancia.
Reentrenar sin él produjo resultados equivalentes o mejores: late rate 2.9% vs 3.0%,
ventana +1 min. El umbral de 2 min para definir "el rider esperó" es arbitrario;
eliminar el feature simplifica el modelo sin costo en performance.

### 4. Split temporal train/val/test

| Split | Período |
|-------|---------|
| Train | 1 ene - 28 feb |
| Val   | 1 mar - 15 mar |
| Test  | 16 mar - 31 mar |

Se evitó el split aleatorio para prevenir data leakage: los agregados históricos
de cada orden se computan solo con órdenes anteriores a ella.

---

## Resultados

| Modelo | Late Rate | Ventana prom. | Split |
|--------|-----------|---------------|-------|
| Baseline lineal (TAU = 0.97) | 3.1% | 53.1 min | Val |
| XGBoost (TAU = 0.97, sin tunear) | 3.7% | 33.4 min | Val |
| **XGBoost final (TAU = 0.975)** | **2.7%** | **36.4 min** | **Test** |

---

## Estructura del proyecto

```
├── sql/
│   └── build_dataset.sql          # Query de extracción de features (ventana 30 días)
├── data/
│   └── generate_synthetic_data.py
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb          # Hyperparameter tuning, SHAP, evaluación en test
├── model/
│   ├── xgb_high.pkl
│   ├── xgb_low.pkl
│   └── feature_cols.pkl
├── api/
│   └── main.py                    # FastAPI - POST /delivery-promise
├── Dockerfile
├── requirements.txt               # Dependencias de runtime de la API
└── requirements-dev.txt           # + notebooks, SHAP, visualización
```

---

## API

```bash
docker build -t delivery-promise .
docker run -p 8000:8000 delivery-promise
```

### Diseño del endpoint

El cliente envía únicamente los datos disponibles en el momento del checkout.
Los agregados históricos (`merchant_avg_prep_min`, `zone_avg_delivery_min`, etc.)
son calculados internamente por la API mediante una query al historial de órdenes,
replicando la misma lógica del SQL de feature engineering.

En producción esta query apuntaría a BigQuery o a un feature store actualizado
diariamente; en la implementación actual se reemplaza por un placeholder con
valores dummy para permitir el testing sin base de datos.

**POST** `/delivery-promise`

```json
{
    "order_id":           "ORD-001",
    "merchant_id":        "M-42",
    "zone_id":            "Z-3",
    "checkout_timestamp": "2026-03-20T20:00:00",
    "order_category":     "food",
    "distance_km":        3.5
}
```

```json
{
    "promise_start":          "2026-03-20T20:31:53.920097",
    "promise_end":            "2026-03-20T21:00:57.166443",
    "estimated_minutes_low":  31.9,
    "estimated_minutes_high": 61.0,
    "notify_merchant_at":     "2026-03-20T20:00:00"
}
```

**GET** `/health` — devuelve estado del modelo y cantidad de features.

Documentación interactiva: `http://localhost:8000/docs`

---

## Limitaciones conocidas y trabajo futuro

- **Feature store**: en producción, los agregados históricos deben precomputarse en
  un feature store actualizado diariamente. La recomputación en tiempo real por orden
  no es viable a escala.
- **Cap de ventana máxima**: si negocio define un ancho máximo de intervalo (ej. 30 min),
  aplicar `promise_start = promise_end - MAX_WINDOW`. Con el modelo actual, el 2% de
  las órdenes supera los 60 min.
- **Cold-start de merchants**: usan el promedio de categoría como fallback para
  `merchant_avg_prep_min` y `merchant_std_prep_min`.
- **Cold-start de zonas**: sin historial de zona, se usa el promedio global. Requeriría
  imputación por zona vecina en producción.

---

## Asistencia de IA

Esta solución fue desarrollada con el apoyo de **Claude**, utilizado como
colaborador técnico a lo largo del proyecto. Claude contribuyó en el diseño del SQL,
las decisiones de feature engineering, la generación de los datos sintéticos y la
sintaxis de la implementación.

Todas las decisiones fueron validadas empíricamente, adaptadas y comprendidas antes
de ser adoptadas. El autor del challenge conserva pleno entendimiento y responsabilidad
sobre cada decisión de diseño documentada.