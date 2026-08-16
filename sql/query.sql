-- ============================================================
-- CONSTRUCCIÓN DEL DATASET PARA EL MODELO DE PROMESA DE ENTREGA
--
-- Supuestos:
--   - La tabla fuente `orders` tiene una fila por orden completada
--     con los timestamps de cada evento observable.
--   - Determino que el rider esperó sólo si pasaron 2 minutos desde que llegó
--     y todavía no sale con la órden, esto para asegurar que estamos en el
--     escenario en que el pedido no está listo cuando el rider llega (configurable).
--   - Merchants sin historia usan promedio de su categoría como fallback.
-- ============================================================

WITH

-- 1. SEGMENTOS BASE
-- Calcula los segmentos de tiempo observables y la variable target por orden.

order_segments AS (
  SELECT
    order_id,
    merchant_id,
    zone_id,
    checkout_timestamp,
    order_category,
    distance_km,

    -- VARIABLES CALCULADAS:

    -- Target: minutos totales desde el checkout hasta la entrega
    TIMESTAMP_DIFF(delivery_timestamp, checkout_timestamp, MINUTE)
      AS total_delivery_minutes,

    -- Tiempo de espera del repartidor (pickup - llegada al comercio).
    -- Si > 2 min: el repartidor esperó -> obs_prep_min es confiable.
    -- Si <= 2:    el pedido estaba listo antes -> obs_prep_min sobreestima.
    TIMESTAMP_DIFF(out_for_delivery_time, rider_arrival_time, MINUTE)
      AS rider_wait_min,

    -- Proxy del tiempo de preparación (confiable solo cuando rider esperó)
    TIMESTAMP_DIFF(out_for_delivery_time, merchant_notification_time, MINUTE)
      AS obs_prep_min,

    -- Tiempo de despacho del repartidor: notificación -> llegada al comercio
    TIMESTAMP_DIFF(rider_arrival_time, rider_notification_time, MINUTE)
      AS rider_dispatch_min,

    -- Tramo de entrega: retiro -> entrega final al cliente
    TIMESTAMP_DIFF(delivery_timestamp, out_for_delivery_time, MINUTE)
      AS delivery_leg_min,

    EXTRACT(HOUR FROM checkout_timestamp)      AS checkout_hour,
    EXTRACT(DAYOFWEEK FROM checkout_timestamp) AS day_of_week

  FROM orders
  WHERE delivery_timestamp IS NOT NULL              -- solo órdenes completadas
    AND TIMESTAMP_DIFF(delivery_timestamp, checkout_timestamp, MINUTE)
        BETWEEN 5 AND 180                           
),

-- 2. ESTADÍSTICOS HISTÓRICOS DE TIEMPO DE PREPARACIÓN POR MERCHANT
-- Agrega métricas por merchant agrupando por órden.

merchant_prep_time_stats AS (
  SELECT
    a.order_id,
    AVG(CASE WHEN b.rider_wait_min > 2 THEN b.obs_prep_min END)
      AS merchant_avg_prep_min,
    STDDEV(CASE WHEN b.rider_wait_min > 2 THEN b.obs_prep_min END)
      AS merchant_std_prep_min,
    -- Proporción de órdenes donde el rider esperó al pedido:
    -- alta = proxy preciso; baja = prep time mayormente no observable
    AVG(CASE WHEN b.rider_wait_min > 2 THEN 1.0 ELSE 0.0 END)
      AS merchant_pct_rider_waits
  FROM order_segments a
  JOIN order_segments b -- Uso un self join para comparar 'orden actual' vs historia
    ON  a.merchant_id = b.merchant_id
    AND b.checkout_timestamp <  a.checkout_timestamp
    -- En principio uso 30 días para los estadísticos
    AND b.checkout_timestamp >= TIMESTAMP_SUB(a.checkout_timestamp, INTERVAL 30 DAY)
  GROUP BY a.order_id
),

-- 3. ESTADÍSTICOS HISTÓRICOS POR ZONA Y HORA
-- Agrega tiempos de entrega y despacho por zona y hora del día.

zone_stats AS (
  SELECT
    a.order_id,
    AVG(b.delivery_leg_min)   AS zone_avg_delivery_min,
    AVG(b.rider_dispatch_min) AS zone_avg_rider_dispatch_min
  FROM order_segments a
  JOIN order_segments b
    ON  a.zone_id = b.zone_id -- misma zona
    AND EXTRACT(HOUR FROM b.checkout_timestamp) = a.checkout_hour
    AND b.checkout_timestamp <  a.checkout_timestamp
    AND b.checkout_timestamp >= TIMESTAMP_SUB(a.checkout_timestamp, INTERVAL 30 DAY)
  GROUP BY a.order_id
),

-- 4. PROXY DE DEMANDA DEL MERCHANT EN TIEMPO REAL
-- Cantidad de órdenes del mismo merchant ongoing cuando llegó el pedido.
-- Captura la carga operativa del merchant al momento del pedido.

demand_stats AS (
  SELECT
    a.order_id,
    COUNT(b.order_id) AS merchant_orders_ongoing
  FROM order_segments a
  JOIN order_segments b
    ON  b.merchant_id = a.merchant_id
    AND b.checkout_timestamp <= a.checkout_timestamp        -- que la órden haya llegado antes de la actual
    AND (
      b.out_for_delivery_time IS NULL                       -- aún no retirado
      OR b.out_for_delivery_time > a.checkout_timestamp     -- se retira después de la actual
    )
    AND b.order_id != a.order_id
  GROUP BY a.order_id
),

-- 5. FALLBACKS POR CATEGORÍA
-- Para merchants sin historial, uso el promedio
-- de todos los merchants de la misma categoría.

category_defaults AS (
  SELECT
    order_category,
    AVG(CASE WHEN rider_wait_min > 2 THEN obs_prep_min END) AS avg_prep_min,
    STDDEV(CASE WHEN rider_wait_min > 2 THEN obs_prep_min END) AS std_prep_min,
    AVG(CASE WHEN rider_wait_min > 2 THEN 1.0 ELSE 0.0 END) AS avg_pct_rider_waits,
    AVG(delivery_leg_min)   AS avg_delivery_min,
    AVG(rider_dispatch_min) AS avg_dispatch_min
  FROM order_segments
  GROUP BY order_category
)

-- DATASET FINAL
SELECT
  os.order_id,
  os.total_delivery_minutes,                               -- variable target (y)

  -- Features temporales
  os.checkout_hour,
  os.day_of_week,

  -- Features de la orden
  os.order_category,
  os.distance_km,

  -- Features del merchant
  COALESCE(ms.merchant_avg_prep_min,    cd.avg_prep_min)  AS merchant_avg_prep_min,
  COALESCE(ms.merchant_std_prep_min,    cd.std_prep_min)  AS merchant_std_prep_min,
  COALESCE(ms.merchant_pct_rider_waits, cd.avg_pct_rider_waits) AS merchant_pct_rider_waits,

  -- Features de zona
  COALESCE(zs.zone_avg_delivery_min,       cd.avg_delivery_min)  AS zone_avg_delivery_min,
  COALESCE(zs.zone_avg_rider_dispatch_min, cd.avg_dispatch_min)  AS zone_avg_rider_dispatch_min,

  -- Proxy de demanda
  COALESCE(ds.merchant_orders_ongoing, 0) AS merchant_orders_ongoing

FROM order_segments os
LEFT JOIN merchant_prep_time_stats  ms ON os.order_id       = ms.order_id
LEFT JOIN zone_stats                zs ON os.order_id       = zs.order_id
LEFT JOIN demand_stats              ds ON os.order_id       = ds.order_id
LEFT JOIN category_defaults         cd ON os.order_category = cd.order_category

ORDER BY os.checkout_timestamp