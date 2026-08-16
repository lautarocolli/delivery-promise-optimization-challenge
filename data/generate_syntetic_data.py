"""
generate_synthetic_data.py

Genera órdenes sintéticas con correlaciones operativas.
Produce la tabla `orders` que consume la query SQL de construcción
del dataset. Los timestamps reflejan los eventos observables definidos en el challenge.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# PARÁMETROS GENERALES
N_ORDERS    = 10_000
N_MERCHANTS = 75
N_ZONES     = 8
DATE_START  = datetime(2026, 1, 1)
DATE_END    = datetime(2026, 3, 31)   # 90 días
CATEGORIES  = ['food', 'pharmacy', 'supermarket']

# MERCHANTS
# Cada merchant tiene zona, categoría y un factor de velocidad fijo.
# speed_factor < 1: más rápido que el promedio de su categoría
# speed_factor > 1: más lento

merchants = pd.DataFrame({
    'merchant_id':   [f'M{i:03d}' for i in range(N_MERCHANTS)],
    'zone_id':       np.random.choice([f'Z{i}' for i in range(1, N_ZONES + 1)], N_MERCHANTS),
    'category':      np.random.choice(CATEGORIES, N_MERCHANTS, p=[0.6, 0.2, 0.2]),
    'speed_factor':  np.random.lognormal(mean=0, sigma=0.3, size=N_MERCHANTS),
})

# ZONAS
# traffic_factor: multiplica los tiempos de movimiento (rider dispatch + delivery leg)
# Factor: a mayor factor, más denso el tráfico.

zones = pd.DataFrame({
    'zone_id':        [f'Z{i}' for i in range(1, N_ZONES + 1)],
    'traffic_factor': np.random.uniform(0.8, 1.4, N_ZONES),
})
zone_map = zones.set_index('zone_id')['traffic_factor'].to_dict()

# FUNCIONES DE SAMPLING

def sample_checkout_hour() -> int:
    """
    Distribución bimodal que concentra pedidos en horarios de almuerzo
    (12-14hs) y cena (19-23hs), con actividad moderada en el resto del día.
    """
    weights = np.zeros(24)
    weights[12:15] = 3.0   # pico almuerzo
    weights[19:24] = 4.0   # pico cena
    weights[10:12] = 1.5   # media mañana
    weights[15:19] = 1.0   # tarde
    weights /= weights.sum()
    return int(np.random.choice(24, p=weights))


def peak_multiplier(hour: int) -> float:
    """
    Multiplica los tiempos de preparación durante horas pico.
    Modela el efecto de mayor carga sobre la cocina del merchant.
    """
    if hour in range(12, 15) or hour in range(19, 24):
        return np.random.uniform(1.3, 1.6)
    return 1.0


def sample_prep_time(category: str, hour: int, speed_factor: float) -> float:
    """
    Tiempo de preparación del pedido (variable latente en producción).
    Distribución LogNormal: asegura valores positivos y cola derecha
    que modela órdenes ocasionalmente lentas.
    """
    base_params = {
        'food':        (np.log(20), 0.40),
        'pharmacy':    (np.log(8),  0.30),
        'supermarket': (np.log(12), 0.35),
    }
    mu, sigma = base_params[category]
    return np.random.lognormal(mu, sigma) * speed_factor * peak_multiplier(hour)


def sample_rider_dispatch(traffic_factor: float, hour: int) -> float:
    """
    Tiempo desde notificación al rider hasta su llegada al comercio.
    Depende del tráfico de la zona y la hora (más riders ocupados en pico).
    """
    base = np.random.lognormal(np.log(10), 0.4)
    hour_factor = 1.2 if hour in range(12, 15) or hour in range(19, 24) else 1.0
    return base * traffic_factor * hour_factor


def sample_delivery_leg(distance_km: float, traffic_factor: float, hour: int) -> float:
    """
    Tiempo desde retiro en el comercio hasta la entrega al cliente.
    Componente lineal de distancia + ruido + efecto de tráfico.
    """
    base = distance_km * 2.5 + np.random.lognormal(np.log(2), 0.3) # 2.5 min/km + ruido
    hour_factor = 1.2 if hour in range(12, 15) or hour in range(19, 24) else 1.0
    return base * traffic_factor * hour_factor


# GENERACIÓN

rows = []
total_days = (DATE_END - DATE_START).days

for i in range(N_ORDERS):

    merchant   = merchants.sample(1).iloc[0]
    traffic    = zone_map[merchant.zone_id]

    # Timestamp del checkout
    day        = np.random.randint(0, total_days)
    hour       = sample_checkout_hour()
    minute     = np.random.randint(0, 60)
    checkout   = DATE_START + timedelta(days=day, hours=hour, minutes=minute)

    distance_km = round(np.random.uniform(0.5, 8.0), 2)

    # Tiempos en minutos desde el checkout
    t_merchant_notif = np.random.uniform(0.5, 2.0)   # asumo notificación casi inmediata antes
                                                     # de solución de notificación.
    t_rider_notif    = np.random.uniform(1.0, 3.0)   # asignación del rider

    # Variable latente: cuándo está listo el pedido
    prep_time        = sample_prep_time(merchant.category, hour, merchant.speed_factor)
    t_pedido_listo   = t_merchant_notif + prep_time   # NO observable

    # Rider en camino al comercio
    rider_dispatch   = sample_rider_dispatch(traffic, hour)
    t_rider_arrival  = t_rider_notif + rider_dispatch

    # El retiro ocurre cuando ambos están listos
    t_out_for_delivery = max(t_rider_arrival, t_pedido_listo)

    # Entrega final
    delivery_leg  = sample_delivery_leg(distance_km, traffic, hour)
    t_delivery    = t_out_for_delivery + delivery_leg

    def to_ts(delta_min: float) -> datetime:
        return checkout + timedelta(minutes=delta_min)

    rows.append({
        'order_id':                   f'ORD{i:06d}',
        'merchant_id':                merchant.merchant_id,
        'zone_id':                    merchant.zone_id,
        'checkout_timestamp':         checkout,
        'order_category':             merchant.category,
        'distance_km':                distance_km,
        'merchant_notification_time': to_ts(t_merchant_notif),
        'rider_notification_time':    to_ts(t_rider_notif),
        'rider_arrival_time':         to_ts(t_rider_arrival),
        'out_for_delivery_time':      to_ts(t_out_for_delivery),
        'delivery_timestamp':         to_ts(t_delivery),
    })

orders = pd.DataFrame(rows)

# VALIDACIÓN BÁSICA
total_min = (orders.delivery_timestamp - orders.checkout_timestamp).dt.total_seconds() / 60
rider_wait = (orders.out_for_delivery_time - orders.rider_arrival_time).dt.total_seconds() / 60

print(f"Órdenes generadas:            {len(orders):,}")
print(f"Tiempo entrega promedio:      {total_min.mean():.1f} min")
print(f"Tiempo entrega mediana:       {total_min.median():.1f} min")
print(f"% con rider esperando:        {(rider_wait > 2).mean() * 100:.1f}%")
print(f"Merchants únicos:             {orders.merchant_id.nunique()}")
print(f"Distribución de categorías:")
print(orders.order_category.value_counts())

# GUARDAR 
orders.to_csv('orders.csv', index=False)
print("\nGuardado en data/orders.csv")