# ============================================================
# REGRESIÓN
# Entradas:
#   1) artifacts_preprocesamiento.zip   (contiene las tablas procesadas)
# Salida:
#   resultados.zip (historial de entrenamiento, metadatos y modelo final)
# ============================================================

# ============================================================
# Diseñador de red para regresión basado en heurísticas
# ============================================================

import math
from dataclasses import dataclass
from typing import List, Union


@dataclass
class DisenoRedRegresion:
    capas: List[int]
    P: int
    rho: float
    l2: float
    dropouts: List[float]
    patience: int
    min_delta: float
    max_epochs: int


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def estimar_parametros(n0: int, capas: List[int]) -> int:
    """
    Cuenta parámetros de una red densa:
      - incluye sesgos en cada capa
      - incluye capa de salida (1 neurona)
    """
    if not capas:
        return 0

    P = (n0 + 1) * capas[0]
    for i in range(1, len(capas)):
        P += (capas[i - 1] + 1) * capas[i]
    P += (capas[-1] + 1) * 1
    return int(P)


def disenar_red_regresion(
    d: int,
    n0: int,
    *,
    k: int = 10,
    c1: float = 2.0,
    r: float = 0.5,
    n_min: int = 8,
    L_max: int = 4,
) -> Union[DisenoRedRegresion, str]:
    """
    Traducción directa del pseudocódigo:
      - d  : número de muestras
      - n0 : número de variables de entrada (features)
    Devuelve:
      - DisenoRedRegresion si es viable
      - str con mensaje si no es viable
    """

    # 0. Tope adaptativo de ancho (según tamaño de muestra)
    n_max = min(1024, max(64, math.floor(0.25 * d)))

    # 1. Verificación mínima de viabilidad (suave)
    if d < 2 * n0:
        return "Dataset muy pequeño: alto riesgo de sobreajuste"

    # 2. Presupuesto total de parámetros
    P_max = math.floor(k * d)

    # 3. Tamaño de la primera capa oculta (capado por presupuesto y por n_max)
    n1_cap_presupuesto = math.floor(P_max / (n0 + 1))
    n1 = min(math.floor(c1 * n0), n1_cap_presupuesto, n_max)

    if n1 < n_min:
        return "Presupuesto insuficiente: no se puede ni una capa >= n_min"

    capas = [int(n1)]

    # 4. Construcción iterativa de capas ocultas (embudo)
    while True:
        if len(capas) >= L_max:
            break

        n_prev = capas[-1]
        n_new = math.floor(r * n_prev)

        if n_new < n_min:
            break

        n_new = min(n_new, n_max)
        capas.append(int(n_new))

    # 5. Estimación de parámetros (incluye sesgos y salida)
    P = estimar_parametros(n0, capas)

    # 6. Validación de complejidad (recorte iterativo)
    while P > P_max:
        if len(capas) > 1:
            capas.pop()
        else:
            n_old = capas[0]
            capas[0] = math.floor(0.9 * capas[0])  # reducción suave
            if capas[0] >= n_old:
                capas[0] = n_old - 1               # garantiza progreso
            if capas[0] < n_min:
                return "Presupuesto insuficiente: no cabe una capa >= n_min"

        P = estimar_parametros(n0, capas)

    # ========================================================
    # 7. HIPERPARÁMETROS DE REGULARIZACIÓN (L2, Dropout, ES)
    # ========================================================

    # 7.1 Ocupación del presupuesto
    rho = P / P_max if P_max > 0 else 1.0

    # -------- Dropout base por tamaño de muestra --------
    if d < 2000:
        drop_base = 0.35
    elif d < 20000:
        drop_base = 0.25
    else:
        drop_base = 0.15

    # -------- Ajuste por ocupación rho --------
    if rho >= 0.8:
        drop = drop_base + 0.10
    elif rho >= 0.4:
        drop = drop_base
    else:
        drop = drop_base - 0.10
    drop = clip(drop, 0.05, 0.50)

    # Dropout por capa (más alto al inicio)
    dropouts: List[float] = []
    for i in range(1, len(capas) + 1):
        di = drop * (1.0 - 0.15 * (i - 1))
        di = clip(di, 0.05, 0.50)
        dropouts.append(float(di))

    # -------- L2 base por tamaño de muestra --------
    if d < 2000:
        l2_base = 1e-3
    elif d < 20000:
        l2_base = 3e-4
    else:
        l2_base = 1e-4

    # -------- Ajuste por ocupación rho --------
    if rho >= 0.8:
        l2 = 3.0 * l2_base
    elif rho >= 0.4:
        l2 = 1.0 * l2_base
    else:
        l2 = 0.3 * l2_base
    l2 = clip(l2, 1e-6, 3e-3)

    # -------- Early stopping (patience) --------
    if d < 2000:
        patience = 20
        max_epochs = 400
    elif d < 20000:
        patience = 15
        max_epochs = 200
    else:
        patience = 10
        max_epochs = 100

    # (opcional) min_delta fijo simple
    min_delta = 1e-4

    return DisenoRedRegresion(
        capas=capas,
        P=int(P),
        rho=float(rho),
        l2=float(l2),
        dropouts=dropouts,
        patience=int(patience),
        min_delta=float(min_delta),
        max_epochs=int(max_epochs),
    )
     