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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class NNDesing:
    layers: List[int]
    P: int
    rho: float
    l2: float
    dropouts: List[float]
    patience: int
    min_delta: float
    max_epochs: int
    
######################
# funciones en común #
######################

def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parameters_estimator(n0: int, layers: List[int]) -> int:
    """
    Cuenta parámetros de una red densa:
      - incluye sesgos en cada capa
      - incluye capa de salida (1 neurona)
    """
    if not layers:
        return 0

    P = (n0 + 1) * layers[0]
    for i in range(1, len(layers)):
        P += (layers[i - 1] + 1) * layers[i]
    P += (layers[-1] + 1) * 1
    return int(P)


#######################
# template + strategy #
####################### 

class NetworkDesigner(ABC): 
    """
    Template medthod to create deep layers
    """
    
    def __init__(
        self, 
        k: int = 10, 
        c1: float = 2.0,
        r: float = 0.5,
        n_min: int = 8,
        L_max: int = 4):
        
        self.k = k
        self.c1 = c1 
        self.r  = r 
        self.n_min = n_min 
        self.L_max = L_max 
        
    # ---------------- Template Method ---------------- # 
    
    def desing(self, d:int, n0:int) -> NNDesing: 
        
        self._dataset_validation(d, n0)
        
        n_max  = min(1024, max(64, math.floor(0.25*d)))
        P_max = math.floor(self.k*d)
        
        layers = self._layers_building(n0, n_max, P_max)
        P = self._fitting_complexity(layers, n0, P_max)
        
        rho =  P/P_max if P_max > 0 else 1.0 
        dropouts = self._dropouts_estimator(d, layers, rho)
        l2 = self._l2_computing(d, rho)
        
        patience, max_epochs = self._early_stopping(d)
        
        return NNDesing(
            layers = layers,
            P=int(P),
            rho=float(rho),
            l2=float(l2),
            dropouts=dropouts,
            patience=int(patience),
            min_delta=1e-4, # optional
            max_epochs=int(max_epochs),
        )

    def _layers_building(self, n0:int, n_max:int, P_max:int) -> List[int]: 
        n1 = min(
            math.floor(self.c1 * n0),
            math.floor(P_max / (n0 + 1)),
            n_max,
        )

        if n1 < self.n_min:
            raise ValueError("Presupuesto insuficiente")
        
        layers: List[int] = [int(n1)]
        
        while True: 
            if len(layers) >= self.L_max:
                break
            n_new = math.floor(self.r * layers[-1])
            if n_new < self.n_min:
                break
            
            layers.append(min(n_new, n_max))
        
        return layers    
    
    def _fitting_complexity(self, layers: List[int], n0:int, P_max:int) -> int: 
        
        P = parameters_estimator(n0 = n0, layers = layers)
        
        while P> P_max: 
            if len(layers) > 1: 
                layers.pop()
            else: 
                n_old = layers[0]
                layers[0] = math.floor(0.9 * layers[0])
                if layers[0]>= n_old: 
                    layers[0] = n_old-1
                if layers[0] < self.n_min: 
                    raise ValueError("Not allowed layer >= n_min")
                
            P = parameters_estimator(n0 = n0, layers = layers)
        
        return P 
    
    def _dropouts_estimator(self, d:int, layers: List[int], rho:float) -> List[float]: 
        
        if d < 2000:
            drop_base = 0.35
        elif d < 20000:
            drop_base = 0.25
        else:
            drop_base = 0.15

        if rho >= 0.8:
            drop = drop_base + 0.10
        elif rho >= 0.4:
            drop = drop_base
        else:
            drop = drop_base - 0.10

        drop = clip(drop, 0.05, 0.50)

        return [
            clip(drop * (1 - 0.15 * i), 0.05, 0.50)
            for i in range(len(layers))
        ]
    
    def _l2_computing(self, d:int, rho:float) -> float: 
        if d < 2000:
            l2_base = 1e-3
        elif d < 20000:
            l2_base = 3e-4
        else:
            l2_base = 1e-4

        if rho >= 0.8:
            return clip(3 * l2_base, 1e-6, 3e-3)
        if rho >= 0.4:
            return clip(l2_base, 1e-6, 3e-3)
        
        return clip(0.3 * l2_base, 1e-6, 3e-3)
        
        
    def _early_stopping(self, d:int) -> Tuple[int,int]: 
        if d < 2000:
            return (20, 400)
        elif d < 20000:
            return (15, 200)
        return (10, 100)
        
    
    # ---------------- Strategy Methods ---------------- #
    
    @abstractmethod
    def _dataset_validation(self, d:int, n0:int) -> None: 
        pass
    
    

# Strategies

class RegressionDesigner(NetworkDesigner):     
   
    def desing(self, d: int, n0: int) -> NNDesing:
        return super().desing(d, n0)
    
    def _dataset_validation(self, d:int, n0: int) -> None: 
        if d<2*n0: 
            raise ValueError("Small dataset for regression task")
    
    

class BinaryClassificationDesigner(NetworkDesigner):
     
    def desing(self, d: int, n0: int) -> NNDesing:
        return super().desing(d, n0)
   
    def _dataset_validation(self, d: int, n0: int) -> None:
        if d<2*n0: 
            raise ValueError("Small dataset for binary classification task")
        

    
    
    
    
    
    
    
    
    
    
    
    
    
    
### ESTO SERA SUSTITUIDO POR UNA MEJOR ESTRATEGIA

def disenar_red_regresion(
    d: int,
    n0: int,
    *,
    k: int = 10,
    c1: float = 2.0,
    r: float = 0.5,
    n_min: int = 8,
    L_max: int = 4,
) -> NNDesing:
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
        raise ValueError("Dataset muy pequeño: alto riesgo de sobreajuste")

    # 2. Presupuesto total de parámetros
    P_max = math.floor(k * d)

    # 3. Tamaño de la primera capa oculta (capado por presupuesto y por n_max)
    n1_cap_presupuesto = math.floor(P_max / (n0 + 1))
    n1 = min(math.floor(c1 * n0), n1_cap_presupuesto, n_max)

    if n1 < n_min:
        raise ValueError("Presupuesto insuficiente: no se puede ni una capa >= n_min")

    layers = [int(n1)]

    # 4. Construcción iterativa de layers ocultas (embudo)
    while True:
        if len(layers) >= L_max:
            break

        n_prev = layers[-1]
        n_new = math.floor(r * n_prev)

        if n_new < n_min:
            break

        n_new = min(n_new, n_max)
        layers.append(int(n_new))

    # 5. Estimación de parámetros (incluye sesgos y salida)
    P = parameters_estimator(n0, layers)

    # 6. Validación de complejidad (recorte iterativo)
    while P > P_max:
        if len(layers) > 1:
            layers.pop()
        else:
            n_old = layers[0]
            layers[0] = math.floor(0.9 * layers[0])  # reducción suave
            if layers[0] >= n_old:
                layers[0] = n_old - 1               # garantiza progreso
            if layers[0] < n_min:
                raise ValueError("Presupuesto insuficiente: no cabe una capa >= n_min")

        P = parameters_estimator(n0, layers)

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
    for i in range(1, len(layers) + 1):
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

    return NNDesing(
        layers=layers,
        P=int(P),
        rho=float(rho),
        l2=float(l2),
        dropouts=dropouts,
        patience=int(patience),
        min_delta=float(min_delta),
        max_epochs=int(max_epochs),
    )
     
     
def disenar_red_binaria(
    d: int,
    n0: int,
    *,
    k: int = 10,
    c1: float = 2.0,
    r: float = 0.5,
    n_min: int = 8,
    L_max: int = 4,
) -> NNDesing:
    """
    Heurística análoga a la de regresión:
      - d  : número de muestras (train)
      - n0 : número de variables de entrada
    Devuelve:
      - DisenoRedBinaria
    """

    # 0. Tope adaptativo de ancho (según tamaño de muestra)
    n_max = min(1024, max(64, math.floor(0.25 * d)))

    # 1. Verificación mínima de viabilidad (suave)
    if d < 2 * n0:
        raise ValueError("Dataset muy pequeño: alto riesgo de sobreajuste")

    # 2. Presupuesto total de parámetros
    P_max = math.floor(k * d)

    # 3. Tamaño de la primera capa oculta (capado por presupuesto y por n_max)
    n1_cap_presupuesto = math.floor(P_max / (n0 + 1))
    n1 = min(math.floor(c1 * n0), n1_cap_presupuesto, n_max)

    if n1 < n_min:
        raise ValueError("Presupuesto insuficiente: no se puede ni una capa >= n_min")

    layers = [int(n1)]

    # 4. Construcción iterativa de layers ocultas (embudo)
    while True:
        if len(layers) >= L_max:
            break

        n_prev = layers[-1]
        n_new = math.floor(r * n_prev)

        if n_new < n_min:
            break

        n_new = min(n_new, n_max)
        layers.append(int(n_new))

    # 5. Estimación de parámetros (incluye sesgos y salida)
    P = parameters_estimator(n0, layers)

    # 6. Validación de complejidad (recorte iterativo)
    while P > P_max:
        if len(layers) > 1:
            layers.pop()
        else:
            n_old = layers[0]
            layers[0] = math.floor(0.9 * layers[0])  # reducción suave
            if layers[0] >= n_old:
                layers[0] = n_old - 1               # garantiza progreso
            if layers[0] < n_min:
                raise ValueError("Presupuesto insuficiente: no cabe una capa >= n_min")

        P = parameters_estimator(n0, layers)

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
    for i in range(1, len(layers) + 1):
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

    return NNDesing(
        layers=layers,
        P=int(P),
        rho=float(rho),
        l2=float(l2),
        dropouts=dropouts,
        patience=int(patience),
        min_delta=float(min_delta),
        max_epochs=int(max_epochs),
    )