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
#  Common functions  #
######################

def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def parameters_estimator(n0: int, layers: List[int], K:int) -> int:
    
    if not layers:
        return 0

    P = (n0 + 1) * layers[0]
    for i in range(1, len(layers)):
        P += (layers[i - 1] + 1) * layers[i]
        
    P += (layers[-1] + 1) * K
    return int(P)


#######################
# template + strategy #
####################### 

class NetworkDesigner(ABC): 
   
    def __init__(self,  k: int = 10,  c1: float = 2.0, r: float = 0.5, n_min: int = 8, L_max: int = 4):      
        """ Architecture designer of neural network for different tasks based in heuristic. 

        Args:
            k (int): Max of parameters for data. By default 10.
            c1 (float): Initial factor expansion. By default  2.0.
            r (float): Reduction rate between layers. By default 0.5.
            n_min (int): Min size of hidden layer . By default  8.
            L_max (int): Max of hidden layers. By defaults 4.
        """
        self.k = k
        self.c1 = c1 
        self.r  = r 
        self.n_min = n_min 
        self.L_max = L_max 
        
    # ---------------- Template Method ---------------- # 
    
    def desing(self, d:int, n0:int, K: int) -> NNDesing: 
        
        self._dataset_validation(d, n0, K)
        
        n_max  = min(1024, max(64, math.floor(0.25*d)))
        P_max = math.floor(self.k*d)
        
        layers = self._layers_building(n0, n_max, P_max)
        P = self._fitting_complexity(layers, n0, K, P_max)
        
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
            raise ValueError("Insufficient budget")
        
        layers: List[int] = [int(n1)]
        
        while True: 
            if len(layers) >= self.L_max:
                break
            n_new = math.floor(self.r * layers[-1])
            if n_new < self.n_min:
                break
            
            layers.append(min(n_new, n_max))
        
        return layers    
    
    def _fitting_complexity(self, layers: List[int], n0:int, K: int, P_max:int) -> int: 
        
        P = parameters_estimator(n0 = n0, layers = layers, K=K)
        
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
                
            P = parameters_estimator(n0 = n0, layers = layers, K=K)
        
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
        
        
    def _early_stopping(self, d:int) -> Tuple[int,int]: # patience and max epoch
        if d < 2000:
            return (20, 400)
        elif d < 20000:
            return (15, 200)
        return (10, 100)
        
    
    # ---------------- Strategy Methods ---------------- #
    
    @abstractmethod
    def _dataset_validation(self, d:int, n0:int, K:int) -> None: 
        pass
    
    
################
#  Strategies  #
################

class RegressionDesigner(NetworkDesigner):     
    
    def desing(self, d: int, n0: int, K:int = 1) -> NNDesing:
        return super().desing(d, n0, K)
    
    def _dataset_validation(self, d:int, n0: int, K:int = 1) -> None: 
        if d<2*n0: 
            raise ValueError("Small dataset for regression task")
        
        if K > 1: 
            raise ValueError("For regression task, K must be equal to 1")
        

class BinaryClassificationDesigner(NetworkDesigner):
     
    def desing(self, d: int, n0: int, K:int =1) -> NNDesing:
        return super().desing(d, n0, K)
   
    def _dataset_validation(self, d: int, n0: int, K:int =1) -> None:
        if d<2*n0: 
            raise ValueError("Small dataset for binary classification task")
        if K > 1: 
            raise ValueError("For binary classification task, K must be equal to 1")
        
class MulticlassClassificationDesigner(NetworkDesigner): 
    
    def desing(self, d: int, n0: int, K: int) -> NNDesing:
        return super().desing(d, n0, K)
    
    def _dataset_validation(self, d: int, n0: int, K:int) -> None:
        if d<2*n0: 
            raise ValueError("Small dataset for binary classification task")
        if K < 3: 
            raise ValueError("For multiclass classification task, K must be grater than 2")
        
        
        
        
        
if __name__ == "__main__":
    """
    this is a simple test
    """
    
    d =  4436
    n0 =  44
    K1 = 1
    K2 = 4
    
    regression_task = RegressionDesigner()
    regression_desing = regression_task.desing(d = d, n0=n0, K = K1)
    print('='*60)
    print("   Regression desing   ")
    print(regression_desing)
    print('-'*60, '\n')
    
    
    binary_task = BinaryClassificationDesigner()
    binary_desing = binary_task.desing(d = d, n0=n0, K=K1)
    print('='*60)
    print("   Binary desing   ")
    print(binary_desing)
    print('-'*60, '\n')
    
    
    multiclass_task = MulticlassClassificationDesigner()
    multiclass_desing = multiclass_task.desing(d = d, n0=n0, K = K2)
    print('-'*60)
    print("   Multiclassification desing   ")
    print(multiclass_desing)
    print('-'*60, '\n')