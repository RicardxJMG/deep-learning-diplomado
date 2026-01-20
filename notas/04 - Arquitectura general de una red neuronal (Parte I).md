# Arquitectura de una red neuronal artificial 

Una red neuronal artificial se define como una composición de funciones parametrizadas organizadas en capas, cuyo objetivo es aproximar una función
$$

f: \mathbb{R}^n \to \mathbb{R}^m

$$
a partir de datos. La arquitectura describe **el número de capas**, **el número de neuronas por capa** y **la conectividad entre ellas**.

---

## Capa de entrada

La **capa de entrada** recibe directamente el vector de características

$$
\boldsymbol{x} \in \mathbb{R}^n

$$
y lo propaga a la siguiente capa sin aplicar transformaciones paramétricas.  
Su rol es exclusivamente estructural y se formaliza como se describe en [[capa de entrada]].

---

## Capas ocultas

Las **capas ocultas** realizan el procesamiento principal de la red. Cada capa implementa una transformación afín seguida de una función de activación no lineal.

### Primera capa oculta

La primera capa oculta calcula

$$
\boldsymbol{z}_1 = W_1\boldsymbol{x} + \boldsymbol{b}_1, \qquad
\boldsymbol{a}_1 = f_1(\boldsymbol{z}_1),
$$

donde los pesos y sesgos constituyen los primeros parámetros entrenables del modelo.  

La formalización completa se encuentra en [[Primera Capa Intermedia de la Red Neuronal]].

### Capas intermedias adicionales

Para capas ocultas subsecuentes, el cálculo se generaliza como

$$
\boldsymbol{a}_r = f_r(W_r\boldsymbol{a}_{r-1} + \boldsymbol{b}_r),
$$

manteniendo la misma estructura matemática en cada nivel.  
Este patrón recurrente está descrito en [[capas Intermedias]].

---

## Capa de salida

La **capa de salida** transforma la última activación oculta en el vector de salida final

$$
\boldsymbol{y} = f_{\text{out}}(W_{\text{out}}\boldsymbol{a}_R + \boldsymbol{b}_{\text{out}}),
$$
donde la elección de la función de activación depende del tipo de problema (regresión, clasificación binaria o multiclase).  
El tratamiento formal se detalla en [[capa de salida]].

---
## Ejemplo de arquitectura

Como ejemplo, una red con arquitectura $3-2-3-4$ contiene tres capas ocultas/salida con un total de 33 parámetros entrenables.  
El conteo exacto de parámetros y el flujo numérico asociado se presentan en [[Ejemplo]].

---

## Observación final

La arquitectura de una red neuronal determina su capacidad de representación, mientras que el entrenamiento ajusta los parámetros para minimizar una función de pérdida definida sobre los datos.



---

Nota anterior: [[03 - Diferencias entre el Machine Learning y Deep Learning]]
Siguiente nota: [[05 - Forward Propagation]]
