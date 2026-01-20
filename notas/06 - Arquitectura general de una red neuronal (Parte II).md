
# Arquitectura general de una red neuronal artificial (Parte II)

Esta segunda parte describe el **mecanismo de retropropagación del error (backpropagation)**, el cual permite ajustar los parámetros entrenables de una red neuronal.
A diferencia de la Parte I, aquí el flujo de información ocurre **desde la salida hacia las capas anteriores**, con el objetivo de reducir el error del modelo.

---

## Propagación hacia atrás del error

Una vez realizada la propagación hacia adelante y obtenida la salida $\boldsymbol{y}$.

La red compara este resultado con el valor objetivo esperado. Esta comparación produce una señal escalar de error, la cual se propaga hacia atrás a través de la red.

---

## Dependencia jerárquica de los parámetros

Cada parámetro de la red contribuye indirectamente al error final.
Los parámetros de capas cercanas a la salida influyen de manera más directa, mientras que los de capas iniciales lo hacen de forma compuesta a través de múltiples transformaciones intermedias (véase [[capa de salida]] y [[capas Intermedias]]).

---

## Gradientes como mecanismo de ajuste

El ajuste de los parámetros se basa en el [[gradiente del error]] con respecto a cada peso y sesgo.

Este gradiente indica la dirección en la que el error aumenta más rápidamente; por lo tanto, los parámetros se ajustan en la dirección opuesta.

---

## Actualización iterativa de parámetros

Los ajustes de los parámetros se realizan de manera incremental y repetida a lo largo del entrenamiento.
Este proceso, junto con la propagación hacia adelante descrita en la Parte I, constituye el ciclo fundamental de aprendizaje.

---

## Observación final

La retropropagación convierte la arquitectura de una red neuronal en un sistema adaptable.

Sin este mecanismo, la arquitectura sería una composición estática de funciones sin capacidad de aprendizaje.
