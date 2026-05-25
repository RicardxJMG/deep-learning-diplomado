Ir al [[Módulo I - DDL - Introducción|temario]] 🔖
# Arquitectura general de una red neuronal artificial (Parte II)

Esta segunda parte describe el **mecanismo de retropropagación del error (backpropagation)**, el cual permite ajustar los parámetros entrenables de una red neuronal. A diferencia de la [[05 - Forward Propagation|Parte I]], aquí el flujo de información ocurre **desde la salida hacia las capas anteriores**, con el objetivo de reducir el error del modelo.

---

## Propagación hacia atrás del error

Una vez realizada la [[05 - Forward Propagation|propagación hacia adelante]], la red obtiene una predicción $\hat{\boldsymbol{y}}$.

Esta predicción se compara con el valor objetivo $\boldsymbol{y}$ mediante una [[función de pérdida]] $\mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})$, que produce un escalar que mide qué tan mala fue la decisión de la red.

El objetivo del entrenamiento es minimizar este escalar ajustando todos los pesos y sesgos de la red.

---

## Dependencia jerárquica de los parámetros

La pérdida $\mathcal{L}$ no depende directamente de los pesos: su efecto se propaga a través de una cadena de transformaciones.

$$\mathcal{L} \leftarrow \hat{\boldsymbol{y}} \leftarrow \boldsymbol{a}_R \leftarrow \boldsymbol{a}_{R-1} \leftarrow \cdots \leftarrow \boldsymbol{a}_1 \leftarrow \boldsymbol{x}$$

Los parámetros de capas cercanas a la salida influyen de manera más directa, mientras que los de capas iniciales lo hacen de forma compuesta a través de múltiples transformaciones intermedias (véase [[Capa de Salida]] y [[Capas Intermedias]]).

---

## Gradientes como mecanismo de ajuste

El ajuste de los parámetros se basa en el [[gradiente del error]] con respecto a cada peso y sesgo:

$$\frac{\partial \mathcal{L}}{\partial w_{i,s}^{(j)}} \qquad \text{y} \qquad \frac{\partial \mathcal{L}}{\partial b_i^{(j)}}$$

Estos valores indican cuánto y en qué dirección cambia el error si se modifica ese parámetro. El cálculo sistemático de todos estos gradientes usando la [[regla de la cadena]] es lo que formalmente se llama **backpropagation**.

Para los parámetros de la **capa de salida**, el gradiente se calcula con un solo paso de la regla de la cadena. Para los parámetros de **capas intermedias**, el camino es más largo: el gradiente de $\mathcal{L}$ respecto a un peso de la capa $r$ debe atravesar todas las capas desde $r+1$ hasta la salida (véase [[derivación de gradientes por capa]]).

---

## Actualización de parámetros

Una vez calculados los gradientes, cada peso y sesgo se actualiza en la dirección opuesta al gradiente, con un paso controlado por la [[tasa de aprendizaje]] $\eta > 0$:

$$w_{i,s}^{(j)} \leftarrow w_{i,s}^{(j)} - \eta,\frac{\partial \mathcal{L}}{\partial w_{i,s}^{(j)}} \qquad \text{y} \qquad b_i^{(j)} \leftarrow b_i^{(j)} - \eta,\frac{\partial \mathcal{L}}{\partial b_i^{(j)}}$$

Si un peso contribuyó mucho al error (gradiente grande), se ajusta más. Si su influencia fue pequeña, el ajuste es menor. Los cambios siempre son pequeños y controlados: modificar demasiado en un solo paso desestabiliza la red.

---

## Métodos de actualización según el número de ejemplos

En la práctica, la red entrena con múltiples ejemplos. Existen tres estrategias que difieren en **cuántos ejemplos se usan para estimar el gradiente antes de actualizar** (véase [[métodos de descenso de gradiente]]):

|Método|Ejemplos usados|Característica|
|---|---|---|
|**Batch**|Todos los $n$ ejemplos|Estable, pero lento y costoso en memoria|
|**SGD** (estocástico)|1 ejemplo aleatorio|Rápido y ruidoso; no garantiza reducir la pérdida en cada paso|
|**Minibatch**|Subconjunto $B$ aleatorio|Balance entre estabilidad y eficiencia; estándar en la práctica|

En minibatch, el tamaño de $B$ suele elegirse entre ${16, 32, 64, 128, 256}$, siendo 32 y 64 los más comunes.

Una pasada completa por todos los datos de entrenamiento se denomina **época**. El aprendizaje emerge de repetir este ciclo durante muchas épocas.

---

## Actualización iterativa de parámetros

Los ajustes se realizan de manera incremental y repetida a lo largo del entrenamiento. Este proceso, junto con la [[05 - Forward Propagation|propagación hacia adelante]] descrita en la Parte I, constituye el ciclo fundamental de aprendizaje.

---

## Observación final

La retropropagación convierte la arquitectura de una red neuronal en un sistema adaptable.

Sin este mecanismo, la arquitectura sería una composición estática de funciones sin capacidad de aprendizaje.

---

Nota anterior: [[05 - Forward Propagation]]