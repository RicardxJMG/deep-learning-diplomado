# Función de pérdida

La **función de pérdida** $\mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}})$ es el mecanismo que permite cuantificar qué tan mala fue la predicción de la red. Produce un escalar no negativo: cuanto más grande, mayor el error.

Sin una función de pérdida bien definida, no hay gradiente que calcular y, por tanto, no hay aprendizaje.

---

## Definición general

Dada la predicción de la red $\hat{\boldsymbol{y}}$ y el valor objetivo real $\boldsymbol{y}$, la función de pérdida asigna un número:

$$\mathcal{L} : (\boldsymbol{y}, \hat{\boldsymbol{y}}) \mapsto \mathbb{R}_{\geq 0}$$

El entrenamiento consiste en encontrar los parámetros $(W, \boldsymbol{b})$ que minimizan $\mathcal{L}$.

---

## Pérdidas para regresión

### Error cuadrático medio (MSE)

$$\mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}}) = \frac{1}{2} \sum_{i=1}^{n_{out}} (y_i - \hat{y}_i)^2$$

El factor $\frac{1}{2}$ es convencional: simplifica la derivada al cancelarse con el exponente.

Su derivada respecto a la predicción es directa:

$$\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \hat{y}_i - y_i$$

Es la pérdida más común para problemas de regresión. Penaliza errores grandes de forma cuadrática (errores pequeños se penalizan poco, errores grandes se penalizan mucho).

---

## Pérdidas para clasificación

### Entropía cruzada binaria (Binary Cross-Entropy)

Para clasificación binaria ($n_{out} = 1$, salida en $(0,1)$ con sigmoid):

$$\mathcal{L}(y, \hat{y}) = -\left[ y \log(\hat{y}) + (1 - y) \log(1 - \hat{y}) \right]$$

### Entropía cruzada categórica (Categorical Cross-Entropy)

Para clasificación multiclase ($n_{out} = k$, salida con softmax):

$$\mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}}) = -\sum_{i=1}^{k} y_i \log(\hat{y}_i)$$

donde $\boldsymbol{y}$ es un vector one-hot (un solo 1, el resto 0s).

---

## Error promedio sobre un conjunto de datos

Cuando la red entrena con $n$ ejemplos $(\boldsymbol{x}^{(k)}, \boldsymbol{y}^{(k)})$, se calcula la pérdida individual de cada ejemplo y se promedian:

$$\mathcal{L}_{\text{prom}} = \frac{1}{n} \sum_{k=1}^{n} \mathcal{L}(\boldsymbol{y}^{(k)}, \hat{\boldsymbol{y}}^{(k)})$$

Este promedio es el que se minimiza durante el entrenamiento. Ver [[métodos de descenso de gradiente]] para los distintos esquemas de actualización según cuántos ejemplos se usan por paso.

---

## Relación con el resto del flujo

$$\boldsymbol{x} \xrightarrow{\text{forward}} \hat{\boldsymbol{y}} \xrightarrow{\mathcal{L}} \text{escalar de error} \xrightarrow{\text{backprop}} \nabla_{W,b}\,\mathcal{L}$$

Véase también: [[gradiente del error]], [[06 - Arquitectura general de una red neuronal (Parte II)]]
