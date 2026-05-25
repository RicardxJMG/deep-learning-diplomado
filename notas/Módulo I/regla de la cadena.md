# Regla de la cadena

La **regla de la cadena** es el teorema del cálculo que permite derivar composiciones de funciones. Es el fundamento matemático de [[06 - Arquitectura general de una red neuronal artificial (Parte II)|backpropagation]].

---
## Caso univariable

Si $y = f(u)$ y $u = g(x)$, entonces:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

---

## Caso multivariable (necesario en redes neuronales)

Si $\mathcal{L}$ depende de varias variables intermedias $z_1, z_2, \ldots, z_m$ que a su vez dependen de $x$:

$$\frac{\partial \mathcal{L}}{\partial x} = \sum_{k=1}^{m} \frac{\partial \mathcal{L}}{\partial z_k} \cdot \frac{\partial z_k}{\partial x}$$

Esto ocurre en capas intermedias: un solo peso $w_{r,s}^{(j)}$ afecta a $a_r^{(j)}$, que a su vez es entrada para *todas* las neuronas de la capa $r+1$. Por tanto, su contribución al error final se acumula por todos esos caminos.

---

## Aplicación en backpropagation

El flujo completo de la red es una composición anidada:

$$\mathcal{L} = \mathcal{L}\!\left(\boldsymbol{y},\, f_{out}\!\left(W_{out}\, f_R\!\left(W_R \cdots f_1(W_1 \boldsymbol{x} + \boldsymbol{b}_1) \cdots + \boldsymbol{b}_R\right) + \boldsymbol{b}_{out}\right)\right)$$

Aplicar la regla de la cadena desde $\mathcal{L}$ hasta cada parámetro $w_{i,s}^{(j)}$ produce los [[gradiente del error|gradientes]] necesarios para la actualización.

El cálculo se hace de **atrás hacia adelante**: primero se obtienen los gradientes de la capa de salida, luego se reutilizan para calcular los de la capa anterior, y así sucesivamente. Ver [[derivación de gradientes por capa]].

---

## Por qué es eficiente

Sin la regla de la cadena aplicada de atrás hacia adelante, habría que recalcular todo el camino desde cero para cada parámetro. La propagación hacia atrás *reutiliza* los gradientes intermedios ya calculados, haciendo que el costo total sea proporcional al número de parámetros, no exponencial.
