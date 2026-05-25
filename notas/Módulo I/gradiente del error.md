# Gradiente del error

El **gradiente del error** es el vector de derivadas parciales de la [[función de pérdida]] $\mathcal{L}$ con respecto a cada parámetro entrenable de la red (pesos y sesgos).

Formalmente, para el peso $w_{i,s}^{(j)}$ (el peso $s$ de la neurona $j$ en la capa $i$) y el sesgo $b_i^{(j)}$:

$$\nabla_{W,b}\,\mathcal{L} = \left\{ \frac{\partial \mathcal{L}}{\partial w_{i,s}^{(j)}},\; \frac{\partial \mathcal{L}}{\partial b_i^{(j)}} \right\}_{\text{para toda } i, j, s}$$

El signo del gradiente indica la dirección en que el error *aumenta*. Por eso la actualización va en dirección **opuesta**. Véase [[tasa de aprendizaje]] y [[06 - Arquitectura general de una red neuronal (Parte II)]].

---

## ¿Por qué hace falta la regla de la cadena?

Los pesos de una capa interior no afectan directamente a $\mathcal{L}$. Su influencia es indirecta:

$$w_{i,s}^{(j)} \;\longrightarrow\; z_i^{(j)} \;\longrightarrow\; a_i^{(j)} \;\longrightarrow\; \cdots \;\longrightarrow\; \hat{\boldsymbol{y}} \;\longrightarrow\; \mathcal{L}$$

Calcular $\frac{\partial \mathcal{L}}{\partial w_{i,s}^{(j)}}$ requiere descomponer ese camino paso a paso usando la [[regla de la cadena]]. Ver [[derivación de gradientes por capa]] para el desarrollo explícito.

---

## Gradiente en la capa de salida

Para la neurona $j$ de la capa de salida, el gradiente respecto a su peso $s$ es:

$$\frac{\partial \mathcal{L}}{\partial w_{out,s}^{(j)}} = \underbrace{\frac{\partial \mathcal{L}}{\partial \hat{y}_j}}_{\text{error en la salida}} \cdot \underbrace{\frac{\partial \hat{y}_j}{\partial z_{out}^{(j)}}}_{f'_{out}(z_{out}^{(j)})} \cdot \underbrace{\frac{\partial z_{out}^{(j)}}{\partial w_{out,s}^{(j)}}}_{a_{R}^{(s)}}$$

Definiendo el **error local de la capa de salida**:

$$\delta_{out}^{(j)} = \frac{\partial \mathcal{L}}{\partial z_{out}^{(j)}} = \frac{\partial \mathcal{L}}{\partial \hat{y}_j} \cdot f'_{out}(z_{out}^{(j)})$$

Se tiene compactamente:

$$\frac{\partial \mathcal{L}}{\partial w_{out,s}^{(j)}} = \delta_{out}^{(j)} \cdot a_R^{(s)} \qquad \text{y} \qquad \frac{\partial \mathcal{L}}{\partial b_{out}^{(j)}} = \delta_{out}^{(j)}$$

---

## Gradiente en capas intermedias

Para la neurona $j$ de la capa $r$ (interior), el gradiente de $\mathcal{L}$ respecto a $w_{r,s}^{(j)}$ debe acumular la contribución de **todas las neuronas** de la capa siguiente que reciben $a_r^{(j)}$:

$$\frac{\partial \mathcal{L}}{\partial w_{r,s}^{(j)}} = \left( \sum_{k} \frac{\partial \mathcal{L}}{\partial z_{r+1}^{(k)}} \cdot w_{r+1,j}^{(k)} \right) \cdot f'_r(z_r^{(j)}) \cdot a_{r-1}^{(s)}$$

El término entre paréntesis es el **error que llega desde la capa siguiente**, propagado hacia atrás a través de los pesos $w_{r+1,j}^{(k)}$.

Ver el desarrollo completo con ejemplo numérico en [[derivación de gradientes por capa]].
