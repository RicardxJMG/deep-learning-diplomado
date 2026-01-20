## Primera capa

Sea la primera capa oculta una capa con $n_1$ neuronas.  
Cada neurona $j$ en esta capa realiza una transformación afín sobre el vector de entrada:

$$
z_1^{(j)} = \mathbf{w}_1^{(j)} \cdot \boldsymbol{x} + b_1^{(j)},
$$

donde  
- $\mathbf{w}_1^{(j)} = (w_{1,1}^{(j)}, w_{1,2}^{(j)}, \ldots, w_{1,n}^{(j)})$ es el vector de pesos,  
- $b_1^{(j)}$ es su sesgo.

Agrupando los pesos de toda la capa obtenemos la matriz


$$
W_1 = \begin{pmatrix}
w_{1,1}^{(1)} & w_{1,2}^{(1)} & \cdots & w_{1,n}^{(1)}\\
w_{1,1}^{(2)} & w_{1,2}^{(2)} & \cdots & w_{1,n}^{(2)}\\
\vdots & \vdots & \ddots & \vdots \\
w_{1,1}^{(n_1)} & w_{1,2}^{(n_1)} & \cdots & w_{1,n}^{(n_1)}
\end{pmatrix},
$$

y el vector de sesgos  

$$
\boldsymbol{b}_1 = (b_1^{(1)}, b_1^{(2)}, \ldots, b_1^{(n_1)}).
$$

Así, la capa completa calcula:

$$
\boldsymbol{z}_1 = W_1 \boldsymbol{x} + \boldsymbol{b}_1.
$$

A continuación, cada neurona aplica su [[Funciones de Activación|función de activación]] $f_1$:

$$
\boldsymbol{a}_1 = f_1(\boldsymbol{z}_1).
$$

Este vector $\boldsymbol{a}_1$ constituye la salida de la primera capa oculta.


---

[[04 - Arquitectura general de una red neuronal (Parte I)|Regresar a la nota principal]]
