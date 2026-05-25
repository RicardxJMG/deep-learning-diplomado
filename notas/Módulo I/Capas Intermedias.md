## La $r$-ésima capa de la red

Para una red con capas ocultas enumeradas como $1,2,\ldots,R$, la operación de la capa $r$ es análoga a la primera.

Sea $\boldsymbol{a}_{r-1}$ la salida de la capa previa. La capa $r$, con $n_r$ neuronas, calcula para cada neurona $j$:


$$
z_r^{(j)} = \mathbf{w}_r^{(j)} \cdot \boldsymbol{a}_{r-1} + b_r^{(j)}.
$$

Agrupando las neuronas:

$$
\boldsymbol{z}_r = W_r \boldsymbol{a}_{r-1} + \boldsymbol{b}_r,
$$

donde  
- $W_r$ es la matriz de pesos de tamaño $n_r \times n_{r-1}$,  
- $\boldsymbol{b}_r \in \mathbb{R}^{n_r}$ es el vector de sesgos.

Aplicando la función de activación de la capa \(r\):

$$
\boldsymbol{a}_r = f_r(\boldsymbol{z}_r).
$$

Por lo tanto, el flujo general de una red neuronal es:

$$
\boldsymbol{a}_r = f_r(W_r \boldsymbol{a}_{r-1} + \boldsymbol{b}_r).
$$

---

[[04 - Arquitectura general de una red neuronal (Parte I)|Regresar a la nota principal]]

