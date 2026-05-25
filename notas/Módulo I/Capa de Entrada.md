## Input Layer

La capa de entrada, o también conocido como *input layer* de una red neuronal recibe el vector de características del ejemplo de entrenamiento. 

Si el vector de entrada es 

$$
\boldsymbol{x} = (x_1, x_2, \ldots, x_n) \in \mathbb{R}^n,
$$

entonces cada componente $x_i$ representa una característica numérica del dato. En otras palabras, el total de neuronas para la capa de entrada es $n_{0} = n$.

La capa de entrada no realiza transformaciones: simplemente entrega $\boldsymbol{x}$ a la primera capa oculta, donde comienza el procesamiento paramétrico mediante pesos y sesgos.


---

[[04 - Arquitectura general de una red neuronal (Parte I)|Regresar a la nota principal]]