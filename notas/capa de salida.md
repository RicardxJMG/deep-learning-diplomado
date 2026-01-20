## Output Layer

El proceso es muy similar a las capas anteriores. 
Sea la última capa oculta la capa $R$, cuya salida es el vector $\boldsymbol{a}_R$.  

La capa de salida contiene $n_{\text{out}}$ neuronas y realiza:

$$
\boldsymbol{z}_{\text{out}} = W_{\text{out}} \boldsymbol{a}_R + \boldsymbol{b}_{\text{out}},
$$
donde

- $W_{\text{out}}$ es una matriz de tamaño $n_{\text{out}} \times n_R$,  
- $\boldsymbol{b}_{\text{out}} \in \mathbb{R}^{n_{\text{out}}}$.

Luego aplica su [[función de activación]] $f_{\text{out}}$:

$$
\boldsymbol{y} = \boldsymbol{a}_{\text{out}} = f_{\text{out}}(\boldsymbol{z}_{\text{out}}).
$$

El flujo completo de la red, desde $\boldsymbol{x}$ hasta la salida, es:

$$
\boldsymbol{y} = 
f_{\text{out}}\!\left(
W_{\text{out}}
f_R\!\left(
W_R \cdots f_1(W_1 \boldsymbol{x} + \boldsymbol{b}_1)
+ \boldsymbol{b}_R
\right)
+ \boldsymbol{b}_{\text{out}}
\right).
$$

