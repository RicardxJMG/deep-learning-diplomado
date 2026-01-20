## Ejemplo de arquitectura de una red neuronal.

## Diseño de la Red 

Considérese una red neuronal con arquitectura $3-2-3-4$.  
Esto implica:

- Capa de entrada: $n_0 = 3$ neuronas  
- Capa oculta 1: $n_1 = 2$ neuronas  
- Capa oculta 2: $n_2 = 3$ neuronas  
- Capa de salida: $n_3 = 4$ neuronas  


![[nnexample.png|450]]

### Conteo de parámetros

Cada capa $k$ introduce  

$$
n_{k}(n_{k-1}+1)
$$
parámetros (pesos + sesgos).

Entonces:

- Primera capa oculta: $2(3+1) = 8$  
- Segunda capa oculta: $3(2+1) = 9$  
- Capa de salida: $4(3+1) = 16$

Total:

$$
8 + 9 + 16 = 33 \text{ parámetros}.
$$

### Ejemplo numérico

Dada la entrada

$$
\boldsymbol{x} = (2,\,1,\,5),
$$
la red calcula:

$$
\begin{align*}

\boldsymbol{z}_{1} &=  W_1 \boldsymbol{x} + \boldsymbol{b}_1,\qquad 
\boldsymbol{a}_1 = f_1(\boldsymbol{z}_{1}) \\[2mm]
\boldsymbol{z}_{2} &=  W_2 \boldsymbol{a}_1 + \boldsymbol{b}_2,\qquad 
\boldsymbol{a}_2 = f_2(\boldsymbol{z}_{2})\\[2mm]
\boldsymbol{z}_{\text{out}} &=  W_{\text{out}} \boldsymbol{a}_2 + \boldsymbol{b}_{\text{out}},\qquad 
\boldsymbol{y} = f_{\text{out}}(\boldsymbol{z}_{\text{out}}).
\end{align*}


$$


Este procedimiento ilustra explícitamente el flujo de una red neuronal estándar mediante transformaciones afines y funciones de activación.

---

[[04 - Arquitectura general de una red neuronal (Parte I)|Regresar a la nota principal]]

