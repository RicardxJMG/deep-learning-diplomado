---
theme: serif
margin: "0.1"
center: "false"
maxScale: "1.0"
---
## Tarea 1
#### Deep Learning con Python 2025


---

::: 
#### **1: Dibujar la red neuronal**
:::

![[nn_tarea1.excalidraw.png|600]]

---

:::
##### **2: ¿Cuántos parámetros tiene?**
:::

:::
<p style="text-align: left;">
La red neuronal tiene en total $23$ parámetros ya que: 
</p>
:::

:::
- Por la primera capa tenemos $9$ parámetros ($6$ pesos y $3$ sesgos)
- Por la segunda capa tenemos $8$ parámetros ($6$ pesos y $2$ sesgos)
- Por la capa de salida tenemos $6$ parámetros ($4$ pesos y $2$ sesgos)
:::

---

:::
##### **2: ¿Cuántos parámetros tiene?**
:::


:::
<p style="text-align: left;">
De manera más técnica, usando la siguiente expresión para estimar el número de parámetros para esta red neuronal
</p>
:::

$$\scriptsize\sum_{k=0}^{R} n_{k+1}(n_k+1) \quad \text{ para } R=2$$

:::
<p style="text-align: left;">
se obtiene
</p>
:::

$$
\scriptsize
\begin{align*}
	\sum_{k=0}^{2} n_{k+1}(n_k+1) 
	& = n_1(n_0+1) + n_2(n_1 +1) + n_3(n_2 +1) \\[-2mm] 
	& = 3(2+1) + 2(3+1) + 2(2+1) \\
	& = 9 + 8 + 6 = 23
\end{align*}

	 
$$


---
:::
##### **3: Según los valores del enunciado, ¿quiénes son $W_1$ y $b_1$?**
:::
<br>

::: 
La matriz de pesos $\small W_{1}$ es:  $\hspace{3.4cm}$  
:::
$$\small
W_1 = \begin{pmatrix}
	0.6 & -0.4 \\
	-0.1 & 0.5 \\
	0.7 & 0.2 
\end{pmatrix} 
$$
<br>
y $\small b_1 = \begin{pmatrix}0.2,\, -0.3,\,  0.1\end{pmatrix} \hspace{3.3cm}$

---
:::
##### **4: ¿Quiénes son los pesos de su segunda neurona y cuál es el sesgo de la primera?**
:::
<br>

<p style="text-align: left;">
Para la segunda neurona de la capa 2 su peso es:
</p>

$$W_2^{(2)} = (-0.5,\, 0.1, \, 0.8).$$

:::
<p style="text-align: left;">
Por otro lado, el sesgo de la primera neurona de esta capa es
</p>

$$
b_1^{(1)} = -0.2.
$$
:::

---



:::
##### **5: ¿Cuántas neuronas tiene la capa de salida?, ¿Cuántos pesos y sesgos tiene?**
:::
<br>
:::
<p style="text-align: left;">
La capa de salida tiene dos neuronas, cuatro pesos y dos sesgos.
</p>
:::


---
:::
##### **6: Crea tu propia matriz de pesos para la capa de salida y los sesgos de sus neuronas**
:::

<br>

:::
<p style="text-align: left;">
Propuesta de matriz de salida 
</p>

$$
W_{\text{out}} = \begin{pmatrix}
	0.01 & -0.7 \\
	0.7 & 0.5 
\end{pmatrix} 
$$

<p style="text-align: left;">
y la propuesta del vector de sesgos es: 
</p>

$$
b_{\text{out}} = (0.33,\, -0.13)
$$

:::

---
