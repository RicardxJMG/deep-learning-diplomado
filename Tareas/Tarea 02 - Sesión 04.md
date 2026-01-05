---
theme: nord-dark
margin: "0.05"
center: "false"
maxScale: "2.0"
---
 

## **Tarea 2 - Sesión 04**

#### Nuevos pesos y sesgo de la cuarta neurona de la capa de salida.

---

Recordemos que tenemos los vectores de entrenamiento y prueba $y$ e  $\hat{y}$, respectivamente

$$
y = (20,60,40,0), \quad \hat{y} = (18.6, 63, 47.7, -4.7).
$$

De aqui que $y_{4}= 0$ y $\hat{y}_4 = -4.7$.

---

También, nuestra función de pérdida esta dada por

$$\small \mathcal{L}(\boldsymbol{y},\boldsymbol{\hat{y}})=\frac{1}{2}\left((y_1-\hat{y}_1)^2+(y_2-\hat{y}_2)^2+(y_3-\hat{y}_3)^2+(y_4-\hat{y}_4)^2\right)$$

y nuestra tasa de aprendizaje es $\eta = 0.001$

---

Para actualizar los nuevos pesos y el sesgo de esta neurona se tiene que calcular:


$$\small
\frac{\partial\mathcal{L}}{\partial \mathbf{w}_{\text{out}}^{(4)}}
\quad\text{ y }\quad
\frac{\partial\mathcal{L}}{\partial b_{\text{out}}^{(4)}}
$$


con $\mathbf{w}_{\text{out}}^{(4)} =( w_{\text{out},1}^{(4)}, w_{\text{out},2}^{(4)}, w_{\text{out},3}^{(4)})$. Luego,  por la regla de la cadena 

$$
\small
\begin{gather*}
\frac{\partial\mathcal{L}}{\partial \mathbf{w}_{\text{out}}^{(4)}}=
\textcolor{BF616A}{\frac{\partial\mathcal{L}}{\partial z_{\text{out}}^{(4)}}}\,
\textcolor{5E81AC}{\frac{\partial z_{\text{out}}^{(4)}}{\partial \mathbf{w}_{\text{out}}^{(4)}}}
=
\textcolor{BF616A}{\frac{\partial\mathcal{L}}{\partial \hat{y}_2}\,\frac{\partial \hat{y}_2}{\partial z_{\text{out}}^{(4)}}}\,
\textcolor{5E81AC}{\frac{\partial z_{\text{out}}^{(4)}}{\partial \mathbf{w}_{\text{out}}^{(4)}}},
\\[5mm]
\frac{\partial\mathcal{L}}{\partial b_{\text{out}}^{(4)}}=
\textcolor{BF616A}{\frac{\partial\mathcal{L}}{\partial z_{\text{out}}^{(4)}}}\,
\textcolor{A3BE8C}{\frac{\partial z_{\text{out}}^{(4)}}{\partial b_{\text{out}}^{(4)}}}

\end{gather*}
$$

---
Por otra parte

$$\frac{\partial\mathcal{L}}{\partial \hat{y}_4}=\hat{y}_4-y_4=-4.7-0=-4.7$$

Como $\hat{y}_4=a_{\text{out}}^{(4)}=f_{\text{out}}(z_{\text{out}}^{(4)})=100z_{\text{out}}^{(4)}\,$, entonces

$$\frac{\partial\hat{y}_4}{\partial z_{\text{out}}^{(4)}}=100$$

$$\therefore \textcolor{#BF616A}{\frac{\partial\mathcal{L}}{\partial \hat{y}_2}\,\frac{\partial \hat{y}_2}{\partial z_{\text{out}}^{(4)}}} = -4.7\cdot100 = -470$$ 
---

Por definición, 

$$z_{out}^{(4)}=\mathbf{w}_{\text{out}}^{4}\cdot a_{2} + b_{\text{out}}^{(4)}$$

  

entonces

$$\textcolor{#5E81AC}{\frac{\partial z_{\text{out}}^{(4)}}{\partial \mathbf{w}_{\text{out}}^{(4)}} = a_2 }
\quad\text{ y }\quad
\textcolor{#A3BE8C}{\frac{\partial z_{\text{out}}^{(4)}}{\partial b_{\text{out}}^{(4)}} = 1}
$$



---

Con lo anterior, concluimos que

$$
\begin{align*}
\frac{\partial\mathcal{L}}{\partial \mathbf{w}_{\text{out}}^{(4)}} 
& = -470 a_{2}= -470 \cdot (0.148, - 0.448, -0.973) \\[7mm]
\frac{\partial\mathcal{L}}{\partial \mathbf{w}_{\text{out}}^{(4)}} & = (-69.56,\, 210.56,\, 457.31)
\end{align*}
$$


$$
\begin{equation*}
\text{ y } \frac{\partial\mathcal{L}}{\partial b_{out}^{(4)}}=-470\cdot1=-470
\end{equation*}.
$$

---
El peso y sesgo de la neurona cuatro de la capa de salida se ajusta ligeramente con:
  

$$
\mathbf{w}_{\text{out}}^{(4)} \leftarrow 
\mathbf{w}_{\text{out}}^{(4)} - \eta \frac{\partial\mathcal{L}}{\partial \mathbf{w}_{\text{out}}^{(4)}}
\quad\text{ y }\quad
b_{\text{out}}^{(4)} \leftarrow 
b_{\text{out}}^{(4)} - \eta \frac{\partial\mathcal{L}}{\partial b_{\text{out}}^{(4)}}

$$


Donde $\mathbf{w}_{\text{out}}^{(4)}=(-0.5, 0.2, 0.6)$ y sesgo es $b_{\text{out}}^{(4)}=0.7$ para esta neurona.

---
Luego, los nuevos parámetros de neurona 4 de la capa de salida son

$$
\begin{align*}
w_{\text{out},1}^{(4)} &= -0.5 - 0.001\cdot(-69.56) = -0.43044 \\
w_{\text{out},2}^{(4)} &= 0.2 - 0.001\cdot 210.56 = -0.01056 \\
w_{\text{out},3}^{(4)} &= -0.6 - 0.001\cdot 457.31 = 0.14269 \\
\\
b_{\text{out}}^{(4)} &= 0.7 - 0.001\cdot(-470) = 1.17
\end{align*}
$$