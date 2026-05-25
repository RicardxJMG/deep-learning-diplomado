# Derivación de gradientes por capa

Este documento detalla cómo aplicar la [[regla de la cadena]] para calcular el [[gradiente del error]] en cada capa de la red. Se apoya en el ejemplo numérico de la arquitectura **3-2-3-4** del curso.

---

## Contexto: el ejemplo numérico

Red con arquitectura 3-2-3-4:

- **Capa 1** (2 neuronas): $f_1(t) = t^2$
- **Capa 2** (3 neuronas): $f_2(t) = \cos(t)$
- **Capa de salida** (4 neuronas): $f_{out}(t) = 100t$

Con entrada $\boldsymbol{x} = (2, 1, 5)$ la red produce $\hat{\boldsymbol{y}} = (18.6,\; 63,\; 47.7,\; -4.7)$ y el valor real es $\boldsymbol{y} = (20,\; 60,\; 40,\; 0)$.

Función de pérdida:

$$\mathcal{L}(\boldsymbol{y}, \hat{\boldsymbol{y}}) = \frac{1}{2}\sum_{i=1}^{4}(y_i - \hat{y}_i)^2 \qquad \Rightarrow \qquad \frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \hat{y}_i - y_i$$

Tasa de aprendizaje: $\eta = 0.001$.

---

## Paso 1: Gradientes en la capa de salida

Para la neurona $j$ de la capa de salida, por la regla de la cadena:

$$\frac{\partial \mathcal{L}}{\partial w_{out,s}^{(j)}} = \underbrace{\frac{\partial \mathcal{L}}{\partial \hat{y}_j} \cdot \frac{\partial \hat{y}_j}{\partial z_{out}^{(j)}}}_{\delta_{out}^{(j)}} \cdot \frac{\partial z_{out}^{(j)}}{\partial w_{out,s}^{(j)}}$$

Como $\hat{y}_j = f_{out}(z_{out}^{(j)}) = 100\,z_{out}^{(j)}$, entonces $f'_{out} = 100$.

Y como $z_{out}^{(j)} = \boldsymbol{w}_{out}^{(j)} \cdot \boldsymbol{a}_2 + b_{out}^{(j)}$, entonces $\frac{\partial z_{out}^{(j)}}{\partial w_{out,s}^{(j)}} = a_2^{(s)}$ y $\frac{\partial z_{out}^{(j)}}{\partial b_{out}^{(j)}} = 1$.

**Ejemplo — neurona 2 de la capa de salida** ($j=2$):

$$\delta_{out}^{(2)} = (\hat{y}_2 - y_2) \cdot 100 = (63 - 60) \cdot 100 = 300$$

Con $\boldsymbol{a}_2 = (0.148,\; -0.448,\; -0.973)$:

$$\frac{\partial \mathcal{L}}{\partial w_{out,1}^{(2)}} = 300 \cdot 0.148 = 44.4$$
$$\frac{\partial \mathcal{L}}{\partial w_{out,2}^{(2)}} = 300 \cdot (-0.448) = -134.4$$
$$\frac{\partial \mathcal{L}}{\partial w_{out,3}^{(2)}} = 300 \cdot (-0.973) = -291.9$$
$$\frac{\partial \mathcal{L}}{\partial b_{out}^{(2)}} = 300$$

Actualización con $\eta = 0.001$:

$$w_{out,1}^{(2)} = -0.2 - 0.001 \cdot 44.4 = -0.2444$$
$$w_{out,2}^{(2)} = 0.5 - 0.001 \cdot (-134.4) = 0.6344$$
$$w_{out,3}^{(2)} = -0.6 - 0.001 \cdot (-291.9) = -0.3081$$
$$b_{out}^{(2)} = 0.3 - 0.001 \cdot 300 = 0$$

---

## Paso 2: Gradientes en capas intermedias

Los parámetros de la capa $r$ no afectan directamente a $\mathcal{L}$. El camino es:

$$(w_{r,s}^{(j)},\, b_r^{(j)}) \;\longrightarrow\; z_r^{(j)} \;\longrightarrow\; a_r^{(j)} \;\longrightarrow\; z_{out}^{(k)} \;\text{para todo } k \;\longrightarrow\; \hat{y}_k \;\longrightarrow\; \mathcal{L}$$

Como $a_r^{(j)}$ alimenta a **todas** las neuronas de la capa siguiente, la derivada acumula todas esas contribuciones:

$$\frac{\partial \mathcal{L}}{\partial w_{r,s}^{(j)}} = \left(\sum_{k} \delta_{out}^{(k)} \cdot w_{out,j}^{(k)}\right) \cdot f'_r(z_r^{(j)}) \cdot a_{r-1}^{(s)}$$

**Ejemplo — neurona 1 de la capa 2** ($f_2 = \cos$, por tanto $f'_2 = -\sin$):

El camino es $(w_{2,s}^{(1)}, b_2^{(1)}) \to z_2^{(1)} \to a_2^{(1)} \to z_{out}^{(k)}$ para $k=1,2,3,4$.

Con $\hat{\boldsymbol{y}} - \boldsymbol{y} = (-1.4,\; 3,\; 7.7,\; -4.7)$, $z_2^{(1)} = 20.273$, $\sin(20.273) \approx 0.989$, $\boldsymbol{a}_1 = (24.01,\; 2.56)$ y primera columna de $W_{out} = (0.7,\; -0.2,\; 0.4,\; -0.5)$:

$$\sum_{k} \delta_{out}^{(k)} \cdot w_{out,1}^{(k)} = 100\left[0.7(-1.4) + (-0.2)(3) + 0.4(7.7) + (-0.5)(-4.7)\right] = 100 \cdot 3.85 = 385$$

$$\frac{\partial \mathcal{L}}{\partial w_{2,1}^{(1)}} = 385 \cdot (-\sin(20.273)) \cdot a_1^{(1)} = 385 \cdot (-0.989) \cdot 24.01 \approx -9142.17$$

$$\frac{\partial \mathcal{L}}{\partial w_{2,2}^{(1)}} = 385 \cdot (-0.989) \cdot 2.56 \approx -974.76$$

$$\frac{\partial \mathcal{L}}{\partial b_2^{(1)}} = 385 \cdot (-0.989) \cdot 1 \approx -380.77$$

Actualizaciones:

$$w_{2,1}^{(1)} = 0.9 - 0.001 \cdot (-9142.17) = 10.042$$
$$w_{2,2}^{(1)} = -0.6 - 0.001 \cdot (-974.76) = 0.375$$
$$b_2^{(1)} = 0.2 - 0.001 \cdot (-380.77) = 0.581$$

---

## Patrón general

| Capa | Fórmula del gradiente |
|---|---|
| Salida | $\delta_{out}^{(j)} \cdot a_R^{(s)}$ |
| Intermedia $r$ | $\left(\sum_k \delta_{r+1}^{(k)} \cdot w_{r+1,j}^{(k)}\right) \cdot f'_r(z_r^{(j)}) \cdot a_{r-1}^{(s)}$ |

Donde $\delta_{r+1}^{(k)} = \frac{\partial \mathcal{L}}{\partial z_{r+1}^{(k)}}$ es el error local ya calculado en la capa siguiente. Esto es lo que hace que el algoritmo sea eficiente: los $\delta$ se reutilizan de capa en capa.
