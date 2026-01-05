---
theme: nord-dark
margin: "0.05"
center: "false"
maxScale: "2.0"
---
## **Tarea 03 - Sesión**

### Primer Algoritmo de una Red Neuronal
---
### **Instrucciones**

1. Añade una característica a cada cliente. Es decir, añade un número a cada fila de $X$. 
2. Añade dos nuevos clientes. Es decir, añade dos filas a $X$ dos filas a $y$.

---

### Nuevos valores de $X$ e $y$.

```python  
X = np.array([
                   # nueva característica
    [ 1.0, 0.0, 1.0,      0.0],
    [ 0.0, 1.0, 1.0,      1.0],
    [ 1.0, 1.0, 0.0,     -0.6],
    [-1.0, 1.0, 0.5,      1.0],
    [0.0, 1.0, -1.0,      0.3], # nuevo cliente
    [1.0, 0.5, -0.5,     -0.7]  # nuevo cliente
], dtype = np.float32)

y = np.array([
    [ 0.5],
    [ 0.0],
    [ 0.4],
    [-0.3],
    [-0.4], # nuevo cliente
    [-0.2]  # nuevo cliente
], dtype=np.float32)

```


---
#### **1. ¿Cuántas capas requiere la capa de entrada?**

- Al añadir una nueva característica a los clientes se tendrá que agregar una neurona adicional a la capa de entrada, es decir, cuatro neuronas en la capa de entrada.

- Asi que, el código se modifica como 

```python
#------------------#
#   Modelo 4-4-1   #
#------------------#
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),   # Ahora son cuatro neuronas
    tf.keras.layers.Dense(4, activation="tanh", use_bias=True, name="hidden"),
    tf.keras.layers.Dense(1, activation="linear", use_bias=True, name="out"),])
```

---
#### **2. ¿Cómo se tiene que modificar la matriz de pesos y el vector de sesgos?**
<br>

- Manteniendo la primera capa oculta con cuatro neuronas, se agrega un peso adicional a cada neurona por la nueva característica del cliente. 

- Al no modificarse la primera capa oculta el vector sesgo se mantiene como

- En código esto se ve como
```python

W1 = np.array([       # nuevo peso añadido
    [ 0.2, -0.1,  0.0,      -0.2],
    [-0.3,  0.1,  0.2,       0.1],
    [ 0.0,  0.2, -0.2,       0.4],
    [ 0.1,  0.0,  0.3,      -0.7],
], dtype=np.float32)		

# no se modifica
b1 = np.array([0.0, 0.1, -0.1, 0.05], dtype=np.float32) 
```

---
#### **3. ¿Cómo queda $B$ para elegir gradiente descendente?**


- Al agregar dos clientes adicionales se tiene que $\small N_{\text{cliente}} = \{1,2,3,4,5,6\}$

- Para gradiente descendente se tiene que $B = |N_{\text{cliente}}| = 6$

---
#### **4. Ejecutar el código el código a 15 épocas para:**

- Gradiente Descendiente  

```python 
B = 6

model.fit(X, y, epochs=15, batch_size=B, verbose=0, callbacks=[PrintEpoch()])
```

```txt 
Época 15/15 
 
W1 (4x4): 
    [[ 2.00832590e-01, -1.04054809e-01, 2.18622107e-02, -1.93471089e-01],
 	[-3.02392423e-01, 1.08505294e-01, 1.59219250e-01, 8.89788866e-02],
 	[-4.47162711e-05, 2.00333402e-01, -2.02323824e-01, 3.99220735e-01], 
 	[ 1.00405715e-01, -9.99024790e-03, 3.60579014e-01, -6.79350197e-01]]
b1 (4,): [-0.002141628, 0.10424267 , -0.0998262 , 0.04454007 ] 
 
Wout (1x4): [ [ 0.09402765 , -0.16095346 , -0.022624772, 0.32323366 ] ] 
bout (1,): [0.025219731] 
 
############## RESULTADOS ############## 
 
Pérdida (L_prom): 0.031976498663 
y_hat: [ 0.20929977, -0.16560799, 0.23071429, -0.2869944 , -0.1708445 , 0.23129791]
```


---


2.  Gradiente Descendiente Estocástico

```python 
B = 1

model.fit(X, y, epochs=15, batch_size=B, verbose=0, callbacks=[PrintEpoch()])
```


```txt

Época 15/15 W1 (4x4): 
	[[ 2.0243400e-01, -1.0544383e-01, 1.0919593e-01, -1.6169186e-01],
	 [-3.0631417e-01, 1.1426777e-01, 6.2478531e-02, 5.6646127e-02],
	 [ 2.9751132e-04, 1.9855973e-01, -2.5946832e-01, 3.7698835e-01],
	 [ 6.7378342e-02, -1.1221306e-02, 6.9403762e-01, -5.3275639e-01]] 
b1 (4,): [-0.007949337, 0.10869703 , -0.09447365 , -0.014571084]

Wout (1x4): [ [ 0.10409161 , -0.057077963, -0.09864424 , 0.55667603 ] ]
bout (1,): [-0.00700059] 

############## RESULTADOS ############## 

Pérdida (L_prom): 0.012874942273 
y_hat: [ 0.41736192 , 0.010546306, 0.2235494 , -0.2645478 , -0.48147848 , 0.07875269 ]

```


---

3. Minibatch Estándar

```python 
B = 3

model.fit(X, y, epochs=15, batch_size=B, verbose=0, callbacks=[PrintEpoch()])
```


```txt
Época 15/5

W1 (4x4):
	[[ 2.02418953e-01, -1.07081115e-01,  5.66280186e-02, -1.82013780e-01],
	[-3.05786937e-01,  1.14306934e-01,  1.08049735e-01,  7.35326335e-02],
	[-5.67533949e-04,  2.00800791e-01, -2.16428220e-01,  3.94167930e-01], 
	[ 9.70181376e-02, -1.60099734e-02,  4.74721789e-01, -6.32849216e-01]]
b1 (4,):
 [-0.00289217,  0.10569257, -0.09993935,  0.04064686]

Wout (1x4): 
	[ [ 0.09097181 , -0.108483195, -0.05559705 ,  0.38860017 ] ]
bout (1,):
 [0.015146608]

##############    RESULTADOS    ##############

Pérdida (L_prom): 0.023462714627
y_hat: [ 0.2770634, -0.113911696, 0.23478253, -0.2768255, -0.2614019, 0.1982821]
```