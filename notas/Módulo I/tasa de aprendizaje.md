# Tasa de aprendizaje

La **tasa de aprendizaje** $\eta > 0$ es un hiperparámetro que controla el tamaño de cada actualización de los parámetros durante el entrenamiento.

Aparece en la regla de actualización de [[06 - Arquitectura general de una red neuronal (Parte II)|backpropagation]]:

$$w \leftarrow w - \eta\,\frac{\partial \mathcal{L}}{\partial w}$$

---

## Intuición

El [[gradiente del error]] indica la *dirección* en que el error crece más rápido. La tasa de aprendizaje decide *qué tan lejos* se da un paso en la dirección opuesta.

- Si $\eta$ es muy **grande**: los pasos son grandes, la red puede saltarse el mínimo y oscilar o diverger.
- Si $\eta$ es muy **pequeña**: los pasos son diminutos, el entrenamiento converge pero muy lento (o puede quedar atascado en mínimos locales).

---

## Valores típicos

En la práctica se suelen explorar valores en el rango $[10^{-5},\; 10^{-1}]$. Valores comunes de partida: $0.01$, $0.001$, $0.0001$.

No existe un $\eta$ universalmente óptimo: depende de la arquitectura, la función de pérdida y el método de optimización.

---

## Estrategias avanzadas

Mantener $\eta$ constante durante todo el entrenamiento no siempre es lo mejor. Existen estrategias para ajustarlo dinámicamente:

- **Learning rate decay**: reducir $\eta$ conforme avanza el entrenamiento.
- **Warm-up**: comenzar con $\eta$ pequeño y aumentarlo gradualmente.
- **Optimizadores adaptativos** (Adam, RMSProp, Adagrad): ajustan $\eta$ de forma diferente para cada parámetro según su historial de gradientes.

---

## Relación con los métodos de actualización

La tasa de aprendizaje interactúa con el esquema de actualización elegido. En [[métodos de descenso de gradiente|minibatch]], el gradiente es un estimado ruidoso del gradiente real, por lo que valores de $\eta$ más conservadores suelen ser más estables.
