# Métodos de descenso de gradiente

Los **métodos de descenso de gradiente** son los algoritmos que actualizan los parámetros de la red durante el entrenamiento. Todos siguen la misma idea: moverse en la dirección opuesta al [[gradiente del error]], pero difieren en **cuántos ejemplos se usan para estimar ese gradiente** antes de cada actualización.

---

## Contexto: múltiples ejemplos

Supongamos $n$ ejemplos de entrenamiento $(\boldsymbol{x}^{(k)}, \boldsymbol{y}^{(k)})$ con $k = 1, \ldots, n$. Cada ejemplo produce su propia pérdida individual $\mathcal{L}^{(k)}$. La pérdida promedio es:

$$\mathcal{L}_{\text{prom}} = \frac{1}{n}\sum_{k=1}^{n} \mathcal{L}^{(k)}$$

El objetivo es minimizar $\mathcal{L}_{\text{prom}}$ ajustando todos los parámetros.

---

## Batch Gradient Descent (gradiente por lotes)

Se usan **todos los $n$ ejemplos** para calcular el gradiente antes de actualizar:

$$w \leftarrow w - \eta\,\frac{\partial \mathcal{L}_{\text{prom}}}{\partial w} = w - \frac{\eta}{n}\sum_{k=1}^{n}\frac{\partial \mathcal{L}^{(k)}}{\partial w}$$

**Ventajas:**
- Gradiente exacto (sin ruido).
- Convergencia estable y predecible.

**Desventajas:**
- Requiere procesar todos los datos antes de actualizar: muy lento y costoso en memoria con datasets grandes.
- Puede quedar atrapado en mínimos locales al no tener ruido que ayude a escapar.

---

## Stochastic Gradient Descent — SGD (gradiente estocástico)

Se elige **un solo ejemplo aleatorio** $k$ y se actualiza con su gradiente individual:

$$w \leftarrow w - \eta\,\frac{\partial \mathcal{L}^{(k)}}{\partial w}$$

**Ventajas:**
- Muy rápido por actualización.
- El ruido puede ayudar a escapar de mínimos locales.

**Desventajas:**
- Alta varianza: la pérdida puede aumentar en algunos pasos.
- No garantiza reducir $\mathcal{L}_{\text{prom}}$ en cada actualización.
- Convergencia errática; requiere reducir $\eta$ con el tiempo.

---

## Minibatch Gradient Descent (estándar en la práctica)

Se elige un **subconjunto aleatorio** $B \subset \{1, \ldots, n\}$ de tamaño $|B|$ y se actualiza con el promedio de sus gradientes:

$$w \leftarrow w - \frac{\eta}{|B|}\sum_{k \in B}\frac{\partial \mathcal{L}^{(k)}}{\partial w}$$

**Ventajas:**
- Balance entre estabilidad (batch) y velocidad (SGD).
- Aprovecha las operaciones matriciales de GPU eficientemente.
- Es el método estándar en deep learning moderno.

**Tamaños de batch típicos:** $|B| \in \{16, 32, 64, 128, 256\}$. Los valores 32 y 64 son los más comunes.

---

## Comparación

| Método | Ejemplos por update | Estabilidad | Velocidad | Uso |
|---|---|---|---|---|
| Batch | $n$ (todos) | Alta | Lenta | Datos pequeños |
| SGD | 1 | Baja | Muy rápida | Raro en DL |
| Minibatch | $\|B\|$ (subconjunto) | Media | Rápida | Estándar |

---

## Época

Una **época** (*epoch*) es una pasada completa por todos los datos de entrenamiento. En minibatch, una época consiste en $\lceil n / |B| \rceil$ actualizaciones.

El entrenamiento típicamente corre durante decenas o cientos de épocas.

---

## Relación con otros conceptos

- [[tasa de aprendizaje]]: controla el tamaño del paso en cada actualización.
- [[gradiente del error]]: el valor que se estima con estos métodos.
- [[06 - Arquitectura general de una red neuronal (Parte II)]]: contexto de uso dentro del ciclo de entrenamiento.
