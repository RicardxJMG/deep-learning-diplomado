# Forward Propagation

El proceso completo corresponde a una **propagación hacia adelante (forward propagation)**, donde la información fluye de manera determinista desde la entrada hasta la salida, sin ciclos.

De forma más explícita, si queremos mandar información (datos) a la red neuronal este es procesado por la red neuronal en los siguientes pasos:

1. La información entra en la [[capa de entrada]]
2. La [[Primera Capa Intermedia de la Red Neuronal| primera capa oculta]] procesa la información de la capa de entrada
3. En caso de haber más de una capa intermedias, la información continua transformando en las [[capas Intermedias]]
4. Finalmente, la información procesada por las capas anteriores pasa por la [[capa de salida]] y entrega el resultado

Este proceso solo es la primera parte, puesto que todavía no se realiza el ajuste de los parámetros de nuestra red. 

---

Nota anterior: [[04 - Arquitectura general de una red neuronal (Parte I)]]
Siguiente nota: [[06 - Arquitectura general de una red neuronal (Parte II)]]


