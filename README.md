# Bandits multibrazo: experimentos del capítulo 2

Implementación en Python de un entorno estacionario de $k$ brazos y de cuatro estrategias de selección de acciones: Greedy, ε-Greedy, límite superior de confianza (UCB) y bandido por gradiente. Código elaborado por Delena a partir de los conceptos y experimentos del capítulo 2 de Sutton y Barto (2018).

## Modelo y experimentos

Para cada ejecución se generan $k=10$ valores verdaderos de acción:

$
q_*(a) \sim \mathcal{N}(\mu, 1), \qquad
R_t\mid A_t=a \sim \mathcal{N}(q_*(a), 1).
$

Cada experimento usa 2000 problemas independientes y 1000 pasos por problema. Todos los agentes de una ejecución comparten los mismos $q_*(a)$, pero tienen generadores independientes para sus decisiones y recompensas. Una semilla fija permite repetir los resultados.

| Experimento | Estrategias | Parámetros principales |
| --- | --- | --- |
| Estándar ($\mu=0$) | Greedy, Greedy con valores iniciales optimistas, ε-Greedy y UCB | $Q_1(a)=5$ y $\alpha=0.1$ para la variante optimista; $\varepsilon=0.1$; $c=2$. Las otras estimaciones de valor usan promedio muestral. |
| Gradiente ($\mu=4$) | Bandido por gradiente con y sin recompensa media como referencia | $\alpha=0.1$; referencia igual a la media de recompensas observadas o a cero. |

Las dos gráficas de cada experimento muestran la recompensa promedio y el porcentaje de elecciones óptimas **en cada paso**:

$
\overline R_t=\frac{1}{M}\sum_{i=1}^{M}R_t^{(i)}, \qquad
P_t=100\,\frac{1}{M}\sum_{i=1}^{M}
\mathbf{1}\!\left\{A_t^{(i)}=\arg\max_a q_*^{(i)}(a)\right\},\quad M=2000.
$

La comparación agrupa estrategias que en el libro aparecen en distintas figuras del capítulo 2. El segundo experimento usa solo $\alpha=0.1$; por ello las gráficas **no son reproducciones exactas** de las figuras 2.2–2.5.

## Ejecutar

Requiere Python 3.10 o posterior. En la carpeta del proyecto:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python bandits.py
```

En Windows, se puede usar `py` en lugar de `python`. El programa abre dos figuras consecutivas; cierre la primera para ver la segunda. El número de ejecuciones y pasos puede ajustarse en `main()` para una prueba más rápida.

## Archivos

- `bandits.py`: entorno, agentes, evaluación y visualizaciones.
- `requirements.txt`: bibliotecas necesarias.
- `.gitignore`: archivos locales excluidos del repositorio.
- `LICENSE`: archivo de la licencia MIT.

## Referencia

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2.ª ed.), cap. 2. MIT Press. [Página de la editorial](https://mitpress.mit.edu/9780262352703/reinforcement-learning/).

Este repositorio contiene una implementación independiente inspirada en el libro. El PDF del libro y sus figuras no forman parte del repositorio. El código de este repositorio se distribuye bajo la licencia MIT; consulta el archivo LICENSE
