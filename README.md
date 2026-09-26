# Multi-Armed Bandits: Chapter 2 Experiments

A Python implementation of a stationary $k$-armed bandit environment and four action-selection methods: greedy, ε-greedy, upper confidence bound (UCB), and gradient bandit. Code written by Delena based on the concepts and experiments in Chapter 2 of Sutton and Barto (2018).

## Model and experiments

Each run samples the true action values for $k=10$ arms:

```math
q_{*}(a) \sim \mathcal{N}(\mu, 1), \qquad
R_t \mid A_t=a \sim \mathcal{N}\bigl(q_{*}(a), 1\bigr).
```

Each experiment consists of 2,000 independent runs with 1,000 steps per run. Within a run, all agents face the same true action values $q_{*}(a)$, but use independent random number generators for action selection and rewards. Fixed seeds make the results reproducible.

| Experiment | Methods | Main parameters |
| --- | --- | --- |
| Standard ($\mu=0$) | Greedy, greedy with optimistic initial values, ε-greedy, and UCB | $Q_1(a)=5$ and $\alpha=0.1$ for optimistic greedy; $\varepsilon=0.1$; $c=2$. The other action-value estimates use sample averages. |
| Gradient ($\mu=4$) | Gradient bandit with and without an average-reward baseline | $\alpha=0.1$; the baseline is either the observed average reward or zero. |

The two plots for each experiment show the mean reward and the percentage of optimal-action selections **at each step**:

```math
\overline{R}_t
= \frac{1}{M}\sum_{i=1}^{M} R_t^{(i)},
\qquad
P_t
= \frac{100}{M}\sum_{i=1}^{M}
\mathbf{1}\!\left\{
A_t^{(i)} = \arg\max_a q_{*}^{(i)}(a)
\right\},
\qquad M=2000.
```

This comparison brings together methods shown in different figures in Chapter 2. The gradient experiment uses only $\alpha=0.1$; therefore, these plots are **not exact reproductions** of Figures 2.2–2.5.

## Run

Requires Python 3.10 or later. From the project directory:

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe bandits.py
```

The program displays two figures in sequence; close the first one to view the second. To run a quicker experiment, reduce the number of runs and steps in `main()`.

## Files

- `bandits.py`: environment, agents, evaluation, and plots.
- `requirements.txt`: required Python packages.
- `.gitignore`: local files excluded from version control.
- `LICENSE`: MIT License.

## Reference

Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.), Chapter 2. MIT Press. [Publisher's page](https://mitpress.mit.edu/9780262352703/reinforcement-learning/).

This repository contains an independent implementation inspired by the book. The book's PDF and figures are not included. The code is distributed under the MIT License; see `LICENSE`.
