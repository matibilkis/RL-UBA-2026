import random
import matplotlib.pyplot as plt
import numpy as np

NT = 5
rng = np.random.default_rng(0)
increments = rng.choice((-1, 1), size=(NT, 500))
walks = np.c_[np.zeros(NT), increments.cumsum(axis=1)]

for k in range(NT):
    plt.plot(walks[k],label="día #{}".format(k))
plt.xlabel("Paso")
plt.ylabel("Posición")
plt.legend()
plt.show()


def random_walk_window():
    x = steps = 0
    while -4 < x < 13:
        x += random.choice((-1, 1))
        steps += 1
    return steps if x == 13 else None

random.seed(0)
N = 100_000
times = [t for _ in range(N) if (t := random_walk_window()) is not None]

steps, counts = np.unique(times, return_counts=True)
probabilities = counts / counts.sum()

L, start = 17, 4
Q = np.diag(np.ones(L - 2) / 2, 1) + np.diag(np.ones(L - 2) / 2, -1)
state = np.zeros(L - 1)
state[start - 1] = 1
analytic = []

for _ in range(steps[-1]):
    analytic.append(0.5 * state[-1] / (start / L))
    state = state @ Q

n_analytic = np.arange(1, len(analytic) + 1)
analytic = np.array(analytic)
mask = analytic > 1e-14

plt.figure(figsize=(8,4))
plt.bar(steps, probabilities, width=2, color="red", alpha=0.75, label="Monte Carlo")
plt.plot(n_analytic[mask], analytic[mask], "k.-", label="Analítica")
plt.ylabel("Probabilidad")
plt.xlabel("Cantidad de pasos")
plt.title("Probabilidad (estimada) \nllegar al carrefour\nSIN CONTROL POLICIAL", size=15)
plt.legend()
plt.show()
