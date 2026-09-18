# %% Lo mismo que en 1D: sortear incrementos y acumularlos con cumsum.
import matplotlib.pyplot as plt
import numpy as np

rng = np.random.default_rng(0)
n_walks, n_steps = 8, 2000
colors = ["#00e5ff", "#ff4fa3", "#b9ff44", "#ffb347",
          "#ad8cff", "#fff06a", "#ff6262", "#51ffbd"]

increments.shape

# %% Cada coordenada da un paso +1 o -1: caminata diagonal en la grilla.
with plt.style.context("dark_background"):
    fig = plt.figure(figsize=(14, 6), facecolor="black")
    for panel, dimension in enumerate((2, 3), start=1):
        increments = rng.choice((-1, 1), size=(n_walks, n_steps, dimension))
        walks = np.zeros((n_walks, n_steps + 1, dimension))
        walks[:, 1:] = increments.cumsum(axis=1)
        ax = fig.add_subplot(1, 2, panel, projection="3d" if dimension == 3 else None)
        for walk, color in zip(walks, colors):
            ax.plot(*walk.T, color=color, lw=5, alpha=0.06)  # halo
            ax.plot(*walk.T, color=color, lw=0.9, alpha=0.9)
            ax.scatter(*walk[-1], color=color, s=25)  # punto final
        ax.scatter(*np.zeros(dimension), color="white", s=70, marker="*")
        ax.set(title=f"Random walks / {dimension}D", xlabel="x", ylabel="y")
        if dimension == 2:
            ax.set_aspect("equal")
            ax.grid(alpha=0.1)
        else:
            ax.set_zlabel("z")
            ax.set_box_aspect(np.ptp(walks, axis=(0, 1)))
            ax.view_init(elev=22, azim=-60)
            ax.grid(False)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.pane.fill = False
    fig.tight_layout()

plt.savefig("walks.png")
import os
os.getcwd()
