# %% El gridworld del Gólem (Clase 4) como gymnasium.Env
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

GOLEM = plt.imread("golem.png")
MOVES = np.array([(-1, 0), (1, 0), (0, -1), (0, 1)])   # 0 ↑, 1 ↓, 2 ←, 3 →

# %% Arranca arriba a la izquierda, termina abajo a la izquierda, -1 por paso.
# Chocar contra un borde o una pared = quedarse quieto (y pagar el -1 igual)
class GridWorld(gym.Env):
    def __init__(self, N=5, walls=((2, 0), (2, 1), (2, 2))):     # walls=() es el de Clase 4
        self.N, self.walls = N, set(walls)
        self.start, self.goal = (0, 0), (N - 1, 0)
        self.observation_space = spaces.Discrete(N * N)
        self.action_space = spaces.Discrete(4)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.start
        self.path = [self.pos]
        return self.pos[0] * self.N + self.pos[1], {}

    def step(self, action):
        r, c = np.add(self.pos, MOVES[action])
        if 0 <= r < self.N and 0 <= c < self.N and (r, c) not in self.walls:
            self.pos = (int(r), int(c))
        self.path.append(self.pos)
        terminated = self.pos == self.goal
        return self.pos[0] * self.N + self.pos[1], -1.0, terminated, False, {}

    def render(self):
        img = np.ones((self.N, self.N, 3))                        # blanco
        for w in self.walls:
            img[w] = 0.3                                          # paredes grises
        img[self.goal] = (0.6, 1, 0.6)                            # meta verde
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img)
        ax.set_xticks(np.arange(self.N + 1) - 0.5)
        ax.set_yticks(np.arange(self.N + 1) - 0.5)
        ax.tick_params(labelbottom=False, labelleft=False, length=0)
        ax.grid(color="k")
        rows, cols = np.array(self.path).T
        ax.plot(cols, rows, "o-", color="royalblue", alpha=0.6)
        ax.add_artist(AnnotationBbox(OffsetImage(GOLEM, zoom=0.12), self.pos[::-1], frameon=False))
        plt.show()

# %%
env = GridWorld()
print(env.observation_space, env.action_space)
env.reset()
env.render()

# %% register + make: ahora es un entorno más, con time limit incluido
gym.register(id="GolemGrid-v0", entry_point=GridWorld, max_episode_steps=50)
env = gym.make("GolemGrid-v0")
print(env)
check_env(env.unwrapped)                                          # ¿cumple el contrato de Gymnasium?

# %% Un episodio con policy random
obs, info = env.reset(seed=0)
G, done = 0, False
while not done:
    obs, r, terminated, truncated, info = env.step(env.action_space.sample())
    G += r
    done = terminated or truncated
print("retorno", G, "| terminated", terminated, "| truncated", truncated)
env.unwrapped.render()
