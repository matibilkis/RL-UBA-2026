# %% Cliff Walking: el entorno (S&B Ejemplo 6.6)
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np


# ###
# Grilla de 4x12, 48 estados numerados por filas: estado = fila * 12 + columna.
#
#   0  1  2 ...                 11
#  12 13 14 ...                 23
#  24 25 26 ...                 35
#  36 [37 38 ... cliff ... 46]  47        S = 36 (inicio), G = 47 (meta)
#
# - Cada paso da reward -1.
# - Si caés al cliff (37..46): reward -100 y volvés a S. El episodio NO termina.
# - Acciones: 0 = arriba, 1 = derecha, 2 = abajo, 3 = izquierda.
# ###

env = gym.make("CliffWalking-v1", render_mode="rgb_array")  # v1: en gymnasium 1.3 la v0 está deprecada
n_states, n_actions = env.observation_space.n, env.action_space.n
env.reset(seed=0)
plt.imshow(env.render())


plt.axis("off")
plt.show()


# %% Sarsa y Q-learning: el mismo código, cambia solo la acción del target
def epsilon_greedy(Q, state, epsilon, rng):
    if rng.random() < epsilon:
        return int(rng.integers(n_actions))
    best = np.flatnonzero(Q[state] == Q[state].max())  # empates: al azar
    return int(rng.choice(best))


def train(algorithm, n_episodes=500, epsilon=0.1, alpha=0.5, gamma=1.0, seed=0, decay_alpha=False):
    rng = np.random.default_rng(seed)
    Q = np.zeros((n_states, n_actions))
    returns = []
    for episode in range(n_episodes):
        # alpha constante (como en el libro), o decreciente para que Q asiente
        step_size = alpha * (1 - episode / n_episodes) + 0.01 if decay_alpha else alpha
        state, info = env.reset(seed=int(rng.integers(1 << 30)))
        action = epsilon_greedy(Q, state, epsilon, rng)
        G = 0
        while True:
            next_state, reward, terminated, truncated, info = env.step(action)
            G += reward
            next_action = epsilon_greedy(Q, next_state, epsilon, rng)

            if algorithm == "sarsa":
                target = Q[next_state, next_action]  # la acción que realmente va a tomar b
            else:
                target = Q[next_state].max()  # la greedy
            if terminated:
                target = 0
            Q[state, action] += step_size * (reward + gamma * target - Q[state, action])

            state, action = next_state, next_action
            if terminated or truncated:
                break
        returns.append(G)
    return Q, np.array(returns)


# %% Retorno durante el entrenamiento (promedio de 50 seeds): Sarsa vs Q-learning
mean_returns = {}
for algorithm in ["sarsa", "qlearning"]:
    runs = [train(algorithm, seed=seed)[1] for seed in range(50)]
    mean_returns[algorithm] = np.mean(runs, axis=0)
    print(algorithm, "retorno promedio, últimos 100 episodios:", round(mean_returns[algorithm][-100:].mean(), 1))

for algorithm, color in [("sarsa", "tab:blue"), ("qlearning", "tab:red")]:
    smooth = np.convolve(mean_returns[algorithm], np.ones(10) / 10, mode="valid")
    plt.plot(smooth, color=color, label=algorithm)
plt.ylim(-100, 0)
plt.xlabel("episodio")
plt.ylabel("retorno durante el episodio")
plt.legend()
plt.show()


# %% Camino de la política greedy (sin exploración) que aprendió cada uno
# alpha decreciente: con alpha constante Q queda ruidoso y la greedy a veces se traba contra una pared
for algorithm in ["sarsa", "qlearning"]:
    Q, _ = train(algorithm, n_episodes=3000, decay_alpha=True)
    state, info = env.reset()
    path = [state]
    for _ in range(60):
        state, reward, terminated, truncated, info = env.step(int(Q[state].argmax()))
        path.append(state)
        if terminated:
            break

    grid = np.full(n_states, ".", dtype="<U1")
    grid[37:47] = "C"
    grid[path] = "*"
    grid[36], grid[47] = "S", "G"
    print(f"\n{algorithm}: {len(path) - 1} pasos")
    print("\n".join("".join(row) for row in grid.reshape(4, 12)))
