# %% First-visit MC control: la misma política epsilon-greedy actúa y aprende.
import os

import gymnasium as gym
import imageio
import matplotlib.pyplot as plt
import numpy as np

environment = "FrozenLake-v1"  # cambiar por "Taxi-v4"
options = {"is_slippery": False} if environment == "FrozenLake-v1" else {}
# Sin TimeLimit: aprendemos con episodios COMPLETOS, no retornos cortados.
env = gym.make(environment, max_episode_steps=-1, **options)
env.reset(seed=0)
rng = np.random.default_rng(0)
n_states, n_actions = env.observation_space.n, env.action_space.n
Q = np.zeros((n_states, n_actions))
N = np.zeros_like(Q)
pi = np.ones_like(Q) / n_actions
gamma = 0.95
epsilon = 0.2 if environment == "FrozenLake-v1" else 0.5
n_episodes = 5_000 if environment == "FrozenLake-v1" else 500
# Taxi necesita más exploración; incluso 500 episodios pueden tardar un minuto.
scores = []

# %% Generar un episodio, calcular sus retornos y promediar primeras visitas.
for _ in range(n_episodes):
    state, info = env.reset()
    episode = []
    while True:
        action = rng.choice(n_actions, p=pi[state])
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append((state, action, reward))
        state = next_state
        if terminated:
            break

    returns = np.zeros(len(episode))
    G = 0
    for t in reversed(range(len(episode))):
        G = episode[t][2] + gamma * G
        returns[t] = G
    seen = np.zeros_like(Q, dtype=bool)
    for (state, action, reward), G in zip(episode, returns):
        if not seen[state, action]:  # primera visita en orden cronológico
            seen[state, action] = True
            N[state, action] += 1
            Q[state, action] += (G - Q[state, action]) / N[state, action]
            best = rng.choice(np.flatnonzero(Q[state] == Q[state].max()))
            pi[state] = epsilon / n_actions
            pi[state, best] += 1 - epsilon
    scores.append(sum(r for s, a, r in episode))

# %% Ver el aprendizaje y evaluar por separado la política greedy.
plt.plot(np.convolve(scores, np.ones(100) / 100, mode="valid"))
plt.xlabel("Episodio (ventana de 100)")
plt.ylabel("Reward total promedio")
plt.title(f"On-policy MC / {environment}")
plt.show()
successes = 0
for _ in range(100):
    state, info = env.reset()
    for step in range(200):  # límite SOLO para evaluar posibles bucles greedy
        state, reward, terminated, truncated, info = env.step(int(Q[state].argmax()))
        if terminated:
            successes += reward > 0
            break
print(f"Greedy: {successes}/100 éxitos en hasta 200 pasos")
if environment == "FrozenLake-v1":
    policy = np.array(["←", "↓", "→", "↑"])[Q.argmax(axis=1)].reshape(4, 4)
    tiles = env.unwrapped.desc.astype(str)
    policy[(tiles == "H") | (tiles == "G")] = tiles[(tiles == "H") | (tiles == "G")]
    print(policy)
env.close()

# %% GIF: la política greedy aprendida, jugando un episodio.
render_env = gym.make(environment, render_mode="rgb_array", **options)
state, info = render_env.reset(seed=0)
frames = [render_env.render()]
for step in range(200):  # mismo límite que en la evaluación
    state, reward, terminated, truncated, info = render_env.step(int(Q[state].argmax()))
    frames.append(render_env.render())
    if terminated:
        break
path = f"{environment}_greedy.gif"
imageio.mimsave(path, frames, duration=1000 / render_env.metadata["render_fps"], loop=0)
print("GIF guardado en", os.path.abspath(path))
render_env.close()
