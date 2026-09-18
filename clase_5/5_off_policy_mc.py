# %% Off-policy MC control: target greedy, behavior epsilon-greedy; weighted IS.
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

environment = "FrozenLake-v1"  # cambiar por "Taxi-v4"
options = {"is_slippery": False} if environment == "FrozenLake-v1" else {}
env = gym.make(environment, max_episode_steps=-1, **options)
env.reset(seed=0)
rng = np.random.default_rng(0)
n_states, n_actions = env.observation_space.n, env.action_space.n
Q = np.zeros((n_states, n_actions))
C = np.zeros_like(Q)  # suma de pesos; reemplaza al conteo N
pi = Q.argmax(axis=1)  # target determinista; empates siempre al primer máximo
epsilon, gamma = 0.2, 0.95
n_episodes = 5_000  # Taxi puede tardar más; episodios completos, sin TimeLimit.
scores = []

# %% Guardamos también b(a|s): debe ser la probabilidad al generar la acción.
for _ in range(n_episodes):
    state, info = env.reset()
    episode = []
    while True:
        best = Q[state] == Q[state].max()
        b = epsilon / n_actions + (1 - epsilon) * best / best.sum()
        action = rng.choice(n_actions, p=b)
        next_state, reward, terminated, truncated, info = env.step(action)
        episode.append((state, action, reward, b[action]))
        state = next_state
        if terminated:
            break

    G, W = 0, 1
    for state, action, reward, probability in reversed(episode):
        G = reward + gamma * G
        C[state, action] += W
        Q[state, action] += W / C[state, action] * (G - Q[state, action])
        pi[state] = Q[state].argmax()
        if action != pi[state]:  # pi(a|s)=0: el resto del prefijo tiene peso 0
            break
        W /= probability  # pi(a|s)=1; para Q no incluimos la acción actual
    scores.append(sum(r for s, a, r, p in episode))

# %% La curva corresponde a b; la evaluación sigue pi sin exploración.
plt.plot(np.convolve(scores, np.ones(100) / 100, mode="valid"))
plt.xlabel("Episodio (ventana de 100)")
plt.ylabel("Reward total promedio de b")
plt.title(f"Off-policy MC / {environment}")
plt.show()
successes = 0
for _ in range(100):
    state, info = env.reset()
    for step in range(200):  # límite de evaluación, no de entrenamiento
        state, reward, terminated, truncated, info = env.step(int(pi[state]))
        if terminated:
            successes += reward > 0
            break
print(f"Target greedy: {successes}/100 éxitos en hasta 200 pasos")
if environment == "FrozenLake-v1":
    policy = np.array(["←", "↓", "→", "↑"])[pi].reshape(4, 4)
    tiles = env.unwrapped.desc.astype(str)
    policy[(tiles == "H") | (tiles == "G")] = tiles[(tiles == "H") | (tiles == "G")]
    print(policy)
env.close()
