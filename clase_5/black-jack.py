# %% Blackjack (S&B Ejemplo 5.1): MC prediction de la policy "pedir hasta 20"
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("Blackjack-v1", sab=True)     # sab=True: reglas exactas de Sutton & Barto
print(env.observation_space)                 # (suma del jugador, carta visible del dealer, ¿as usable?)
print(env.action_space)                      # 0 = stick (plantarse), 1 = hit (pedir)
print(hasattr(env.unwrapped, "P"))           # False: acá no hay p(s',r|s,a) escrita a mano

# %% La policy a evaluar
def policy(obs):
    player_sum, dealer_card, usable_ace = obs
    return 0 if player_sum >= 20 else 1

# %% Una mano
obs, info = env.reset(seed=18)
done = False
while not done:
    a = policy(obs)
    obs_next, r, terminated, truncated, info = env.step(a)
    print(obs, "hit " if a else "stick", "->", obs_next, "reward", r)
    obs, done = obs_next, terminated or truncated

# %% First-visit MC prediction (acá = every-visit: no hay loops)
def mc_prediction(env, policy, n_episodes, gamma=1.0):
    V = np.zeros((32, 11, 2))
    N = np.zeros((32, 11, 2))
    for _ in range(n_episodes):
        episode = []                                    # 1) jugar una mano con la policy
        obs, info = env.reset()
        done = False
        while not done:
            obs_next, r, terminated, truncated, info = env.step(policy(obs))
            episode.append((obs, r))
            obs, done = obs_next, terminated or truncated
        G = 0                                           # 2) hacia atrás: G_t = R_{t+1} + gamma G_{t+1}
        for obs, r in reversed(episode):
            G = r + gamma * G
            N[obs] += 1
            V[obs] += (G - V[obs]) / N[obs]             # promedio incremental (Clase 2)
    return V

V_10k = mc_prediction(env, policy, 10_000)
V_500k = mc_prediction(env, policy, 500_000)            # ~1 min

# %% Figura 5.1 de S&B
player, dealer = np.meshgrid(np.arange(12, 22), np.arange(1, 11), indexing="ij")
fig = plt.figure(figsize=(10, 8))
for i, (V, title) in enumerate([(V_10k, "10.000 episodios"), (V_500k, "500.000 episodios")]):
    for j, ace in enumerate([1, 0]):
        ax = fig.add_subplot(2, 2, 2 * j + i + 1, projection="3d")
        ax.plot_surface(dealer, player, V[12:22, 1:11, ace], cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_zlim(-1, 1)
        ax.set_title(f"{title} — {'con' if ace else 'sin'} as usable")
        ax.set_xlabel("Carta del dealer")
        ax.set_ylabel("Suma del jugador")
plt.show()


#### graficando
import imageio

env = gym.make("Blackjack-v1", render_mode="rgb_array")
obs, info = env.reset(seed=18)
frames = [env.render()]
done = False
while not done:
    obs, r, terminated, truncated, info = env.step(0 if obs[0] >= 20 else 1)
    frames.append(env.render())
    done = terminated or truncated
imageio.mimsave("/home/mbilkis/reluba/experiments/clase5/data/blackjack.gif", frames, duration=1000, loop=0)   # 1 s por frame
