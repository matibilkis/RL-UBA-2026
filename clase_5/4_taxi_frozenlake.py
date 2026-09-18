# %% FrozenLake y Taxi: lo básico de cada entorno
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

def random_episode(env):
    obs, info = env.reset()
    G, T, done = 0, 0, False
    while not done:
        obs, r, terminated, truncated, info = env.step(env.action_space.sample())
        G, T = G + r, T + 1
        done = terminated or truncated
    return G, T, truncated

# %% FrozenLake: S inicio, F hielo, H agujero (termina), G meta (reward 1). Todo lo demás: reward 0
env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=True, render_mode="rgb_array")
print(env.observation_space, env.action_space)           # Discrete(16) Discrete(4): 0 ←, 1 ↓, 2 →, 3 ↑
print("time limit:", env.spec.max_episode_steps)
obs, info = env.reset(seed=0)
plt.imshow(env.render())




# %% El modelo está a la vista: P[s][a] = [(prob, s', r, terminated), ...]
print(env.unwrapped.P[14][2])                             # al lado de la meta, "→": 1/3 de llegar
print(gym.make("FrozenLake-v1", is_slippery=False).unwrapped.P[14][2])

# %% Policy random: ¿cuántas veces llega?
for slippery in [True, False]:
    env = gym.make("FrozenLake-v1", is_slippery=slippery)
    G = [random_episode(env)[0] for _ in range(10_000)]
    print(f"is_slippery={slippery}: llega a la meta {np.mean(G):.1%} de las veces")



# %% Taxi: buscar al pasajero y dejarlo en su destino (R, G, Y, B)
env = gym.make("Taxi-v4", render_mode="ansi")            # ojo: Taxi-v3 ya no existe en gymnasium 1.3
print(env.observation_space, env.action_space)           # 500 = 25 lugares x 5 (pasajero) x 4 (destino)
print("acciones: 0 sur, 1 norte, 2 este, 3 oeste, 4 levantar, 5 dejar")
print("time limit:", env.spec.max_episode_steps)
obs, info = env.reset(seed=0)
print(env.render())
print("decode:", list(env.unwrapped.decode(obs)))       # [fila taxi, col taxi, pasajero, destino]
print("info:", info)                                     # action_mask: qué acciones cambian algo

# %% Rewards: -1 por paso, +20 por dejar bien al pasajero, -10 por levantar/dejar mal
print(env.unwrapped.P[obs][5])                           # "dejar" sin pasajero: -10

# %% Policy random: casi nunca termina, la corta el time limit
G, T, truncated = np.array([random_episode(env) for _ in range(1_000)]).T
print(f"retorno medio {G.mean():.0f} | largo medio {T.mean():.0f} | truncados {truncated.mean():.0%}")

# %% Los mismos entornos, en dibujito (render_mode="rgb_array", necesita pygame)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (name, kw) in zip(axes, [("FrozenLake-v1", {}), ("FrozenLake-v1", {"map_name": "8x8"}), ("Taxi-v4", {})]):
    env = gym.make(name, render_mode="rgb_array", **kw)
    env.reset(seed=0)
    ax.imshow(env.render())
    ax.set_title(f"{name} {kw.get('map_name', '')}")
    ax.axis("off")
plt.show()
