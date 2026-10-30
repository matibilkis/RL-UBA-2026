# %% 0. Clase 9 · práctica · 4 (paso 2) — + experience replay: Q-learning con red y un buffer de transiciones en Cliff Walking
# Lo mismo que 2_DQN-cliff-1-online.py (MLP one-hot(48) -> 64 -> 64 -> 4, SGD, el target sale de la MISMA red con stop_gradient,
# sin red objetivo), con UN cambio: en vez de actualizar con la transición que acaba de pasar, guardamos cada transición en un
# buffer y actualizamos con un minibatch de transiciones al azar de todo el buffer.
# Cota: todas las recompensas son <= 0, entonces Q <= 0. Un Q positivo es imposible; si crece, algo se rompió.
import pathlib
import numpy as np
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import tqdm


FIGS = pathlib.Path.cwd() / "figs"   # correr desde clase_9/practica/ (no __file__: falla celda por celda en Hydrogen)
FIGS.mkdir(exist_ok=True)


def guardar(fig, nombre):
    """Guarda la figura en figs/ y la vuelve a abrir desde el archivo, para ver lo que quedó guardado."""
    ruta = FIGS / nombre
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    print("guardada:", ruta)
    plt.figure(figsize=fig.get_size_inches())
    plt.imshow(plt.imread(ruta))
    plt.axis("off")
    plt.show()


# %% 1. El entorno (el mismo de 2_QL-cliff.py): estado 0..47, acciones 0 arriba, 1 derecha, 2 abajo, 3 izquierda
env = gym.make("CliffWalking-v1", is_slippery=False, max_episode_steps=200)
env_eval = gym.make("CliffWalking-v1", is_slippery=False, max_episode_steps=200)   # para medir sin mover el de entrenamiento
GAMMA = 0.99

# %% 2. La red: Q(s,·;θ) = MLP(one_hot(s)), 48 -> 64 -> 64 -> 4. Con one-hot puede representar la tabla exacta.
def iniciar(key, tamanos=(48, 64, 64, 4)):
    return [(jax.random.normal(k, (n, m)) * jnp.sqrt(2.0 / n), jnp.zeros(m))
            for k, n, m in zip(jax.random.split(key, len(tamanos) - 1), tamanos[:-1], tamanos[1:])]


@jax.jit
def q_red(params, s):
    x = jax.nn.one_hot(s, 48)
    for W, b in params[:-1]:
        x = jax.nn.relu(x @ W + b)
    W, b = params[-1]
    return x @ W + b           # los 4 Q(s,a) juntos (para un estado, o para un lote de estados)


accion_greedy = jax.jit(lambda params, s: jnp.argmax(q_red(params, s)))

# %% 3. El replay buffer: cinco arreglos circulares (s, a, r, s', terminó). Cada transición se guarda una vez y se usa en muchos updates
BUFFER = 40_000     # capacidad: cuando se llena, la transición nueva pisa la más vieja
BATCH = 64          # transiciones por update
EMPEZAR = 1_000     # no se actualiza hasta tener esta cantidad de transiciones guardadas
CADA = 4            # un update cada tantos steps de entorno (como la Tabla 1 de Mnih)


def nueva_memoria():
    """El buffer: cinco arrayhs de largo BUFFER."""
    return {"s": np.zeros(BUFFER, np.int32),
            "a": np.zeros(BUFFER, np.int32),
            "r": np.zeros(BUFFER, np.float32),
            "s2": np.zeros(BUFFER, np.int32),
            "terminó": np.zeros(BUFFER, np.float32)}


def guardar_transicion(memoria, step, transicion):
    """Guarda la transición del step `step` en el lugar i.
    Si el buffer se llena, i vuelve a 0 y pisa la más vieja."""
    s, a, r, s2, terminó = transicion
    i = (step - 1) % BUFFER
    memoria["s"][i] = s
    memoria["a"][i] = a
    memoria["r"][i] = r
    memoria["s2"][i] = s2
    memoria["terminó"][i] = terminó


def muestrear(memoria, step, rng, batch=BATCH):
    """`batch` transiciones al azar entre las que hay guardadas (los primeros min(step, BUFFER) lugares)."""
    i = rng.integers(min(step, BUFFER), size=batch)
    return memoria["s"][i], memoria["a"][i], memoria["r"][i], memoria["s2"][i], memoria["terminó"][i]


# %% 4. La pérdida de un MINIBATCH: promedio de (y − Q(s,a;θ))², con y = r + γ max_a' Q(s',a';θ). Sigue sin red objetivo
def costo(params, s, a, r, s2, terminó):
    y = r + jnp.where(terminó, 0.0, GAMMA * q_red(params, s2).max(axis=1))   # el target sale de la misma red
    y = jax.lax.stop_gradient(y)                                             # semi-gradiente: no se deriva y
    q = q_red(params, s)[jnp.arange(s.shape[0]), a]
    return jnp.mean((y - q) ** 2)


# %% 5. Un update de SGD: θ ← θ − η ∇θ L
LR = 1e-3     # η. Si lo cambiás, volvé a correr esta celda: jit compila `actualizar` con el valor de LR que tenía la primera vez

@jax.jit
def actualizar(params, minibatch):
    """actualización gradient descent"""
    grad = jax.grad(costo)(params, *minibatch)
    return jax.tree_util.tree_map(lambda p, g: p - LR * g, params, grad)


# %% 6. Cómo medimos: retorno de la política greedy (ε = 0) y el Q más alto del mapa
def evaluar(params):
    """
    acá nos fijamos cuál es el valor más alto de Q, para ver si la red predice
    valores que no pueden ser (en cliff ningún Q puede ser más grande que -1),
    pero de la manera que estamos entrenando puede diverger incluso NaN!
    """
    Q = np.asarray(q_red(params, jnp.arange(48)))
    s, _ = env_eval.reset()
    retorno, fin = 0.0, False
    while not fin:
        s, r, terminó, truncado, _ = env_eval.step(int(Q[s].argmax()))
        retorno += r
        fin = terminó or truncado
    return retorno, Q.max()


# %% 7. Entrenar: 40 000 steps, ε de 1 a 0,1 en 20 000 steps, un update cada 4 steps con un minibatch del buffer. Se corta en el primer NaN
PASOS = 40_000
EVAL_CADA = 500
SEMILLA = 0
rng = np.random.default_rng(SEMILLA)
params = iniciar(jax.random.PRNGKey(SEMILLA))
memoria = nueva_memoria()
s, _ = env.reset(seed=SEMILLA)
log = []   # (step, retorno greedy, Q más alto)
for step in tqdm(range(1, PASOS + 1)):
    eps = max(0.1, 1.0 - 0.9 * (step - 1) / 20_000)
    a = int(rng.integers(4)) if rng.random() < eps else int(accion_greedy(params, s))
    s2, r, terminó, truncado, _ = env.step(a)
    guardar_transicion(memoria, step, (s, a, r, s2, terminó))   # truncado (corte a 200 pasos) NO corta el bootstrap, terminó sí
    s = s2
    if terminó or truncado:
        s, _ = env.reset()
    if step >= EMPEZAR and step % CADA == 0:
        params = actualizar(params, muestrear(memoria, step, rng))
    if step % EVAL_CADA == 0:
        log.append((step, *evaluar(params)))
        if not np.isfinite(log[-1][2]):
            print("cortamos")
            break

log = np.array(log)
finito = np.isfinite(log[:, 2])
print("Q más alto: pico", log[finito, 2].max().round(1), " al final", log[-1, 2].round(2), "  (cota: 0)")
print("Q = NaN en el step", int(log[-1, 0])) if not finito[-1] else print("no se rompió; retorno greedy al final:", log[-1, 1])

# %% 8. Ver la figura: el Q más alto contra la cota 0, y cómo juega la política greedy
fig_replay, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
ejes[0].plot(log[:, 0], log[:, 2], "o-", ms=4)
ejes[0].axhline(0, color="gray", ls="--", label="cota: 0")
if not finito[-1]:
    ejes[0].axvline(log[-1, 0], color="tab:red", ls=":", label=f"NaN en el step {int(log[-1, 0])}")
ejes[0].set_yscale("symlog", linthresh=1)   # lineal entre -1 y 1, logarítmica afuera: se ve la cota 0 y los valores enormes
ejes[0].set(xlabel="steps de entorno", ylabel="Q más alto del mapa", title="¿Se pasa de la cota?")
ejes[0].legend()
ejes[1].plot(log[:, 0], np.maximum(log[:, 1], -200), "o-", ms=4, color="C1")   # caer 200 veces daría −20 000: recortamos en −200
ejes[1].axhline(-13, color="gray", ls="--", label="óptimo: −13")
ejes[1].set(xlabel="steps de entorno", ylabel="retorno greedy (recortado en −200)", ylim=(-210, 5), title="Cómo juega")
ejes[1].legend()
fig_replay.suptitle(f"Paso 2: + replay (buffer {BUFFER}, minibatch {BATCH}), el target de la misma red (SGD, η = {LR:g})")
fig_replay.tight_layout()
plt.show()

# %% 8. Guardar la figura
guardar(fig_replay, "4_dqn_cliff_2_replay.png")


# %% 9. Mostrando divergencia con MB = 1: el mismo buffer, pero cada update usa UNA sola transición (sacada al azar de todo el buffer)
# Decorrelamos los datos pero no promediamos: igual que en el paso 1 (online), cada update depende de una transición,
# y un solo error grande (la caída de −100) puede mover todo el mapa. Un update por step, como en el paso 1.
MB = 1
CADA_MB1 = 1
rng_mb1 = np.random.default_rng(SEMILLA)
params_mb1 = iniciar(jax.random.PRNGKey(SEMILLA))
memoria_mb1 = nueva_memoria()
s_mb1, _ = env.reset(seed=SEMILLA)
log_mb1 = []   # (step, retorno greedy, Q más alto)
for step in tqdm(range(1, PASOS + 1)):
    eps = max(0.1, 1.0 - 0.9 * (step - 1) / 20_000)
    a = int(rng_mb1.integers(4)) if rng_mb1.random() < eps else int(accion_greedy(params_mb1, s_mb1))
    s2, r, terminó, truncado, _ = env.step(a)
    guardar_transicion(memoria_mb1, step, (s_mb1, a, r, s2, terminó))
    s_mb1 = s2
    if terminó or truncado:
        s_mb1, _ = env.reset()
    if step >= EMPEZAR and step % CADA_MB1 == 0:
        params_mb1 = actualizar(params_mb1, muestrear(memoria_mb1, step, rng_mb1, batch=MB))
    if step % EVAL_CADA == 0:
        log_mb1.append((step, *evaluar(params_mb1)))
        if not np.isfinite(log_mb1[-1][2]):
            print("cortamos")
            break

log_mb1 = np.array(log_mb1)
finito_mb1 = np.isfinite(log_mb1[:, 2])
print("Q más alto antes de romper: pico", log_mb1[finito_mb1, 2].max().round(1), "  (cota: 0)")
print(f"Q = NaN detectado en la evaluación del step {int(log_mb1[-1, 0])} (se rompió entre el {int(log_mb1[-1, 0]) - EVAL_CADA} y el {int(log_mb1[-1, 0])})") if not finito_mb1[-1] else print("no se rompió; retorno greedy al final:", log_mb1[-1, 1])

# %% 10. Ver la figura (MB = 1): el Q más alto contra la cota 0, y cómo juega la política greedy
fig_mb1, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
ejes[0].plot(log_mb1[:, 0], log_mb1[:, 2], "o-", ms=4)
ejes[0].axhline(0, color="gray", ls="--", label="cota: 0")
if not finito_mb1[-1]:
    ejes[0].axvline(log_mb1[-1, 0], color="tab:red", ls=":", label=f"NaN antes del step {int(log_mb1[-1, 0])}")
ejes[0].set_yscale("symlog", linthresh=1)   # lineal entre -1 y 1, logarítmica afuera: se ve la cota 0 y los valores enormes
ejes[0].set(xlabel="steps de entorno", ylabel="Q más alto del mapa", title="¿Se pasa de la cota?")
ejes[0].legend()
ejes[1].plot(log_mb1[:, 0], np.maximum(log_mb1[:, 1], -200), "o-", ms=4, color="C1")   # caer 200 veces daría −20 000: recortamos en −200
ejes[1].axhline(-13, color="gray", ls="--", label="óptimo: −13")
ejes[1].set(xlabel="steps de entorno", ylabel="retorno greedy (recortado en −200)", ylim=(-210, 5), title="Cómo juega")
ejes[1].legend()
fig_mb1.suptitle(f"Paso 2, mostrando divergencia con MB = {MB}: buffer {BUFFER}, un update por step (SGD, η = {LR:g})")
fig_mb1.tight_layout()
plt.show()

# %% 10. Guardar la figura (MB = 1)
guardar(fig_mb1, "4_dqn_cliff_2_replay_mb1.png")
