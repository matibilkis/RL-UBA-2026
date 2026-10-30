# %% 0. Clase 9 · práctica · 2 (paso 1) — la tabla pasa a ser una red: Q-learning online en Cliff Walking
# Misma regla que 2_QL-cliff.py, pero Q(s,a;θ) es un MLP: entrada one-hot (48) -> 64 -> 64 -> 4 salidas (una por acción).
# Una transición, un update de SGD, y el target sale de la MISMA red (con stop_gradient). Sin replay y sin red objetivo.
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
    return x @ W + b           # los 4 Q(s,a) juntos

##la función del jax.jit es "just in time", y es importante para acelerar!
accion_greedy = jax.jit(lambda params, s: jnp.argmax(q_red(params, s)))
print("Q(inicio) con pesos al azar:", q_red(iniciar(jax.random.PRNGKey(0)), 36))

# %% 3. La pérdida de UNA transición: (y − Q(s,a;θ))², con y = r + γ max_a' Q(s',a';θ)
def costo(params, s, a, r, s2, terminó):
    y = r + jnp.where(terminó, 0.0, GAMMA * q_red(params, s2).max())   # el target sale de la misma red
    y = jax.lax.stop_gradient(y)                                       # semi-gradiente: no se deriva y
    return (y - q_red(params, s)[a]) ** 2


# %% 4. Un update de SGD: θ ← θ − η ∇θ L
LR = 1e-3     # η. Si lo cambiás, volvé a correr esta celda: jit compila `actualizar` con el valor de LR que tenía la primera vez


@jax.jit
def actualizar(params, transicion):
    """actualización gradient descent"""
    grad = jax.grad(costo)(params, *transicion)
    return jax.tree_util.tree_map(lambda p, g: p - LR * g, params, grad)


# %% 5. Cómo medimos: retorno de la política greedy (ε = 0) y el Q más alto del mapa
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



# %% 6. Entrenar: hasta 500 steps, ε de 1 a 0,1 en 20 000 steps, un update por step. Se corta en el primer NaN
PASOS = 500
EVAL_CADA = 1     # evaluar en cada step: cuesta ≈ 3 ms (una política completa); con 500 steps son ~2 s. Con 40 000 steps usá 500
SEMILLA = 0
rng = np.random.default_rng(SEMILLA)
params = iniciar(jax.random.PRNGKey(SEMILLA))
s, _ = env.reset(seed=SEMILLA)
log = []   # (step, retorno greedy, Q más alto)
for step in tqdm(range(1, PASOS + 1)):
    eps = max(0.1, 1.0 - 0.9 * (step - 1) / 20_000) ### reducimos el epsilon
    a = int(rng.integers(4)) if rng.random() < eps else int(accion_greedy(params, s))
    s2, r, terminó, truncado, _ = env.step(a)
    params = actualizar(params, (s, a, r, s2, terminó))
    s = s2
    if terminó or truncado:
        s, _ = env.reset()
    if step % EVAL_CADA == 0:
        log.append((step, *evaluar(params)))
        if not np.isfinite(log[-1][2]):
            print("cortamos")
            break

log = np.array(log)
finito = np.isfinite(log[:, 2])
print("Q más alto antes de romper: pico", log[finito, 2].max().round(1), "  (cota: 0)")
print("Q = NaN en el step", int(log[-1, 0])) if not finito[-1] else print("no se rompió; retorno greedy al final:", log[-1, 1])

# %% 7. Ver la figura: el Q más alto contra la cota 0, y cómo juega la política greedy
fig_online, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
ejes[0].plot(log[:, 0], log[:, 2], "o-", ms=4)
ejes[0].axhline(0, color="gray", ls="--", label="cota: 0")
if not finito[-1]:
    ejes[0].axvline(log[-1, 0], color="tab:red", ls=":", label=f"NaN en el step {int(log[-1, 0])}")
ejes[0].set_yscale("symlog", linthresh=1)   # lineal entre -1 y 1, logarítmica afuera: se ve la cota 0 y los 10^10
ejes[0].set(xlabel="steps de entorno", ylabel="Q más alto del mapa", ylim=(-1, None), title="¿Se pasa de la cota?")
ejes[0].legend()
ejes[1].plot(log[:, 0], np.maximum(log[:, 1], -200), "o-", ms=4, color="C1")   # caer 200 veces daría −20 000: recortamos en −200
ejes[1].axhline(-13, color="gray", ls="--", label="óptimo: −13")
ejes[1].set(xlabel="steps de entorno", ylabel="retorno greedy (recortado en −200)", ylim=(-210, 5), title="Cómo juega")
ejes[1].legend()
fig_online.suptitle(f"Paso 1: red con un update por transición y el target de la misma red (SGD, η = {LR:g})")
fig_online.tight_layout()
plt.show()

# %% 7. Guardar la figura
guardar(fig_online, "2_dqn_cliff_1_online.png")
