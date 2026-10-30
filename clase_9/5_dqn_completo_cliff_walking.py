# %% 0. Clase 9 · práctica · 5 (paso 3) — DQN completo en Cliff Walking: MLP + replay buffer + red objetivo
# Es 4-DQL-CLIFF-REPLAY.py (paso 2: MLP one-hot(48) -> 64 -> 64 -> 4, SGD, replay buffer) con UN cambio: el target
# y = r + γ max_a' Q(s',a') ya no sale de la red que estamos entrenando (θ) sino de una COPIA congelada de ella (θ⁻),
# que se actualiza cada tanto. Mismos hiperparámetros y misma semilla que el paso 2; lo único nuevo es COPIAR.
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


# %% 1. Hiperparámetros: los del paso 2, más COPIAR. Si cambiás alguno, volvé a correr todo desde acá (jit se queda con GAMMA y LR)
GAMMA = 0.99
LR = 1e-3                          # η de SGD
PASOS = 40_000                     # steps de entorno
EPS_INICIAL, EPS_FINAL = 1.0, 0.1  # ε baja linealmente de uno al otro...
EPS_PASOS = 20_000                 # ...en esta cantidad de steps
BUFFER = 40_000                    # capacidad del replay buffer: cuando se llena, la transición nueva pisa la más vieja
BATCH = 64                         # transiciones por update
EMPEZAR = 1_000                    # no se actualiza hasta tener esta cantidad de transiciones guardadas
CADA = 4                           # un update cada tantos steps de entorno (como la Tabla 1 de Mnih)
COPIAR = 200                       # NUEVO: la red objetivo se pone igual a la red cada tantos steps de entorno
EVAL_CADA = 500
SEMILLA = 0


def epsilon(step):
    return max(EPS_FINAL, EPS_INICIAL - (EPS_INICIAL - EPS_FINAL) * (step - 1) / EPS_PASOS)


# %% 2. El entorno (el mismo de 2_QL-cliff.py): estado 0..47, acciones 0 arriba, 1 derecha, 2 abajo, 3 izquierda
env = gym.make("CliffWalking-v1", is_slippery=False, max_episode_steps=200)
env_eval = gym.make("CliffWalking-v1", is_slippery=False, max_episode_steps=200)   # para medir sin mover el de entrenamiento


# %% 3. La red: Q(s,·;θ) = MLP(one_hot(s)), 48 -> 64 -> 64 -> 4. Con one-hot puede representar la tabla exacta
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


# %% 4. El replay buffer: cinco arreglos circulares (s, a, r, s', terminó). Cada transición se guarda una vez y se usa en muchos updates
class Replay:
    def __init__(self, capacidad):
        self.capacidad = capacidad
        self.guardadas = 0     # cuántas transiciones se guardaron en total (puede pasar la capacidad)
        self.s = np.zeros(capacidad, np.int32)
        self.a = np.zeros(capacidad, np.int32)
        self.r = np.zeros(capacidad, np.float32)
        self.s2 = np.zeros(capacidad, np.int32)
        self.terminó = np.zeros(capacidad, bool)

    def guardar(self, s, a, r, s2, terminó):
        i = self.guardadas % self.capacidad    # al llenarse, vuelve a 0 y pisa la más vieja
        self.s[i], self.a[i], self.r[i], self.s2[i], self.terminó[i] = s, a, r, s2, terminó
        self.guardadas += 1

    def muestrear(self, rng, batch):
        """`batch` transiciones al azar (con reposición) entre las que hay guardadas."""
        i = rng.integers(min(self.guardadas, self.capacidad), size=batch)
        return self.s[i], self.a[i], self.r[i], self.s2[i], self.terminó[i]


# %% 5. La pérdida de un MINIBATCH: promedio de (y − Q(s,a;θ))², con y = r + γ max_a' Q(s',a';θ⁻)
def costo(params, params_obj, s, a, r, s2, terminó):
    q_siguiente = q_red(params_obj, s2).max(axis=1)      # el target sale de la red OBJETIVO θ⁻, no de θ
    y = r + jnp.where(terminó, 0.0, GAMMA * q_siguiente)   # y no depende de θ: jax.grad deriva solo respecto de `params`
    q = q_red(params, s)[jnp.arange(s.shape[0]), a]
    return jnp.mean((y - q) ** 2)


# %% 6. Un update de SGD: θ ← θ − η ∇θ L (θ⁻ no se toca)
@jax.jit
def actualizar(params, params_obj, minibatch):
    grad = jax.grad(costo)(params, params_obj, *minibatch)
    return jax.tree_util.tree_map(lambda p, g: p - LR * g, params, grad)


# %% 7. Cómo medimos: retorno de la política greedy (ε = 0) y el Q más alto del mapa
def evaluar(params):
    """Devuelve el retorno de jugar un episodio greedy y el mayor Q de los 48 estados.
    En Cliff ningún Q puede ser positivo; si el mayor Q lo es, o es NaN, el entrenamiento divergió."""
    Q = np.asarray(q_red(params, jnp.arange(48)))
    s, _ = env_eval.reset()
    retorno, fin = 0.0, False
    while not fin:
        s, r, terminó, truncado, _ = env_eval.step(int(Q[s].argmax()))
        retorno += r
        fin = terminó or truncado
    return retorno, Q.max()


# %% 8. Entrenar. En cada step: actuar, guardar, (cada CADA) aprender de un minibatch, (cada COPIAR) copiar la red objetivo. Se corta en el primer NaN
rng = np.random.default_rng(SEMILLA)
params = iniciar(jax.random.PRNGKey(SEMILLA))
params_obj = params            # θ⁻ ← θ. Los arrays de JAX son inmutables: params_obj no se mueve cuando params se actualiza
memoria = Replay(BUFFER)
s, _ = env.reset(seed=SEMILLA)
log = []   # (step, retorno greedy, Q más alto)
for step in tqdm(range(1, PASOS + 1)):
    # actuar: ε-greedy sobre la red que se entrena
    a = int(rng.integers(4)) if rng.random() < epsilon(step) else int(accion_greedy(params, s))
    s2, r, terminó, truncado, _ = env.step(a)

    # guardar: truncado (corte a 200 pasos) NO corta el bootstrap, terminó sí
    memoria.guardar(s, a, r, s2, terminó)
    s = s2
    if terminó or truncado:
        s, _ = env.reset()

    # aprender: un update con un minibatch del buffer, con el target de la red objetivo
    if step >= EMPEZAR and step % CADA == 0:
        params = actualizar(params, params_obj, memoria.muestrear(rng, BATCH))

    # copiar: la red objetivo queda igual a la red actual hasta la próxima copia
    if step % COPIAR == 0:
        params_obj = params

    # medir
    if step % EVAL_CADA == 0:
        log.append((step, *evaluar(params)))
        if not np.isfinite(log[-1][2]):
            print("cortamos")
            break

log = np.array(log)
finito = np.isfinite(log[:, 2])
print("Q más alto: pico", log[finito, 2].max().round(1), " al final", log[-1, 2].round(2), "  (cota: 0)")
if finito[-1]:
    print("no se rompió; retorno greedy al final:", log[-1, 1])
else:
    print("Q = NaN en el step", int(log[-1, 0]))

# %% 9. Ver la figura: el Q más alto contra la cota 0, y cómo juega la política greedy
fig_dqn, ejes = plt.subplots(1, 2, figsize=(11, 3.8))
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
fig_dqn.suptitle(f"Paso 3: DQN completo, replay (buffer {BUFFER}, minibatch {BATCH}) + red objetivo (copia cada {COPIAR} steps), SGD η = {LR:g}")
fig_dqn.tight_layout()
plt.show()

# %% 10. Guardar la figura
guardar(fig_dqn, "5_dqn_completo_cliff_walking.png")
