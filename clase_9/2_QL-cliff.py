# %% 0. Clase 9 · práctica · 2 — Q-learning con tabla en Cliff Walking (gymnasium + JAX)
# Primer escalón: la tabla Q es un array de JAX y la regla de Q-learning es una función. Nada más.
# El escalón siguiente cambia esa tabla por una red; el resto del script queda igual.
import pathlib
import numpy as np
import gymnasium as gym
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

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


# %% 1. El entorno: grilla de 4 x 12, estado = 12 * fila + columna (0 a 47)
#   fila 3:   I ######## M      I = inicio (36), # = acantilado (37..46), M = meta (47)
# Acciones: 0 arriba, 1 derecha, 2 abajo, 3 izquierda. Cada paso da -1; caer al acantilado da -100 y vuelve al inicio.
env = gym.make("CliffWalking-v1", is_slippery=False, max_episode_steps=200)
s, _ = env.reset(seed=0)
print("estado inicial:", s)
print("un paso a la derecha desde el inicio (estado, recompensa, terminó):", env.step(1)[:3])   # cae: vuelve al inicio con -100

# %% 2. La tabla y la regla de Q-learning: Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
GAMMA = 0.99
ALFA = 0.5            # paso grande: el entorno es determinista, no hay ruido que promediar


@jax.jit
def actualizar(Q, s, a, r, s2, terminó):
    """Un update: toca UNA celda de la tabla.
    Si el episodio terminó no hay futuro que sumar.
    terminó es verdadero/falso"""
    objetivo = r + jnp.where(terminó, 0.0, GAMMA * Q[s2].max()) ## jnp.where(cond,x,y), si cond --> x, sino y
    return Q.at[s, a].add(ALFA * (objetivo - Q[s, a]))


@jax.jit
def greedy(Q, s):
    return Q[s].argmax()


Q = jnp.zeros((48, 4))   # 48 estados x 4 acciones = 192 números; arrancar en 0 es optimista (todo lo demás es negativo)
print("tabla:", Q.shape)

# %% 3. Un episodio entero con ε-greedy: con prob. EPS una acción al azar, si no la mejor según la tabla
EPS = 0.1

def episodio(Q, rng):
    s, _ = env.reset()
    retorno, fin = 0.0, False
    while not fin:
        a = int(rng.integers(4)) if rng.random() < EPS else int(greedy(Q, s))
        s2, r, terminó, truncado, _ = env.step(a)
        Q = actualizar(Q, s, a, r, s2, terminó)   # truncado (corte a 200 pasos) NO corta el bootstrap, terminó sí
        retorno += r
        s, fin = s2, terminó or truncado
    return Q, retorno


# %% 4. Cómo medimos la política que va quedando: un episodio sin exploración (ε = 0)
def retorno_greedy(Q):
    pi = np.asarray(Q).argmax(axis=1)
    s, _ = env.reset()
    retorno, fin = 0.0, False
    while not fin:
        s, r, terminó, truncado, _ = env.step(int(pi[s]))
        retorno += r
        fin = terminó or truncado
    return retorno


# %% 5. Entrenar: 300 episodios
EPISODIOS = 300
SEMILLA = 0
rng = np.random.default_rng(SEMILLA)
env.reset(seed=SEMILLA)
Q = jnp.zeros((48, 4))
con_eps = np.zeros(EPISODIOS)    # retorno de cada episodio de entrenamiento (con exploración)
sin_eps = np.zeros(EPISODIOS)    # retorno de la política greedy después de cada episodio
for ep in range(EPISODIOS):
    Q, con_eps[ep] = episodio(Q, rng)
    sin_eps[ep] = retorno_greedy(Q)
print("retorno greedy al final (óptimo: −13):", sin_eps[-1])

# %% 6. La learning curve: ver la figura
VENTANA = 10
suave = np.convolve(con_eps, np.ones(VENTANA) / VENTANA, mode="valid")
fig_curva, ejes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
ejes[0].plot(con_eps, color="C0", alpha=0.3, label="cada episodio")
ejes[0].plot(np.arange(VENTANA - 1, EPISODIOS), suave, color="C0", label=f"promedio de {VENTANA}")
ejes[0].set(title=f"Entrenamiento & acciones (ε = {EPS})")
ejes[1].plot(sin_eps, color="C1")
ejes[1].set(title="Politica greedy (ε = 0)")
for ax in ejes:
    ax.axhline(-13, color="gray", ls="--", label="óptimo: −13")
    ax.set(xlabel="episodios", ylim=(-250, 0))
ejes[0].set_ylabel("retorno del episodio")
ejes[0].legend()
fig_curva.suptitle("Q-learning con tabla en Cliff Walking (una semilla)")
fig_curva.tight_layout()
plt.show()

# %% 6. La learning curve: guardar la figura
guardar(fig_curva, "2_ql_cliff_curva.png")
