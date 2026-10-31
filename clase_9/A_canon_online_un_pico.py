# %% Clase 9 · práctica · 0 — El cañón: ¿se asienta Q_hat con epsilon-greedy online y un MLP?
# Viene de clase_8/practica/5_canon.py (mismo cañón, D y V fijos, sigue siendo un bandit).
# Ahí ajustábamos la red con un batch fijo de 100 tiros i.i.d.: convergía bien.
# Acá el agente tira, ajusta con los últimos K tiros (ventana deslizante, no un batch grande)
# y vuelve a tirar con la red actualizada. El mensaje: la ACCION elegida converge razonablemente
# bien (el problema es simple, un solo pico), pero el VALOR Q_hat nunca se asienta, ni siquiera
# con mucha experiencia — motivación directa de replay buffer en DQN (ver 0_poc_dqn_cartpole.py).
# Para correr en Hydrogen: Shift+Enter en cada celda, de arriba hacia abajo.
import pathlib
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

FIGS = pathlib.Path.cwd() / "figs"   # correr desde clase_9/practica/
FIGS.mkdir(exist_ok=True)


def guardar(fig, nombre):
    "Guarda la figura en figs/ y muestra el archivo que quedó."
    ruta = FIGS / nombre
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    print("guardada:", ruta)
    plt.figure(figsize=fig.get_size_inches())
    plt.imshow(plt.imread(ruta))
    plt.axis("off")
    plt.show()


# %% 1. El problema: igual que clase_8/practica/5_canon.py
G = 9.8
V = 10.0
D = 7.0
RUIDO = 0.5
theta_fino = np.linspace(0, np.pi / 4, 1000)


def alcance(theta):
    return V**2 * np.sin(2 * theta) / G


def Q_verdadera(theta):
    return -np.abs(alcance(theta) - D)


def tirar(theta, rng):
    "Un solo tiro (o varios) con el theta elegido."
    return Q_verdadera(theta) + RUIDO * rng.normal(size=np.shape(theta))


# %% 2. La red: un MLP chico en JAX (1-16-1)
CAPAS = (1, 16, 1)


def iniciar(key):
    params = []
    for n_in, n_out in zip(CAPAS[:-1], CAPAS[1:]):
        clave, sub = jax.random.split(key)
        W = jax.random.normal(sub, (n_in, n_out)) / jnp.sqrt(n_in)
        params.append((W, jnp.zeros(n_out)))
    return params


def red(params, theta):
    h = (theta / (np.pi / 8) - 1)[:, None]    # [0, pi/4] -> [-1, 1]
    for W, b in params[:-1]:
        h = jnp.tanh(h @ W + b)
    W, b = params[-1]
    return (h @ W + b)[:, 0]


def perdida(params, theta, r):
    return jnp.mean((red(params, theta) - r) ** 2)


@jax.jit
def paso_sgd(params, theta, r, lr=0.1):
    "Un paso de descenso por gradiente: cada parámetro p se mueve como p <- p - lr * dL/dp."
    grad = jax.grad(perdida)(params, theta, r)
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grad)


# %% 3. El loop online: tira, ajusta con la ventana de los últimos K, vuelve a tirar
from collections import deque

K = 8                 # tamaño de la ventana (minibatch)
PASOS_POR_TIRO = 5    # pasos de SGD sobre esa ventana, cada vez que llega un tiro nuevo
EPS = 0.2
T = 3000              # mucha experiencia: 3000 tiros, no 500
SNAPSHOTS = [10, 100, 500, 1500, 2999]   # en qué pasos guardamos la curva Q_hat completa

theta_optimo = theta_fino[np.argmax(Q_verdadera(theta_fino))]
Q_optimo = Q_verdadera(theta_optimo)   # el valor verdadero en el óptimo: la vara contra la que comparamos

rng = np.random.default_rng(0)
params = iniciar(jax.random.PRNGKey(1))
ventana_theta, ventana_r = deque(maxlen=K), deque(maxlen=K)
historial_theta = np.zeros(T)
q_hat_en_optimo = np.zeros(T)   # Q_hat(theta_optimo) en cada paso: ¿se asienta en Q_optimo?
curvas = {}   # paso -> Q_hat(theta_fino) en ese momento

for i in range(T):
    if rng.random() < EPS:
        theta_i = rng.uniform(0, np.pi / 4)
    else:
        q_grid = red(params, jnp.asarray(theta_fino))
        theta_i = theta_fino[int(jnp.argmax(q_grid))]

    r_i = tirar(theta_i, rng)
    ventana_theta.append(theta_i)
    ventana_r.append(r_i)

    theta_batch = jnp.array(ventana_theta)
    r_batch = jnp.array(ventana_r)
    for _ in range(PASOS_POR_TIRO):
        params = paso_sgd(params, theta_batch, r_batch)

    historial_theta[i] = theta_i
    q_hat_en_optimo[i] = float(red(params, jnp.array([theta_optimo]))[0])
    if i in SNAPSHOTS:
        curvas[i] = np.array(red(params, jnp.asarray(theta_fino)))

print(f"theta optimo verdadero: {theta_optimo:.3f} rad, Q ahi: {Q_optimo:.3f}")
print(f"theta promedio, últimos 200 tiros: {historial_theta[-200:].mean():.3f} rad")
print(f"theta std, últimos 200 tiros: {historial_theta[-200:].std():.3f} rad")
print(f"Q_hat(theta_optimo) promedio, primeros 200 pasos:  {q_hat_en_optimo[:200].mean():.3f}")
print(f"Q_hat(theta_optimo) promedio, últimos 200 pasos:   {q_hat_en_optimo[-200:].mean():.3f}")
print(f"Q_hat(theta_optimo) std, primeros 200 pasos:       {q_hat_en_optimo[:200].std():.3f}")
print(f"Q_hat(theta_optimo) std, últimos 200 pasos:        {q_hat_en_optimo[-200:].std():.3f}")


# %% 4. Figura A: la curva Q_hat en distintos pasos, ¿se asienta en la verdadera?
fig_a, ax = plt.subplots(figsize=(8, 4))
ax.plot(theta_fino, Q_verdadera(theta_fino), "k", linewidth=2, label="Q verdadera")
colores = plt.cm.viridis(np.linspace(0.15, 0.85, len(SNAPSHOTS)))
for paso, color in zip(SNAPSHOTS, colores):
    ax.plot(theta_fino, curvas[paso], color=color, label=f"Q_hat, paso {paso}")
ax.set_xlabel("ángulo theta (rad)")
ax.legend(fontsize=9)
plt.show()

# %% 4. Figura A: guardar
guardar(fig_a, "canon_online_a_curvas.png")


# %% 5. Figura B: qué theta elige el agente a lo largo del tiempo
fig_b, ax = plt.subplots(figsize=(8, 3.5))
ax.scatter(np.arange(T), historial_theta, s=6, alpha=0.4, label="theta elegido")
ax.axhline(theta_optimo, color="k", linestyle="--", label="theta óptimo")
ax.set_xlabel("paso (tiro)")
ax.set_ylabel("theta (rad)")
ax.legend(fontsize=9)
plt.show()

# %% 5. Figura B: guardar
guardar(fig_b, "canon_online_b_theta_t.png")


# %% 6. Figura C: Q_hat en el theta óptimo, a lo largo de 3000 tiros. ¿Se asienta?
fig_c, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(np.arange(T), q_hat_en_optimo, linewidth=0.8, alpha=0.8, label="Q_hat(theta_optimo)")
ax.axhline(Q_optimo, color="k", linestyle="--", label="Q verdadera en theta_optimo")
ax.set_xlabel("paso (tiro)")
ax.set_ylabel("Q(theta_optimo)")
ax.legend(fontsize=9)
plt.show()

# %% 6. Figura C: guardar
guardar(fig_c, "canon_online_c_q_optimo.png")


# %% 7. Video: repetimos el mismo entrenamiento (misma semilla) guardando la curva cada
# FRAME_EVERY pasos, para animar cómo Q_hat nunca se asienta en la verdadera.
import matplotlib.animation as animation

FRAME_EVERY = 10   # 3000 / 10 = 300 cuadros

rng = np.random.default_rng(0)
params = iniciar(jax.random.PRNGKey(1))
ventana_theta, ventana_r = deque(maxlen=K), deque(maxlen=K)
cuadros = []   # cada cuadro: (paso, curva Q_hat completa, ventana actual, Q_hat(theta_optimo) hasta ahora)
q_hat_hasta_ahora = []

for i in range(T):
    if rng.random() < EPS:
        theta_i = rng.uniform(0, np.pi / 4)
    else:
        q_grid = red(params, jnp.asarray(theta_fino))
        theta_i = theta_fino[int(jnp.argmax(q_grid))]

    r_i = tirar(theta_i, rng)
    ventana_theta.append(theta_i)
    ventana_r.append(r_i)

    theta_batch = jnp.array(ventana_theta)
    r_batch = jnp.array(ventana_r)
    for _ in range(PASOS_POR_TIRO):
        params = paso_sgd(params, theta_batch, r_batch)

    q_hat_hasta_ahora.append(float(red(params, jnp.array([theta_optimo]))[0]))
    if i % FRAME_EVERY == 0:
        curva = np.array(red(params, jnp.asarray(theta_fino)))
        cuadros.append((i, curva, np.array(ventana_theta), np.array(ventana_r), list(q_hat_hasta_ahora)))

print(f"{len(cuadros)} cuadros")

# %% 7. Video: armar la animación (panel izq: la curva; panel der: Q_hat(theta_optimo) creciendo)
fig_v, (ax_izq, ax_der) = plt.subplots(1, 2, figsize=(11, 4))

ax_izq.plot(theta_fino, Q_verdadera(theta_fino), "k", linewidth=2, label="Q verdadera")
linea_qhat, = ax_izq.plot([], [], color="C0", label="Q_hat")
puntos_ventana = ax_izq.scatter([], [], color="C1", s=25, zorder=3, label=f"últimos {K} tiros")
ax_izq.set_xlim(0, np.pi / 4)
ax_izq.set_ylim(-8, 1.5)
ax_izq.set_xlabel("ángulo theta (rad)")
ax_izq.legend(fontsize=8, loc="lower right")
titulo = ax_izq.set_title("")

ax_der.axhline(Q_optimo, color="k", linestyle="--", label="Q verdadera en theta_optimo")
linea_q_opt, = ax_der.plot([], [], color="C0", linewidth=0.8)
ax_der.set_xlim(0, T)
ax_der.set_ylim(-6, 2)
ax_der.set_xlabel("paso (tiro)")
ax_der.set_ylabel("Q_hat(theta_optimo)")
ax_der.legend(fontsize=8, loc="lower right")


def actualizar(cuadro):
    paso, curva, ws_theta, ws_r, q_hasta_ahora = cuadro
    linea_qhat.set_data(theta_fino, curva)
    puntos_ventana.set_offsets(np.column_stack([ws_theta, ws_r]))
    titulo.set_text(f"tiro {paso}/{T}")
    linea_q_opt.set_data(np.arange(len(q_hasta_ahora)), q_hasta_ahora)
    return linea_qhat, puntos_ventana, titulo, linea_q_opt


anim = animation.FuncAnimation(fig_v, actualizar, frames=cuadros, blit=False)
anim.save(FIGS / "canon_online_evolucion.mp4", writer=animation.FFMpegWriter(fps=20))
print("guardado:", FIGS / "canon_online_evolucion.mp4")
