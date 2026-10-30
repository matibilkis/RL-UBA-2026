# %% Clase 9 · práctica · 3 — ¿Arregla prioritized experience replay la trampa de los dos picos?
# Viene de 1_canon_dos_picos.py: ahí la red con memoria COMPLETA (minibatch muestreado uniforme
# de todo el historial) igual quedaba atrapada en el pico facil 13/20 veces -- darle mas memoria
# no alcanzaba, porque la evidencia rara del pico angosto se diluia contra la abundante del facil.
# Prioritized Experience Replay (Schaul et al. 2015) prioriza las transiciones por error de
# prediccion |r - Q_hat|, en vez de muestrear parejo. La idea: una vez que la red ya aprendio bien
# el pico facil (error bajo ahi), cualquier tiro que haya caido cerca del pico angosto tiene error
# alto (la red todavia no lo sabe) y deberia samplearse mucho mas seguido que por simple frecuencia.
# Para correr en Hydrogen: Shift+Enter en cada celda, de arriba hacia abajo.
import pathlib

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.animation as animation

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


# %% 1. El problema: la misma Q de dos picos de 1_canon_dos_picos.py
TH1, S1, A1 = 0.15, 0.09, 1.0   # pico trampa: ancho, bajo
TH2, S2, A2 = 0.55, 0.025, 1.6  # pico optimo: angosto, mas alto
RUIDO = 0.15
theta_fino = np.linspace(0, np.pi / 4, 1000)


def Q_verdadera(theta):
    return A1 * np.exp(-((theta - TH1) / S1) ** 2) + A2 * np.exp(-((theta - TH2) / S2) ** 2)


def tirar(theta, rng):
    "Un solo tiro (o varios) con el theta elegido."
    return Q_verdadera(theta) + RUIDO * rng.normal(size=np.shape(theta))


EPS = 0.1
T = 3000
K = 8
CAPAS = (1, 16, 1)
PASOS_POR_TIRO = 5


# %% 2. La red: igual que en 1_canon_dos_picos.py
def iniciar(key):
    params = []
    for n_in, n_out in zip(CAPAS[:-1], CAPAS[1:]):
        key, sub = jax.random.split(key)
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


# %% 3a. Memoria completa, sampleo UNIFORME (el que se atrapaba en 1_canon_dos_picos.py)
def correr_red_uniforme(semilla, eps=EPS, T=T, k=K, pasos=PASOS_POR_TIRO):
    rng = np.random.default_rng(semilla)
    params = iniciar(jax.random.PRNGKey(semilla + 100))
    hist_theta, hist_r = [], []
    historial = np.zeros(T)
    for i in range(T):
        if rng.random() < eps:
            th = rng.uniform(0, np.pi / 4)
        else:
            q_grid = red(params, jnp.asarray(theta_fino))
            th = theta_fino[int(jnp.argmax(q_grid))]
        r = tirar(th, rng)
        hist_theta.append(th)
        hist_r.append(r)
        n_hist = len(hist_theta)
        idxs = np.arange(n_hist) if n_hist <= k else rng.choice(n_hist, size=k, replace=False)
        theta_batch = jnp.array([hist_theta[j] for j in idxs])
        r_batch = jnp.array([hist_r[j] for j in idxs])
        for _ in range(pasos):
            params = paso_sgd(params, theta_batch, r_batch)
        historial[i] = th
    return historial, params


# %% 3b. Memoria completa, sampleo PRIORIZADO por error |r - Q_hat| (prioritized experience replay)
# Usamos buffers de tamaño fijo T (en vez de listas que crecen) para poder compilar con @jax.jit
# la cuenta de prioridad; si no, JAX recompila en cada paso al cambiar el largo del historial y
# el entrenamiento se vuelve carisimo (nos paso en el primer intento: quedo colgado ~50 minutos).
@jax.jit
def calcular_prioridad(params, theta_buf, r_buf, mascara):
    q_pred = red(params, theta_buf)
    return mascara * (jnp.abs(r_buf - q_pred) + 0.05)   # +0.05: piso para que nada tenga prob 0


def correr_red_per(semilla, eps=EPS, T=T, k=K, pasos=PASOS_POR_TIRO, guardar_cuadros=False, cada=10):
    rng = np.random.default_rng(semilla)
    params = iniciar(jax.random.PRNGKey(semilla + 100))
    buffer_theta, buffer_r = np.zeros(T), np.zeros(T)
    historial = np.zeros(T)
    cuadros = [] if guardar_cuadros else None
    for i in range(T):
        if rng.random() < eps:
            th = rng.uniform(0, np.pi / 4)
        else:
            q_grid = red(params, jnp.asarray(theta_fino))
            th = theta_fino[int(jnp.argmax(q_grid))]
        r = tirar(th, rng)
        buffer_theta[i], buffer_r[i] = th, r
        n_hist = i + 1
        if n_hist <= k:
            idxs = np.arange(n_hist)
        else:
            mascara = (np.arange(T) < n_hist).astype(np.float32)
            prioridad = np.array(calcular_prioridad(
                params, jnp.asarray(buffer_theta), jnp.asarray(buffer_r), jnp.asarray(mascara)))
            probs = prioridad / prioridad.sum()
            idxs = rng.choice(T, size=k, replace=True, p=probs)
        theta_batch, r_batch = jnp.asarray(buffer_theta[idxs]), jnp.asarray(buffer_r[idxs])
        for _ in range(pasos):
            params = paso_sgd(params, theta_batch, r_batch)
        historial[i] = th
        if guardar_cuadros and i % cada == 0:
            curva = np.array(red(params, jnp.asarray(theta_fino)))
            cuadros.append((i, curva, buffer_theta[idxs].copy(), buffer_r[idxs].copy()))
    return historial, params, cuadros


# %% 4. Comparación semilla 1: uniforme queda atrapada, ¿PER la rescata?
SEMILLA_DEMO = 1
historial_uniforme, params_uniforme = correr_red_uniforme(SEMILLA_DEMO)
historial_per, params_per, cuadros_per = correr_red_per(SEMILLA_DEMO, guardar_cuadros=True)
print(f"uniforme, ultimos 500: theta promedio={historial_uniforme[-500:].mean():.3f}")
print(f"PER,      ultimos 500: theta promedio={historial_per[-500:].mean():.3f}")

fig_curvas, ax = plt.subplots(figsize=(8, 4))
ax.plot(theta_fino, Q_verdadera(theta_fino), "k", linewidth=2, label="Q verdadera")
ax.plot(theta_fino, np.array(red(params_uniforme, jnp.asarray(theta_fino))), color="C3", label="memoria completa, uniforme")
ax.plot(theta_fino, np.array(red(params_per, jnp.asarray(theta_fino))), color="C0", label="memoria completa, PER")
ax.set_xlabel("ángulo theta (rad)")
ax.legend(fontsize=9)
plt.show()
guardar(fig_curvas, "per_1_curvas.png")

# %% 4. Figura: a dónde tira cada uno a lo largo del tiempo
fig_theta, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.5), sharey=True)
for ax, historial, titulo, color in [(ax1, historial_uniforme, "uniforme", "C3"), (ax2, historial_per, "PER", "C0")]:
    ax.scatter(np.arange(T), historial, s=6, alpha=0.4, color=color)
    ax.axhline(TH1, color="k", linestyle=":", alpha=0.6)
    ax.axhline(TH2, color="k", linestyle="--", alpha=0.6)
    ax.set_title(titulo)
    ax.set_xlabel("paso (tiro)")
ax1.set_ylabel("theta elegido (rad)")
plt.show()
guardar(fig_theta, "per_2_theta_t.png")


# %% 5. Resumen de 20 semillas: uniforme vs PER, ¿encuentran el pico bueno o quedan atrapadas?
# Esto tarda unos minutos (dos barridos de 20 corridas de 3000 tiros cada uno).
def clasificar(historial):
    c1 = np.mean(np.abs(historial[-500:] - TH1) < 0.08)
    c2 = np.mean(np.abs(historial[-500:] - TH2) < 0.08)
    if c1 > 0.7:
        return "atrapado"
    if c2 > 0.7:
        return "encontro"
    return "oscila"


SEMILLAS = range(20)
resumen = {"uniforme": {"atrapado": 0, "encontro": 0, "oscila": 0},
           "PER": {"atrapado": 0, "encontro": 0, "oscila": 0}}

for semilla in SEMILLAS:
    h, _ = correr_red_uniforme(semilla)
    resumen["uniforme"][clasificar(h)] += 1
    h, _, _ = correr_red_per(semilla)
    resumen["PER"][clasificar(h)] += 1

print(resumen)

# %% 5. Figura: barras apiladas uniforme vs PER
fig_resumen, ax = plt.subplots(figsize=(6, 4))
metodos = ["uniforme", "PER"]
categorias = ["encontro", "oscila", "atrapado"]
colores = {"encontro": "C2", "oscila": "C1", "atrapado": "C3"}
base = np.zeros(2)
for cat in categorias:
    valores = [resumen[m][cat] for m in metodos]
    ax.bar(metodos, valores, bottom=base, color=colores[cat], label=cat)
    base += valores
ax.set_ylabel(f"semillas de {len(list(SEMILLAS))}")
ax.set_title("red, memoria completa: uniforme vs priorizado")
ax.legend(fontsize=9)
plt.show()
guardar(fig_resumen, "per_3_resumen.png")


# %% 6. Video: PER rescatando a la semilla 5 del pico malo, cuadro a cuadro
fig_v, ax = plt.subplots(figsize=(8, 4))
ax.plot(theta_fino, Q_verdadera(theta_fino), "k", linewidth=2, label="Q verdadera")
linea_qhat, = ax.plot([], [], color="C0", label="Q_hat (PER)")
puntos_batch = ax.scatter([], [], color="C1", s=25, zorder=3, label="minibatch priorizado")
ax.set_xlim(0, np.pi / 4)
ax.set_ylim(-0.3, 2.0)
ax.set_xlabel("ángulo theta (rad)")
ax.legend(fontsize=8, loc="upper left")
titulo = ax.set_title("")


def actualizar(cuadro):
    paso, curva, ws_theta, ws_r = cuadro
    linea_qhat.set_data(theta_fino, curva)
    puntos_batch.set_offsets(np.column_stack([ws_theta, ws_r]))
    titulo.set_text(f"tiro {paso}/{T}")
    return linea_qhat, puntos_batch, titulo


anim = animation.FuncAnimation(fig_v, actualizar, frames=cuadros_per, blit=False)
anim.save(FIGS / "per_video.mp4", writer=animation.FFMpegWriter(fps=6))
print("guardado:", FIGS / "per_video.mp4")
