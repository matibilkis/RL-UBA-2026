
# %% jax 101: grad, y grad de grad
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit
import matplotlib.pyplot as plt

def p(x):
    return x**3 - 2*x

d1 = grad(p)
d2 = grad(d1)
d3 = grad(d2)
d4 = grad(d3)
print(p(2.0), d1(2.0), d2(2.0), d3(2.0), d4(2.0))

# %% jax 101: la sigmoide y sus derivadas
def sigmoide(z):
    return 1 / (1 + jnp.exp(-z))

curvas = [sigmoide, grad(sigmoide), grad(grad(sigmoide)), grad(grad(grad(sigmoide)))]
zs = jnp.linspace(-8, 8, 400)
valores = [vmap(f)(zs) for f in curvas]
s = sigmoide(zs)
print("max |sigma' - sigma(1-sigma)| =", float(jnp.max(jnp.abs(valores[1] - s*(1 - s)))))

# %% figura
fig, ax = plt.subplots(figsize=(8, 4))
for v, etiqueta, c in zip(valores, [r'$\sigma$', r"$\sigma'$", r"$\sigma''$", r"$\sigma'''$"],
                          ["black", "tab:blue", "tab:red", "tab:green"]):
    ax.plot(zs, v, linewidth=3, alpha=0.7, color=c, label=etiqueta)
ax.axhline(0, color="gray", linewidth=1)
ax.set_xlabel(r'$z$', size=20)
ax.legend(fontsize=18, frameon=False)
plt.tight_layout()
plt.show()

# %% guardar
fig.savefig("figs/jax_sigmoide_derivadas.png", dpi=200, bbox_inches="tight")

# %% jax 101: grad respecto de un diccionario de parametros
Xs = jnp.array([[-1.0, 0.5], [0.3, 1.2], [1.5, -0.4], [2.0, 1.0]])
ys = jnp.array([0.0, 0.0, 1.0, 1.0])
params = {"w": jnp.array([0.5, -0.3]), "b": 0.1}

def bce(params, X, y):
    z = X @ params["w"] + params["b"]
    return jnp.mean(jnp.logaddexp(0.0, z) - y * z)

g = grad(bce)(params, Xs, ys)
print(g)

import numpy as np

## %%%%% ESPIRAL

def espiral(n, ruido, semilla):
    r = np.random.default_rng(semilla)
    t = np.sqrt(r.uniform(0.02, 1, n))
    ang = 4*np.pi*t
    A = np.c_[t*np.cos(ang), t*np.sin(ang)]*4
    B = np.c_[t*np.cos(ang + np.pi), t*np.sin(ang + np.pi)]*4
    X = np.vstack([A, B]) + r.normal(0, ruido, (2*n, 2))
    return jnp.array(X, dtype=jnp.float32), jnp.r_[jnp.zeros(n), jnp.ones(n)]

Xe, ye = espiral(250, 0.2, 0)
Xe_t, ye_t = espiral(250, 0.2, 1)

# %% el mismo modelo y el mismo costo para los dos casos
def init(key, tam):
    """inicializamos les parametros"""
    ps = []
    for a, b_ in zip(tam[:-1], tam[1:]):
        key, k = jax.random.split(key)
        ps.append((jax.random.normal(k, (b_, a)) / np.sqrt(a), jnp.zeros(b_)))
    return ps

def forward(ps, X):
    for W, b_ in ps[:-1]:
        X = jnp.tanh(X @ W.T + b_)
    W, b_ = ps[-1]
    return (X @ W.T + b_)[:, 0]

def loss(ps, X, y):
    """BCE: -(1/N) sum_i [ y_i log sigma(z_i) + (1 - y_i) log(1 - sigma(z_i)) ]"""
    z = forward(ps, X)
    log_p1 = jax.nn.log_sigmoid(z)      # log sigma(z)
    log_p0 = jax.nn.log_sigmoid(-z)     # log (1 - sigma(z)), estable
    return -jnp.mean(y * log_p1 + (1 - y) * log_p0)

def acierto(ps, X, y):
    return float(jnp.mean((forward(ps, X) > 0) == y))

# %% entrenar: descenso por gradiente
def entrenar(ps, X, y, pasos, lr, X_t, y_t, cada=50):
    paso = jit(jax.value_and_grad(loss))
    curva = {"paso": [], "train": [], "test": []}
    for i in range(pasos):
        costo, g_ = paso(ps, X, y)
        if i % cada == 0:
            curva["paso"].append(i)
            curva["train"].append(float(costo))
            curva["test"].append(float(loss(ps, X_t, y_t)))
        ps = jax.tree.map(lambda p_, gp: p_ - lr * gp, ps, g_)
    return ps, float(costo), curva

clave = jax.random.PRNGKey(0)
ps_log, c_log, curva_log = entrenar(init(clave, [2, 1]), Xe, ye, 3000, 0.1, Xe_t, ye_t)
ps_mlp, c_mlp, curva_mlp = entrenar(init(clave, [2, 64, 64, 1]), Xe, ye, 5000, 0.5, Xe_t, ye_t)

# %% resultados
print("sin capa oculta   train", acierto(ps_log, Xe, ye), "test", acierto(ps_log, Xe_t, ye_t), "costo", round(c_log, 3))
print("MLP 2-64-64-1     train", acierto(ps_mlp, Xe, ye), "test", acierto(ps_mlp, Xe_t, ye_t), "costo", round(c_mlp, 4))

# %% figura
gx = jnp.linspace(-5, 5, 300)
GX, GY = jnp.meshgrid(gx, gx)
malla = jnp.c_[GX.ravel(), GY.ravel()]
def dibujar_red(ax, tam, max_nodos=5, ancho=6.6):
    """esquema de la red: una columna por capa, '...' si hay muchas neuronas"""
    columnas = []
    for k, n in enumerate(tam):
        mostrar = min(n, max_nodos)
        ys_ = np.linspace(1, -1, mostrar) * (0.45 * (mostrar - 1) / 2 + 0.2) if mostrar > 1 else np.array([0.0])
        columnas.append((k * ancho / (len(tam) - 1), n, ys_))
    for (xa, n, ya), (xb, n2, yb) in zip(columnas[:-1], columnas[1:]):
        for y1 in ya:
            for y2 in yb:
                ax.plot([xa, xb], [y1, y2], color="black", alpha=0.15, linewidth=1, zorder=1)
    for k, (x_, n, yk) in enumerate(columnas):
        for i, y_ in enumerate(yk):
            if n > max_nodos and i == len(yk) // 2:
                ax.text(x_, y_, r'$\vdots$', size=22, ha="center", va="center")
                continue
            ax.add_patch(plt.Circle((x_, y_), 0.23, fc="white" if k == 0 else "lightgray", ec="black", lw=1.5, zorder=3))
            if k == 0:
                ax.text(x_, y_, r'$x_%d$' % (i + 1), size=12, ha="center", va="center", zorder=4)
            if k == len(tam) - 1:
                ax.text(x_, y_, r'$\sigma$', size=13, ha="center", va="center", zorder=4)
        etiqueta = "entrada" if k == 0 else ("salida" if k == len(tam) - 1 else "oculta")
        ax.text(x_, -2.05, "%s\n%d" % (etiqueta, n), size=14, ha="center", va="center", color="gray")
    ax.set_xlim(-1, ancho + 1)
    ax.set_ylim(-2.6, 1.4)
    ax.set_aspect("equal")
    ax.axis("off")

fig, axs = plt.subplots(3, 2, figsize=(12, 11.5), gridspec_kw=dict(height_ratios=[5, 2.2, 3]))
dibujar_red(axs[1, 0], [2, 1])
dibujar_red(axs[1, 1], [2, 64, 64, 1])
for ax, ps, titulo in [(axs[0, 0], ps_log, "Sin capa oculta"), (axs[0, 1], ps_mlp, "MLP 2-64-64-1")]:
    Z = jax.nn.sigmoid(forward(ps, malla)).reshape(GX.shape)
    ax.pcolormesh(GX, GY, Z, cmap="coolwarm", vmin=0, vmax=1, shading="auto", alpha=0.7)
    ax.contour(GX, GY, Z, levels=[0.5], colors="black", linewidths=3, alpha=0.6)
    ax.scatter(*np.array(Xe[ye == 0]).T, color="tab:blue", edgecolor="black", alpha=0.8, s=30)
    ax.scatter(*np.array(Xe[ye == 1]).T, color="tab:red", edgecolor="black", alpha=0.8, s=30)
    ax.set_title(titulo + ": acierto en test %.2f" % acierto(ps, Xe_t, ye_t), size=18)
    ax.set_xlabel(r'$x_1$', size=20)
    ax.set_aspect("equal")
axs[0, 0].set_ylabel(r'$x_2$', size=20)
for ax, curva in [(axs[2, 0], curva_log), (axs[2, 1], curva_mlp)]:
    ax.plot(curva["paso"], curva["train"], linewidth=3, alpha=0.7, color="black", label="train")
    ax.plot(curva["paso"], curva["test"], linewidth=3, alpha=0.7, color="tab:orange", label="test")
    ax.set_yscale("log")
    ax.set_xlabel("paso de descenso", size=16)
    ax.legend(fontsize=14, frameon=False)
axs[2, 0].sharey(axs[2, 1])
axs[2, 0].set_ylabel("BCE", size=16)
plt.tight_layout()
plt.show()

# %% guardar
fig.savefig("figs/espiral_logistica_vs_mlp.png", dpi=200, bbox_inches="tight")
