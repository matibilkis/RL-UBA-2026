# %% features: el mismo modelo (regresion logistica) sobre distintas features de la espiral
import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, jit
import matplotlib.pyplot as plt

os.makedirs("figs", exist_ok=True)

# %% datos: la espiral doble de 1_jax-101.py
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

# %% el modelo y el costo de siempre (BCE). Con tam = [d, 1] es regresion logistica sobre d features
def init(key, tam):
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
    z = forward(ps, X)
    return -jnp.mean(y * jax.nn.log_sigmoid(z) + (1 - y) * jax.nn.log_sigmoid(-z))

def acierto(ps, X, y):
    return float(jnp.mean((forward(ps, X) > 0) == y))

grad_L = jit(grad(loss))

def entrenar(ps, X, y, pasos, lr):
    for i in range(pasos):
        g_ = grad_L(ps, X, y)
        ps = [(W - lr * gW, b_ - lr * gb) for (W, b_), (gW, gb) in zip(ps, g_)]
    return ps

# %% las features: cada una es una funcion X (N, 2) -> phi (N, d)
def crudas(X):
    return X

def a_mano(X):
    """la espiral es theta = pi * r (y theta = pi * r + pi la otra): cos(theta - pi r) vale +1 en una y -1 en la otra"""
    r = jnp.sqrt(X[:, 0]**2 + X[:, 1]**2)
    theta = jnp.arctan2(X[:, 1], X[:, 0])
    return jnp.cos(theta - jnp.pi * r)[:, None]

def polinomios(X, grado):
    """todos los x1^i x2^j con 1 <= i + j <= grado"""
    cols = []
    for i in range(grado + 1):
        for j in range(grado + 1 - i):
            if i + j > 0:
                cols.append(X[:, 0]**i * X[:, 1]**j)
    return jnp.stack(cols, axis=1)

def fourier_aleatorias(n_feat, sigma, semilla):
    """cos(x W + b) con W y b sorteados una vez y despues fijos (no se entrenan)"""
    k1, k2 = jax.random.split(jax.random.PRNGKey(semilla))
    W = sigma * jax.random.normal(k1, (2, n_feat))
    b_ = jax.random.uniform(k2, (n_feat,), minval=0, maxval=2*np.pi)
    return lambda X: jnp.cos(X @ W + b_)

# %% entrenar la misma regresion logistica sobre cada phi (se estandarizan las features para que el descenso ande bien)
def probar(phi, oculto=[], pasos=5000, lr=0.5):
    F = phi(Xe)
    mu, sd = F.mean(0), F.std(0)
    phi_e = lambda X: (phi(X) - mu) / sd
    ps = init(jax.random.PRNGKey(0), [F.shape[1]] + oculto + [1])
    ps = entrenar(ps, phi_e(Xe), ye, pasos, lr)
    return phi_e, ps

modelos = {
    "crudas (x1, x2)": probar(crudas),
    "a mano: cos(θ − πr)": probar(a_mano),
    "polinomios grado 5": probar(lambda X: polinomios(X, 5), pasos=20000),
    "Fourier aleatorias (100)": probar(fourier_aleatorias(100, 2.0, 0)),
    "red 2-64-64-1 (aprendidas)": probar(crudas, oculto=[64, 64]),
}

# %% resultados
for nombre, (phi_e, ps) in modelos.items():
    d = phi_e(Xe[:1]).shape[1]
    print("%-28s %3d entradas   train %.3f   test %.3f" % (nombre, d, acierto(ps, phi_e(Xe), ye), acierto(ps, phi_e(Xe_t), ye_t)))

# %% figura: regiones de decision en el plano (x1, x2)
gx = jnp.linspace(-5, 5, 300)
GX, GY = jnp.meshgrid(gx, gx)
malla = jnp.c_[GX.ravel(), GY.ravel()]

fig, axs = plt.subplots(1, 5, figsize=(24, 5), sharey=True)
for ax, (nombre, (phi_e, ps)) in zip(axs, modelos.items()):
    z = forward(ps, phi_e(malla)).reshape(GX.shape)
    ax.contourf(GX, GY, z > 0, levels=[-0.5, 0.5, 1.5], colors=["tab:blue", "tab:red"], alpha=0.25)
    ax.scatter(Xe_t[:, 0], Xe_t[:, 1], c=ye_t, cmap="coolwarm", s=8)
    ax.set_title("%s\ntest %.3f" % (nombre, acierto(ps, phi_e(Xe_t), ye_t)), size=18)
    ax.set_xlabel(r'$x_1$', size=20)
axs[0].set_ylabel(r'$x_2$', size=20)
plt.tight_layout()
plt.show()

# %% guardar
fig.savefig("figs/features_espiral.png", dpi=200, bbox_inches="tight")
