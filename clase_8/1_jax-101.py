
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
