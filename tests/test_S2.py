from math import pi

from jax import vmap
from jax.config import config
import jax.numpy as jnp
import numpy.random as np_random

from diffjeom import (
    check_diff_bianchi,
    check_riemann_sym,
    get_christoffel2,
    get_ricci_scalar,
    get_ricci_tensor,
    get_riemann,
)

# Necessary for these tests
config.update("jax_enable_x64", True)

"""
Test differential geometry functions on S^2, the unit 2-sphere.
"""


def get_xs(n):
    return jnp.stack(
        [
            np_random.rand(n) * pi,
            np_random.rand(n) * 2 * pi,
        ],
        axis=1,
    )


def get_g(x):
    th, _ = x
    return jnp.diag(jnp.array([1, jnp.cos(th) ** 2]))


def get_christoffel2_true(x):
    """
    See https://math.stackexchange.com/questions/3057630/calculation-of-christoffel-symbol-for-unit-sphere.
    """
    th, _ = x
    c_th = jnp.cos(th)
    s_th = jnp.sin(th)
    Gamma_th = jnp.diag(jnp.array([0, s_th * c_th]))
    Gamma_ph = jnp.array([[0, -s_th / c_th], [-s_th / c_th, 0]])
    return jnp.stack([Gamma_th, Gamma_ph], axis=0)


def get_ricci_tensor_true(x):
    th, _ = x
    return jnp.diag(jnp.array([1, jnp.cos(th) ** 2]))


def test_test_christoffel2(n=10):
    xs = get_xs(n)
    Gammas = vmap(lambda x: get_christoffel2(x, get_g))(xs)
    Gamma_trues = vmap(get_christoffel2_true)(xs)

    jnp.allclose(Gammas, Gamma_trues)


def test_riemann(n=10):
    xs = get_xs(n)
    g = vmap(get_g)(xs)
    get_riemann_pt = lambda x: get_riemann(x, get_g)
    R = vmap(get_riemann_pt)(xs)

    check_riemann_sym(g, R)
    check_diff_bianchi(xs, get_riemann_pt)


def test_ricci_tensor(n=10):
    xs = get_xs(n)
    Rt_trues = vmap(get_ricci_tensor_true)(xs)
    Rts = vmap(lambda x: get_ricci_tensor(x, get_g))(xs)

    jnp.allclose(Rts, Rt_trues)


def test_ricci_scalar(n=10):
    """
    See https://math.stackexchange.com/questions/3057630/calculation-of-christoffel-symbol-for-unit-sphere.
    """
    xs = get_xs(n)
    Rss = vmap(lambda x: get_ricci_scalar(x, get_g))(xs)

    jnp.allclose(Rss, 2.0)
