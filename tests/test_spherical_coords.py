from math import pi

from jax import vmap
from jax.config import config
import jax.numpy as jnp
import numpy.random as np_random

from diffjeom import (
    check_diff_bianchi,
    check_ricci_tensor_sym,
    check_riemann_sym,
    get_christoffel2,
    get_ricci_scalar,
    get_ricci_tensor,
    get_riemann,
)

# Necessary for these tests
config.update("jax_enable_x64", True)


"""
Test differential geometry functions with spherical coordinates and Euclidean
metric.
"""


def get_xs(n):
    return jnp.stack(
        [
            np_random.rand(n) * 2,
            np_random.rand(n) * pi,
            np_random.rand(n) * 2 * pi,
        ],
        axis=1,
    )


def get_g(x):
    r, th, _ = x
    return jnp.diag(jnp.array([1, r ** 2, r ** 2 * jnp.cos(th) ** 2]))


def get_christoffel2_true(x):
    """
    See
    https://en.wikipedia.org/wiki/Christoffel_symbols#In_earth_surface_coordinates.
    """
    r, th, _ = x
    Gamma_r = jnp.array([[0, 0, 0], [0, -r, 0], [0, 0, -r * jnp.cos(th) ** 2]])
    Gamma_th = jnp.array(
        [[0, 1 / r, 0], [1 / r, 0, 0], [0, 0, jnp.cos(th) * jnp.sin(th)]]
    )
    Gamma_ph = jnp.array(
        [[0, 0, 1 / r], [0, 0, -jnp.tan(th)], [1 / r, -jnp.tan(th), 0]]
    )
    return jnp.stack([Gamma_r, Gamma_th, Gamma_ph])


def test_christoffel2(n=10):
    xs = get_xs(n)
    Gamma_trues = vmap(get_christoffel2_true)(xs)
    Gammas = vmap(lambda x: get_christoffel2(x, get_g))(xs)

    assert jnp.allclose(Gamma_trues, Gammas)


def test_riemann(n=10):
    xs = get_xs(n)
    g = vmap(get_g)(xs)
    get_riemann_pt = lambda x: get_riemann(x, get_g)
    R = vmap(get_riemann_pt)(xs)

    check_riemann_sym(g, R)
    check_diff_bianchi(xs, get_riemann_pt)


def test_ricci_tensor(n=10):
    xs = get_xs(n)
    Rt = vmap(lambda x: get_ricci_tensor(x, get_g))(xs)

    check_ricci_tensor_sym(Rt)


def test_ricci_scalar(n=10):
    xs = get_xs(n)
    Rs = vmap(lambda x: get_ricci_scalar(x, get_g))(xs)

    assert jnp.allclose(Rs, 0)  # Euclidean space is flat
