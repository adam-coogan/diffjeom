from jax import jacfwd, vmap
import jax.numpy as jnp
from typing import Callable


"""
Functions for computing differential geometric quantities from a metric and
checking that they obey basic identities.
"""


def get_christoffel2(
    x: jnp.ndarray, get_g: Callable[[jnp.ndarray], jnp.ndarray]
) -> jnp.ndarray:
    """
    Computes the Christoffel symbols of the second kind:

        Gamma[i, j, k] = dGamma^i_jk
    """
    g = get_g(x)
    g_inv = jnp.linalg.inv(g)
    # dg[i, j, k] = dg_ij / dx^k
    dg = jacfwd(get_g)(x)
    return (
        1
        / 2
        * (
            jnp.einsum(
                "im,mkl->ikl",
                g_inv,
                dg + jnp.einsum("mlk->mkl", dg) - jnp.einsum("klm->mkl", dg),
            )
        )
    )


def get_riemann(
    x: jnp.ndarray, get_g: Callable[[jnp.ndarray], jnp.ndarray]
) -> jnp.ndarray:
    """
    Computes the Riemann curvature tensor:

        R[r, s, m, n] = R^r_smn
    """
    # Gamma[i, j, k] = Gamma^i_jk
    Gamma = get_christoffel2(x, get_g)
    # dGamma[i, j, k, l] = dGamma^i_jk / dx^l
    dGamma = jacfwd(lambda x: get_christoffel2(x, get_g))(x)

    dGamma_term = jnp.einsum("rnsm->rsmn", dGamma) - jnp.einsum("rmsn->rsmn", dGamma)
    Gamma2_term = jnp.einsum("rml,lns->rsmn", Gamma, Gamma) - jnp.einsum(
        "rnl,lms->rsmn", Gamma, Gamma
    )
    return dGamma_term + Gamma2_term


def get_ricci_tensor(
    x: jnp.ndarray, get_g: Callable[[jnp.ndarray], jnp.ndarray]
) -> jnp.ndarray:
    """
    Computes the Ricci tensor:

        Rt[i, j] =  R_ij
    """
    R = get_riemann(x, get_g)
    return jnp.einsum("cacb->ab", R)


def get_ricci_scalar(
    x: jnp.ndarray, get_g: Callable[[jnp.ndarray], jnp.ndarray]
) -> jnp.ndarray:
    """
    Computes the Ricci scalar R.
    """
    g = get_g(x)  # g_ij
    g_inv = jnp.linalg.inv(g)  # g^ij
    R = get_riemann(x, get_g)  # R^a_bcd
    # g^bd R^a_bad
    return jnp.einsum("bd,abad", g_inv, R)


def check_christoffel2_sym(Gamma: jnp.ndarray):
    """
    Checks symmetry of a Christoffel symbol. Batched.
    """
    assert jnp.allclose(jnp.einsum("...acb->...abc", Gamma), Gamma)


def check_riemann_sym(g: jnp.ndarray, R: jnp.ndarray):
    """
    Checks symmetries of a Riemann curvature tensor. Batched.
    """
    R_lower = jnp.einsum("...as,...sbcd->...abcd", g, R)
    assert jnp.allclose(R_lower, -jnp.einsum("...abdc->...abcd", R_lower))
    assert jnp.allclose(R_lower, -jnp.einsum("...bacd->...abcd", R_lower))
    assert jnp.allclose(
        R_lower
        + jnp.einsum("...acdb->...abcd", R_lower)
        + jnp.einsum("...adbc->...abcd", R_lower),
        jnp.zeros_like(R_lower),
    )
    assert jnp.allclose(R_lower, jnp.einsum("...cdab->...abcd", R_lower))


def check_ricci_tensor_sym(Rt: jnp.ndarray):
    """
    Checks symmetry of a Ricci tensor. Batched.
    """
    assert jnp.allclose(jnp.einsum("...ij->...ji", Rt), Rt), Rt


def check_diff_bianchi(
    x: jnp.ndarray, get_riemann_pt: Callable[[jnp.ndarray], jnp.ndarray]
):
    """
    Checks the differential Bianchi identity. Batched.
    """
    dR = vmap(jacfwd(get_riemann_pt))(x)
    assert jnp.allclose(
        dR
        + jnp.einsum("...abdec->...abcde", dR)
        + jnp.einsum("...abecd->...abcde", dR),
        jnp.zeros_like(dR),
    )
