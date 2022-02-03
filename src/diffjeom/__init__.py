from typing import Callable

from jax import jacfwd, vmap
import jax.numpy as jnp

Array = jnp.ndarray
"""
jax array type
"""


"""
Functions for computing differential geometric quantities from a metric and
checking that they obey basic identities.
"""


def get_christoffel2(x: Array, get_g: Callable[[Array], Array]) -> Array:
    r"""
    Computes the Christoffel symbols of the second kind, `Gamma[i,j,k]`, defined
    as:

        :math:`\Gamma^i_{\ jk} = \frac{1}{2} g^{im} \left( \frac{\partial g_{mk}}{\partial x^l} + \frac{\partial g_{ml}}{\partial x^k} - \frac{\partial g_{kl}}{\partial x^m} \right)`

    Args:
        x: point at which to evaluate the Christoffel symbol
        get_g: function returning the metric :math:`g_{ab}` at a point

    Returns:
        `Gamma`
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


def get_riemann(x: Array, get_g: Callable[[Array], Array]) -> Array:
    r"""
    Computes the Riemann curvature tensor, `R[r,s,m,n]`, defined as:

        :math:`R^r_{\ smn} = \partial_m \Gamma^r_{\ ns} - \partial_n \Gamma^r_{\ ms} + \Gamma^r_{\ ml} \Gamma^l_{\ ns} - \Gamma^r_{\ nl} \Gamma^l_{\ ms}`

    Args:
        x: point at which to evaluate the Christoffel symbol
        get_g: function returning the metric :math:`g_{ab}` at a point

    Returns:
        `R`
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


def get_ricci_tensor(x: Array, get_g: Callable[[Array], Array]) -> Array:
    r"""
    Computes the Ricci tensor, `Rt[i,j]`, defined as:

        :math:`R_{ij} = R^a_{\ iaj}`

    Args:
        x: point at which to evaluate the Christoffel symbol
        get_g: function returning the metric :math:`g_{ab}` at a point

    Returns:
        `Rt`
    """
    R = get_riemann(x, get_g)
    return jnp.einsum("cacb->ab", R)


def get_ricci_scalar(x: Array, get_g: Callable[[Array], Array]) -> Array:
    r"""
    Computes the Ricci scalar (also called the scalar curvature), `R`, defined as:

        :math:`R = g^{ij} R_{ij}`

    Args:
        x: point at which to evaluate the Christoffel symbol
        get_g: function returning the metric :math:`g_{ab}` at a point

    Returns:
        `R`
    """
    g = get_g(x)  # g_ij
    g_inv = jnp.linalg.inv(g)  # g^ij
    R = get_riemann(x, get_g)  # R^a_bcd
    # g^bd R^a_bad
    return jnp.einsum("bd,abad", g_inv, R)


def check_christoffel2_sym(Gamma: Array, rtol: float = 1e-5, atol: float = 1e-8):
    r"""
    Checks symmetry of a Christoffel symbol. Batched.

    Args:
        Gamma: the Christoffel symbol :math:`\Gamma^i_{\ jk}`
        rtol: relative tolerance
        atol: absolute tolerance

    Raises:
        ValueError: if the Christoffel symbol is not symmetric in the last two indices
    """
    if not jnp.allclose(jnp.einsum("...acb->...abc", Gamma), Gamma, rtol, atol):
        raise ValueError("Gamma is not symmetric in the last two indices")


def check_riemann_sym(g: Array, R: Array, rtol: float = 1e-5, atol: float = 1e-8):
    r"""
    Checks four symmetries/identities of a Riemann curvature tensor. Batched.

    Args:
        g: the metric tensor :math:`g_{ab}(x)`
        R: the Riemann tensor at the same point, :math:`R^r_{\ smn}(x)`
        rtol: relative tolerance
        atol: absolute tolerance

    Raises:
        ValueError: if one of the symmetries/identities does not hold
    """
    R_lower = jnp.einsum("...as,...sbcd->...abcd", g, R)
    if not jnp.allclose(R_lower, -jnp.einsum("...abdc->...abcd", R_lower), rtol, atol):
        raise ValueError("R_rsmn is not antisymmetric in the last two indices")
    if not jnp.allclose(R_lower, -jnp.einsum("...bacd->...abcd", R_lower), rtol, atol):
        raise ValueError("R_rsmn is not antisymmetric in the first two indices")
    if not jnp.allclose(
        R_lower
        + jnp.einsum("...acdb->...abcd", R_lower)
        + jnp.einsum("...adbc->...abcd", R_lower),
        jnp.zeros_like(R_lower),
        rtol,
        atol,
    ):
        raise ValueError("R_rsmn does not satisfy the first Bianchi identity")
    if not jnp.allclose(R_lower, jnp.einsum("...cdab->...abcd", R_lower), rtol, atol):
        raise ValueError("R_rsmn does not satisfy interchange symmetry")


def check_diff_bianchi(
    x: Array,
    get_riemann_pt: Callable[[Array], Array],
    rtol: float = 1e-5,
    atol: float = 1e-8,
):
    """
    Checks the differential/second Bianchi identity. Batched.

    Args:
        x: point at which to check identity
        get_riemann_pt: function returning the Riemann tensor at a point
        rtol: relative tolerance
        atol: absolute tolerance

    Raises:
        ValueError: if the identity does not hold
    """
    dR = vmap(jacfwd(get_riemann_pt))(x)
    if not jnp.allclose(
        dR
        + jnp.einsum("...abdec->...abcde", dR)
        + jnp.einsum("...abecd->...abcde", dR),
        jnp.zeros_like(dR),
        rtol,
        atol,
    ):
        raise ValueError("the differential Bianchi identity does not hold")


def check_ricci_tensor_sym(Rt: Array, rtol: float = 1e-5, atol: float = 1e-8):
    """
    Checks symmetry of a Ricci tensor. Batched.

    Args:
        Rt: the Ricci tensor :math:`R_{ij}`
        rtol: relative tolerance
        atol: absolute tolerance

    Raises:
        ValueError: if Rt is not symmetric
    """
    if not jnp.allclose(jnp.einsum("...ij->...ji", Rt), Rt, rtol, atol):
        raise ValueError("Rt is not symmetric")
