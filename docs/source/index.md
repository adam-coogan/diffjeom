# diffjeom

Automated differential geometry, powered by [`jax`](https://github.com/google/jax).
Given a metric function, `diffjeom` can compute:

- [Christoffel symbols](<https://en.wikipedia.org/wiki/Christoffel_symbols#Christoffel_symbols_of_the_second_kind_(symmetric_definition)>)
  ({func}`~diffjeom.get_christoffel2`);
- [Riemann curvature tensor](https://en.wikipedia.org/wiki/Riemann_curvature_tensor#Coordinate_expression)
  ({func}`~diffjeom.get_riemann`);
- [Ricci tensor](https://en.wikipedia.org/wiki/Ricci_curvature)
  ({func}`~diffjeom.get_ricci_tensor`);
- [Ricci scalar/scalar curvature](https://en.wikipedia.org/wiki/Scalar_curvature)
  ({func}`~diffjeom.get_ricci_scalar`).

These work with the usual `jax` transformations like `jit` and `vmap`.

The functions {func}`~diffjeom.check_christoffel2_sym`, {func}`~diffjeom.check_riemann_sym`,
{func}`~diffjeom.check_diff_bianchi` and {func}`~diffjeom.check_ricci_tensor_sym`
are provided to check identities for these objects.

Take a look at the {doc}`full API documentation <api>` for more details.

## Installation

Install `diffjeom` and its few dependencies (`jax`, `jaxlib`, `numpy`) from PyPI
with

```bash
pip install diffjeom
```

Requires python 3.6 or newer.

## Citing

Please acknowledge `diffjeom` if you use it in your work.

```{eval-rst}
.. toctree::
   :maxdepth: 1
   :hidden:

   api
```
