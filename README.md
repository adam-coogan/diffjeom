# diffjeom

Differential geometry powered by [jax](https://github.com/google/jax). Given a
metric function, this package lets you compute:

- [Christoffel symbols of the second kind](<https://en.wikipedia.org/wiki/Christoffel_symbols#Christoffel_symbols_of_the_second_kind_(symmetric_definition)>) (`get_christoffel2`);
- [The Riemann curvature tensor](https://en.wikipedia.org/wiki/Riemann_curvature_tensor#Coordinate_expression) (`get_riemann`);
- [The Ricci tensor](https://en.wikipedia.org/wiki/Ricci_curvature) (`get_ricci_tensor`);
- [The Ricci scalar/scalar curvature](https://en.wikipedia.org/wiki/Scalar_curvature) (`get_ricci_scalar`).

These work with the usual jax transformations like `jit` and `vmap`.

The functions `check_christoffel2_sym`, `check_riemann_sym`,
`check_diff_bianchi` and `check_ricci_tensor_sym` are provided to check
identities for these objects.

## Getting started

Install with
```bash
pip install diffjeom
```
This will install jax and jaxlib if you don't already have them. The tests
require numpy and can be run with `pytest`. Check them out for some usage
examples.
