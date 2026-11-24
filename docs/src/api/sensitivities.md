# [API — Sensitivities](@id api-sensitivities)

Public lenses and autodiff entry points provided by
`MeanFieldHomogenization.Schemes`.  `ForwardDiff` is a **strong dependency** of
`MeanFieldHomogenization` — the four `derivative` / `gradient` /
`jacobian` / `sensitivity` functions are available out of the box,
and the built-in [`SelfConsistent`](@ref) Newton-Raphson solver
([`NewtonDefault`](@ref)) uses the same machinery internally.

## Lenses

```@docs
AbstractParameter
AmountParameter
AbstractPropertyParameter
PropertyParameter
PropertyDirection
GeometryParameter
DistributionShapeParameter
amount
property
geometry
shape_param
get_param
set_param
get_coefficient
set_coefficient
```

## Autodiff entry points

```@docs
derivative
gradient
jacobian
sensitivity
```

## Second moments

Theory in [Second moments of the local fields](@ref th-second-moments), syntax
in [Sensitivities](@ref man-sensitivities-moments).

```@docs
second_moment
MeanFieldHomogenization.Schemes.moment_along
```
