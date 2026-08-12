# [API — Conductivity](@id api-conductivity)

The 2nd-order Hill tensor is obtained through the same entry point
[`hill_tensor`](@ref) — this module only registers additional
`_kernel` methods.

The 2nd-order Eshelby tensor ``\boldsymbol{s} = \boldsymbol{P} \cdot \boldsymbol{K}_0``
is likewise obtained via the dispatching
[`eshelby_tensor`](@ref MeanFieldHomogenization.Core.eshelby_tensor) wrapper on a
2nd-order `K₀`.

```@docs
MeanFieldHomogenization.Conductivity
```
