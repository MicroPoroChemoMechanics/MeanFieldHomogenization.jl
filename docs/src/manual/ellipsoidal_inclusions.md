# [Ellipsoidal inclusions](@id man-ellipsoidal-inclusions)

!!! info "Before this page"
    [Hill polarization tensors](@ref th-hill-tensors), where the tensor returned
    by `hill_tensor` below is derived, and, for a first reading,
    [Getting started](@ref getting-started).

The ellipsoid is the one shape for which the strain is uniform inside the
inclusion, which is what makes a Hill tensor exist at all
([The Eshelby inclusion problem](@ref th-eshelby-problem)). An
[`Ellipsoid`](@ref) is built from its semi-axes, in any order — the constructor
sorts them decreasing and permutes the local frame to match, so ``a \ge b \ge c``
always holds downstream.

```julia
using MeanFieldHomogenization, TensND
E, ν = 210e3, 0.3
λ = E*ν/((1+ν)*(1-2ν)); μ = E/(2*(1+ν))
C₀ = TensISO{3}(3*(λ+2μ/3), 2μ)

# Sphere
hill_tensor(Ellipsoid(1.0), C₀)

# Prolate spheroid
hill_tensor(Ellipsoid(3.0, 1.0, 1.0), C₀)

# 2D ellipse
hill_tensor(Ellipsoid(1.0, 0.5), TensISO{2}(3*(λ+2μ/3), 2μ))
```

The two aspect ratios that recur everywhere are the in-plane ``\eta = b/a`` and
the flatness ``\omega = c/a``; both are read off the stored semi-axes:

```@setup ellipsoids
using MeanFieldHomogenization
using TensND
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
```

```@example ellipsoids
ell = Ellipsoid(3.0, 1.5, 0.8)
a, b, c = ell.semi_axes
(a = a, b = b, c = c, η = b / a, ω = c / a, shape = MeanFieldHomogenization.shape_trait(ell))
```

```@example ellipsoids
plotly_scene(shape_traces(ell); uid = "man-ellipsoid", height = 440,
    title = "Ellipsoid(3.0, 1.5, 0.8) — dashed guides are the principal semi-axes")
```

Every other spheroid and the degenerate limits below are drawn side by side in
[The inclusion zoo](@ref man-inclusion-gallery).

## Degenerate limits

When an `Ellipsoid` constructor receives a real semi-axis equal to
`Inf` or `0`, it returns the appropriate dedicated type:

| Call | Returned type | See |
| :-- | :-- | :-- |
| `Ellipsoid(Inf, b, c)` with `b, c > 0` | `Cylinder` | [cylindrical inclusions](@ref man-cylindrical-inclusions) |
| `Ellipsoid(a, b, 0)` with `a, b > 0` | `EllipticCrack` | [cracks](@ref man-cracks) |
| `Ellipsoid(Inf, b, 0)` with `b > 0` | `RibbonCrack` | [cracks](@ref man-cracks) |
| `Ellipsoid(Inf, Inf, c)` | `ArgumentError` (slab, out of scope) | |
| `Ellipsoid(a, 0, 0)` | `ArgumentError` (needle, out of scope) | |

The detection is active only for real element types; with symbolic
types (`SymPy.Sym`, `Symbolics.Num`) call the dedicated constructor
(`Cylinder`, `EllipticCrack`, `RibbonCrack`) explicitly.

## Where to go next

The two families an `Ellipsoid` call redirects to are described next:
[Cylindrical inclusions](@ref man-cylindrical-inclusions) for the fiber limit,
then [Cracks](@ref man-cracks) for the flat one. The tutorial
[Hill polarization tensors in practice](@ref tut-hill-tensors) evaluates the
tensors of this page across shapes and matrix symmetries.
