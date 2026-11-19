# [Heterogeneities of any shape](@id man-gaussian-inclusions)

[`GaussianInclusion`](@ref) is a heterogeneity whose shape is any region of
space: a torus, a concave particle, a shape given by a function or by voxels.
Its concentration tensors are computed by the
[Gaussian approximation](@ref th-gaussian-volume) of its integral equation, in
elasticity and in conduction, and it enters the schemes through the
[custom-inclusion contract](@ref man-custom-inclusions), as the
[finite-element inclusions](@ref man-fe-inclusions) do. It needs neither a mesh
nor an extension.

```@setup mangaussincl
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Plots
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
```

## Describing a shape

A shape is an [`AbstractRegion`](@ref) of space, built as the contour of a
[planar crack](@ref man-planar-cracks) is:

| Constructor | Region |
| :--- | :--- |
| [`Region`](@ref)`(Ellipsoid(a, b, c))` | an ellipsoid, with its orientation |
| [`Region`](@ref)`(Supersphere(a, p))`, `Region(Superspheroid(a, c, p))` | a supersphere, a superspheroid |
| [`Torus`](@ref)`(R, a)` | a torus |
| [`Region`](@ref)`(inside, lo, hi)` | any function `inside(x)`, in the box of corners `lo`, `hi` |
| [`PixelRegion`](@ref)`(mask, h)` | voxels: a Boolean array and the size of a voxel |
| `∪`, `∩`, `setdiff` | set operations on regions |

`GaussianInclusion` also takes an `Ellipsoid`, a `Supersphere` or a
`Superspheroid` directly.

```@example mangaussincl
surface = param_surface((u, v) -> ((1 + 0.35cos(v)) * cos(u), (1 + 0.35cos(v)) * sin(u), 0.35sin(v)),
    range(0, 2π; length = 61), range(0, 2π; length = 25))
plotly_scene([surface_trace(surface...; color = "#4063D8")]; uid = "man-gauss-torus", height = 380,
    title = "Torus(1.0, 0.35)")
```

```@example mangaussincl
torus = GaussianInclusion(Torus(1.0, 0.35); n = 16)
```

`n` is the number of nodes along the longest side of the box of the region.
The slice of the torus through its axis, as the grid sees it (the fraction of
each cell that the region covers):

```@example mangaussincl
region = Torus(1.0, 0.35)
grid = GaussianGrid(region, 48)
θ = coverage(region, grid; samples = 3)
heatmap(θ[:, 24, :]'; aspect_ratio = :equal, color = :grays, colorbar = false, axis = false)
```

## Concentration tensors

The two tensors of the contract, for a heterogeneity of property `P₁` in a
matrix of property `P₀`:

```@example mangaussincl
K₀, K₁ = TensISO{3}(1.0), TensISO{3}(50.0)
A = gradient_gradient_concentration(torus, K₁, K₀)       # mean gradient / remote gradient
round.(get_array(A); sigdigits = 3)
```

The mean gradient is four times larger in the plane of the torus than along its
axis. One solve gives both tensors and is kept by the inclusion
for the pair of properties:

```@example mangaussincl
flux_gradient_concentration(torus, K₁, K₀)
fe_assembly_count(torus)
```

## In a scheme

```@example mangaussincl
rve = RVE()
add_phase!(rve, :matrix, Ellipsoid(1.0), Dict(:K => K₀); fraction = :rest)
add_phase!(rve, :tori, torus, Dict(:K => K₁); fraction = 0.1, symmetrize = IsoSymmetrize())
k_tori = first(get_data(homogenize(rve, MoriTanaka(), :K)))

rve = RVE()
add_phase!(rve, :matrix, Ellipsoid(1.0), Dict(:K => K₀); fraction = :rest)
add_phase!(rve, :spheres, Ellipsoid(1.0), Dict(:K => K₁); fraction = 0.1)
k_spheres = first(get_data(homogenize(rve, MoriTanaka(), :K)))
k_tori, k_spheres
```

At the same volume fraction, randomly oriented tori raise the conductivity more
than spheres do.

## Accuracy and cost

The error of the default grid is a few percent against Eshelby's solution, for
contrasts between ``10^{-4}`` and ``10^{3}``
([theory](@ref th-gaussian-volume), §4). On a tilted ellipsoid fifty times more
conducting than the matrix:

```@example mangaussincl
exact = Ellipsoid(1.0, 0.7, 0.5; euler_angles = (0.4, 0.9, 0.0))
A_exact = gradient_gradient_concentration(exact, K₁, K₀)
for n in (12, 16, 24)
    A_n = gradient_gradient_concentration(GaussianInclusion(exact; n), K₁, K₀)
    println("n = ", n, "   relative error: ", round(norm(A_n - A_exact) / norm(A_exact); sigdigits = 2))
end
```

A solve takes seconds in conduction and tens of seconds in elasticity at 24 to
32 nodes; its memory grows as ``n^{3}`` and the inclusion is refused, before
anything is allocated, when it would exceed `memory_limit`. Three limits:

- the reference medium is isotropic (`symmetrize = IsoSymmetrize()` otherwise);
- a contrast beyond ``10^{4}`` on the stiff side is slow, and ``10^{3}`` already
  gives the rigid limit within the error of the method;
- a derivative with respect to the shape is refused; derivatives with respect
  to the properties are available through `ForwardDiff`.

## Which route for a shape that is not an ellipsoid

| Need | Route |
| :--- | :--- |
| any shape, no mesh, a few percent | [`GaussianInclusion`](@ref), this page |
| an axisymmetric pore, to the accuracy of a mesh | [Finite-element inclusions](@ref man-fe-inclusions) |
| many evaluations, differentiable in the shape | [Neural-surrogate inclusions](@ref man-neural-inclusions) |
| a flat crack of any contour | [Planar cracks of any contour](@ref man-planar-cracks) |

## Where to go next

- [Heterogeneities of any shape](@ref th-gaussian-volume): the equations.
- [Custom inclusions](@ref man-custom-inclusions): the contract this inclusion
  enters through.
