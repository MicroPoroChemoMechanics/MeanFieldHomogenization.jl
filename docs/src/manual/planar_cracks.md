# [Planar cracks of any contour](@id man-planar-cracks)

[`PlanarCrack`](@ref) is a flat crack whose contour is any region of its plane:
a rectangle, a ring, a star, a region read from an image, a crack whose faces
touch on islands. Its opening is computed by the
[Gaussian approximation](@ref th-gaussian-cracks) of the integral equation of
the crack, in elasticity and in conduction, in a matrix of any anisotropy. It
then behaves as the cracks of the [Cracks](@ref man-cracks) page: same COD
tensor, same contributions, same schemes.

```@setup manplanar
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Plots
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
```

## Describing a contour

A contour is an [`AbstractRegion`](@ref) of the plane of the crack, given in the
coordinates of its first two basis vectors. No mesh is needed: a region only
answers whether a point is inside it. There are four ways to build one.

**A shape of the catalog.**

| Constructor | Region |
| :--- | :--- |
| [`Disk`](@ref)`(a)` | disk of radius `a` |
| [`Ellipse`](@ref)`(a, b)` | ellipse of semi-axes `a`, `b` |
| [`Rectangle`](@ref)`(a, b)` | rectangle of half-sides `a`, `b` |
| [`Superellipse`](@ref)`(a, b, p)` | ``\lvert x/a\rvert^{2p}+\lvert y/b\rvert^{2p}\le 1`` |
| [`Annulus`](@ref)`(a, c)` | ring between the radii `c` and `a` |
| [`StarShape`](@ref)`(ρ, ρmax)` | polar curve ``r\le\rho(\varphi)`` |
| [`Polygon`](@ref)`(points)` | polygon given by its vertices |

Each takes a `center`, and those with axes an `angle`.

**Set operations** on regions, with `∪`, `∩` and `setdiff`. A crack with a
contact island is the contour without the island.

**Any function**, with [`Region`](@ref)`(inside, lo, hi)`: `inside(x)` returns
`true` in the region, contained in the box of corners `lo` and `hi`.

**An image**, with [`PixelRegion`](@ref)`(mask, h)`: a Boolean array and the
size of a pixel, for a contour segmented on a micrograph.

```@example manplanar
regions = [
    "Superellipse, p = 3" => Superellipse(1.0, 0.6, 3.0),
    "ring" => Annulus(1.0, 0.45),
    "star" => StarShape(φ -> 1 + 0.3cos(5φ), 1.3),
    "polygon" => Polygon([(0, 0), (2, 0), (2, 0.7), (0.7, 0.7), (0.7, 2), (0, 2)]),
    "two islands" => setdiff(Disk(1.0), Disk(0.2; center = (0.4, 0.2)) ∪ Disk(0.15; center = (-0.4, -0.3))),
    "function" => Region(x -> abs(x[2]) ≤ 0.25 + 0.2cos(3x[1]), (-1.5, -0.5), (1.5, 0.5)),
]
panels = map(regions) do (name, region)
    grid = GaussianGrid(region, 80)
    heatmap(coverage(region, grid)'; aspect_ratio = :equal, colorbar = false, axis = false,
        color = :grays, title = name, titlefontsize = 9)
end
plot(panels...; layout = (2, 3), size = (720, 460))
```

The panels show what the crack will see of each region: the fraction of every
cell of a grid that the region covers ([`coverage`](@ref)).

## A crack and its COD tensor

```@example manplanar
C₀ = iso_stiffness_E_nu(10.0, 0.3)
crack = PlanarCrack(Rectangle(1.0, 0.5))
```

```@example manplanar
B = cod_tensor(crack, C₀)
round.(get_array(B); sigdigits = 4)
```

The components are those of the frame of the crack: the two axes of the region,
then the normal.

**The reference length.** ``\boldsymbol{B}`` is normalized by a length ``\ell``,
`crack.size`, exactly as the COD tensor of a penny-shaped crack is by its
radius:
``\pi\ell^{3}\boldsymbol{B}\cdot\boldsymbol{\Sigma}\cdot\underline{n}
= \int[\![\underline{u}]\!]\,\mathrm{d}S``. The crack density of a family is
``\varepsilon = N\ell^{3}`` with the same ``\ell``. By default ``\ell`` is the
radius of the disk of the same area; `size = …` sets it. The contribution of a
family, ``\propto\varepsilon\,\boldsymbol{B}``, does not depend on the choice.

```@example manplanar
crack.size, sqrt(crack_area(crack) / π)
```

**Orientation**, by `euler_angles`, as for every crack. **Conduction**: the same
call with a conductivity returns the thermal COD scalar.

```@example manplanar
tilted = PlanarCrack(Rectangle(1.0, 0.5); euler_angles = (0.3, 0.9, 0.0))
cod_tensor(tilted, TensISO{3}(2.5))
```

**Anisotropic matrix**: nothing changes in the call.

```@example manplanar
C_ti = TensTI{4}(30.0, 8.0, 5.0, 12.0, 4.0, (0.0, 0.0, 1.0))
round.(get_array(cod_tensor(tilted, C_ti)); sigdigits = 4)
```

## Accuracy and cost

`n` is the number of nodes along the longest side of the box of the region
(64 by default). The mean openings are computed on the grids of `n` and
`n ÷ 2` nodes and extrapolated ([the edge](@ref th-gaussian-cracks), §4);
`extrapolate = false` keeps the fine grid alone.

```@example manplanar
exact = cod_tensor(EllipticCrack(1.0, 0.5), C₀)        # ℓ³ B = a b² Bᴱ
for n in (16, 32, 64)
    crack_n = PlanarCrack(Ellipse(1.0, 0.5); n, size = 1.0)
    err = norm(get_array(cod_tensor(crack_n, C₀)) - 0.25 * get_array(exact)) / norm(0.25 * get_array(exact))
    println("n = ", n, "   relative error on B: ", round(err; sigdigits = 2))
end
```

What matters is the number of nodes across the narrowest part of the crack: a
ligament between two islands, the width of a ring. Below about eight nodes
there, raise `n`.

A crack keeps its solutions: one solve per reference medium, reused by every
later call. A self-consistent or a differential scheme, which change the
reference medium at each iteration, cost one solve per iteration.

```@example manplanar
cod_tensor(crack, C₀); cod_tensor(crack, C₀)
fe_assembly_count(crack)                                  # one solve, two calls
```

## Contacts between the faces

Faces bonded on islands are the contour without the islands. The reduction of
the normal compliance with the radius of a central contact:

```@example manplanar
radii = 0.1:0.1:0.8
Bnn(crack) = get_array(cod_tensor(crack, C₀))[3, 3]
free = Bnn(PlanarCrack(Disk(1.0)))
reduction = [Bnn(PlanarCrack(Annulus(1.0, c); size = 1.0)) / free for c in radii]
plot(radii, reduction; marker = :circle, xlabel = "contact radius / crack radius",
    ylabel = "B_nn with contact / without", label = "central contact", ylims = (0, 1))
```

A contact of a tenth of the radius, a hundredth of the area, already removes
more than a tenth of the compliance.

Faces linked by springs of stiffness ``\boldsymbol{K}`` take the keyword
`K_interface` of [`cod_tensor`](@ref) (`α_interface` in conduction), as every
crack does; for a `PlanarCrack` the springs enter the integral equation itself
([theory](@ref th-gaussian-cracks), §6).

## In a scheme

A `PlanarCrack` is a crack: it is added to a cell with a crack density.

```@example manplanar
rve = RVE(; distribution_shape = Ellipsoid(1.0))
add_phase!(rve, :matrix, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
add_phase!(rve, :cracks, PlanarCrack(Rectangle(1.0, 0.5)), Dict(:C => C₀);
    density = 0.1, symmetrize = IsoSymmetrize())           # randomly oriented
C_hom = homogenize(rve, MoriTanaka(), :C)
E_nu(best_fit_iso(C_hom))
```

## Opening and intensity factors

[`crack_opening`](@ref) returns the opening on the finest grid: the positions of
the nodes and, at each, the displacement jump in the frame of the crack.
[`opening_surface`](@ref) lays one of its components on the grid. A disk with a
contact island, under a unit normal traction:

```@example manplanar
crack = PlanarCrack(setdiff(Disk(1.0), Disk(0.25; center = (0.35, 0.0))); size = 1.0)
n̂ = tens_basis(crack.basis, 3)
points, u = crack_opening(crack, C₀, n̂ ⊗ n̂)
x, y, Z = opening_surface(crack, last.(u))                 # the normal component
plotly_surface(x, y, Z'; title = "Opening of a disk with a contact island", zlabel = "normal opening",
    uid = "man-planar-opening", colorscale = "Viridis", reversescale = false)
```

[`sif`](@ref) gives the intensity factors at a point of the front, from its
position and the outward normal to the front there, both in the plane of the
crack:

```@example manplanar
ellipse = PlanarCrack(Ellipse(1.0, 0.5); size = 1.0)
K, (K_I, K_II, K_III) = sif(ellipse, C₀, n̂ ⊗ n̂; point = (0.0, 0.5), normal = (0.0, 1.0))
exact = EllipticCrack(1.0, 0.5)                             # the closed form, at the same point
_, (K_I_exact, _, _) = sif(exact, C₀, n̂ ⊗ n̂; y₀ = tens_basis(exact.basis, 2))
K_I, K_I_exact
```

They follow from the opening near the front and the ribbon tangent to it
([`sif_from_dif`](@ref)), within 1 to 2 % of the exact values of an ellipse for
the default grid. The point must be a regular point of the front, not a corner.

## Derivatives

Derivatives with respect to the properties of the matrix go through the solve
by the implicit-function theorem, with `ForwardDiff`:

```@example manplanar
using ForwardDiff
B_of(ν) = get_array(cod_tensor(PlanarCrack(Rectangle(1.0, 0.5); n = 32), iso_stiffness_E_nu(10.0, ν)))[1, 1]
ForwardDiff.derivative(B_of, 0.3)
```

A derivative with respect to the contour is refused with an error: the nodes a
contour contains do not vary continuously with it.

## Which route for a crack that is not an ellipse

| The crack | Route |
| :--- | :--- |
| any flat contour, with or without contacts | [`PlanarCrack`](@ref), this page |
| a ring | [`AnnularCrack`](@ref) (Galerkin, isotropic matrix), or `PlanarCrack(Annulus(a, c))` |
| known intensity factors along circular fronts | [`RiceCrack`](@ref) |
| a curved crack | [`CircularArcCrack`](@ref) |
| an ellipse meshed in a finite cell | [`FEEllipticCrack`](@ref) |

## Where to go next

- [Planar cracks of any contour](@ref th-gaussian-cracks): the equations.
- [Approximation by Gaussians](@ref th-gaussian-approximation): the
  approximation itself.
- [Cracks](@ref man-cracks): what every crack provides.
