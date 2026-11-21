# [Conformal holes](@id man-conformal-holes)

!!! info "Before this page"
    [Holes given by a conformal map](@ref th-conformal-mapping), where the
    tensors returned below are derived, and
    [Cylindrical inclusions](@ref man-cylindrical-inclusions), of which this
    page is the extension to sections that are not ellipses.

A [`ConformalHole`](@ref) is an infinite cylindrical cavity whose
cross-section is given by a conformal map with a finite number of terms: a
prismatic pore with rounded corners, a star-shaped channel, a slot. It is a
cavity in elasticity and an insulator in conduction, in an isotropic matrix.
Its field is not uniform and it has no Hill tensor; its concentration tensors
are exact.

```@setup manconformal
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Plots
gr()
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
# rounded for display, without negative zeros
tidy(M; digits = 3) = round.(Matrix(M); digits) .+ 0.0
```

## Declaring a hole

The contour of the section is the image of the unit circle by

```math
z=R\,\Bigl(\zeta+\sum_{k=1}^{N}m_k\,\zeta^{-k}\Bigr),\qquad|\zeta|=1 ,
```

and the hole is declared by the coefficients ``m_k``, all of them or the
nonzero ones as pairs `degree => coefficient`. The size ``R`` is not given:
the tensors are per unit volume of the cavity.

```@example manconformal
ellipse = ConformalHole((0.5,))                    # m₁ = 0.5: semi-axes 1.5 R and 0.5 R
triangle = ConformalHole(2 => 0.25)                # m₂ = 0.25: three rounded corners
square = ConformalHole(polygon_map(4, 3)...)       # the first three terms of the map of a square
```

[`polygon_map`](@ref) returns the terms of the map of a regular polygon; the
more terms, the sharper the corners. The coefficients must satisfy
``\sum k\,|m_k|\le1``, which makes the map one-to-one
([Holes given by a conformal map](@ref th-conformal-mapping), section 2); a
complex coefficient turns the corresponding lobe.

```@example manconformal
contour(hole) = [ζ + sum(m * ζ^(-k) for (k, m) in pairs(hole.coefficients)) for ζ in cis.(range(0, 2π; length = 361))]
holes = ["m₁ = 0.5" => ellipse, "m₂ = 0.25" => triangle, "square, 3 terms" => square,
    "pentagon, 20 terms" => ConformalHole(polygon_map(5, 20)...), "m₃ = 0.3, m₁ = 0.1i" => ConformalHole(1 => 0.1im, 3 => 0.3)]
plot([plot(real.(contour(h)), imag.(contour(h)); aspect_ratio = 1, fill = (0, 0.35, :orange), color = :black,
        legend = false, axis = false, grid = false, title = name, titlefontsize = 8) for (name, h) in holes]...;
    layout = (1, 5), size = (860, 200))
```

As for a `Cylinder`, the axis of the cavity is the first vector of the basis
and the section lies in the plane of the two others, with
``z=x_2+\mathrm{i}\,x_3``. The basis is given by the ZYZ `euler_angles`, or by
`basis`:

```@example manconformal
tilted = ConformalHole(polygon_map(3, 4)...; euler_angles = (0.0, π / 3, 0.0))
plotly_scene(shape_traces(tilted); uid = "man-conformal", height = 420, title = "A prismatic cavity of triangular section, its axis <u><i>e</i></u><sub>1</sub> tilted by π/3") # hide
```

## Its concentration tensors

In elasticity the strain concentration tensor is the mean strain of the
cavity, defined from the displacement of its surface, per unit remote strain:

```@example manconformal
C₀ = iso_stiffness(20.0, 12.0)             # bulk and shear moduli of the matrix
void = 0 * C₀                              # the property of the cavity, which is not read
A = gradient_gradient_concentration(square, void, C₀)
tidy(TensND.mandel(A))
```

The matrix above is in the Kelvin-Mandel notation, in the order
``11, 22, 33, 23, 13, 12``. The first row says that the axial strain of the
cavity is the remote one. No stress acts in the cavity; its compliance
contribution, per unit volume, follows:

```@example manconformal
H = inverse_property_contribution(square, void, C₀)
tidy(TensND.mandel(H); digits = 4)
```

In conduction the hole is an insulator, and its gradient concentration tensor
depends on the first coefficient of the map and on the area of the section
only:

```@example manconformal
K₀ = TensISO{3}(2.0)
tidy(canonical_components(gradient_gradient_concentration(square, 0 * K₀, K₀)))
```

For the ellipse both are those of the elliptic `Cylinder`, obtained from its
Hill tensor:

```@example manconformal
norm(canonical_components(gradient_gradient_concentration(ellipse, void, C₀)) -
    canonical_components(gradient_gradient_concentration(Cylinder(1.5, 0.5), void, C₀)))
```

## In a scheme

The hole is added to a cell as any inclusion. Here aligned prismatic pores of
square section, at a porosity of 10 %:

```@example manconformal
rve = RVE()
add_phase!(rve, :matrix, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
add_phase!(rve, :pores, square, Dict(:C => void); fraction = 0.1)
C = homogenize(rve, MoriTanaka(), :C)
tidy(TensND.mandel(C))
```

The effective stiffness has the symmetry of the square about the axis of the
pores. Randomly oriented pores are obtained with `symmetrize = IsoSymmetrize()`
on the phase, which the self-consistent and differential schemes need: the
hole is solved in an isotropic medium only.

## Derivatives

The coefficients of the map and the moduli of the matrix are differentiated
through the solution:

```@example manconformal
using ForwardDiff
modulus(m) = canonical_components(homogenize(
    (r = RVE(); add_phase!(r, :matrix, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest);
        add_phase!(r, :pores, ConformalHole(3 => m), Dict(:C => void); fraction = 0.1); r),
    MoriTanaka(), :C))[2, 2, 2, 2]
ForwardDiff.derivative(modulus, 0.1)
```

## What is not covered

- an anisotropic reference medium, refused with a message;
- a section that conducts or carries stress: the hole is empty;
- a corner, which a finite map rounds: the tensors converge with the number of
  terms ([Prismatic pores with corners](@ref app-conformal-holes));
- a section given by a function or by pixels rather than by a map: the grid of
  Gaussians treats it in three dimensions
  ([Heterogeneities of any shape](@ref man-gaussian-inclusions)).

## Where to go next

- [Holes given by a conformal map](@ref th-conformal-mapping): the derivation.
- [Prismatic pores with corners](@ref app-conformal-holes): published values,
  polygons, and the effect of the shape of the section.
- [Choosing an inclusion](@ref man-choosing-inclusion): the hole among the
  other routes.
