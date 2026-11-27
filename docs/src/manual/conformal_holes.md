# [Conformal holes](@id man-conformal-holes)

!!! info "Before this page"
    [Holes given by a conformal map](@ref th-conformal-mapping), where the
    tensors returned below are derived, and
    [Cylindrical inclusions](@ref man-cylindrical-inclusions), of which this
    page is the extension to sections that are not ellipses.

A [`ConformalHole`](@ref) is an infinite cylindrical cavity whose
cross-section is given by a conformal map with a finite number of terms: a
prismatic pore with rounded corners, a star-shaped channel, a slot. It is a
cavity in elasticity and an insulator in conduction, in a matrix that is
isotropic or transversely isotropic about the axis of the cavity; in a cell of
the plane the elliptic hole takes a matrix of any anisotropy. Its field is
not uniform and it has no Hill tensor; its concentration tensors are exact,
and so is the stress along its edge and in the matrix around it.

```@example manconformal
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Plots
gr()  # hide
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))  # hide
# rounded for display, without negative zeros
tidy(M; digits = 3) = round.(Matrix(M); digits) .+ 0.0
nothing # hide
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
pores.

The self-consistent and the differential schemes solve the hole in their own
estimate. The hole accepts it when it is isotropic in the plane of the
section, that is transversely isotropic about the axis of the pores
([Holes given by a conformal map](@ref th-conformal-mapping), section 7): for
sections with an axis of symmetry of order 3, or of order 5 and more, and for
any section oriented at random in the plane with
`symmetrize = TISymmetrize(axis)`. Triangular pores, then square ones turned
at random about ``\underline{e}_1``:

```@example manconformal
function porous(hole; kw...)
    rve = RVE()
    add_phase!(rve, :matrix, Cylinder(1.0, 1.0), Dict(:C => C₀); fraction = :rest)
    add_phase!(rve, :pores, hole, Dict(:C => void); fraction = 0.1, kw...)
    return rve
end
C_triangles = homogenize(porous(ConformalHole(polygon_map(3, 20)...)), SelfConsistent(), :C)
C_squares = homogenize(porous(square; symmetrize = TISymmetrize((1.0, 0.0, 0.0))), SelfConsistent(), :C)
tidy(TensND.mandel(C_triangles))
```

```@example manconformal
tidy(TensND.mandel(C_squares))
```

The matrix is here a phase of circular section, `Cylinder(1.0, 1.0)`, so that
the medium is invariant along the axis. Pores oriented at random in space are
obtained with `symmetrize = IsoSymmetrize()`.

## In a cell of the plane

When the moduli of the plane are what is wanted, the hole is placed in a
two-dimensional cell, whose properties are tensors of the plane
([Plane problems](@ref man-plane-problems)): it is then its section, and its
tensors are those of the plane. A plate with square holes, in plane stress:

```@example manconformal
c = plane_stress_stiffness(C₀)             # the stiffness of the plate
plate = RVE()
add_phase!(plate, :matrix, Ellipsoid(1.0; dim = 2), Dict(:C => c); fraction = :rest)
add_phase!(plate, :holes, square, Dict(:C => 0 * c); fraction = 0.1)
tidy(TensND.mandel(homogenize(plate, MoriTanaka(), :C)))
```

## The stress along the contour

The edge of a hole of the plane is free: the only stress left there is the
normal stress ``\sigma_{tt}`` on the facets normal to the tangent, the hoop
stress. [`contour_stress`](@ref) gives it under a remote stress of the plane,
at the point ``\omega(\mathrm{e}^{\mathrm{i}\theta})`` of the edge, for an
angle ``\theta`` or a collection of angles. The axes are those of the map,
``z=x_1+\mathrm{i}\,x_2``, and the matrix is isotropic: its moduli do not
enter ([Holes given by a conformal map](@ref th-conformal-mapping), section 9).
The ellipse ``m_1=0.5`` pulled across its major axis carries
``1+2a/b=7`` times the pull at the end of that axis:

```@example manconformal
Σ = Tens([0.0 0.0; 0.0 1.0])               # a unit pull along e₂
contour_stress(ellipse, Σ, 0.0)
```

The point ``\theta=0`` is a corner of the square of [`polygon_map`](@ref): the
stress at a corner grows with the number of terms of the map, which sharpen it.
Along the edge of the square pulled along ``\underline{e}_2``:

```@example manconformal
θ = range(0, π; length = 361)
plot(θ, [contour_stress(ConformalHole(polygon_map(4, n)...), Σ, θ) for n in (1, 3, 10, 30)];
    lw = 2, label = ["1 term" "3 terms" "10 terms" "30 terms"], xlabel = "θ", ylabel = "σtt / Σ₂₂",
    xticks = ([0, π / 2, π], ["0", "π/2", "π"]), size = (600, 320))
```

## The stress in the matrix

[`matrix_stress`](@ref)`(hole, Σ, x)` gives the stress tensor at the point
`x = [x₁, x₂]` of the matrix, in the axes and at the scale of the map; a
collection of points gives a collection of tensors. The matrix is isotropic
and the map has any number of terms
([The stress in the matrix](@ref th-matrix-stress)). Ahead of the corner
``\theta=0`` of the square, at ``x_1=\omega(1)=1+\sum_km_k`` on the axis
``\underline{e}_1``, under the pull along ``\underline{e}_2``:

```@example manconformal
corner = 1 + real(sum(square.coefficients))        # the point ω(1) of the edge
x₁ = corner .+ range(0, 3; length = 301)
σ₂₂ = [canonical_components(σ)[2, 2] for σ in matrix_stress(square, Σ, [[x, 0.0] for x in x₁])]
plot(x₁ .- corner, σ₂₂; lw = 2, label = false, xlabel = "distance to the corner", ylabel = "σ₂₂ / Σ₂₂", size = (600, 300))
```

On the edge the stress is the hoop stress of [`contour_stress`](@ref):

```@example manconformal
first(σ₂₂), contour_stress(square, Σ, 0.0)
```

A point inside the hole is refused:

```@example manconformal
try
    matrix_stress(square, Σ, [0.5, 0.0])
catch err
    print(sprint(showerror, err))
end
```

The point of the unit circle that a point of the matrix comes from is found by
comparing moduli: the function takes numbers and dual numbers, not symbols.

## An elliptic hole in an anisotropic matrix

In a cell of the plane the elliptic hole, whose map has its first term only,
takes a matrix of any anisotropy of the plane: orthotropic, square or of no
symmetry. Its complex potentials are those of Lekhnitskii, in closed form for
the ellipse ([The elliptic hole in an anisotropic matrix](@ref th-lekhnitskii-ellipse)).
Its strain concentration tensor is that of the ellipse of the same
semi-axes, `Ellipsoid(1 + m, 1 - m)`, whose Hill tensor is computed in the same
matrix:

```@example manconformal
c₀ = TensOrtho(40.0, 6.0, 2.5, 3.0, Basis(0.37))      # C₁₁₁₁, C₂₂₂₂, C₁₁₂₂, C₁₂₁₂ in a frame at 0.37 rad
A_map = gradient_gradient_concentration(ellipse, 0 * c₀, c₀)
A_hill = gradient_gradient_concentration(Ellipsoid(1.5, 0.5), 0 * c₀, c₀)
norm(canonical_components(A_map) - canonical_components(A_hill))
```

The stress along the edge, and in the matrix, take the stiffness as their
last argument; they then depend on the anisotropy of the matrix:

```@example manconformal
plot(θ, [contour_stress(ellipse, Σ, θ) contour_stress(ellipse, Σ, θ, c₀)]; lw = 2,
    label = ["isotropic matrix" "matrix c₀"], xlabel = "θ", ylabel = "σtt / Σ₂₂",
    xticks = ([0, π / 2, π], ["0", "π/2", "π"]), size = (600, 320))
```

At the end of the major axis, ``x=(1.5,0)``, the edge is normal to
``\underline{e}_1``: the stress in the matrix there has the one component
``\sigma_{22}``, the hoop stress.

```@example manconformal
tidy(canonical_components(matrix_stress(ellipse, Σ, [1.5, 0.0], c₀))), contour_stress(ellipse, Σ, 0.0, c₀)
```

The hole enters a cell of the plane with such a matrix as any inclusion:

```@example manconformal
sheet = RVE()
add_phase!(sheet, :matrix, Ellipsoid(1.0; dim = 2), Dict(:C => c₀); fraction = :rest)
add_phase!(sheet, :holes, ellipse, Dict(:C => 0 * c₀); fraction = 0.1)
tidy(TensND.mandel(homogenize(sheet, MoriTanaka(), :C), CanonicalBasis{2, Float64}()))
```

In conduction the insulating ellipse takes any symmetric conductivity of the
plane ([Conduction around the insulating ellipse](@ref th-ellipse-conduction)).
Its gradient concentration tensor is that of `Ellipsoid(1.5, 0.5)`, and is not
symmetric when the axes of the conductivity are not those of the hole:

```@example manconformal
k₀ = TensOrtho{2}(3.0, 0.5, Basis(0.4))                # conductivities along and across a frame at 0.4 rad
G_map = canonical_components(gradient_gradient_concentration(ellipse, 0 * k₀, k₀))
G_hill = canonical_components(gradient_gradient_concentration(Ellipsoid(1.5, 0.5), 0 * k₀, k₀))
tidy(G_map), norm(G_map - G_hill)
```

A numeric matrix of no symmetry has its characteristic roots solved
numerically, and carries dual numbers; a symbolic matrix is given as a
`TensISO`, a `TensCubic` or a `TensOrtho`, whose roots are in closed form, and
a symbolic conductivity is accepted as it is. A hole of several terms is
refused in an anisotropic matrix, in elasticity and in conduction:

```@example manconformal
try
    contour_stress(square, Σ, 0.0, c₀)
catch err
    print(first(split(sprint(showerror, err), "; ")))
end
```

The isotropic plate and the plywood plate of Lekhnitskii, with his tables
recomputed, are in
[The elliptic hole, from an isotropic to an anisotropic plate](@ref app-elliptic-hole).

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

- a hole of several terms in a matrix anisotropic in the plane of the
  section, in elasticity and in conduction, refused with a message: the
  complex potentials of such a matrix are functions of other variables, which
  one map brings to the same circle as the variable of the hole for the
  ellipse only ([What is restricted, and why](@ref man-plane-restricted));
- in three dimensions, a cavity in a matrix anisotropic in the plane of its
  section, the elliptic one included: the elliptic [`Cylinder`](@ref) has a
  Hill tensor in such a matrix;
- the stress around a hole of space: [`contour_stress`](@ref) and
  [`matrix_stress`](@ref) are written for a hole of the plane;
- symbols in [`matrix_stress`](@ref), whose point of the circle is chosen by a
  comparison; [`contour_stress`](@ref) accepts them;
- a section that conducts or carries stress: the hole is empty;
- a corner, which a finite map rounds: the tensors converge with the number of
  terms ([Prismatic pores with corners](@ref app-conformal-holes));
- a section given by a function or by pixels rather than by a map: the grid of
  Gaussians treats it in three dimensions
  ([Heterogeneities of any shape](@ref man-gaussian-inclusions)).

## Where to go next

- [Holes given by a conformal map](@ref th-conformal-mapping): the derivation,
  and [The elliptic hole in an anisotropic matrix](@ref th-lekhnitskii-ellipse)
  for an anisotropic matrix.
- [The elliptic hole, from an isotropic to an anisotropic plate](@ref app-elliptic-hole):
  the stress along the edge against the books of Muskhelishvili and
  Lekhnitskii, the stress ahead of the tip, and the hole as an insulator.
- [Prismatic pores with corners](@ref app-conformal-holes): published values,
  polygons, and the effect of the shape of the section.
- [Plane problems](@ref man-plane-problems): sheets and long bodies with such
  holes, in plane stress and in plane strain, as cells of the plane.
- [Choosing an inclusion](@ref man-choosing-inclusion): the hole among the
  other routes.
