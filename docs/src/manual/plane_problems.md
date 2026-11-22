# [Plane problems](@id man-plane-problems)

!!! info "Before this page"
    [Cells and schemes](@ref man-schemes), for the cell and the schemes used
    here, and [Cylindrical inclusions](@ref man-cylindrical-inclusions), whose
    sections are the inclusions of the plane.

A medium invariant along an axis is studied in the plane normal to it: a
fiber composite across its fibers, a rock with parallel channels, a perforated
sheet. This page shows how such a two-dimensional medium is homogenized, in
plane strain or in plane stress. The cell holds tensors and inclusions of the
plane, and [`homogenize`](@ref) is called as in three dimensions: the schemes
are the same, on smaller tensors.

```@setup manplane
using MeanFieldHomogenization
using TensND
using Printf
```

## Two states, one problem of the plane

![A long prism in plane strain, a thin plate in plane stress, and the section they share](../assets/theory/plane_problem.svg)

The axis is ``\underline{e}_1`` and the plane is that of
``(\underline{e}_2,\underline{e}_3)``, with Greek indices; in a cell of the
plane these two vectors are the first and the second of its basis. The
materials are here isotropic in the plane: isotropic, or transversely isotropic
about the axis. Two states are met in practice.

- **Plane strain**, ``\varepsilon_{11}=\varepsilon_{12}=\varepsilon_{13}=0``:
  a body that is long along the axis. The law of the plane is

  ```math
  \sigma_{\alpha\beta}=(k-\mu)\,\varepsilon_{\gamma\gamma}\,\delta_{\alpha\beta}+2\mu\,\varepsilon_{\alpha\beta},
  \qquad k=\frac{C_{2222}+C_{2233}}{2},\quad\mu=\frac{C_{2222}-C_{2233}}{2} ,
  ```

  with ``k`` the plane-strain bulk modulus and ``\mu`` the shear modulus of
  the plane.
- **Plane stress**, ``\sigma_{11}=\sigma_{12}=\sigma_{13}=0``: a thin plate
  loaded in its plane, for the mean fields through its thickness
  ([muskhelishvili1977](@cite), section 26). The axial strain follows from
  ``\sigma_{11}=0``, and the same law holds with ``k`` replaced by

  ```math
  k^*=k-\frac{C_{1122}^2}{C_{1111}} .
  ```

For an isotropic material of Young modulus ``E`` and Poisson ratio ``\nu``,
``\mu=E/(2(1+\nu))`` in both states, ``k=\mu/(1-2\nu)`` and
``k^*=E/(2(1-\nu))``. The two states are thus one problem of the plane, with
two moduli, and the stiffness of the plane is the isotropic tensor
``2k\,\mathbb{J}+2\mu\,\mathbb{K}`` of dimension 2, a `TensISO{4,2}`. TensND
builds it from the three-dimensional stiffness:

```@example manplane
C = iso_stiffness_young_poisson(70.0, 0.25)                  # E and ν
bulk_shear(plane_strain_stiffness(C)), bulk_shear(plane_stress_stiffness(C))
```

A transversely isotropic material is isotropic in the plane normal to its
axis:

```@example manplane
Cₜ = ti_from_constants(30.0, 60.0, 0.25, 0.2, 9.0, (1.0, 0.0, 0.0); form = :engineering, role = :stiffness)
bulk_shear(plane_stress_stiffness(Cₜ))                       # E₁/(2(1 - ν₁₂)) and E₁/(2(1 + ν₁₂))
```

With symbols the two moduli come out in closed form:

```@example manplane
import Symbolics
Symbolics.@variables E ν
Symbolics.simplify.(bulk_shear(plane_stress_stiffness(iso_stiffness_young_poisson(E, ν))))
```

The moduli of a tensor of the plane are read back with the same functions.
`young_poisson` returns the constants of the three-dimensional material that
has this stiffness in plane strain; `plane_strain_to_plane_stress` converts
them into those of the plate that has it in plane stress:

```@example manplane
c = plane_stress_stiffness(C)
young_poisson(c), plane_strain_to_plane_stress(young_poisson(c)...)
```

## The cell of a plane medium

The phases of the cell are inclusions of the plane, and their properties
tensors of the plane:

| section | phase |
| :--- | :--- |
| disk | [`Ellipsoid`](@ref)`(r; dim = 2)` |
| ellipse of semi-axes `a` and `b`, turned by `angle` | `Ellipsoid(a, b; angle)` |
| hole given by a conformal map | [`ConformalHole`](@ref) |

A sheet with circular holes, at the porosity ``f``, in plane stress:

```@example manplane
disk = Ellipsoid(1.0; dim = 2)
function sheet(c, f; hole = disk, kw...)
    rve = RVE()
    add_phase!(rve, :matrix, disk, Dict(:C => c); fraction = :rest)
    add_phase!(rve, :holes, hole, Dict(:C => 0 * c); fraction = f, kw...)
    return rve
end
homogenize(sheet(c, 0.1), MoriTanaka(), :C)
```

The estimate is a tensor of the plane, in the state of the properties given:
here the plane-stress stiffness of the perforated sheet. For circular holes
Young's modulus of the sheet does not depend on Poisson's ratio, and the
schemes have the closed forms ``(1-f)/(1+2f)`` (Mori and Tanaka), ``1-3f``
(self-consistent) and ``(1-f)^3`` (differential):

```@example manplane
f = 0.1
plate_modulus(t) = first(plane_strain_to_plane_stress(young_poisson(t)...))
for (name, scheme, closed_form) in (("Mori-Tanaka", MoriTanaka(), (1 - f) / (1 + 2f)), ("self-consistent", SelfConsistent(), 1 - 3f), ("differential", DifferentialScheme(), (1 - f)^3))
    @printf "%-16s  E/E₀ = %.6f   closed form %.6f\n" name plate_modulus(homogenize(sheet(c, f), scheme, :C)) / 70.0 closed_form
end
```

The same cell with the plane-strain stiffness gives the moduli of the long
body:

```@example manplane
bulk_shear(homogenize(sheet(plane_strain_stiffness(C), 0.1), SelfConsistent(), :C))
```

## Sections that are not disks

A hole given by a conformal map is declared as in three dimensions
([Conformal holes](@ref man-conformal-holes)); in a cell of the plane it is
its section. The estimate is then a general tensor of the plane; a section
with an axis of symmetry of order 3, or of order 5 and more, leaves it
isotropic, and its moduli are read on its isotropic part. Triangular holes:

```@example manplane
triangle = ConformalHole(polygon_map(3, 30)...)
perforated = homogenize(sheet(c, 0.1; hole = triangle), SelfConsistent(), :C)
bulk_shear(isotropify(perforated))
```

Aligned ellipses make the sheet anisotropic in its plane, and the estimate is
a general tensor of the plane:

```@example manplane
aligned = homogenize(sheet(c, 0.1; hole = Ellipsoid(3.0, 1.0)), MoriTanaka(), :C)
round.(TensND.mandel(aligned); digits = 3)
```

Sections oriented at random in the plane are obtained with
`symmetrize = IsoSymmetrize()` on the phase, the average over the rotations
and the reflections of the plane:

```@example manplane
bulk_shear(homogenize(sheet(c, 0.1; hole = Ellipsoid(3.0, 1.0), symmetrize = IsoSymmetrize()), MoriTanaka(), :C))
```

## Conduction

In conduction the property of the plane is the conductivity across the axis,
a second-order tensor of dimension 2, and the two states do not differ.
Conducting disks in a matrix:

```@example manplane
rve = RVE()
add_phase!(rve, :matrix, disk, Dict(:K => TensISO{2}(1.0)); fraction = :rest)
add_phase!(rve, :disks, disk, Dict(:K => TensISO{2}(5.0)); fraction = 0.2)
homogenize(rve, MoriTanaka(), :K)
```

## The same medium in three dimensions

A medium invariant along an axis is also a three-dimensional cell of aligned
cylinders ([`Cylinder`](@ref), [`LayeredCylinder`](@ref), `ConformalHole`),
which is in plane strain in the plane of the sections. Its estimate carries,
besides the moduli of the plane, the axial modulus and the shear modulus along
the axis; its block of the plane is the estimate of the plane cell.

```@example manplane
function prisms(f)
    rve = RVE()
    add_phase!(rve, :matrix, Cylinder(1.0, 1.0), Dict(:C => C); fraction = :rest)
    add_phase!(rve, :holes, Cylinder(1.0, 1.0), Dict(:C => 0 * C); fraction = f)
    return rve
end
C₃ = canonical_components(homogenize(prisms(0.1), SelfConsistent(), :C))    # axis 1, plane (2, 3)
C₂ = canonical_components(homogenize(sheet(plane_strain_stiffness(C), 0.1), SelfConsistent(), :C))
C₃[2, 2, 2, 2] - C₂[1, 1, 1, 1], C₃[2, 2, 3, 3] - C₂[1, 1, 2, 2]
```

The plane cell is the route to take when the moduli of the plane are what is
wanted: it carries tensors of dimension 2 only, and it is the one that gives
plane stress.

## What is not covered

- coated sections: a [`LayeredCylinder`](@ref) is a three-dimensional
  inclusion, to be used in the cell of the previous section, in plane strain;
- cracks of the plane, and aging viscoelasticity in the plane;
- the average over the rotations alone of a section without a mirror line:
  `IsoSymmetrize` also averages over the reflections.

## Where to go next

- [Holes given by a conformal map](@ref th-conformal-mapping), sections 7 and
  8: the hole in a matrix isotropic in the plane, and in a plate.
- [Prismatic pores with corners](@ref app-conformal-holes): sheets with
  polygonal holes.
- The manual of TensND on the elastic moduli, for the conversions between the
  two states: [`plane_stress_stiffness`](@extref TensND TensND.plane_stress_stiffness).
