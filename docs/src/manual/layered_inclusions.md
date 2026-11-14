# [Layered inclusions](@id man-layered)

!!! info "Before this page"
    [Layered spheres](@ref th-layered-sphere),
    [Layered cylinder](@ref th-layered-cylinder),
    [Layered spheroids](@ref th-layered-spheroid) and
    [The elastic layered spheroid](@ref th-spheroid-elasticity), where the
    algebra behind the two families used below is derived — the Hervé–Zaoui
    transfer matrices, the confocal spheroidal harmonics and the
    Papkovich–Neuber elastic solution.

## What this is

A layered inclusion is a **pattern**, not a shape: a core surrounded by
concentric shells, each with its own moduli, embedded in the matrix as one
object. It is what a coated particle, a hydrated cement grain or an
interphase-bearing fiber actually is.

The strain is **not uniform inside it**, so Eshelby's result does not apply and
it has no Hill tensor. What it does have is a volume-averaged concentration
tensor, computed exactly by a recurrence over the layers, and that is all a
scheme needs — the family enters through
[level B](@ref man-custom-inclusions), supplying both concentration tensors
directly.

Three families are shipped, and they share their whole surface:

| Type | Geometry | Layers | Physics |
|:--|:--|:--|:--|
| [`LayeredSphere`](@ref MeanFieldHomogenization.LayeredSpheres.LayeredSphere) | concentric spheres | isotropic | elasticity, conduction, aging viscoelasticity |
| [`LayeredCylinder`](@ref MeanFieldHomogenization.LayeredCylinders.LayeredCylinder) | coaxial infinite cylinders | isotropic or transversely isotropic about the axis | elasticity, conduction, aging viscoelasticity |
| [`LayeredSpheroid`](@ref MeanFieldHomogenization.LayeredSpheroids.LayeredSpheroid) | **confocal** spheroids | isotropic | elasticity and conduction |

The general syntax comes first below; the three geometries, and what is
specific to each, come after.

## The general syntax

### 1. Build it: radii, then moduli, one per layer

```@example layered
using MeanFieldHomogenization, TensND

sphere = LayeredSphere(
    (0.5, 0.75, 1.0),                                   # ascending radii
    (iso_stiffness(10.0, 5.0),                          # core
     iso_stiffness(20.0, 9.0),                          # first shell
     iso_stiffness(35.0, 15.0)),                        # outer shell
)
```

Radii ascend, with ``r_0 = 0`` implicit; layer ``k`` lies between ``r_{k-1}``
and ``r_k``. **The constituent properties live in the geometry object**, not in
the `Dict` handed to `add_phase!` — a layered inclusion has no single stiffness
for a phase property to carry, and the `Dict` is a placeholder the kernel
ignores.

That also means **one object per physics**: a tuple of `Tens{4,3}` for
elasticity, a tuple of `Tens{2,3}` for conduction. Mixing them is a constructor
error rather than a wrong answer.

A coated fiber is built with the same two tuples, its radii being measured from
its axis; the axis, ``\underline e_3`` unless `axis` gives another one, is the
one thing a cylinder adds ([Coaxial cylinders](@ref man-layered-cylinders) below):

```@example layered
cylinder = LayeredCylinder(
    (0.5, 0.75, 1.0),                                   # ascending radii, from the axis
    (iso_stiffness(10.0, 5.0),                          # core
     iso_stiffness(20.0, 9.0),                          # first coating
     iso_stiffness(35.0, 15.0)),                        # outer coating
)
```

Everything that follows on this page is written for the sphere and holds for
the cylinder unchanged.

### 2. Use it like any other inclusion

```@example layered
C₀ = iso_stiffness(30.0, 12.0)

rve = RVE()
add_phase!(rve, :m, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
add_phase!(rve, :grains, sphere, Dict(:C => C₀); fraction = 0.3)

k_mu(homogenize(rve, MoriTanaka(), :C))    # bulk and shear moduli
```

Every scheme accepts it, orientation averaging and the sensitivity API included.
Because the inclusion is heterogeneous the package uses its exact identities
``\mathbb N = \mathbb A_{\sigma\varepsilon} - \mathbb C_0:\mathbb A_{\varepsilon\varepsilon}``
and ``\mathbb H = (\mathbb A_{\varepsilon\varepsilon} - \mathbb S_0:\mathbb
A_{\sigma\varepsilon}):\mathbb S_0``, which need no ``\mathbb C_1`` at all.

Supplying the layer fractions and properties — which these types know from their
own geometry — also unlocks the `Voigt` and `Reuss` bounds, which a
heterogeneous inclusion cannot otherwise serve.

### 3. Imperfect interfaces

One interface per radius, the last being the boundary with the matrix:

```@example layered
coated = LayeredSphere(
    (0.5, 1.0), (iso_stiffness(10.0, 5.0), iso_stiffness(35.0, 15.0));
    interfaces = (PerfectInterface{Float64}(), SpringInterface(1e3, 5e2)),
)
```

| Type | Models | Jump |
|:--|:--|:--|
| `PerfectInterface` | continuity | none |
| `SpringInterface` | a compliant film | displacement |
| `MembraneInterface` | a stiff film | traction |
| `KapitzaInterface` | thermal contact resistance | temperature |
| `SurfaceConductiveInterface` | a conductive coating | flux |

!!! warning "`SpringInterface` takes stiffnesses, not compliances"
    `SpringInterface(kn, kt)` — normal and tangential **stiffnesses**. The
    compliances are what the solver stores internally, and
    [`spring_compliances`](@ref MeanFieldHomogenization.LayeredSpheres.spring_compliances)
    / [`spring_stiffnesses`](@ref MeanFieldHomogenization.LayeredSpheres.spring_stiffnesses)
    convert. Passing a compliance where a stiffness is expected is silent and
    inverts the physics: a very compliant film reads as a very stiff one.

An opening spring adds strain that belongs to no layer. The concentration
tensor of the whole sphere, which the schemes use, counts it, including the
opening of the outer interface, as Echoes does. `external = false` leaves that
last one to the matrix:

```@example layered
gradient_gradient_concentration(coated, C₀, C₀)                     # what the schemes use
gradient_gradient_concentration(coated, C₀, C₀; external = false)   # outer opening excluded
```

The averages of the next section are over the material and count no opening.
[Layered spheres §7](@ref th-layered-sphere-jumps) derives the jump terms and
lists the options, per layer included. A Kapitza resistance is treated the
same way in conduction.

### 4. Reading inside the inclusion

The layer averages, and the pointwise fields, are the reason to model a pattern
rather than an equivalent homogeneous particle:

The **averages** read off the geometry directly:

```@example layered
ε∞ = TensISO{3}(1.0)                       # a hydrostatic remote strain

layer_gradient_average(sphere, C₀, ε∞, 2)    # mean strain in shell 2
sphere_gradient_average(sphere, C₀, ε∞)      # over the whole inclusion
```

The **pointwise** fields go through a solved object, built once and reused —
it holds the harmonic amplitudes, so evaluating a profile costs one solve and
not one per point:

```@example layered
fields = LayeredSphereFields(sphere, C₀)

local_gradient(fields, [0.0, 0.0, 0.6], ε∞)  # at a point, Cartesian
local_flux(fields, 0.6, 0.0, 0.0, ε∞)    # or spherical, r θ φ
get_layer(sphere, 0.6)                     # which layer that radius is in
```

with `local_displacement`, `local_temperature`, `local_gradient`, `local_flux`
and the four `local_*_*_concentration` couplings alongside. One binding serves both
families, and the transport fields come from `LayeredSphereTransportFields`.

!!! tip "`side` at an interface"
    A radius that falls *on* an interface is ambiguous, and across a
    `SpringInterface` the displacement genuinely jumps. `side = :outer`
    (the default) or `:inner` selects the limit; `layer = k` forces one.

Three checks come free and are worth using on a new configuration:
``\operatorname{div}\boldsymbol\sigma = \underline 0``, ``\boldsymbol\varepsilon = \operatorname{sym}\nabla\underline u``,
and the continuity of tractions across every interface — with the displacement
jump matching the interface law where there is one.

## The three geometries

### Concentric spheres

[`LayeredSphere`](@ref MeanFieldHomogenization.LayeredSpheres.LayeredSphere) is
the classical Hervé–Zaoui pattern: isotropic layers, any number of them, both
physics. Nothing constrains the radii but that they ascend. Its tensors are
isotropic and have no axis, and the reference medium must be isotropic too;
since a cell of isotropic phases keeps every scheme in that class, the
self-consistent and differential schemes take it with an ordinary spherical
matrix ([Layered sphere, section 9](@ref th-layered-sphere)). The tutorials
[n-layer sphere: volume-averaged concentration tensors](@ref tut-layered-sphere)
and [n-layer sphere: pointwise fields](@ref tut-layered-sphere-local-fields)
work through one in detail.

### [Coaxial cylinders](@id man-layered-cylinders)

[`LayeredCylinder`](@ref MeanFieldHomogenization.LayeredCylinders.LayeredCylinder)
is a coated fiber: an infinite core and coatings of ascending radii around an
axis, ``\underline e_3`` unless `axis` says otherwise. Its layers, and the
reference medium, may be transversely isotropic about that axis, which is the
symmetry of an aligned-fiber composite; a self-consistent or a differential
scheme on aligned fibers then stays within it.

```@example layered
ti(k, l, m, n, p) = ti_from_constants(k + m, k - m, l, n, p, (0.0, 0.0, 1.0))  # Hill's moduli
fiber = LayeredCylinder(
    (0.8, 1.0),
    (ti(30.0, 12.0, 9.0, 150.0, 14.0),                  # carbon-like core
     iso_stiffness(4.0, 1.5)),                          # isotropic sizing
    interfaces = (SpringInterface(20.0, 12.5), PerfectInterface()),
)
N = property_contribution(fiber, iso_stiffness(3.0, 1.2))
ti_constants(best_fit_ti(N, (0.0, 0.0, 1.0)))
```

Its tensors are transversely isotropic about the axis; the concentration
tensor has no major symmetry, the axial and in-plane loadings being coupled.
The derivation is on [Layered cylinder](@ref th-layered-cylinder).

The axis may point in any direction ``\underline n``, given by `axis`. A layer
that is transversely isotropic must then be so about ``\underline n``:
`ti_from_constants` takes the axis as its last argument, as `ti` above does
with ``\underline e_3``. The same fiber, turned to
``\underline n = (1, -2, 2)/3``:

```@example layered
n = (1.0, -2.0, 2.0) ./ 3
ti_n(k, l, m, n₃₃, p) = ti_from_constants(k + m, k - m, l, n₃₃, p, n)
fiber_n = LayeredCylinder(
    (0.8, 1.0),
    (ti_n(30.0, 12.0, 9.0, 150.0, 14.0), iso_stiffness(4.0, 1.5)),
    interfaces = (SpringInterface(20.0, 12.5), PerfectInterface()),
    axis = n,
)
N_n = property_contribution(fiber_n, iso_stiffness(3.0, 1.2))
ti_constants(best_fit_ti(N_n, n))
```

Drag to rotate: the turned fiber, its core and its coating cut open, and its
axis ``\underline n``.

```@example layered
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))  # hide
plotly_scene(shape_traces(fiber_n); uid = "layered-fiber-turned", height = 420,
    title = "The fiber turned to n = (1, −2, 2)/3")
```

The constants about ``\underline n`` are those found about ``\underline e_3``:
turning the fiber turns its contribution and changes nothing else. A
transversely isotropic tensor is stored by its coefficients and its axis, so
no rotation is computed in elasticity. The aging case is in
[Coaxial layered cylinders: a transversely isotropic reference](@ref man-alv-cylinders-ti),
and the tutorial
[Coated fibers: the layered cylinder at work](@ref tut-layered-cylinder) follows
one fiber from its fields to the composite and its creep.

### Confocal spheroids

[`LayeredSpheroid`](@ref MeanFieldHomogenization.LayeredSpheroids.LayeredSpheroid)
takes two tuples — the axis radii and the disk radii — and they are **not
free**:

```@example layered
focal² = 0.27
axis = (0.6, 0.9, 1.2)
disk = map(a -> sqrt(a^2 - focal²), axis)
spheroid = LayeredSpheroid(axis, disk, (TensISO{3}(1.0), TensISO{3}(3.0), TensISO{3}(2.0)))
```

Every layer must share the same ``\text{axis}^2 - \text{disk}^2``: the surfaces
are **confocal**, not similar. That is a modeling constraint and not a
convenience — a family of similar spheroids does not admit the harmonic
separation the solution rests on. A consequence worth expecting: the inner
shells are *more* elongated than the outer ones.

[`layered_spheroid_from_fractions`](@ref MeanFieldHomogenization.LayeredSpheroids.layered_spheroid_from_fractions)
builds one from volume fractions instead, which is usually what is known.

Both physics are served, and elasticity is a genuinely different solve from
conduction: the confocal surfaces are not homothetic, so the harmonic degrees
couple across a *perfect* interface too, and one global system is assembled
rather than a chain of transfer matrices. `Nseries` sets the truncation.

## What is not available, and why

- **Imperfect interfaces on the elastic spheroid.** They fit the sphere and the
  spheroid in conduction, and not the elastic spheroid: a spring or membrane law
  puts an *odd* power of the metric factor ``w = \sqrt{q^2-p^2}`` into the
  matching condition, so it stops being a polynomial identity in ``p`` and takes
  both the exactness of the projection and the banding with it. The workable
  approach is a thin confocal interphase — an extra layer, already handled — but
  not an *equivalent* one: such a shell is exactly ``\omega`` times thicker at
  the equator than at the pole, so it models a compliance that varies along the
  interface.
- **Pointwise elastic fields on the spheroid**, and an arbitrary symmetry axis:
  the solver returns harmonic amplitudes and the sums are simply not written
  yet. Conduction already has them.
- **The exact spherical limit of a confocal spheroid**: the coordinates
  degenerate. Use `LayeredSphere`, or a nearly spherical aspect ratio.
- **A cylinder with incompressible or void layers**, or a reference that is
  not transversely isotropic about its axis: the radial problems divide by the
  plane-strain bulk and the shear moduli, and their decoupling rests on that
  symmetry. The axis itself may point in any direction, in aging
  viscoelasticity as well.

## Where to go next

A morphology outside the shipped families enters the schemes through
[Custom inclusions](@ref man-custom-inclusions), the interface the two layered
families themselves use through level B. The tutorials
[n-layer sphere: volume-averaged concentration tensors](@ref tut-layered-sphere),
[Coated fibers: the layered cylinder at work](@ref tut-layered-cylinder)
and
[The n-layer confocal spheroid: geometry and effective conductivity](@ref tut-layered-spheroid-effective)
work through the three geometries, and
[The interfacial transition zone in concrete](@ref app-itz-concrete) uses a
coated sphere for an aggregate and its transition zone.

- [Layered spheres](@ref th-layered-sphere),
  [Layered cylinder](@ref th-layered-cylinder),
  [Layered spheroids](@ref th-layered-spheroid),
  [The elastic layered spheroid](@ref th-spheroid-elasticity) — the derivations
- [API — LayeredSphere](@ref api-layered-sphere),
  [API — LayeredCylinder](@ref api-layered-cylinder),
  [API — LayeredSpheroid](@ref api-layered-spheroid) — the docstrings
