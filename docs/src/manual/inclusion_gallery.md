# [The inclusion zoo](@id man-inclusion-gallery)

Every morphology `MeanFieldHomogenization` knows, drawn from the object the code actually
computes with. The figures below are **interactive** — drag to rotate, scroll to
zoom — and each one is built by passing the very same inclusion instance that a
[`hill_tensor`](@ref) or [`homogenize`](@ref) call would receive. There is no
separate drawing to keep in sync: if a constructor sorts semi-axes or permutes a
frame, the picture moves with it.

```@setup zoo
using MeanFieldHomogenization
using TensND
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
```

The zoo splits in three, along the line that matters for the theory:

| Family | Why it belongs here | Enters a scheme through |
| :--- | :--- | :--- |
| ellipsoids and their degenerate limits | Eshelby's uniformity result holds — a Hill tensor exists | gate A, [`hill_tensor`](@ref) |
| composite inclusions (layered spheres and spheroids) | a *pattern* has no Hill tensor; its concentration tensors do exist | gate B |
| everything else (superspheres, meshes, surrogates) | no closed form at all | gate A, B or C — see [custom inclusions](@ref man-custom-inclusions) |

## Ellipsoids

The general case: three distinct semi-axes ``a \ge b \ge c``, with the two
aspect ratios the rest of the documentation uses — the in-plane
``\eta = b/a`` and the flatness ``\omega = c/a``
([Notation](../theory/notation.md#Ellipsoid-geometry)).

```@example zoo
ell = Ellipsoid(3.0, 1.5, 0.8)
plotly_scene(shape_traces(ell); uid = "zoo-triaxial",
    title = "Triaxial ellipsoid — a = 3, b = 1.5, c = 0.8 (η = 0.5, ω ≈ 0.27)")
```

A **spheroid** has an axis of revolution. `Spheroid` takes the single
aspect ratio ``\omega``, fixes the two equatorial semi-axes to 1, and lets the
`Ellipsoid` constructor do the sorting — which is worth watching, because it is
what every closed form downstream assumes:

```@example zoo
oblate, prolate = Spheroid(0.2), Spheroid(5.0)
(oblate.semi_axes, MeanFieldHomogenization.shape_trait(oblate)),
(prolate.semi_axes, MeanFieldHomogenization.shape_trait(prolate))
```

`Spheroid(5.0)` is stored as ``(a, b, c) = (5, 1, 1)`` with its frame permuted,
so the long axis still points along the global ``\underline{e}_3`` — the sorting
is bookkeeping, not a rotation of the physics.

```@example zoo
plotly_scene(shape_traces(oblate); uid = "zoo-oblate", height = 420,
    title = "Oblate spheroid, ω = 0.2 — a disc-like inclusion")
```

```@example zoo
plotly_scene(shape_traces(prolate); uid = "zoo-prolate", height = 420,
    title = "Prolate spheroid, ω = 5 — a needle-like inclusion")
```

## Degenerate limits

Sending a semi-axis to `Inf` or `0` leaves the ellipsoid family and lands on a
dedicated type — the redirection table is in
[Ellipsoidal inclusions](@ref man-ellipsoidal-inclusions). The three shapes it
produces are the ones below.

An **infinite cylinder** is `Ellipsoid(Inf, b, c)`: the fiber limit, unbounded
along its axis. Only the transverse semi-axes are stored, since the Hill tensor
no longer depends on the length.

```@example zoo
cyl = Cylinder(1.5, 0.8)
plotly_scene(shape_traces(cyl); uid = "zoo-cylinder", height = 420,
    title = "Elliptic cylinder — transverse semi-axes b = 1.5, c = 0.8, L → ∞")
```

A **penny-shaped crack** is `Ellipsoid(a, a, 0)`: zero volume, finite surface,
and therefore a *density* rather than a volume fraction. Its normal
``\hat{\underline{n}}`` is the third column of its frame, and it is the direction
everything about a crack is written in.

```@example zoo
pc = PennyCrack(1.0)
plotly_scene(shape_traces(pc); uid = "zoo-penny", height = 420,
    title = "Penny-shaped crack — a = b = 1, c = 0, with its normal n̂")
```

Tilting a crack is a matter of Euler angles, and the frame carries the
orientation into every tensor built from it:

```@example zoo
tilted = EllipticCrack(1.0, 0.4; euler_angles = (π / 4, π / 3))
plotly_scene(shape_traces(tilted); uid = "zoo-elliptic-crack", height = 420,
    title = "Elliptic crack, a = 1, b = 0.4, tilted by (π/4, π/3)")
```

A **ribbon crack** is the remaining limit, `Ellipsoid(Inf, b, 0)` — a tunnel
crack, flat *and* unbounded:

```@example zoo
rb = RibbonCrack(0.5)
plotly_scene(shape_traces(rb); uid = "zoo-ribbon", height = 420,
    title = "Ribbon crack — half-width b = 0.5, unbounded along ℓ̂")
```

## Composite inclusions

A layered pattern has **no Hill tensor**: the strain is not uniform inside it, so
Eshelby's result does not apply. What does exist is its volume-averaged
concentration tensor, delivered by the Hervé–Zaoui recurrences, and that is
enough for every scheme ([Layered spheres](../theory/layered_sphere.md)).

```@example zoo
ls = LayeredSphere(
    (0.5, 0.75, 1.0),
    (iso_stiffness(10.0, 5.0), iso_stiffness(20.0, 9.0), iso_stiffness(35.0, 15.0)),
)
plotly_scene(shape_traces(ls); uid = "zoo-layered-sphere", height = 460,
    title = "Three-layer composite sphere — shells cut open at r = 0.5, 0.75, 1")
```

The **confocal spheroid** is the anisotropic counterpart: the layer boundaries
share their foci rather than their center, which is what keeps the transfer
matrices tractable ([Layered spheroids](../theory/layered_spheroid.md)). Confocal
means ``\text{axis}^2 - \text{disk}^2`` is the same for every layer, so the inner
shells are *more* elongated than the outer ones.

```@example zoo
focal² = 0.27
axis_radii = (0.6, 0.9, 1.2)
disk_radii = map(a -> sqrt(a^2 - focal²), axis_radii)
sp = LayeredSpheroid(axis_radii, disk_radii,
    (TensISO{3}(1.0), TensISO{3}(3.0), TensISO{3}(2.0)))
plotly_scene(shape_traces(sp); uid = "zoo-layered-spheroid", height = 460,
    title = "Three-layer confocal prolate spheroid (conduction)")
```

## Beyond the catalog

Nothing in the schemes requires an ellipsoid. A morphology answers one of the
three gates and becomes a first-class citizen — the
[custom-inclusion contract](@ref man-custom-inclusions). The **supersphere**

```math
|x/a|^{2p} + |y/a|^{2p} + |z/a|^{2p} \le 1
```

is the family the package's non-ellipsoidal routes were built for. It has no
closed-form Hill tensor at any ``p`` but the sphere, and it spans a wide range
of morphologies from one parameter: ``p = 1`` is the sphere, ``p = 1/2`` the
regular octahedron, ``p < 1/2`` a **concave** body with conical points on the
axes, and ``p \to \infty`` the cube.

!!! warning "Two conventions for `p`, differing by a factor of two"
    The implicit exponent is ``2p``, after the
    Sevostianov–Giraud–Chen–Grgic literature, and that is what
    [`Supersphere`](@ref MeanFieldHomogenization.Superspheres.Supersphere) and
    [`shape_exponent`](@ref MeanFieldHomogenization.Superspheres.shape_exponent)
    use. The plotting helper `supersphere_surface` in
    `scripts/common/docviz.jl` takes the **exponent itself**, so its sphere is
    at `2.0` and the figure below, drawn at `0.6`, shows ``p = 0.3``.

```@example zoo
X, Y, Z = supersphere_surface(0.6, 1.0, 1.0, 1.0)
plotly_scene([surface_trace(X, Y, Z; color = "#f0ad4e", opacity = 0.9)];
    uid = "zoo-supersphere", height = 440,
    title = "Supersphere, p = 0.3 — concave, with conical points on the axes")
```

A cavity of this shape reaches every scheme through
[`FESupershapePore`](@ref MeanFieldHomogenization.FESupershapePore), which
solves it on a truncated cell — see
[finite-element inclusions](@ref man-fe-inclusions). Its compliance
contribution is **cubic**, so it has three independent constants and not two;
that third constant is real, and measured
[far above its own error bar](@ref api-cubic).

A **laminate** is the other extreme: a periodic stack, exact rather than
estimated, handled by its own algebra ([Laminates](@ref man-laminates)).

```@example zoo
plotly_scene(laminate_traces([0.30, 0.15, 0.40, 0.15]);
    uid = "zoo-laminate", height = 420,
    title = "Four-layer periodic cell, normal n̂")
```

## What a scheme actually sees

None of the above is what a real material looks like. A scheme sees a
*population*: many inclusions, each treated as if alone in a reference medium.
The realization below — 70 non-overlapping spheres inside a cell, drawn with a
fixed seed — is the picture behind a volume fraction.

```@example zoo
plotly_scene(rve_traces(; n = 70, semi_axes = (0.055, 0.055, 0.055));
    uid = "zoo-rve-spheres", height = 480,
    title = "RVE realization: isotropic distribution of spherical inclusions")
```

Flatten the inclusions and the same picture becomes a cracked medium, where the
amount is a density and no volume is involved:

```@example zoo
plotly_scene(rve_traces(; n = 70, semi_axes = (0.10, 0.10, 0.004), seed = 2024);
    uid = "zoo-rve-cracks", height = 480,
    title = "RVE realization: randomly oriented flat cracks")
```

Align them and the effective behavior turns transversely isotropic — same
density, different tensor:

```@example zoo
plotly_scene(rve_traces(; n = 70, semi_axes = (0.10, 0.10, 0.004), seed = 2024,
        orientation = (0.0, 0.0, 1.0));
    uid = "zoo-rve-cracks-aligned", height = 480,
    title = "RVE realization: parallel cracks, all normals along e₃")
```

Both crack populations are worked out, numbers included, in
[Crack distributions: isotropic or parallel](@ref tut-crack-distributions).

## See also

- [Ellipsoidal inclusions](@ref man-ellipsoidal-inclusions) — constructors and
  degenerate-limit redirections
- [Cylindrical inclusions](@ref man-cylindrical-inclusions), [Cracks](@ref man-cracks)
- [Custom inclusions](@ref man-custom-inclusions), [finite elements](@ref man-fe-inclusions),
  [neural surrogates](@ref man-neural-inclusions)
- `scripts/common/docviz.jl` — the figure helpers used on this page, reusable in
  your own scripts
