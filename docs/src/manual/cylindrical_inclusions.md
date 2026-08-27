# [Cylindrical inclusions](@id man-cylindrical-inclusions)

Infinite cylinders are handled by the dedicated `Cylinder` type —
a subtype of `AbstractEllipsoidalInclusion{3, T}` that stores only
the two transverse semi-axes and the local basis (the infinite axis
is implicit).

## Construction

```julia
using MeanFieldHomogenization

# Elliptic cross-section — returns Cylinder{EllipticCylindrical, …}
cyl_ell = Cylinder(2.0, 1.0)

# Circular cross-section — returns Cylinder{CircularCylindrical, …}
cyl_circ = Cylinder(1.5)          # shortcut
cyl_circ2 = Cylinder(1.5, 1.5)    # equivalent, detected numerically

# Oriented cylinder — axis along the third global direction
cyl_oriented = Cylinder(2.0, 1.0; euler_angles=(π/2, 0.0, 0.0))

# Via rotation matrix
cyl_rot = Cylinder(2.0, 1.0, [0 0 1; 1 0 0; 0 1 0])
```

The local frame convention mirrors `Prolate`: the cylinder axis is the
first column of the basis, and the transverse semi-axes `(b, c)` (with
`b ≥ c` for real element types) are associated with the second and
third columns respectively.

```@setup cylinders
using MeanFieldHomogenization
using TensND
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
```

```@example cylinders
cyl = Cylinder(2.0, 1.0)
plotly_scene(shape_traces(cyl); uid = "man-cylinder", height = 420,
    title = "Cylinder(2.0, 1.0) — the drawn length is arbitrary, the Hill tensor does not see it")
```

The length shown is a drawing convenience only: the cylinder is the ``a \to \infty``
limit, and no closed form below depends on how much of it is displayed.

## Redirection from `Ellipsoid`

Passing an infinite or zero semi-axis to the `Ellipsoid` constructor
redirects to the appropriate dedicated type — the caller does not
have to switch constructors manually:

| Call | Returned type |
| :-- | :-- |
| `Ellipsoid(Inf, 2.0, 1.0)` | `Cylinder{EllipticCylindrical}` |
| `Ellipsoid(1.0, 1.0, Inf)` | `Cylinder{CircularCylindrical}` |
| `Ellipsoid(2.0, 1.0, 0.0)` | `EllipticCrack{EllipticShape}` |
| `Ellipsoid(1.0, 1.0, 0.0)` | `EllipticCrack{Penny}` |
| `Ellipsoid(Inf, 1.0, 0.0)` | `RibbonCrack` |
| `Ellipsoid(Inf, Inf, 1.0)` | `ArgumentError` |
| `Ellipsoid(2.0, 0.0, 0.0)` | `ArgumentError` |

As for ellipsoids, the detection is active only for real element types (see
[Ellipsoidal inclusions](ellipsoidal_inclusions.md)).

## Hill tensor

```julia
using TensND
K, μ = 175.0, 80.0
C_iso = TensISO{3}(3K, 2μ)

cyl = Cylinder(2.0, 1.0)
P   = hill_tensor(cyl, C_iso)        # TensOrtho (elliptic)
P_c = hill_tensor(Cylinder(1.5), C_iso)  # TensTI{4} (circular, TI axis e₁)
```

All components involving the cylinder axis (`P_{11kl}`) are exactly
zero — reflecting the fact that an infinite cylinder cannot sustain a
non-uniform axial strain.

For a general anisotropic matrix the call
`hill_tensor(cyl, C_aniso)` transparently routes to a dedicated 1D
QuadGK quadrature over the transverse plane.  The `method=:residues`
option is remapped to the same routine (the residue algorithm is not
applicable to a cylinder — see
[theory / Hill polarization tensors](../theory/hill_tensors.md)).

## Conductivity

The 2nd-order Hill tensor is computed via the same
`hill_tensor(…, K₀)` entry point:

```julia
K = TensISO{3}(3.5)
H = hill_tensor(cyl, K)
# H[1, 1]  ≈ 0     (axial component)
# H[2, 2]  ≈ c / (k (b + c))
# H[3, 3]  ≈ b / (k (b + c))
```

Anisotropic conductors use a `K⁻¹/²` change of variable on the
transverse 2×2 sub-matrix, reusing the 2D Newton potentials.

## Auxiliary tensors

`tens_IA`, `tens_UA`, `tens_VA` are all defined for cylinders:

```julia
IA = tens_IA(cyl)   # 2nd-order, Iₐ = 0, sum = 1
UA = tens_UA(cyl)   # 4th-order — TensTI{4} (circular) or TensOrtho (elliptic)
VA = tens_VA(cyl)
```

## ForwardDiff and symbolic element types

Every cylinder path (isotropic, anisotropic, auxiliary tensors) is
ForwardDiff- and symbolic-compatible.  Derivatives with respect to
`b` and `c` propagate through the analytical formulas; symbolic
construction uses structural equality (`isequal(b, c)`) to select the
circular branch.  For symbolic differentiation at `b = c`, prefer the
single-argument constructor `Cylinder(b)` which forces the circular
trait at compile time.
