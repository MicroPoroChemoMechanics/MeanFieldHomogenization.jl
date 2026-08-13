# [Laminates — periodic multilayer cells](@id man-laminates)

A [`Laminate`](@ref) is a periodic unit cell of parallel layers of common
normal `n`: no matrix, no reference medium, and an **exact** effective
behavior rather than an estimate. It is an
[`AbstractHomogenizationCell`](@ref MeanFieldHomogenization.Core.AbstractHomogenizationCell)
alongside [`RVE`](@ref), solved by the [`Laminated`](@ref) scheme.

The theory, including the closed forms and the corrected pseudo-inverse
argument, is on the [laminate theory page](@ref th-laminate).

## Building a cell

Construction mirrors an `RVE`: an empty cell, then layers in stacking order.

```julia
using MeanFieldHomogenization, TensND

C_A = TensISO{3}(3 * 2.0, 2 * 0.8)      # 3κ, 2μ
C_B = TensISO{3}(3 * 0.5, 2 * 0.2)

lam = Laminate(; normal = (0, 0, 1))
add_layer!(lam, :A, Dict(:C => C_A); fraction = 0.3)
add_layer!(lam, :B, Dict(:C => C_B); fraction = 0.7)

C_eff = homogenize(lam, Laminated(), :C)
```

`homogenize(lam, :laminated, :C)` and the aliases `:lam`, `:multilayer` work
too, as for every other scheme.

### Thickness or fraction

Each layer takes **exactly one** of `thickness` (an absolute height) or
`fraction` (a share of the period). Thicknesses are what is stored;
`layer_volume_fraction` derives `f_i = h_i / L` from them.

The distinction matters as soon as an interface is imperfect: interfaces enter
with the weight `1/L`, an interface *density*, so the **absolute period** sets
their size effect. With perfect bonding the result depends on the fractions
alone and `L` is irrelevant.

```julia
lam = Laminate(; normal = (0, 0, 1))
add_layer!(lam, :A, Dict(:C => C_A); thickness = 30.0e-6)     # 30 µm
add_layer!(lam, :B, Dict(:C => C_B); thickness = 70.0e-6)
laminate_period(lam)                # 1.0e-4
layer_volume_fraction(lam, :A)      # 0.3
```

### The frame

Give **one** of:

- `normal = (nx, ny, nz)` — completed into an orthonormal `(ℓ, m, n̂)`. The
  default `(0, 0, 1)` yields a `CanonicalBasis` and the kernel then skips the
  frame rotation entirely;
- `euler_angles = (θ, ϕ, ψ)` — ZYZ angles, as everywhere else in the package;
- `basis = …` — an explicit `TensND` basis whose third axis is the normal
  (the only route for a symbolic frame).

The result is invariant under rotation about `n`, so the discrete choice of
the in-plane pair never leaks into a gradient.

## Elasticity and transport

The physics is selected by the **order of the stored property**, exactly as
for the mean-field schemes — the property key is only a dictionary key.

```julia
lam = Laminate(; normal = (0, 0, 1))
add_layer!(lam, :A, Dict(:C => C_A, :K => TensISO{3}(2.0)); fraction = 0.3)
add_layer!(lam, :B, Dict(:C => C_B, :K => TensISO{3}(0.3)); fraction = 0.7)

homogenize(lam, Laminated(), :C)    # 4th order → elasticity
homogenize(lam, Laminated(), :K)    # 2nd order → transport
```

### The returned type

The symmetry of the result is decided **structurally**, from the declared
classes of the layers:

- one layer with perfect interfaces → the layer property object itself,
  unchanged;
- every layer isotropic, or **major-symmetric** TI about `n` itself
  (`TensISO`, `TensTI{4,T,5}` / `TensTI{2,T,2}` of axis `n`), **and** every
  interface in-plane isotropic → an exact `TensTI` about `n`;
- **anything else → a generic `Tens`** in the laminate basis.

The middle case is worth more than tidiness: a `TensTI` fed back into a
multiscale chain reaches the analytic TI-coaxial Hill branch instead of a
cubature.

The last case is the **general** one, and it covers more than it may look. A
laminate of orthotropic layers is *not* transversely isotropic even when their
axes coincide with the laminate frame, a TI layer whose axis is not `n`
breaks it too, and so does any anisotropic interface however isotropic the
layers. The non-major-symmetric `TensTI{4,T,8}` — what the exact
rotation-group average produces — also falls here deliberately: the
five-coefficient Walpole read-off would discard its `ℓ₃ ≠ ℓ₄` and
antisymmetric content, so the generic wrapper, which is lossless, is used
instead.

## Bounds

`Voigt` and `Reuss` need no matrix phase, so they apply to a laminate. They
bracket the exact answer, and two of the bracketings are **equalities**: the
in-plane shear is exactly Voigt, the out-of-plane response exactly Reuss.

```julia
homogenize(lam, Voigt(), :C)
homogenize(lam, Reuss(), :C)
```

## Imperfect interfaces

The four models of the [layered sphere](@ref th-layered-sphere) are reused
unchanged — a planar interface is the curvature-free case. `interface` is the
condition **on top of** the layer; the last one closes the cell onto the first
by periodicity.

| | primal (field jump) | dual (surface stiffness) |
| :--- | :--- | :--- |
| elasticity | `SpringInterface(kn, kt)` | `MembraneInterface(κs, μs)` |
| transport | `KapitzaInterface(ρ)` | `SurfaceConductiveInterface(ks)` |

```julia
lam = Laminate(; normal = (0, 0, 1))
add_layer!(lam, :A, Dict(:C => C_A); thickness = 0.3,
           interface = SpringInterface(1.0e-3, 2.0e-3))    # compliances
add_layer!(lam, :B, Dict(:C => C_B); thickness = 0.7,
           interface = MembraneInterface(0.07, 0.04))
```

!!! note "The spring fields are compliances"
    `SpringInterface(kn, kt)` follows the `LayeredSpheres` convention:
    ``[\![\underline u]\!] = \boldsymbol{\mathcal K}\cdot(\boldsymbol\sigma\cdot\underline n)``, so `kn = kt = 0` is perfect bonding and `k → ∞`
    decouples the layers.

### Anisotropic interfaces

The four types above are **isotropic in the plane** — that is what the
spherical recurrence requires, since the jump conditions must share the
symmetry of the geometry. A *plane* imposes no such restriction: it has a
well-defined normal and an arbitrary in-plane texture. A laminate therefore
also accepts a full tensor:

| scalar (shared with the sphere) | tensor (laminate only) |
| :--- | :--- |
| `SpringInterface(kn, kt)` | `AnisotropicSpringInterface(𝒦)` — any symmetric 3×3 compliance |
| `MembraneInterface(κs, μs)` | `AnisotropicMembraneInterface(ℂˢ)` — any in-plane surface stiffness (6 coefficients) |
| `SurfaceConductiveInterface(ks)` | `AnisotropicSurfaceConductiveInterface(𝐤ˢ)` — any in-plane surface conductivity |
| `KapitzaInterface(ρ)` | — *already general*: `[T] = ρ qₙ` relates two scalars |

```julia
# a spring with different normal and tangential compliances, and a coupling
𝒦 = [3.0e-3 5.0e-4 0.0; 5.0e-4 8.0e-3 0.0; 0.0 0.0 1.0e-3]
add_layer!(lam, :A, Dict(:C => C_A); thickness = 0.3,
           interface = AnisotropicSpringInterface(𝒦))

# an orthotropic membrane: in-plane Kelvin-Mandel block (ℓ⊗ℓ, m⊗m, √2 ℓ⊗ˢm),
# so the [3,3] entry is 2 Cˢ₁₂₁₂
ℂˢ = [0.20 0.05 0.0; 0.05 0.09 0.0; 0.0 0.0 0.06]
add_layer!(lam, :B, Dict(:C => C_B); thickness = 0.7,
           interface = AnisotropicMembraneInterface(ℂˢ))
```

A tensor field is read as **components in the layer frame** `(ℓ, m, n)` when
given as a plain matrix, or converted from its own basis when given as a
`TensND` tensor. Feeding the tensor form the isotropic values reproduces the
scalar form exactly.

Both oracles stay exact with a full tensor — the compliance simply adds to the
out-of-plane series law, the surface stiffness to the in-plane one. What does
change is the **symmetry of the result**: an anisotropic interface breaks
transverse isotropy just as an anisotropic layer does, so such a cell returns
a generic `Tens` even when every layer is isotropic.

The two families act on complementary halves of the answer: a primal interface
changes the out-of-plane response and leaves the in-plane one untouched, a
dual one does the reverse (and, the interfaces being planar, produces no
traction jump at all).

The displacement jump itself is available:

```julia
interface_jump(lam, 1, E)      # [u] across interface 1 under macroscopic strain E
```

## Per-layer fields

```julia
layer_strain_localization(lam, :A)       # 𝔸_A,  ε_A = 𝔸_A : E,   Σ f 𝔸 = 𝕀
layer_stress_localization(lam, :A)       # 𝔹_A,  σ_A = 𝔹_A : Σ
layer_gradient_localization(lam, :A)     # transport counterparts
layer_flux_localization(lam, :A)
laminate_hill(lam, :A)                   # (ℙ_A, ℚ_A), the two Hill tensors
```

## Sensitivities

Two lenses are specific to a laminate, on top of the shared
`PropertyParameter`:

```julia
thickness(:A)                 # ThicknessParameter — a layer thickness
interface_param(1, :kn)       # InterfaceParameter — an interface scalar
property(:A, :C, :shear)      # shared with the RVE; `phase` names a LAYER here

derivative(lam, Laminated(), thickness(:A); indexer = C -> k_mu(C)[1])
gradient(lam, Laminated(), [thickness(:A), interface_param(1, :kn)];
         indexer = C -> k_mu(C)[2])
```

Differentiating with respect to a thickness is not the same as with respect to
a volume fraction: it also moves the period, hence the interface size effect.
`AmountParameter` therefore **raises** on a laminate, pointing at
`ThicknessParameter`, rather than silently reinterpreting itself.

## Symbolic and autodiff

The kernel is generic in the number type: the pseudo-inverse is a cofactor
inverse (never an SVD) and every intermediate is an `SMatrix` (never an
`MMatrix`, which cannot even be constructed for a symbolic element type). A
`Laminate{Sym}` therefore produces the closed forms directly —
`scripts/38_laminate_symbolic.jl` derives Backus (1962) that way — and
`ForwardDiff` traverses moduli, thicknesses, interface compliances and nested
scales alike.

## Ageing viscoelasticity

Store a [`ViscoLaw`](@ref) per layer and call `homogenize_alv`:

```julia
lam = Laminate(; normal = (0, 0, 1))
add_layer!(lam, :A, Dict(:C => maxwell_relaxation(C_A, [C_A], [3.0])); thickness = 0.4)
add_layer!(lam, :B, Dict(:C => heaviside_law(C_B)); thickness = 0.6)

homogenize_alv(lam, Laminated(), :C; times = 0.0:2.0:10.0)
```

`Voigt` and `Reuss` are available in ALV too. Interfaces stay elastic in this
version.

## In a multiscale chain

A laminate is a cell like any other: chain it explicitly, or nest it
declaratively with [`Homogenized`](@ref) — in either direction. See
[Multiscale models](@ref man-multiscale).

```julia
rve = RVE(:M)
add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_matrix))
add_phase!(rve, :agg, Ellipsoid(1.0),
           Dict(:C => Homogenized(lam, Laminated())); fraction = 0.3)
homogenize(rve, MoriTanaka(), :C)
```

!!! note "A cell, not an inclusion"
    A laminate is homogenized, not embedded. Putting a *laminated inclusion*
    inside a matrix would require its Hill tensor, which is a different
    problem and is not provided.
