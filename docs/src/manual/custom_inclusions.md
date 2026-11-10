# [Custom inclusions](@id man-custom-inclusions)

!!! info "Before this page"
    [Concentration and contribution tensors](@ref th-concentration), which defines
    the concentration and contribution tensors that the three entry levels below
    supply or derive, and [Layered inclusions](@ref man-layered), the built-in
    family that enters through level B.

`MeanFieldHomogenization` ships ellipsoids, cylinders, elliptical and ribbon cracks and
multi-layer patterns. When your morphology is none of those — a non-ellipsoidal
shape, a pattern whose response comes out of a finite-element solve, a
hand-crafted approximate formula — you can still plug it into **every**
homogenization scheme, in elasticity and in transport, without touching the
package.

This is the counterpart of the `user_inclusion` mechanism of the C++/Python
*echoes* codebase. Two options are available:

- **subtype** [`AbstractCustomInclusion`](@ref) and add methods — the durable
  option for a morphology you will reuse;
- instantiate [`CustomInclusion`](@ref MeanFieldHomogenization.CustomInclusion) with
  **callbacks** — the quick option for prototyping.

Both go through the same interface, spelled out in full on the developer page
[Adding a new inclusion](@ref dev-adding-inclusion). This page shows how to use it.

## The idea in one line

An inclusion contributes to a scheme as **a quantitative amount times a
size-independent contribution**:

```math
\Delta \mathbb C = \underbrace{f}_{\text{volume fraction}} \; \mathbb N
\qquad\text{or}\qquad
\Delta \mathbb C = \underbrace{\tfrac{4\pi}{3}\,\varepsilon}_{\mathrm{density}} \; \mathbb N
```

Your job is to supply the contribution — or anything upstream of it that the
package can turn into one. The amount lives on the RVE, not on the geometry.

## Three entry levels

Implement the lowest level you can reach; everything above is derived.

| Level | What you supply | What you get |
|:--|:--|:--|
| **A** | `hill_tensor(incl, P₀)` | the 4 concentration tensors and the 2 contribution tensors of each physics, every scheme |
| **B** | `gradient_gradient_concentration(incl, P₁, P₀)` — plus `flux_gradient_concentration` if the inclusion is heterogeneous | the derived concentrations, the contributions, every scheme |
| **C** | the contribution tensors themselves | the contribution-based schemes (see the caveat below) |

`P₀` is the reference medium: a 4th-order stiffness in elasticity, a 2nd-order
conductivity in transport. Dispatch on it and one implementation serves both,
under the same function names.

## Quick approach — `CustomInclusion`

Callbacks are keyword arguments named after the generic they implement.

```julia
using MeanFieldHomogenization, TensND

# A morphology whose Hill tensor you happen to know (here: delegate to a
# spheroid — replace by your own formula or solver).
ell  = Ellipsoid(3.0, 1.0, 1.0)
blob = CustomInclusion((3.0, 1.0, 1.0);
    basis       = MeanFieldHomogenization.inclusion_basis(ell),
    hill_tensor = (P₀; kw...) -> hill_tensor(ell, P₀; kw...))

rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness(30.0, 10.0)); fraction = :rest)
add_phase!(rve, :I, blob, Dict(:C => iso_stiffness(60.0, 20.0)); fraction = 0.25)

homogenize(rve, MoriTanaka(), :C)
```

The same object works in conduction — feed the RVE a `:K` property and the
callback receives a 2nd-order tensor:

```julia
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => TensISO{3}(2.0)); fraction = :rest)
```

!!! note "Callbacks must accept kw..."
    The schemes forward `method`, tolerances and — for density-based phases —
    `K_interface` / `α_interface`. Always write `(P₀; kw...) -> …`.

### Flat objects and densities

A zero-volume object is measured by a density, not a volume fraction. Set
`density_factor` (the prefactor of the amount × contribution seam) and supply
the **two-argument** contributions:

```julia
crack = EllipticCrack(1.0, 0.25)

flat = CustomInclusion((1.0, 0.25, 0.0);
    basis          = MeanFieldHomogenization.inclusion_basis(crack),
    density_factor = 4π / 3,                      # Budiansky, elliptical crack
    inverse_property_contribution = (P₀; kw...) -> inverse_property_contribution(crack, P₀; kw...),
    property_contribution  = (P₀; kw...) -> property_contribution(crack, P₀; kw...))

add_phase!(rve, :cracks, flat, Dict(:C => C_matrix); density = 0.08)
```

### Internally heterogeneous inclusions

If no single property represents your inclusion (a coated or layered pattern),
pass `homogeneous = false`. The scheme kernels then use the exact identities
instead of ``\langle\mathbb C:\boldsymbol\varepsilon\rangle_i = \mathbb C_i:\mathbb A_i``,
which would otherwise give a wrong answer — occasionally with the wrong sign.

The same caveat applies *inside* level B. The strain concentration fixes the
stress one only through ``\mathbb A_{\sigma\varepsilon} = \mathbb C_1 :
\mathbb A_{\varepsilon\varepsilon}``, so a heterogeneous inclusion must supply
both:

```julia
layered = CustomInclusion(;
    homogeneous                     = false,
    gradient_gradient_concentration = (P₁, P₀; kw...) -> ...,   # ⟨ε⟩_I from ε∞, ⟨∇T⟩_I from ∇T∞
    flux_gradient_concentration     = (P₁, P₀; kw...) -> ...)   # ⟨σ⟩_I from ε∞ or from ∇T∞
```

Each callback serves both physics: `P₁` and `P₀` are fourth-order stiffnesses
in elasticity and second-order conductivities in conduction, as for
`hill_tensor`.

The built-in `LayeredSphere` does so. Omitting the
stress-side callback fails *silently*, and in the most misleading way:
`Dilute` and `MoriTanaka` need only ``(\mathbb A, \mathbb N)`` and stay exactly
right, while `SelfConsistent` and `AsymmetricSelfConsistent` drift by several
percent. `check_inclusion_interface` catches it.

### An arbitrary outer shape

`semi_axes` is optional. It only feeds
[`shape_tensor`](@ref MeanFieldHomogenization.Core.shape_tensor), which describes an
*equivalent ellipsoidal envelope* and which no kernel reads. A morphology with
no such envelope simply omits it:

```julia
blob = CustomInclusion(; hill_tensor = (P₀; kw...) -> my_solver(P₀))
```

## Durable approach — subtyping

```julia
struct MyBlob{T, B <: TensND.AbstractBasis} <: MeanFieldHomogenization.AbstractCustomInclusion{T}
    radius::T
    eccentricity::T
    basis::B
end

_equivalent(b::MyBlob) =
    Ellipsoid(b.radius, b.radius * (1 - b.eccentricity), b.radius * (1 - b.eccentricity)^2)

MeanFieldHomogenization.dimension(::MyBlob)          = 3
MeanFieldHomogenization.inclusion_basis(b::MyBlob)   = b.basis
MeanFieldHomogenization.shape_trait(b::MyBlob)       = MeanFieldHomogenization.shape_trait(_equivalent(b))
MeanFieldHomogenization.shape_tensor(b::MyBlob)      = MeanFieldHomogenization.shape_tensor(_equivalent(b))

MeanFieldHomogenization.Elasticity.hill_tensor(b::MyBlob, P₀::TensND.AbstractTens; kw...) =
    hill_tensor(_equivalent(b), P₀; kw...)
```

Subtyping gives access to sensitivities: because `radius` and `eccentricity` are
plain `<:Number` fields, `derivative`, `gradient` and `jacobian` reach them
with no extra code.

```julia
derivative(rve, Dilute(), geometry(:I, :eccentricity);
           indexer = C -> canonical_components(C)[1, 1, 1, 1])
```

This requires the whole path to be type-generic, so that `ForwardDiff.Dual`
propagates. An external numerical solver in the loop generally breaks that:
give the solve its derivative by the implicit-function theorem — solve on the
values, then one Newton correction carrying the duals — rather than
differentiating it by finite differences, whose step trades truncation against
cancellation and loses the derivative's accuracy.

### A crack, from `cod_tensor` alone

Subtyping [`AbstractCrack`](@ref) and declaring the right
[`shape_trait`](@ref MeanFieldHomogenization.Core.shape_trait) is the most economical case
in the whole package: **one** method unlocks the entire chain — ℍ, ℕ, 𝐑, 𝐍_K,
the bundled pair, and the four `delta_*` with their Budiansky-O'Connell prefactor.

```julia
struct MyCrack{T, B <: TensND.AbstractBasis} <: MeanFieldHomogenization.AbstractCrack{T}
    a::T
    b::T
    basis::B          # column 3 is the crack normal
end

MeanFieldHomogenization.shape_trait(::MyCrack)  = MeanFieldHomogenization.EllipticShape
MeanFieldHomogenization.shape_tensor(c::MyCrack) = ...

# One method per tensor order — a single `AbstractTens` method is ambiguous.
MeanFieldHomogenization.Cracks.cod_tensor(c::MyCrack, C₀::TensND.AbstractTens{4,3}; kw...) = ...
MeanFieldHomogenization.Cracks.cod_tensor(c::MyCrack, K₀::TensND.AbstractTens{2,3}; kw...) = ...
```

## Check before you wire

```julia
check_inclusion_interface(blob)                           # elasticity, fraction
check_inclusion_interface(flat; amount = :density)        # elasticity, density
check_inclusion_interface(blob; physics = :conduction)    # transport
```

It names the missing level-0 methods, the entry level it found, and what that
level leaves unavailable.

## Two points to keep in mind

**Level C does not provide the concentration tensor.** Contribution tensors
alone do not determine ``\mathbb A``, so a *volume-fraction* phase entered
through level C works with `Voigt`, `Reuss`, `Dilute`, `DiluteDual`, `Maxwell`,
`PonteCastanedaWillis` and `DifferentialScheme`, but not with `MoriTanaka` or
the self-consistent schemes. Density-based phases are unaffected.

**Orientation averaging is applied after the inclusion's own method.** `IsoSymmetrize` and
`TISymmetrize` are applied by the scheme to the tensor you returned, so:

- return tensors in the **global** frame;
- under symmetrization the scheme hands your kernel a **pre-projected**
  (isotropic, by default) reference medium — so a whole family of orientations
  may share one and the same solve, which is worth exploiting when your
  response is expensive to compute.

```julia
for (i, bin) in enumerate(polar_orientation_bins(12))
    add_phase!(rve, Symbol(:I, i), my_inclusion_at(bin.θ), props;
               fraction = 0.2 * bin.weight, symmetrize = TISymmetrize((0.0, 0.0, 1.0)))
end
```

**Level B is complete for a heterogeneous inclusion too.** Supply *both*
concentration tensors (`gradient_gradient_concentration` and `flux_gradient_concentration`,
in either physics) and the contribution tensors are derived from
``\mathbb N = \mathbb A_{\sigma\varepsilon} - \mathbb C_0 :
\mathbb A_{\varepsilon\varepsilon}``, which needs no single
``\mathbb C_1``. Supply only the strain side, with
`is_homogeneous_inclusion` still `true`, and the average stress is silently
wrong — `check_inclusion_interface` is there to catch that.

## Worked examples in the package

The first two live in [`MeanFieldHomogenization.FiniteElements`](@ref) and take
the same approach: a finite-element resolution of the Eshelby problem, plugged in
through the interface, with nothing downstream aware of it. The third shows that
the contribution need not be independent of the size: the Brinkman inclusions
of the Stokes-to-Darcy upscaling depend on it through the reference medium,
and enter through level B all the same.

| Page | Covers | Level |
| :--- | :--- | :--- |
| [Finite-element inclusions](@ref man-fe-inclusions) | the principle, the shared syntax, and the five shipped morphologies | the crack algebra from `cod_tensor` alone, or B |
| [A recycled-concrete aggregate](@ref app-recycled-aggregate) | the off-center core, worked through the whole calculation | B, both concentration tensors |
| [Brinkman inclusions](@ref man-brinkman) | size-dependent inclusions of the Stokes flow in the pores | B, both concentration tensors |

A cavity is the case where level B is at its simplest and most exact: its
stress-side concentration is **identically zero**, so the package's own
contribution identities collapse to ``\mathbb N = -\mathbb C_0:\mathbb A``
and ``\mathbb H = \mathbb A:\mathbb S_0`` with no ``\mathbb C_1`` in them at
all. Declaring `is_homogeneous_inclusion` **false** is what reaches that approach —
the flag asks whether a single ``\mathbb C_1`` describes the interior, and
`inv(0)` is meaningless. See
[`FESupershapePore`](@ref MeanFieldHomogenization.FESupershapePore).

## Where to go next

The two pages that follow apply the interface to morphologies with no closed
form: [Finite-element inclusions](@ref man-fe-inclusions) obtains the response
from a mesh, [Neural-surrogate inclusions](@ref man-neural-inclusions) from a
trained network. The tutorial
[The custom-inclusion interface: three entry levels, one answer](@ref tut-custom-inclusion-contract)
enters one morphology through each level and compares the three answers.

- [Adding a new inclusion](@ref dev-adding-inclusion) — the complete interface,
  level by level
- [`CustomInclusion`](@ref MeanFieldHomogenization.CustomInclusion),
  [`check_inclusion_interface`](@ref MeanFieldHomogenization.check_inclusion_interface)
  — the docstrings
- `scripts/27_user_inclusion_sensitivity.jl` — sensitivities through a
  user-defined type
