# [Adding a new inclusion](@id dev-adding-inclusion)

This page is the **contract**: everything an inclusion must provide for the
homogenization schemes to accept it, in elasticity and in transport. It applies
equally to a new built-in family and to a user-defined morphology living
outside the package — there is no privileged path.

For the user-facing tutorial (including the ready-made
[`CustomInclusion`](@ref MeanFieldHomogenization.CustomInclusion) callback type), see
[Custom inclusions](@ref man-custom-inclusions).

Where the answer comes from is free. Four routes already use this contract: the
analytic families (gate A), the layered patterns and the
[finite-element inclusions](@ref man-fe-inclusions) (gate B — neither has a Hill
tensor), and the [neural surrogates](@ref man-neural-inclusions) (gate A or B,
from a trained network). A surrogate is worth knowing about while reading this
page for one reason: it is the only route that is **differentiable in the
morphology**, so it is how an expensive gate-B solver becomes usable inside an
iterative scheme and inside the sensitivity API.

The contract has **four levels**. Implement the lowest one you can reach:
everything above it is derived algebraically by the package.

Before wiring anything into an `RVE`, run

```julia
check_inclusion_interface(my_inclusion; physics = :elasticity, amount = :fraction)
```

It reports which level-0 methods are missing, which entry gate it found, and
what that gate does *not* unlock.

---

## Level 0 — geometric identity

Pick a supertype first:

| Supertype | When |
|:--|:--|
| [`AbstractEllipsoidalInclusion`](@ref) | solid ellipsoids and their degenerate limits |
| [`AbstractCrack`](@ref) | flat, zero-volume objects measured by a density |
| [`AbstractLayeredInclusion`](@ref) | internally layered patterns |
| [`AbstractCustomInclusion`](@ref) | anything else — a neutral branch, disjoint from the other three, so your methods can never become ambiguous with theirs |

Then implement — but only two of these are actually *load-bearing*:

| Function | Returns | Status |
|:--|:--|:--|
| [`shape_trait`](@ref MeanFieldHomogenization.Core.shape_trait) | a `Type` | **required.** Holy-style tag. For a flat object it is more than decoration: the crack algebra (``\mathbb H`` from the COD tensor, and the Budiansky prefactors) is keyed on it — `EllipticShape` / `Penny` buy the elliptical convention, `Ribbon` the ribbon one (see level 2). |
| [`is_homogeneous_inclusion`](@ref) | `Bool` | **defaults to `true`**; override it — and mind level 1 — when the inclusion has no single uniform property. The scheme kernels branch on it. |
| [`dimension`](@ref MeanFieldHomogenization.Core.dimension) | `Int` | accessor. No kernel reads it, but downstream code and scripts do. Already defined for every `AbstractCrack`. |
| [`inclusion_basis`](@ref MeanFieldHomogenization.Core.inclusion_basis) | `TensND.AbstractBasis` | accessor; for a flat object **column 3 is the normal**. Already defined for every `AbstractCrack` with a `basis` field. |
| [`shape_tensor`](@ref MeanFieldHomogenization.Core.shape_tensor) | `Tens{2,dim}` | **optional.** It describes an *equivalent ellipsoidal envelope*, ``\boldsymbol{A} = \boldsymbol{R}\,\mathrm{diag}(a_i)\,\boldsymbol{R}^{T}``. Nothing in the package consumes it, and a morphology that has no such envelope simply does not have one. Implement it when the notion is meaningful — the degenerate conventions are tabulated in its docstring. |
| [`element_type`](@ref MeanFieldHomogenization.Core.element_type) | `Type` | **default provided** from the type parameter — do not implement it. |

The point of the distinction: an inclusion that supplies its own response
tensors owes the package nothing about its *outer shape*. `shape_tensor` is
descriptive, not structural.

---

## Level 1 — the mechanical / transport response

Three entry gates. Implement **one**, per physics (4th order = elasticity,
2nd order = transport).

### Gate A — the Hill tensor

```julia
hill_tensor(incl, C₀::AbstractTens{4,3}; kw...) -> Tens{4,3}
hill_tensor(incl, K₀::AbstractTens{2,3}; kw...) -> Tens{2,3}
```

The cheapest way in, and the most complete: all eight localization tensors and
all four contribution tensors follow, and every scheme becomes available.

Two ways to supply it:

1. **Register kernels** (idiomatic for a new built-in family): add
   `Elasticity._kernel(::YourInclusion, C₀, ::Analytical; kw...)` methods for
   every matrix-symmetry class where a closed form exists, plus
   `::Residue` / `::DECUHR` / `::NestedQuadGK` fallbacks, and — if your
   inclusion needs a specialized rule — a `_resolve_algo` method (see
   [Adding a new algorithm](@ref dev-adding-algorithm)). This works for
   `AbstractEllipsoidalInclusion` and `AbstractCustomInclusion`, whose
   `hill_tensor` entry points both route into the `_kernel` table.
2. **Override `hill_tensor` directly** on your concrete type. Simplest when
   you delegate to an equivalent built-in geometry or to an external solver.

!!! warning "Match the argument types of the generic"
    `cod_tensor`, `eshelby_tensor` and the localization generics are declared
    **separately for `AbstractTens{4,3}` and `AbstractTens{2,3}`**. A method
    typed on the union supertype `AbstractTens` is ambiguous with both. Write
    one method per order.

### Gate B — the localization tensors

```julia
strain_strain_loc(incl, C₁, C₀; kw...)      -> Tens{4,3}   # 𝔸_εε
stress_strain_loc(incl, C₁, C₀; kw...)      -> Tens{4,3}   # 𝔸_σε
gradient_gradient_loc(incl, K₁, K₀; kw...)  -> Tens{2,3}   # 𝔸_∇∇
flux_gradient_loc(incl, K₁, K₀; kw...)      -> Tens{2,3}   # 𝔸_q∇
```

Use this when ``\mathbb P`` has no convenient closed form, or when an external
solver hands you ``\mathbb A`` directly.

**How many of them you owe depends on
[`is_homogeneous_inclusion`](@ref).** The strain side determines the stress
side only through

```math
\mathbb A_{\sigma\varepsilon} = \mathbb C_1 : \mathbb A_{\varepsilon\varepsilon},
```

which presupposes a *single uniform* ``\mathbb C_1``. So:

- **homogeneous inclusion** (the default): the strain-side tensor alone is
  enough. `stress_strain_loc` and the rest follow from the generic methods.
- **heterogeneous inclusion** (`is_homogeneous_inclusion == false`): there is
  no such ``\mathbb C_1``. The average stress has to be assembled from the
  local fields, so you **must** supply `stress_strain_loc` as well (and
  `flux_gradient_loc` in transport). `LayeredSphere` and
  [`FEExcenteredSphere`](@ref) both take that route; the finite-element one
  gets the two tensors out of a single solve, which is the usual situation and
  the reason [`fe_axi_localization`](@ref) returns them as a pair.

Supplying that second tensor buys the **contribution tensors** as well: the
generic `stiffness_contribution` and friends switch, when
`is_homogeneous_inclusion` is `false`, to the identities

```math
\mathbb N = \mathbb A_{\sigma\varepsilon} - \mathbb C_0 : \mathbb A_{\varepsilon\varepsilon},
\qquad
\mathbb H = (\mathbb A_{\varepsilon\varepsilon}
             - \mathbb S_0 : \mathbb A_{\sigma\varepsilon}) : \mathbb S_0 ,
```

which need no ``\mathbb C_1`` at all and reduce to `(C₁ - C₀) : A` whenever a
uniform ``\mathbb C_1`` does exist. So gate B really is a complete entry point
for a heterogeneous morphology — two tensors in, everything else derived. (A
type may still override the contributions for speed or for an extra physical
term, as `LayeredSphere` does to add its interface contributions.)

Getting this wrong is silent, and the silence is well hidden: `Dilute` and
`MoriTanaka` consume only ``(\mathbb A, \mathbb N)`` and stay exactly right,
while `SelfConsistent` and `AsymmetricSelfConsistent` — the two schemes that
consume the stress average — drift by several percent.
`check_inclusion_interface` flags the omission.

A heterogeneous inclusion also has no single property to enter the `Voigt` and
`Reuss` bounds: a bound averages the *constituent* properties over the
inclusion, and that takes its internal volume fractions, which the `RVE` does
not carry. The bounds are therefore unavailable unless the type supplies them —
implement `Schemes._layer_voigt` / `Schemes._layer_reuss` and set
`Schemes.has_layer_average` to `true`, as `LayeredSphere` and
[`FEExcenteredSphere`](@ref) do. Nothing else is affected:
`AsymmetricSelfConsistent` consults a bound only to choose between its
stiffness and compliance iteration, and falls back to the dilute estimate when
none is available.

The remaining localizations are derived: `strain_stress_loc` and
`gradient_flux_loc` are a change of driving variable (valid regardless of
homogeneity), while `stress_stress_loc` and `flux_flux_loc` are expressed in
terms of the stress-side tensor, so overriding that one fixes them too.

### Gate C — the contribution tensors

```julia
# solid inclusion (volume-fraction amount)
stiffness_contribution(incl, P₁, P₀; kw...)
compliance_contribution(incl, P₁, P₀; kw...)
conductivity_contribution(incl, P₁, P₀; kw...)
resistivity_contribution(incl, P₁, P₀; kw...)

# flat object (density amount) — two arguments, no property of its own
stiffness_contribution(incl, C₀; kw...)      # ℕ
compliance_contribution(incl, C₀; kw...)     # ℍ
conductivity_contribution(incl, K₀; kw...)   # 𝐍_K
compliance_contribution(incl, K₀; kw...)     # 𝐑 (transport flavour of ℍ)
```

!!! warning "Gate C does not provide the concentration tensor"
    Contribution tensors alone do not determine ``\mathbb A``. A
    **volume-fraction** phase entered through gate C works with `Voigt`,
    `Reuss`, `Dilute`, `DiluteDual`, `Maxwell`, `PonteCastanedaWillis` and
    `DifferentialScheme`, but **not** with `MoriTanaka`, `SelfConsistent` or
    `AsymmetricSelfConsistent`. Density-based (flat) phases are unaffected:
    those kernels rebuild ``\mathbb A`` from ``\mathbb H : \mathbb C_0``.

!!! note "Swallow the interface keywords"
    For a phase registered with a density the schemes always forward
    `K_interface` and `α_interface` (possibly `nothing`). Your methods must
    accept and ignore them — a trailing `kw...` is enough.

### Elasticity ↔ transport correspondence

| Elasticity (order 4) | Transport (order 2) |
|:--|:--|
| `strain_strain_loc` | `gradient_gradient_loc` |
| `stress_strain_loc` | `flux_gradient_loc` |
| `strain_stress_loc` | `gradient_flux_loc` |
| `stress_stress_loc` | `flux_flux_loc` |
| `stiffness_contribution` | `conductivity_contribution` |
| `compliance_contribution` | `resistivity_contribution` |
| `delta_stiffness` | `delta_conductivity` |
| `delta_compliance` | `delta_resistivity` |

There is no separate code path: `homogenize(rve, scheme, property)` treats
`property` as a dictionary key and dispatches on the **tensor order** of the
matrix property found under it.

---

## Level 2 — the amount × contribution seam

The scheme picks its branch from the **amount type** registered on the RVE,
not from the geometry type.

**`fraction = f`** ([`VolumeFraction`](@ref MeanFieldHomogenization.Schemes.VolumeFraction))
uses the three-argument contributions and the generic two-argument
`delta_*(X, f) = f * X`. **Nothing to implement.**

**`density = ε`** ([`CrackDensity`](@ref MeanFieldHomogenization.Schemes.CrackDensity))
uses the two-argument contributions and the **three-argument** seams

```julia
delta_stiffness(incl, N, ε)      delta_compliance(incl, H, ε)
delta_conductivity(incl, N, ε)   delta_resistivity(incl, R, ε)
```

This is where the geometric prefactor relating a density to the effective
correction lives — `4π/3` for an elliptical crack of Budiansky density
``\varepsilon^{3\mathrm d} = N a b^{2}``, `π` for a ribbon of
``\varepsilon^{2\mathrm d} = N b^{2}``.

For anything subtyping [`AbstractCrack`](@ref) you do **not** write those four
methods: they are dispatched on `shape_trait` through the single hook
[`crack_density_factor`](@ref MeanFieldHomogenization.Cracks.crack_density_factor).
Override that one method if your flat morphology uses a different density
convention.

Nothing ties `CrackDensity` to `AbstractCrack`, so a flat
[`AbstractCustomInclusion`](@ref) can be registered with `density = …` just as
well — it then owns the four `delta_*` methods itself.

---

## Level 3 — optional refinements

- **Bundled seams.** [`loc_and_stiffness`](@ref MeanFieldHomogenization.loc_and_stiffness),
  [`loc_and_stress_average`](@ref MeanFieldHomogenization.loc_and_stress_average) and
  [`compliance_and_stiffness_contribution`](@ref MeanFieldHomogenization.compliance_and_stiffness_contribution)
  let a scheme obtain two objects from one expensive solve. Generic fallbacks
  exist, so specializing is purely a performance decision — and the result
  must be **bitwise identical** to calling the two functions separately.
- **Sensitivities.** `derivative` / `gradient` / `jacobian` reach a geometric
  field automatically as long as the struct's numeric fields are `<:Number`
  (or tuples of them) and it has the default parametric constructor:
  `Schemes._replace_geom_field` rebuilds it by reflection. Add a method for it
  only if your type needs custom reconstruction. ForwardDiff support then
  requires the whole level-1 path to be type-generic — which rules out
  algorithms hard-wired to `Float64` (the `Residue` backend) and external
  solvers that factorize numerically.
- **Ageing viscoelasticity.** The ALV pipeline has its own per-geometry seam:
  either `tens_UA` / `tens_VA`, or a `Viscoelasticity._inclusion_alv_quantities`
  method.
- **Crack API.** Implement `cod_tensor` if you want `sif` and `dif` too. For
  an `AbstractCrack` with a standard `shape_trait`, `cod_tensor` *alone*
  unlocks ℍ, ℕ, 𝐑, 𝐍_K, the bundle and the four `delta_*`.

---

## What you get for free

**Orientation averaging.** `IsoSymmetrize` and `TISymmetrize` are applied by
the scheme, *after* your method returns
(`Schemes._apply_symmetrize`). Two consequences:

- the tensors you return must be expressed in the **global** frame;
- under `IsoSymmetrize` / `TISymmetrize` the scheme hands your kernel a
  **pre-projected** `C₀` (`Schemes._project_matrix`, isotropic by default), so
  an inclusion family at many orientations may see the same reference medium
  many times — worth exploiting if your solve is expensive.

Note that `_apply_symmetrize` does **not** commute with the tensor product:
``\langle R(C_r : A_r)\rangle \neq C_r : \langle R(A_r)\rangle`` unless `C_r`
is isotropic. The scheme handles this; do not attempt to average inside your
kernel.

---

## Checklist

1. Choose the supertype; implement `shape_trait`, plus
   `is_homogeneous_inclusion` if the inclusion is not uniform, plus the
   accessors that make sense for it.
2. Implement one level-1 gate, per physics — remembering that gate B costs
   *two* tensors when the inclusion is heterogeneous.
3. If the phase is measured by a density, provide the level-2 prefactor
   (or `crack_density_factor` for an `AbstractCrack`).
4. Run `check_inclusion_interface`.
5. Add at least one unit test under `test/<SubModule>/` — the strongest form
   is an **equivalence test** against a built-in geometry through every
   scheme, as in `test/CustomInclusions/test_custom_inclusion.jl`.
