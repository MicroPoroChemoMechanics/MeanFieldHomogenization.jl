# [Viscoelastic homogenization — user manual](@id man-viscoelasticity)

The ALV (ageing linear viscoelastic) pipeline reuses the [`RVE`](@ref)
machinery of the elastic side: replace each phase property by a
[`ViscoLaw`](@ref) and pass a `times` grid to [`homogenize_alv`](@ref).

This manual walks through eight use cases, each runnable as is.
A more elaborate version of every example exists under
`scripts/50_visco_law_basics.jl` … `scripts/59_alv_sensitivities.jl`.

## 1. Defining a constitutive law

A [`ViscoLaw`](@ref) wraps a two-argument kernel function `(t, t')` and
a mode flag (`:relaxation` for the relaxation kernel `R(t, t')`,
`:creep` for the compliance kernel `J(t, t')`). The kernel can return:

* a scalar (`Float64` / `Complex{Float64}`) — 1D ALV problems,
* a `TensISO{4, 3}` / `TensTI{4}` / `TensOrtho` / `Matrix{6×6}` — 3D
  4-tensor in Mandel form,
* a `TensISO{2, 3}` / `Matrix{3×3}` — 3D 2-tensor for conductivity /
  diffusion / permittivity (order-2 ALV).

### 1.1 Hand-rolled Maxwell isotropic relaxation

```julia
using MeanFieldHomogenization, TensND

# Iso Maxwell relaxation:
#   R(t,t') = (3 K∞ + (3 K₀ - 3 K∞) exp(-(t-t')/τ_K)) 𝕁
#           + (2 μ∞ + (2 μ₀ - 2 μ∞) exp(-(t-t')/τ_μ)) 𝕂
const k₀ = 5.0;  const μ₀ = 2.0
const k∞ = 3.0;  const μ∞ = 1.0
const τ_K = 1.0; const τ_μ = 0.5

function R_iso(t, tp)
    α = 3 * (k∞ + (k₀ - k∞) * exp(-(t - tp) / τ_K))
    β = 2 * (μ∞ + (μ₀ - μ∞) * exp(-(t - tp) / τ_μ))
    return TensISO{3}(α, β)        # iso 4-tensor with parameters (3K, 2μ)
end
law_M = ViscoLaw(R_iso, :relaxation)
```

### 1.2 Pre-built constructors

The shortest route to a non-ageing law is not to write a kernel at all, but to
take a model from the [rheological library](@ref man-rheological-models) and let
`ViscoLaw` build the kernel:

```julia
m = iso_rheology(zener_maxwell(30.0, 20.0, 1.0), zener_maxwell(10.0, 8.0, 0.7))
law = ViscoLaw(m)          # (t, t') ↦ R(t - t'), ready for `homogenize_alv`
```

The same object also drives the [Laplace-Carson route](@ref man-laplace-inversion)
through `carson_relaxation(m, p)`, so a non-ageing material need only be
described once and the two routes are guaranteed to be comparing the same
thing — which is what the [three-route check](@ref tut-freq-vs-time) relies on.

The hand-written constructors below remain the way to build an **ageing**
kernel, which no model in the library can express.


```julia
# `maxwell_iso(K, μ, τ_K, τ_μ)` —  R = 3K·e^{-t/τ_K} 𝕁 + 2μ·e^{-t/τ_μ} 𝕂
law_max = maxwell_iso(5.0, 2.0, 1.0, 0.5)

# `kelvin_iso(K_∞, μ_∞, K₀, μ₀, τ_K, τ_μ)` — Kelvin (creep) iso
law_kel = kelvin_iso(3.0, 1.0, 5.0, 2.0, 1.0, 0.5)

# Elastic limit : R(t,t') = C · H(t-t')
law_el  = heaviside_law(TensISO{3}(15.0, 4.0))
```

### 1.3 Ageing kernels

The first argument is the current time `t`, the second is the loading
time `t'`. Ageing means the kernel depends on `t'`, not just on the
duration `t − t'` (basic linear viscoelasticity is the special case
where it depends only on `t − t'`):

```julia
# Sanahuja-style solidification : volume fraction of "active" gel grows
# with t' as `f_∞ · t'^α / (1 + t'^α)`.
const α_age = 4.0
const f_∞   = 0.3
@inline solidification(tp) = f_∞ * tp^α_age / (1 + tp^α_age)

function R_aging(t, tp)
    f = solidification(tp)
    α = 3 * (3.0 + (5.0 - 3.0) * f * exp(-(t - tp) / 1.0))
    β = 2 * (1.0 + (2.0 - 1.0) * f * exp(-(t - tp) / 0.5))
    return TensISO{3}(α, β)
end
law_aging = ViscoLaw(R_aging, :relaxation)
```

## 2. Trapezoidal discretization

Given a time grid `times = (t₁, …, tₙ)`, the Stieltjes integral becomes
a `(B·n × B·n)` lower-block-triangular matrix (`B = 6` for a 4-tensor
kernel, `B = 3` for a 2-tensor kernel, `B = 1` for scalar kernels):

```julia
times = collect(range(0.0, 5.0; length = 50))
M = trapezoidal_matrix(law_M, times)      # 300 × 300  (=  6·50)
```

[`volterra_inverse`](@ref) flips a relaxation matrix to the
corresponding creep matrix (and vice-versa) via block forward
substitution. A `LowerTriangular` BLAS path is selected internally for
`B ≥ 2`:

```julia
J = volterra_inverse(M; block_size = 6)            # creep (compliance) matrix
@assert isapprox(M * J,
                 [iszero(rem(i - j, 6)) && (i ÷ 6 == j ÷ 6) for i in 1:300, j in 1:300] |>
                  Matrix{Float64};
                 atol = 1e-10)                       # block-diagonal identity
```

## 3. Building an RVE and homogenizing

```julia
# 50-step time grid; the matrix is the Maxwell iso law from §1.1
times = collect(range(0.0, 5.0; length = 50))

# Inclusions : aligned spheroids (oblate ratio 0.5), elastic
C_I = TensISO{3}(60.0, 20.0)

rve = RVE(:M)
add_matrix!(rve, Ellipsoid(1.0), Dict(:C => law_M))
add_phase!(rve, :I, Ellipsoid(1.0, 1.0, 0.5),
            Dict(:C => heaviside_law(C_I));
            fraction = 0.2)

# Homogenise
C_eff = homogenize_alv(rve, MoriTanaka(), :C; times = times)   # 300 × 300
```

[`homogenize_alv`](@ref) accepts: `Voigt`, `Reuss`, `Dilute`,
`DiluteDual`, `MoriTanaka`, `Maxwell`, `PonteCastanedaWillis`,
`SelfConsistent`, `AsymmetricSelfConsistent`, `DifferentialScheme`.

A symmetric companion exists for **conductivity / diffusion /
permittivity** (order-2 properties, 3 × 3 kernels) — the dispatcher
inspects the sample type returned by `visco_eval(law, t, t)` and
routes to the order-2 pipeline automatically (see §6).

## 4. Reading effective properties

The `(6n × 6n)` output is a Mandel block matrix. Use
[`iso_params_from_blocks`](@ref) to extract the iso `(α, β)` parameter
matrices:

```julia
α, β = iso_params_from_blocks(C_eff)         # n × n each
# 3 K_eff(t,t')   = α(t,t')
# 2 μ_eff(t,t')   = β(t,t')

# Effective shear modulus history : column index = t' = 0 (Heaviside step)
times_keep = times                            # (length n)
μ_eff_t = β[:, 1] ./ 2                        # μ(tᵢ, t₁) for i = 1..n
```

For a **uniaxial creep** test (unit longitudinal stress, all
components 1, 2, 3 stay the same in iso ; 4, 5, 6 are zero):

```julia
n = length(times)
J_eff = volterra_inverse(C_eff; block_size = 6)       # creep matrix
S = zeros(n * 6); for i in 1:n;  S[6 * (i - 1) + 1] = 1.0;  end
ε = J_eff * S
ε_xx_t = ε[1:6:end]                            # ε_xx(tᵢ)
```

For TI (axis = e₃), the 4-tensor Walpole parameters
`(ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆)` are extracted similarly:

```julia
ℓ = ti_params_from_blocks(C_eff)               # NTuple{6, Matrix}
```

## 5. Cracks in ALV

### 5.1 Traction-free penny crack

```julia
add_phase!(rve, :C, PennyCrack(1.0), Dict(:C => law_M);
            density = 0.05, symmetrize = :iso)
C_eff_cracks = homogenize_alv(rve, MoriTanaka(), :C; times = times)
```

`PennyCrack`, `EllipticCrack` and `RibbonCrack` are accepted. The
dispatcher pre-aggregates crack stiffness and compliance contributions
and routes them through the appropriate scheme branch.

### 5.2 Cracks with finite interface stiffness (Sevostianov)

For a flat crack carrying a **spring-like interface stiffness** with
time-dependent normal `Rn(t,t')` and tangential `Rt(t,t')` ageing
kernels, attach the interface laws as `:Rn` / `:Rt` properties on the
crack phase :

```julia
# Interface kernels — same Maxwell-iso ageing form as the matrix law
R_n_kernel(t, tp) = (1 + 0.1 * tp^0.4) *
                     (1.0e10 + (2.0e10 - 1.0e10) * exp(-(t - tp) / 2.0))
R_t_kernel(t, tp) = (1 + 0.1 * tp^0.2) *
                     (1.0e10 + (1.0e10 - 1.0e10) * exp(-(t - tp) / 3.0))
law_Rn = ViscoLaw(R_n_kernel, :relaxation)
law_Rt = ViscoLaw(R_t_kernel, :relaxation)

add_phase!(rve, :CRACK, PennyCrack(1.0),
            Dict(:C => law_M, :Rn => law_Rn, :Rt => law_Rt);
            density = 0.05, symmetrize = :iso)

C_eff = homogenize_alv(rve, MoriTanaka(), :C; times = times)
```

The scalar COD kernels `B̃_n`, `B̃_t` are post-corrected by the
spring-interface construction of [sevostianovIJSS2007](@cite), transposed to
the Volterra algebra: crack-face and interface compliances add up, at the cost
of one extra scalar Volterra inverse per direction.

Limits, all exercised in the test suite :

| Interface | Behavior                                  |
|-----------|--------------------------------------------|
| `Rn = Rt = nothing`   | traction-free penny — `B̃` unchanged |
| `Rn, Rt → 0`          | recovers traction-free                |
| `Rn, Rt → ∞` (rigid)  | `B̃_eff → 0`, cracks behave as bonded |

### 5.3 Notes on Mori-Tanaka and Self-Consistent for cracks

Two well-established but **distinct** formulations co-exist in the
literature for crack-bearing RVEs :

* **Additive (Budiansky-O'Connell)** : MT adds the crack stiffness
  contribution `(4π/3)·ε·Ñ` to the numerator with a zero
  contribution to the denominator (cracks have no volume in the
  strain-concentration sum).  At convergence SC writes
  `J_eff = J_M + ε·H̃(C_eff)`.  This is what MFH currently
  implements.

* **Multiplicative (ECHOES)** : MT adds `(4π/3)·ε·H̃·C_0` to the
  *denominator* via the strain-strain concentration tensor
  (`strain_Strain = H̃·C_0`).  At convergence SC writes
  `C_eff = (B_E)·(A_E)^{-vol}` where the cracks contribute to `A_E`.

They coincide in the dilute limit ``\varepsilon \to 0`` and differ at finite density: at
`d = 0.30, traction-free`, MFH MT gives ``\varepsilon_{xx}(t\to\infty) \approx 0.481`` against `0.559` for
ECHOES MT. PCW coincides between the two implementations over the configurations
of `scripts/60_alv_cracks_interface.jl`.

The additive form is kept for consistency with the (also-additive) MFH
elastic MT. `scripts/60_alv_cracks_interface.jl` runs the same configuration
through both implementations: `rtol ≤ 1e-3` at low density, a few % to ~14 %
at `d ≥ 0.20`.

A static (non-ageing) elastic + conductivity crack benchmark with
matrix-only interface stiffness is in
`scripts/15_cracks_iso_interface.jl`.

| Scheme                         | Crack treatment                                     |
|--------------------------------|-----------------------------------------------------|
| `Voigt`, `Reuss`               | ignored (zero-volume convention)                    |
| `Dilute`, `DiluteDual`         | additive `+ ΔC_cracks`                              |
| `MoriTanaka`, `Maxwell`, `PCW` | virtual phase with `A = 0`, `N = ΔC`, `f = 1`       |
| `SC`, `ASC`                    | re-evaluated against the running effective estimate |

A complete demo with **all seven** crack-aware ALV schemes lives in
`scripts/57_ageing_creep_cracks.jl`.

## 6. Order-2 ALV — conductivity / diffusion

Same API as the order-4 case, but the kernel returns a 2-tensor:

```julia
function K_iso_order2(t, tp)
    κ = 1.0 + 0.5 * exp(-(t - tp))
    return TensISO{2,3}(κ)
end
law_κ = ViscoLaw(K_iso_order2, :relaxation)

rve_κ = RVE(:M)
add_matrix!(rve_κ, Ellipsoid(1.0), Dict(:K => law_κ))
add_phase!(rve_κ, :I, Ellipsoid(1.0), Dict(:K => heaviside_law(TensISO{2,3}(5.0)));
            fraction = 0.3)

K_eff = homogenize_alv(rve_κ, MoriTanaka(), :K; times = times)   # 150 × 150 (= 3·n)
```

The dispatcher sees the 2-tensor sample and routes via the
order-2 pipeline ([`homogenize_alv_order2`](@ref) under the hood).
Result is a `(3n × 3n)` block matrix. See
`scripts/56_ageing_creep_order2.jl`.

The order-2 pipeline implements the bounds, `Dilute`, `DiluteDual`,
`MoriTanaka`, `Maxwell` and `DifferentialScheme`.

`symmetrize` is honored in both orders. The projection is applied to the
dilute quantities (`Ã_α`, `Ñ_α`) block by block, with the projector of the
right tensor order — 6×6 Mandel blocks for order 4, plain 2-tensor blocks
for order 2. It is exact in both cases: the SO(3) average of a 2-tensor is
its spherical part `(tr B / 3) 𝟙`, and the azimuthal average about `n̂`
keeps the axial component, the transverse mean and the axial antisymmetric
part.

Note that the bounds are shape-blind: `Voigt` and `Reuss` average the phase
relaxation matrices directly, so `symmetrize` cannot change their result.

## [7. Isotropic reference: what it costs the reference-updating schemes](@id man-alv-iso-reference)

Every ALV Hill kernel — order 2 and order 4 alike — is built for an
**isotropic** reference medium: that is the condition under which the
time and space parts decouple and a closed form exists at all. This is
not a restriction on the *result*, which is generally anisotropic; it is
a restriction on what the kernel may be evaluated *against*.

That splits the schemes in two:

* `Voigt`, `Reuss`, `Dilute`, `DiluteDual`, `MoriTanaka`, `Maxwell`, `PCW`
  evaluate the kernel against the **matrix**, which is fixed and
  isotropic. Any inclusion shape and orientation is fine.
* `SelfConsistent` and `DifferentialScheme` evaluate it against their
  **running estimate** — the fixed point for the former, ``\tilde{\mathbb C}(\tau)`` for the
  latter. An aligned non-spherical inclusion, or a crack (whose
  contribution is transversely isotropic in its own frame), drags that
  estimate out of the isotropic class, and the kernel would no longer be
  valid there.

!!! warning "Reference-updating ALV schemes need an isotropic running medium"
    With `SelfConsistent` or `DifferentialScheme` in ALV, every inclusion
    phase must keep the running estimate isotropic. Two ways to satisfy
    that: **spherical inclusions** with an isotropic phase law
    (`LayeredSphere` also qualifies — its contribution is isotropic by
    construction), or an **isotropic orientation average**,
    `symmetrize = :iso`, which is also what randomly oriented inclusions or
    cracks mean physically.

    An RVE that satisfies neither raises an explicit `ArgumentError` naming
    the offending phase, rather than silently reading iso parameters off a
    matrix that is no longer isotropic. A non-isotropic ALV *matrix* is
    refused for the same reason.

```julia
# randomly oriented cracks — the orientation average makes this legitimate
add_phase!(rve, :CR, PennyCrack(1.0), Dict(:C => law_M);
           density = 0.1, symmetrize = :iso)

# aligned spheroids, order 2: same requirement, same fix
add_phase!(rve_κ, :I, Spheroid(5.0), Dict(:K => heaviside_law(TensISO{2,3}(10.0)));
           fraction = 0.2, symmetrize = :iso)
homogenize_alv(rve_κ, DifferentialScheme(), :K; times = times)
```

!!! note "The elastic schemes have no such restriction"
    This is specific to ALV. In the elastic pipeline the Hill tensor is
    available for anisotropic references, so `homogenize(rve,
    SelfConsistent())` and `homogenize(rve, DifferentialScheme())` handle a
    running medium of any symmetry class — the differential scheme even
    tracks it, growing its ODE state from the iso to the TI, ortho or fully
    anisotropic layout as the phases require.

## 8. The differential scheme in ALV

`DifferentialScheme` is available in both tensor orders, with the same
keywords as the elastic scheme — `trajectory`, `nsteps`, `abstol` /
`reltol`, `alg`, and `formulation = :stiffness | :compliance` (the dual
form integrates the creep function `J̃ = C̃^{-vol}`):

```julia
homogenize_alv(rve, DifferentialScheme(; formulation = :compliance), :C; times = times)
```

Supported inclusions: ellipsoids and spheroids, `LayeredSphere`, and
crack families through their density — subject to the isotropic-reference
requirement of [section 7](@ref man-alv-iso-reference).

## 9. Symmetry-class fast paths

When all phases share an iso / TI / ortho symmetry with compatible
axes, [`homogenize_alv`](@ref) automatically routes through a fast
path that solves the scheme algebra in the **structured** domain :

The classes, their stored components and the cost of each closure operation
are tabulated in [Symmetry classes and structured
storage](../theory/viscoelasticity.md#th-visco-classes).

Detection is heuristic (`_is_iso_block` / `_is_ti_block` /
`_is_ortho_block`) — the user never asks for a fast path explicitly,
and the output is still a dense `(6n × 6n)` `Matrix{T}`.

For user code that wants to keep the compact storage and the type
information, the structured wrappers
[`ALVKernelISO`](@ref) / [`ALVKernelTI`](@ref) /
[`ALVKernelOrtho`](@ref) are `AbstractMatrix{T}` subtypes:

```julia
M = trapezoidal_matrix(law_M, times)
K_iso = ALVKernelISO(M)            # extracts (α, β), 18× cheaper storage

# Algebra closure stays in the structured class (no (6n × 6n)
# materialisation), with auto-promotion iso ⊂ TI ⊂ ortho
K_prod = K_iso * K_iso             # ALVKernelISO
K_inv  = volterra_inverse(K_iso)   # ALVKernelISO

K_TI = ALVKernelTI(K_iso)          # promote to TI form
K_O  = ALVKernelOrtho(K_iso)       # promote to ortho form
K_iso + K_TI                       # ALVKernelTI (auto-promote)
K_iso * K_O                        # ALVKernelOrtho

Matrix(K_iso)                      # back to dense (6n × 6n) on demand
```

**Prototype**: usable for hand-rolled ALV pipelines, but `homogenize_alv`
does not accept them as inputs — use `Matrix(K)` to cross the boundary
(`scripts/58_alv_kernel_types.jl`).

## 10. Sensitivities (autodiff via ForwardDiff)

The pipeline supports `ForwardDiff.Dual` end-to-end so derivatives of
effective properties wrt RVE parameters are direct.

### 10.1 Sensitivity wrt volume fraction — recommended `set_param` lens

```julia
using ForwardDiff

# Build the RVE once with a Float64 placeholder fraction.
rve_base = RVE(:M)
add_matrix!(rve_base, Ellipsoid(1.0), Dict(:C => law_M))
add_phase!(rve_base, :I, Ellipsoid(1.0), Dict(:C => heaviside_law(TensISO{3}(60.0, 20.0)));
            fraction = 0.20)

# Differentiate by substituting a `Dual` value via `set_param`.
function eff_mu(f)
    rve_f = set_param(rve_base, AmountParameter(:I), f)
    R̃ = homogenize_alv(rve_f, MoriTanaka(), :C; times = times)
    _, β = iso_params_from_blocks(R̃)
    return β[end, end] / 2
end

dμ_df = ForwardDiff.derivative(eff_mu, 0.20)        # ≈ 1.66 (validated FD ≤ 1e-7)
```

### 10.2 Sensitivity wrt a material parameter — closure-captured

When the parameter lives **inside** the kernel function (e.g. a
modulus, relaxation time, ageing exponent), close it into the kernel
and differentiate normally. ForwardDiff lifts the parameter to `Dual`
through the closure:

```julia
function eff_mu_vs_μM(μ_M)
    function R(t, tp)
        TensISO{3}(15.0, 2 * μ_M * (0.5 + 1.5 * exp(-(t - tp) / 0.5)))
    end
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => ViscoLaw(R, :relaxation)))
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => heaviside_law(TensISO{3}(60.0, 20.0)));
                fraction = 0.20)
    R̃ = homogenize_alv(rve, MoriTanaka(), :C; times = times)
    _, β = iso_params_from_blocks(R̃)
    return β[end, end] / 2
end

dμ_dμM = ForwardDiff.derivative(eff_mu_vs_μM, 1.0)
```

### 10.3 Joint gradient over multiple parameters

```julia
function eff_mu_vs_p(p)
    f, k_M, μ_M = p
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => maxwell_iso(k_M, μ_M, 1.0, 0.5)))
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => heaviside_law(TensISO{3}(60.0, 20.0)));
                fraction = 0.20)
    rve_f = set_param(rve, AmountParameter(:I), f)
    R̃ = homogenize_alv(rve_f, MoriTanaka(), :C; times = times)
    _, β = iso_params_from_blocks(R̃)
    return β[end, end] / 2
end

∇ = ForwardDiff.gradient(eff_mu_vs_p, [0.20, 1.0, 1.0])    # 3-vector
```

The complete suite of sensitivity patterns lives in
`scripts/59_alv_sensitivities.jl`. Each derivative is validated against
a central finite difference at `rtol ≤ 1e-7`.

## 11. Validation against ECHOES C++

| script | benchmark | agreement |
| :--- | :--- | :--- |
| `53_ageing_creep_solid.jl` | multi-phase Maxwell + solidifying Maxwell + pore (ECHOES C++ manual) | — |
| `57_ageing_creep_cracks.jl` | seven crack-aware ALV schemes, penny-crack RVE | — |
| `52_rabotnov_mittag_leffler.jl` | Rabotnov / Mittag-Leffler closed form, [barthelemyIJES2019](@cite) §5 | `rtol ≤ 1.3e-3` at `n_times = 200` |

The Rabotnov kernel needed by that benchmark used to come from an external
Python module through PyCall, which made the script machine-dependent. It no
longer does — [`Rabotnov`](@ref) is in the
[model library](@ref man-rheological-models):

```julia
I_Rabotnov(t, α, β) = relaxation(Rabotnov(1.0, 1.0, α, β), t) - 1.0
```

The reason this needs nothing special is that the kernel's Laplace-Carson
transform is elementary, ``R^{*}(p) = \mu_0(1 + \lambda_0/(p^{\alpha+1}+\beta))``,
with no Mittag-Leffler function in it at all. Loading `MittagLeffler.jl` (a weak
dependency) switches on the closed-form *time* value; without it the numerical
inversion of that transform supplies the same number to about `1e-10`.

Random-RVE cross-checks vs the reference implementation live in
`scripts/bench_echoes/benchmark.jl` (relative error `≤ 1e-8` on the
Mandel `(1, 1)` block, `≤ 1e-6` on the full matrix).
