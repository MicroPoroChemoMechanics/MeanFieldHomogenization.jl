```@meta
EditURL = "../../../../scripts/59_alv_sensitivities.jl"
```

# Derivatives through the ageing-viscoelastic pipeline

[Derivatives and sensitivities](@ref tut-sensitivities) differentiates an
*elastic* homogenization. Nothing in that story is specific to elasticity: the
ALV path is built from the same generic Julia code, so `ForwardDiff` propagates
a `Dual` straight through the Volterra assembly, the block inversions and the
scheme, and returns ``\partial \mu^{\hom}/\partial p`` for free — no adjoint,
no hand-written derivative, no finite difference.

Two patterns cover every case, and every derivative below is validated against
a central finite difference.

| Pattern | When | How |
|---|---|---|
| `set_param` lens | the parameter lives on the **RVE** — a volume fraction, a crack density | build once with `Float64`, substitute a `Dual` through [`set_param`](@ref) |
| closure capture | the parameter lives inside the **`ViscoLaw`** — a modulus, a relaxation time | close it into the kernel; it lifts to `Dual` on its own |

The first allocates nothing outside the differentiation and works for every
scheme; the second is the only route to a material parameter, because the
kernel is a user function the package never inspects.

````@example alv_sensitivities
using MeanFieldHomogenization
using TensND
using ForwardDiff
using LinearAlgebra
using Printf
````

## §0 A common ALV setup

An isotropic Maxwell matrix whose bulk and shear kernels relax with distinct
times, parametrized by ``(k_M, \mu_M, \tau_K, \tau_\mu)``, and elastic
spherical inclusions. `build_law_M` returns a **fresh** `ViscoLaw` per call —
that is what lets a `Dual` reach the kernel in patterns 2 to 4.

````@example alv_sensitivities
const N_TIMES = 8
const TIMES = collect(range(0.0, 2.0; length = N_TIMES))
````

Maxwell-iso matrix law parametrized by `(k_M, μ_M, τ_K, τ_μ)`.
Closure pattern: each call returns a fresh `ViscoLaw`.

````@example alv_sensitivities
function build_law_M(k_M, μ_M, τ_K = 1.0, τ_μ = 0.5)
    function R_iso(t, tp)
        α = 3 * k_M * (1.0 + 4.0 * exp(-(t - tp) / τ_K))
        β = 2 * μ_M * (0.5 + 1.5 * exp(-(t - tp) / τ_μ))
        return TensISO{3}(α, β)
    end
    return ViscoLaw(R_iso, :relaxation)
end

const C_INC = TensISO{3}(3 * 10.0, 2 * 4.0)

function build_rve_base(f::Real)
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => build_law_M(1.0, 1.0)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0),
        Dict(:C => heaviside_law(C_INC));
        fraction = f
    )
    return rve
end
````

The scalar being differentiated is the effective shear modulus at the last
instant, ``\mu^{\hom}(t_n, t_n) = \beta_{nn}/2``, read off the isotropic
blocks of the effective relaxation operator.

````@example alv_sensitivities
function effective_mu_final(rve, scheme)
    R̃ = homogenize_alv(rve, scheme, :C; times = TIMES)
    _, β = iso_params_from_blocks(R̃)
    return β[end, end] / 2
end
````

## §1 The `set_param` lens — ``\partial \mu^{\hom} / \partial f``

The RVE is built once, outside the differentiated function; `set_param`
substitutes the `Dual` fraction into a copy. Every scheme accepts it,
including the two bounds.

````@example alv_sensitivities
const RVE_BASE_F = build_rve_base(0.2)

function eff_mu_vs_f(f::Real, scheme)
    rve_f = set_param(RVE_BASE_F, AmountParameter(:I), f)
    return effective_mu_final(rve_f, scheme)
end

const SCHEMES = (Voigt(), Reuss(), Dilute(), MoriTanaka(), Maxwell())
const SCHEME_NAMES = ("Voigt", "Reuss", "Dilute", "MoriTanaka", "Maxwell")

for (sch, name) in zip(SCHEMES, SCHEME_NAMES)
    f₀ = 0.2
    dμ_df_AD = ForwardDiff.derivative(f -> eff_mu_vs_f(f, sch), f₀)
    h = 1.0e-5
    dμ_df_FD = (eff_mu_vs_f(f₀ + h, sch) - eff_mu_vs_f(f₀ - h, sch)) / (2h)
    rel_err = abs(dμ_df_AD - dμ_df_FD) / abs(dμ_df_FD)
    @printf "  %-12s  AD = %+.6e   FD = %+.6e   rel_err = %.2e\n" name dμ_df_AD dμ_df_FD rel_err
end
````

## §2 Closure capture — ``\partial \mu^{\hom} / \partial \mu_M``

A matrix modulus is not an RVE field but an argument of the relaxation kernel.
Closing it into `build_law_M` is enough: `ForwardDiff` lifts it to a `Dual`
and the whole Volterra assembly follows.

````@example alv_sensitivities
function eff_mu_vs_μM(μ_M::Real)
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => build_law_M(1.0, μ_M)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => heaviside_law(C_INC));
        fraction = 0.2
    )
    return effective_mu_final(rve, MoriTanaka())
end

μM₀ = 1.0
dμ_dμM_AD = ForwardDiff.derivative(eff_mu_vs_μM, μM₀)
h = 1.0e-5
dμ_dμM_FD = (eff_mu_vs_μM(μM₀ + h) - eff_mu_vs_μM(μM₀ - h)) / (2h)
@printf "  AD = %+.6e   FD = %+.6e   rel_err = %.2e\n" dμ_dμM_AD dμ_dμM_FD abs(dμ_dμM_AD - dμ_dμM_FD) / abs(dμ_dμM_FD)
````

## §3 Both at once — the joint gradient ``\nabla_{(f,\,k_M,\,\mu_M)}\,\mu^{\hom}``

The two patterns compose: one geometric parameter through the lens, two
material ones through the closure, in a single `ForwardDiff.gradient` — and
in a single forward sweep rather than three.

````@example alv_sensitivities
function eff_mu_vs_fkμ(p::AbstractVector)
    f, k_M, μ_M = p
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => build_law_M(k_M, μ_M)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => heaviside_law(C_INC));
        fraction = 0.2
    )
    rve_f = set_param(rve, AmountParameter(:I), f)
    return effective_mu_final(rve_f, MoriTanaka())
end

p₀ = [0.2, 1.0, 1.0]
∇AD = ForwardDiff.gradient(eff_mu_vs_fkμ, p₀)
∇FD = let h = 1.0e-5
    [
        (
                eff_mu_vs_fkμ(p₀ .+ h .* (i == k for k in 1:3)) -
                eff_mu_vs_fkμ(p₀ .- h .* (i == k for k in 1:3))
            ) / (2h) for i in 1:3
    ]
end
for (i, name) in enumerate(("f", "k_M", "μ_M"))
    @printf "  ∂μ/∂%-3s   AD = %+.6e   FD = %+.6e   rel_err = %.2e\n" name ∇AD[i] ∇FD[i] abs(∇AD[i] - ∇FD[i]) / abs(∇FD[i])
end
````

## §4 A relaxation time — ``\partial \mu^{\hom} / \partial \tau_K``

The derivative that has no elastic counterpart: ``\tau_K`` sets *when* the
bulk kernel relaxes, and it enters only through an exponential inside the
user's own kernel function. The package never sees it, and differentiates it
anyway.

````@example alv_sensitivities
function eff_mu_vs_τK(τ_K::Real)
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => build_law_M(1.0, 1.0, τ_K, 0.5)); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0), Dict(:C => heaviside_law(C_INC));
        fraction = 0.2
    )
    return effective_mu_final(rve, MoriTanaka())
end

τK₀ = 1.0
dμ_dτK_AD = ForwardDiff.derivative(eff_mu_vs_τK, τK₀)
h = 1.0e-5
dμ_dτK_FD = (eff_mu_vs_τK(τK₀ + h) - eff_mu_vs_τK(τK₀ - h)) / (2h)
@printf "  AD = %+.6e   FD = %+.6e   rel_err = %.2e\n" dμ_dτK_AD dμ_dτK_FD abs(dμ_dτK_AD - dμ_dτK_FD) / abs(dμ_dτK_FD)
````

Every relative error above sits at ``10^{-7}`` or below — which is the
accuracy of the *finite difference*, not of the AD: with ``h = 10^{-5}`` the
central-difference truncation error is ``h^2 f'''/6``. The automatic
derivative is exact to machine precision; the finite difference is the
approximation being checked against it.

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

