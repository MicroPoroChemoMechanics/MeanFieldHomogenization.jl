# =============================================================================
#  library.jl — the laws shipped with MaterialPoints.
#
#  Each is an instance of `ResidualLaw` or `PotentialLaw`, built from a few lines
#  of mathematics: nothing here is privileged, and each doubles as an example of
#  how a user writes a law of their own. Every one is checked in the tests
#  against a closed-form response.
# =============================================================================

# Isotropic stiffness from Young's modulus and Poisson's ratio, in Kelvin–Mandel
# components (TensND's isotropic stiffness, converted once at construction).
_iso_mandel(E, ν) = _mandel(TensND.iso_stiffness_young_poisson(E, ν))

# ½ a:𝔸:a, the quadratic form of a stiffness, written with `transpose` rather
# than `dot` (which conjugates, and a symbolic entry must not see that).
_quadratic(a, 𝔸) = transpose(a) * (𝔸 * a) / 2

# ── Potential route ──────────────────────────────────────────────────────────

"""
    LinearElasticity(ℂ)
    LinearElasticity(; E, ν)

Linear elasticity, ``\\psi = \\tfrac12\\,\\boldsymbol{\\varepsilon}:\\mathbb{C}:\\boldsymbol{\\varepsilon}``,
from a stiffness tensor (TensND, any symmetry, any basis) or, isotropic, from
Young's modulus and Poisson's ratio. The control case of the potential route:
its response is ``\\mathbb{C}:\\boldsymbol{\\varepsilon}`` with the tangent
``\\mathbb{C}``, exactly.
"""
LinearElasticity(ℂ::TensND.AbstractTens{4, 3}) =
    PotentialLaw(; name = :LinearElasticity, energy = _elastic_energy, p = (ℂ = _mandel(ℂ),))
LinearElasticity(; E, ν) = LinearElasticity(TensND.iso_stiffness_young_poisson(E, ν))

_elastic_energy(g, α, p) = _quadratic(g.ε, p.ℂ)

"""
    PowerLawElasticity(; k, σ₀, ε₀, m)

Nonlinear elasticity of power-law type (deformation theory of plasticity):

```math
\\psi(\\boldsymbol{\\varepsilon}) = \\frac{k}{2}\\,(\\operatorname{tr}\\boldsymbol{\\varepsilon})^2
+ \\frac{\\sigma_0\\,\\varepsilon_0}{m + 1}\\left(\\frac{\\varepsilon_{eq}}{\\varepsilon_0}\\right)^{m+1},
```

linear in the volume change (bulk modulus `k`) and of exponent `m` in the
deviatoric strain: the equivalent stress is
``\\sigma_{eq} = \\sigma_0\\,(\\varepsilon_{eq}/\\varepsilon_0)^m``, ``m = 1`` being
linear elasticity of shear modulus ``\\sigma_0 / (3\\varepsilon_0)`` and
``m \\to 0`` rigid–perfect plasticity.

!!! note "Exponents below one"
    For ``m < 1`` the shear stiffness is infinite at zero deviatoric strain, so
    the tangent there is not finite; any nonzero strain is fine.
"""
PowerLawElasticity(; k, σ₀, ε₀, m) =
    PotentialLaw(; name = :PowerLawElasticity, energy = _power_law_energy, p = (; k, σ₀, ε₀, m))

# Written with ε_eq² so that no square root is differentiated at zero. At zero
# deviatoric strain the deviatoric term and its gradient vanish, but automatic
# differentiation of x^a at x = 0 gives 0·∞ for a < 1: that point is taken apart
# (a value test, hence a numerical-only branch; a symbolic strain never is zero).
function _power_law_energy(g, α, p)
    εeq² = 2 * _sqnorm_deviator(g.ε) / 3
    volumetric = p.k / 2 * trace_mandel(g.ε)^2
    iszero(εeq²) && return volumetric + zero(εeq²)
    return volumetric + p.σ₀ * p.ε₀ / (p.m + 1) * (εeq² / p.ε₀^2)^((p.m + 1) / 2)
end

"""
    MaxwellViscoelasticity(; E, ν, τ)

A Maxwell viscoelastic solid written as two potentials: a spring of isotropic
stiffness ``\\mathbb{C}`` (`E`, `ν`) in series with a dashpot of viscosity
``\\tau\\,\\mathbb{C}``, so that every mode relaxes with the same time `τ`:

```math
\\psi = \\tfrac12\\,(\\boldsymbol{\\varepsilon} - \\boldsymbol{\\varepsilon}^v):\\mathbb{C}:(\\boldsymbol{\\varepsilon} - \\boldsymbol{\\varepsilon}^v),
\\qquad
\\varphi = \\frac{\\tau}{2}\\,\\dot{\\boldsymbol{\\varepsilon}}^v:\\mathbb{C}:\\dot{\\boldsymbol{\\varepsilon}}^v,
```

hence ``\\dot{\\boldsymbol{\\sigma}} = \\mathbb{C}:\\dot{\\boldsymbol{\\varepsilon}} - \\boldsymbol{\\sigma}/\\tau``:
held at a fixed strain, the stress relaxes as ``e^{-t/\\tau}``. The example of
the potential route with an internal variable.
"""
MaxwellViscoelasticity(; E, ν, τ) = PotentialLaw(;
    name = :MaxwellViscoelasticity, energy = _maxwell_energy, dissipation = _maxwell_dissipation,
    layout = (εᵛ = 6,), p = (ℂ = _iso_mandel(E, ν), τ = τ)
)

_maxwell_energy(g, α, p) = _quadratic(g.ε - α.εᵛ, p.ℂ)
_maxwell_dissipation(α̇, α, p) = p.τ * _quadratic(α̇.εᵛ, p.ℂ)

"""
    BiotPoroelasticity(; E, ν, b, M)

Saturated linear poroelasticity, a law of two coupled primal fields: the strain
``\\boldsymbol{\\varepsilon}`` and the pore pressure ``p``, conjugate to the
stress ``\\boldsymbol{\\sigma}`` and to the change of fluid content ``\\phi``
(volume of fluid per unit volume). With the drained isotropic stiffness
``\\mathbb{C}`` (`E`, `ν`), the Biot coefficient `b` and the Biot modulus `M`
(see [Coussy (2004)](@cite coussy2004)),

```math
\\boldsymbol{\\sigma} = \\mathbb{C}:\\boldsymbol{\\varepsilon} - b\\,p\\,\\boldsymbol{1},
\\qquad
\\phi = b\\,\\operatorname{tr}\\boldsymbol{\\varepsilon} + \\frac{p}{M},
```

derived from the potential
``\\psi = \\tfrac12\\,\\boldsymbol\\varepsilon:\\mathbb{C}:\\boldsymbol\\varepsilon - b\\,p\\operatorname{tr}\\boldsymbol\\varepsilon - p^2/(2M)``,
``\\boldsymbol\\sigma = \\partial\\psi/\\partial\\boldsymbol\\varepsilon`` and
``\\phi = -\\partial\\psi/\\partial p``. Gradients `(ε = 6, p = 1)`, fluxes
`(σ = 6, φ = 1)`; the tangent blocks are those of the Gauss-point contract,
`σε` ``= \\mathbb{C}``, `σp` ``= -b\\,\\boldsymbol{1}``, `φε` ``= b\\,\\boldsymbol{1}``,
`φp` ``= 1/M``. The example of a coupled law.
"""
BiotPoroelasticity(; E, ν, b, M) = PotentialLaw(;
    name = :BiotPoroelasticity, energy = _biot_energy, gradients = (ε = 6, p = 1),
    flux_names = (σ = 6, φ = 1), signs = (φ = -1,), p = (ℂ = _iso_mandel(E, ν), b = b, M = M)
)

_biot_energy(g, α, p) = _quadratic(g.ε, p.ℂ) - p.b * g.p * trace_mandel(g.ε) - g.p^2 / (2 * p.M)

# ── Residual route ───────────────────────────────────────────────────────────

"""
    NortonViscoplasticity(; E, ν, K, n)

Isotropic elasticity (`E`, `ν`) with Norton's power-law creep of the plastic
strain ``\\boldsymbol{\\varepsilon}^p``:

```math
\\dot{\\boldsymbol{\\varepsilon}}^p = \\frac32\\left(\\frac{\\sigma_{eq}}{K}\\right)^{n}
\\frac{\\boldsymbol{s}}{\\sigma_{eq}},
```

integrated by the backward Euler scheme. Under a constant uniaxial stress
``\\sigma`` the creep rate is ``(\\sigma/K)^n``. Requires ``n \\ge 1``.
"""
function NortonViscoplasticity(; E, ν, K, n)
    n ≥ 1 || throw(DomainError(n, "NortonViscoplasticity: the exponent n must be at least 1"))
    return ResidualLaw(;
        name = :NortonViscoplasticity, stress = _norton_stress, residual = _norton_residual,
        elastic = _norton_elastic, layout = (εᵖ = 6,), p = (ℂ = _iso_mandel(E, ν), K = K, n = n)
    )
end

_norton_stress(ε, y, p) = p.ℂ * (ε - y.εᵖ)

# (3/2)(σ_eq/K)ⁿ s/σ_eq = 3/(2K) (σ_eq²/K²)^((n−1)/2) s: no division by σ_eq and
# no square root to differentiate.
function _norton_rate(σ, p)
    s = deviator(σ)
    σeq² = 3 * _sqnorm_deviator(σ) / 2
    return (3 / (2 * p.K)) * (σeq² / p.K^2)^((p.n - 1) / 2) * s
end

_norton_residual(y, g, prev, Δt, p) =
    (εᵖ = (y.εᵖ - prev.y.εᵖ) - Δt * _norton_rate(_norton_stress(g.ε, y, p), p),)
# No time, or no deviatoric stress at the trial state: no creep over the step.
_norton_elastic(g, prev, Δt, p) =
    iszero(Δt) || iszero(_sqnorm_deviator(_norton_stress(g.ε, prev.y, p)))

"""
    J2Plasticity(; E, ν, σ₀, H = 0, Hkin = 0, hardening = nothing)

Rate-independent von Mises plasticity with isotropic elasticity (`E`, `ν`),
isotropic and linear kinematic hardening:

```math
f = \\sigma_{eq}(\\boldsymbol{\\sigma} - \\boldsymbol{X}) - R(p) \\le 0,\\qquad
\\dot{\\boldsymbol{\\varepsilon}}^p = \\dot p\\,\\frac32\\,
\\frac{\\operatorname{dev}(\\boldsymbol{\\sigma} - \\boldsymbol{X})}{\\sigma_{eq}(\\boldsymbol{\\sigma} - \\boldsymbol{X})},\\qquad
\\boldsymbol{X} = \\frac23\\,H_{kin}\\,\\boldsymbol{\\varepsilon}^p,
```

``p`` being the cumulated plastic strain and ``R(p) = \\sigma_0 + H\\,p`` the
yield stress, or `hardening(p)` when a function is given (any law of isotropic
hardening). The state is `(εᵖ = 6, p = 1)`. An elastic predictor decides
whether the step is plastic; a plastic step is solved by Newton on the backward
Euler equations, which for linear hardening reproduce the radial return.
"""
function J2Plasticity(; E, ν, σ₀, H = zero(σ₀), Hkin = zero(σ₀), hardening = nothing)
    return ResidualLaw(;
        name = :J2Plasticity, stress = _j2_stress, residual = _j2_residual, elastic = _j2_elastic,
        layout = (εᵖ = 6, p = 1),
        p = (ℂ = _iso_mandel(E, ν), E = E, σ₀ = σ₀, H = H, Hkin = Hkin, hardening = hardening)
    )
end

_yield_stress(::Nothing, κ, p) = p.σ₀ + p.H * κ
_yield_stress(R, κ, p) = R(κ)

_j2_stress(ε, y, p) = p.ℂ * (ε - y.εᵖ)
# σ − X, the stress relative to the center of the yield surface.
_j2_relative(ε, y, p) = _j2_stress(ε, y, p) - (2 * p.Hkin / 3) * y.εᵖ

# Elastic when the trial stress, plastic strain frozen, is inside the yield surface.
_j2_elastic(g, prev, Δt, p) =
    von_mises(_j2_relative(g.ε, prev.y, p)) <= _yield_stress(p.hardening, prev.y.p, p)

function _j2_residual(y, g, prev, Δt, p)
    ξ = _j2_relative(g.ε, y, p)
    q = von_mises(ξ)
    return (
        εᵖ = (y.εᵖ - prev.y.εᵖ) - (y.p - prev.y.p) * 3 * deviator(ξ) / (2 * q),
        p = (q - _yield_stress(p.hardening, y.p, p)) / p.E,
    )
end
