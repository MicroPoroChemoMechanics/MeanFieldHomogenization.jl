# =============================================================================
#  crack_damage.jl — damage by the growth of microcracks, derived from the
#  homogenized energy of the cracked solid, as a law of MaterialPoints.
#
#  The internal variables are the crack densities of the families of the RVE.
#  The stiffness at given densities is the homogenized one, from any scheme of
#  the package; the force driving the growth of a family is the derivative of
#  the stored energy with respect to its density, at fixed strain; a family
#  grows when that force reaches its resistance. Nothing is fitted but the
#  resistance, which follows from a toughness when the cracks grow in size.
# =============================================================================

"""
    CrackGrowthResistance(; Gc, a₀, d₀)

The resistance to the growth of a family of ``\\mathcal{N}`` penny-shaped cracks
per unit volume whose radius ``a`` grows at fixed number, the crack density
being ``d = \\mathcal{N} a^3``. Griffith's criterion [griffith1921](@cite) on each crack front, of
toughness ``G_c`` (an energy per unit area), reads in terms of the density

```math
R(d) = \\frac{2\\pi G_c}{3 a_0}\\left(\\frac{d_0}{d}\\right)^{1/3},
```

``a_0`` and ``d_0`` being the radius and the density in a reference state (the
initial one, say), which fix ``\\mathcal{N} = d_0 / a_0^3``. The resistance
decreases as the cracks grow: the growth is unstable. Units: ``G_c / a_0`` is a
stress, e.g. `Gc` in N/mm and `a₀` in mm for stresses in MPa.
"""
struct CrackGrowthResistance{T}
    Gc::T
    a₀::T
    d₀::T
    function CrackGrowthResistance{T}(Gc, a₀, d₀) where {T}
        Gc > 0 && a₀ > 0 && d₀ > 0 || throw(
            ArgumentError("CrackGrowthResistance: Gc = $Gc, a₀ = $a₀ and d₀ = $d₀ must all be positive")
        )
        return new{T}(Gc, a₀, d₀)
    end
end

function CrackGrowthResistance(; Gc, a₀, d₀)
    T = promote_type(typeof(Gc), typeof(a₀), typeof(d₀))
    return CrackGrowthResistance{T}(Gc, a₀, d₀)
end

(R::CrackGrowthResistance)(d) = 2 * oftype(R.Gc, π) * R.Gc / (3 * R.a₀) * cbrt(R.d₀ / d)

"""
    CrackGrowthDamage(rve; scheme = MoriTanaka(), resistance, τ = nothing, families)

The law of a solid weakened by families of microcracks that **grow**. `rve` is
a cell of the package whose crack phases (declared with a
[`CrackDensity`](@ref MeanFieldHomogenization.Schemes.CrackDensity)) are the
families; their densities in `rve` are the initial ones. `scheme` is any linear
scheme that handles cracks. `resistance` gives the resistance of a family as a
function of its density: a [`CrackGrowthResistance`](@ref), any function
`d -> R`, or a `NamedTuple` of them keyed by family. `families` defaults to
every crack phase of `rve`.

At the strain ``\\boldsymbol{\\varepsilon}`` and the densities ``d_i``,

```math
\\boldsymbol{\\sigma} = \\mathbb{C}^{\\mathrm{hom}}(d_1, \\ldots, d_n):\\boldsymbol{\\varepsilon},
\\qquad
Y_i = -\\frac{\\partial}{\\partial d_i}\\Big(\\frac12\\,\\boldsymbol{\\varepsilon}:\\mathbb{C}^{\\mathrm{hom}}:\\boldsymbol{\\varepsilon}\\Big),
\\qquad
\\dot d_i \\ge 0,\\quad Y_i \\le R_i(d_i),\\quad \\dot d_i\\,\\big(Y_i - R_i(d_i)\\big) = 0 .
```

This is the rate-independent law, `τ = nothing`. It suits a resistance that
increases with the density. Under a [`CrackGrowthResistance`](@ref), which
decreases, the growth is unstable: the densities jump at the threshold, and
Newton's method cannot follow the jump from the densities of the last step.
A time `τ > 0` then regularizes the growth by a viscosity,

```math
\\dot d_i = \\frac{1}{\\tau}\\,\\Big\\langle \\frac{Y_i}{R_i(d_i)} - 1 \\Big\\rangle,
```

``\\langle x\\rangle = \\max(x, 0)``, which tends to the rate-independent law as
``\\tau`` becomes small against the duration of the loading.

The result is a [`ResidualLaw`](@ref) whose internal variables are the
densities, named after the families: its consistent tangent includes the growth
of the cracks, and it runs in [`point_response`](@ref) and in a finite-element
computation like any other law. The cracks stay open (no unilateral contact).
The approach is that of [dormieuxKondo2016book](@citet); the derivation is in
[the theory page](@ref mp-theory-crack-damage).
"""
function CrackGrowthDamage(
        rve::Schemes.RVE; scheme = Schemes.MoriTanaka(), resistance, τ = nothing, families = nothing,
        name::Symbol = :CrackGrowthDamage
    )
    τ === nothing || τ > 0 || throw(ArgumentError("CrackGrowthDamage: the time τ must be positive; got τ = $τ"))
    fams = families === nothing ? Constitutive._crack_phase_names(rve) : Tuple(families)
    isempty(fams) && throw(ArgumentError("CrackGrowthDamage needs at least one crack family (a phase with a CrackDensity)"))
    for φ in fams
        rve.amounts[φ] isa Schemes.CrackDensity ||
            throw(ArgumentError("phase :$φ is not a crack family: its amount is not a CrackDensity"))
    end
    resistances = NamedTuple{fams}(map(φ -> _family_resistance(resistance, φ), fams))
    return ResidualLaw(;
        name, stress = _cd_stress, residual = _cd_residual, elastic = _cd_below_threshold,
        layout = NamedTuple{fams}(ntuple(_ -> 1, length(fams))),
        y₀ = [Schemes.get_param(rve, Schemes.AmountParameter(φ)) for φ in fams],
        p = (; rve, scheme, resistances, τ)
    )
end

_family_resistance(R::NamedTuple, φ) = R[φ]
_family_resistance(R, φ) = R

# The cracked solid at the densities y.
_cracked_rve(p, y) = foldl((rve, φ) -> Schemes.set_param(rve, Schemes.AmountParameter(φ), y[φ]), keys(p.resistances); init = p.rve)

_cd_stress(ε, y, p) = _effective_stiffness(_cracked_rve(p, y), p.scheme) * ε

# The forces driving the families, Yᵢ = −∂W/∂dᵢ at fixed strain.
function _driving_forces(ε, y, p)
    φs = keys(p.resistances)
    d = SVector(map(φ -> y[φ], φs))
    W(x) = _quadratic(ε, _effective_stiffness(_cracked_rve(p, NamedTuple{φs}(Tuple(x))), p.scheme))
    return NamedTuple{φs}(Tuple(-ForwardDiff.gradient(W, d)))
end

# Elastic when no family has reached its resistance at the densities of the
# last step.
function _cd_below_threshold(g, prev, Δt, p)
    Y = _driving_forces(g.ε, prev.y, p)
    return all(φ -> Y[φ] <= p.resistances[φ](prev.y[φ]), keys(p.resistances))
end

# A density must stay positive: outside, the residual is not finite, which
# makes the local Newton iteration shorten its correction.
function _cd_residual(y, g, prev, Δt, p)
    φs = keys(p.resistances)
    all(φ -> y[φ] > 0, φs) || return NamedTuple{φs}(map(φ -> oftype(y[φ], NaN), φs))
    Y = _driving_forces(g.ε, y, p)
    return NamedTuple{φs}(map(φ -> _cd_growth(p.τ, y[φ] - prev.y[φ], Y[φ] / p.resistances[φ](y[φ]), Δt), φs))
end

# Rate-independent: the complementarity of the growth Δd ≥ 0 and of the
# criterion Y/R ≤ 1, written with `min` (a comparison: numbers only, no
# symbols); both terms are dimensionless, so they share a scale.
_cd_growth(::Nothing, Δd, ratio, Δt) = min(Δd, 1 - ratio)
# Viscous: backward Euler on ḋ = ⟨Y/R − 1⟩ / τ.
_cd_growth(τ, Δd, ratio, Δt) = Δd - Δt / τ * max(ratio - 1, zero(ratio))
