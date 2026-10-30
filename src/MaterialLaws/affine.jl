# =============================================================================
#  affine.jl — the affine formulation of Masson, Bornert, Suquet and Zaoui
#  (2000) for a composite of two nonlinear elastic (or viscous) phases.
#
#  Every phase is linearized about the average strain it carries: its stress is
#  replaced by its tangent at that strain plus a prestress, σ ≈ L_r:ε + τ_r with
#  τ_r = σ_r(ε̄_r) − L_r:ε̄_r. The two phases with their uniform tangents and
#  prestresses form a linear thermoelastic comparison composite, whose average
#  phase strains Levin's theorem gives exactly for two phases; they must be the
#  strains about which the phases were linearized: a fixed point on ε̄₁ alone,
#  ε̄₂ following from the mixture rule.
# =============================================================================

import ..Laminates
import ..Core as MFH_Core

# What the linear comparison composite of the tangents estimates: the stress of
# the phases at their average strains (affine), or the effective potential,
# whose derivative is the stress (second order).
abstract type _ComparisonEstimate end
struct _AffineEstimate <: _ComparisonEstimate end
struct _SecondOrderEstimate <: _ComparisonEstimate end

"""
    AffineComposite(cell, laws; scheme = MoriTanaka(), options...)

The composite of two phases of nonlinear elastic (or viscous) `laws` (a
`NamedTuple` by phase name, laws of the strain alone such as
[`PowerLawElasticity`](@ref)) by the affine formulation of
[massonAffine2000](@citet): each phase is linearized about its average strain,
``\\boldsymbol{\\sigma} \\approx \\mathbb{L}_r:\\boldsymbol{\\varepsilon} + \\boldsymbol{\\tau}_r``,
with the tangent ``\\mathbb{L}_r`` and the prestress
``\\boldsymbol{\\tau}_r = \\boldsymbol{\\sigma}_r(\\bar{\\boldsymbol{\\varepsilon}}_r) - \\mathbb{L}_r:\\bar{\\boldsymbol{\\varepsilon}}_r``,
and the average strains are those of the linear thermoelastic comparison
composite they form, which Levin's theorem gives for two phases:

```math
f_1\\,\\big(\\bar{\\boldsymbol{\\varepsilon}}_1 - \\mathbb{A}_1:\\boldsymbol{E}\\big)
= (\\mathbb{L}_1 - \\mathbb{L}_2)^{-1}:\\big(\\tilde{\\boldsymbol{\\tau}} - \\langle\\boldsymbol{\\tau}\\rangle\\big),
\\qquad
\\tilde{\\boldsymbol{\\tau}} = \\sum_r f_r\\,\\mathbb{A}_r^{T}:\\boldsymbol{\\tau}_r .
```

`cell` is an [`RVE`](@ref MeanFieldHomogenization.Schemes.RVE) or a [`Laminate`](@ref MeanFieldHomogenization.Laminates.Laminate) of the two phases, `scheme`
a linear scheme for it (`Laminated()` for a laminate); the tangents, anisotropic
in general, replace the stiffnesses the cell declares; `options` go to the
computation of the concentration tensors (`reltol` of the Hill tensors of an
anisotropic medium, for instance). The internal variable is the average strain
of the first phase (`M_ε`, after its name). See
[the theory page](@ref ml-theory-affine).
"""
struct AffineComposite{R <: _ComparisonEstimate, P <: Tuple, F <: Tuple, Q <: NamedTuple, L <: NamedTuple, G <: NamedTuple, FL <: NamedTuple, Y} <: AbstractPhaseComposite
    estimate::R
    name::Symbol
    phases::P
    fractions::F
    y₀::SVector{6, Y}
    p::Q
    layout::L
    gradients::G
    fluxes::FL
end

_cell_fraction(cell::Schemes.RVE, φ) = Schemes.volume_fraction(cell, φ)
_cell_fraction(cell::Laminates.Laminate, φ) = Laminates.layer_volume_fraction(cell, φ)
_cell_members(cell::Schemes.RVE) = Schemes.phase_names(cell)
_cell_members(cell::Laminates.Laminate) = Laminates.layer_names(cell)

AffineComposite(cell, laws::NamedTuple; scheme = Schemes.MoriTanaka(), name::Symbol = :AffineComposite, options...) =
    _comparison_composite(_AffineEstimate(), cell, laws, scheme, name, options)

"""
    SecondOrderComposite(cell, laws; scheme = MoriTanaka(), options...)

The composite of two phases of nonlinear elastic (or viscous) `laws` by the
second-order estimate of [ponteCastaneda1996](@citet): the linear comparison
composite of [`AffineComposite`](@ref), each phase replaced by its tangent at
its average strain with the prestress that makes it exact there, and the
average strains those of the comparison composite, estimates the effective
potential,

```math
\\tilde{w}(\\boldsymbol{E}) = \\sum_r f_r\\,\\Big[w_r(\\bar{\\boldsymbol{\\varepsilon}}_r)
  - \\boldsymbol{\\sigma}_r(\\bar{\\boldsymbol{\\varepsilon}}_r):\\bar{\\boldsymbol{\\varepsilon}}_r
  + \\tfrac{1}{2}\\,\\bar{\\boldsymbol{\\varepsilon}}_r:\\mathbb{L}_r:\\bar{\\boldsymbol{\\varepsilon}}_r\\Big]
  + \\tfrac{1}{2}\\,\\boldsymbol{E}:\\tilde{\\mathbb{L}}:\\boldsymbol{E} + \\tilde{\\boldsymbol{\\tau}}:\\boldsymbol{E}
  + \\tfrac{1}{2}\\,(\\boldsymbol{\\tau}_1 - \\boldsymbol{\\tau}_2):(\\mathbb{L}_1 - \\mathbb{L}_2)^{-1}:(\\tilde{\\boldsymbol{\\tau}} - \\langle\\boldsymbol{\\tau}\\rangle),
```

the energy of the comparison composite, and the stress is its derivative
``\\boldsymbol{\\Sigma} = \\mathrm{d}\\tilde{w}/\\mathrm{d}\\boldsymbol{E}``, the average
strains following ``\\boldsymbol{E}``. The arguments are those of
[`AffineComposite`](@ref). See [the theory page](@ref ml-theory-affine).
"""
SecondOrderComposite(cell, laws::NamedTuple; scheme = Schemes.MoriTanaka(), name::Symbol = :SecondOrderComposite, options...) =
    _comparison_composite(_SecondOrderEstimate(), cell, laws, scheme, name, options)

function _comparison_composite(estimate, cell, laws::NamedTuple, scheme, name, options)
    length(laws) == 2 || throw(ArgumentError("the affine formulation is implemented for two phases (Levin's theorem); got $(length(laws))"))
    Set(keys(laws)) == Set(_cell_members(cell)) ||
        throw(ArgumentError("one law per phase of the cell: the cell has $(_cell_members(cell)), the laws $(keys(laws))"))
    fractions = NamedTuple{keys(laws)}(map(φ -> _cell_fraction(cell, φ), keys(laws)))
    first_law = _composite_phases(laws, fractions)
    for (φ, law) in pairs(laws)
        isempty(law.layout) && _sizes(law.gradients) == (ε = 6,) ||
            throw(ArgumentError("phase :$φ: the affine formulation takes laws of the strain alone, without internal variable"))
    end
    numeric = zero(mapreduce(φ -> eltype(MFH_Core.cell_container_property(cell, φ, :C)), promote_type, keys(laws); init = eltype(values(fractions))))
    p = (fractions = fractions, phases = map(law_parameters, laws), cell = cell, scheme = scheme, options = NamedTuple(options), names = keys(laws), numeric = numeric)
    return AffineComposite(
        estimate, name, Tuple(values(laws)), Tuple(values(fractions)), zero(SVector{6, Float64}), p,
        _layout(NamedTuple{(Symbol(first(keys(laws)), :_ε),)}((6,))), first_law.gradients, first_law.fluxes
    )
end

const _NO_STATE = SVector{0, Float64}()

# The strains of the two phases: ε̄₁ the unknown, ε̄₂ from the mixture rule.
_affine_strains(c::AffineComposite, E, ε₁) = (ε₁, (E - c.fractions[1] * ε₁) / c.fractions[2])

# The linear comparison composite at the average strain ε₁ of the first phase:
# the strains, stresses, tangents and prestresses of the phases and their
# strain-concentration tensors; `nothing` where a phase has no finite tangent
# (a power law at zero strain), outside the domain of the linearization.
function _comparison(c::AffineComposite, g, y)
    ε₁, ε₂ = _affine_strains(c, g, y)
    σ₁, σ₂ = _fluxes(c.phases[1], ε₁, _NO_STATE), _fluxes(c.phases[2], ε₂, _NO_STATE)
    L₁, L₂ = _phase_tangents(c, ε₁, ε₂)
    all(isfinite, L₁) && all(isfinite, L₂) || return nothing
    cell = foldl(zip(c.p.names, (L₁, L₂)); init = c.p.cell) do cell, (φ, L)
        MFH_Core.cell_set_property(cell, φ, :C, TensND.from_mandel(L))
    end
    A = Schemes.concentration_tensors(cell, c.p.scheme, :C; c.p.options...).strain
    A₁, A₂ = map(φ -> _mandel(TensND.canonical(A[φ])), c.p.names)
    f₁, f₂ = c.fractions
    τ₁, τ₂ = σ₁ - L₁ * ε₁, σ₂ - L₂ * ε₂
    return (; ε₁, ε₂, σ₁, σ₂, L₁, L₂, A₁, A₂, τ₁, τ₂, τ̃ = f₁ * transpose(A₁) * τ₁ + f₂ * transpose(A₂) * τ₂, τ̄ = f₁ * τ₁ + f₂ * τ₂)
end

_phase_tangents(c, ε₁, ε₂) = map((law, ε) -> ForwardDiff.jacobian(e -> _fluxes(law, e, _NO_STATE), ε), c.phases, (ε₁, ε₂))

# The average strain of the first phase is that of the comparison composite
# (Levin's theorem).
# The element type of the comparison composite at (g, y): the concentration
# tensors come out of the schemes with it lost to inference.
_comparison_type(c, g, y) = promote_type(eltype(y), eltype(g), typeof(c.p.numeric), _numeric_type(c.p.phases))

function _residual(c::AffineComposite, y, g, prev, Δt)
    T = _comparison_type(c, g, y)
    lcc = _comparison(c, g, y)
    lcc === nothing && return SVector{6, T}(ntuple(_ -> T(NaN), 6))
    return SVector{6, T}(c.fractions[1] * (lcc.ε₁ - lcc.A₁ * g) - (lcc.L₁ - lcc.L₂) \ (lcc.τ̃ - lcc.τ̄))::SVector{6, T}
end

# The affine estimate: the stress of the phases at their average strains.
_fluxes(c::AffineComposite, g, y) = _affine_stress(c, g, y)
function _affine_stress(c, g, y)
    ε₁, ε₂ = _affine_strains(c, g, y)
    return c.fractions[1] * _fluxes(c.phases[1], ε₁, _NO_STATE) + c.fractions[2] * _fluxes(c.phases[2], ε₂, _NO_STATE)
end

# The energy of the comparison composite but the potentials of the phases at
# their average strains, Σ_r f_r w_r(ε̄_r): the first two terms of the
# thermoelastic energy are those of the uniform field, the last one Levin's
# energy of the prestresses at zero macroscopic strain.
function _second_order_quadratic(c::AffineComposite, g, y)
    lcc = _comparison(c, g, y)
    f₁, f₂ = c.fractions
    L̃ = f₁ * lcc.L₁ * lcc.A₁ + f₂ * lcc.L₂ * lcc.A₂
    shift(ε, σ, L) = transpose(ε) * (L * ε / 2 - σ)
    return f₁ * shift(lcc.ε₁, lcc.σ₁, lcc.L₁) + f₂ * shift(lcc.ε₂, lcc.σ₂, lcc.L₂) +
        transpose(g) * (L̃ * g / 2 + lcc.τ̃) + transpose(lcc.τ₁ - lcc.τ₂) * ((lcc.L₁ - lcc.L₂) \ (lcc.τ̃ - lcc.τ̄)) / 2
end

# The second-order estimate: Σ = dw̃/dE, the average strains following E. With
# Q the quadratic part above, ε̄₁ = y and ε̄₂ = (E − f₁y)/f₂, and dw_r = σ_r:dε̄_r,
#   Σ = ∂Q/∂E + σ₂ + (dy/dE)ᵀ (∂Q/∂y + f₁(σ₁ − σ₂)),
# dy/dE = −R_y⁻¹ R_E by the implicit-function theorem on the residual R of the
# average strains.
function _fluxes(c::AffineComposite{_SecondOrderEstimate}, g, y)
    (iszero(g) && iszero(y)) && return _affine_stress(c, g, y)
    f₁ = c.fractions[1]
    ε₁, ε₂ = _affine_strains(c, g, y)
    σ₁, σ₂ = _fluxes(c.phases[1], ε₁, _NO_STATE), _fluxes(c.phases[2], ε₂, _NO_STATE)
    # Where the comparison composite does not exist (a phase without finite
    # tangent), neither does the estimate.
    all(L -> all(isfinite, L), _phase_tangents(c, ε₁, ε₂)) || return σ₂ .* oftype(zero(eltype(σ₂)), NaN)
    ∇Q = ForwardDiff.gradient(x -> _second_order_quadratic(c, x[SVector{6}(7:12)], x[SOneTo(6)]), vcat(y, g))
    T = _comparison_type(c, g, y)
    return SVector{6, T}(∇Q[SVector{6}(7:12)] + σ₂ + transpose(_average_strain_sensitivity(c, g, y)) * (∇Q[SOneTo(6)] + f₁ * (σ₁ - σ₂)))::SVector{6, T}
end

# dy/dE = −R_y⁻¹ R_E, the derivatives in (y, E) in one pass.
function _average_strain_sensitivity(c, g, y)
    J = ForwardDiff.jacobian(x -> _residual(c, x[SOneTo(6)], x[SVector{6}(7:12)], nothing, nothing), vcat(y, g))
    T = _comparison_type(c, g, y)
    return SMatrix{6, 6, T, 36}(-(J[SOneTo(6), SOneTo(6)] \ J[SOneTo(6), SVector{6}(7:12)]))::SMatrix{6, 6, T, 36}
end

# The consistent tangent dΣ/dE, column by column along (e_k, dy/dE e_k): the
# stress holds the derivatives of the Hill tensors already, and a derivative of
# it in all six directions at once nests two six- and twelve-wide dual numbers
# through their quadrature, whose compilation takes more memory than a machine
# may have; one direction at a time keeps it within a few gigabytes. At rest the
# stress is the affine one, and so is its tangent.
function _fluxes_and_tangent(c::AffineComposite{_SecondOrderEstimate}, y, g, prev, Δt, frozen::Bool)
    F = _fluxes(c, g, y)
    frozen && return F, ForwardDiff.jacobian(h -> _affine_stress(c, h, y), g)
    dydE = _average_strain_sensitivity(c, g, y)
    columns = ntuple(k -> ForwardDiff.derivative(s -> _fluxes(c, g + s * _unit(g, k), y + s * dydE[:, k]), zero(eltype(g))), Val(6))
    return F, hcat(columns...)
end

# At rest the phases carry no strain: the linearization of a law may not exist
# there (a power law has no finite tangent at zero strain), and none is needed.
_frozen(::AffineComposite, g, prev, Δt) = iszero(g) && iszero(prev.y)

# The first phase starts from its strain of the last step plus the macroscopic
# increment.
_newton_start(c::AffineComposite, g, prev) = TensND.primal.(prev.y + (g - prev.g))

_with_external(c::AffineComposite, ::NamedTuple{()}) = c
_with_external(c::AffineComposite, ext::NamedTuple) = AffineComposite(
    c.estimate, c.name, map(law -> _with_external(law, ext), c.phases), c.fractions, c.y₀, c.p, c.layout, c.gradients, c.fluxes
)
