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
struct AffineComposite{P <: Tuple, F <: Tuple, Q <: NamedTuple, L <: NamedTuple, G <: NamedTuple, FL <: NamedTuple, Y} <: AbstractPhaseComposite
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

function AffineComposite(cell, laws::NamedTuple; scheme = Schemes.MoriTanaka(), name::Symbol = :AffineComposite, options...)
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
        name, Tuple(values(laws)), Tuple(values(fractions)), zero(SVector{6, Float64}), p,
        _layout(NamedTuple{(Symbol(first(keys(laws)), :_ε),)}((6,))), first_law.gradients, first_law.fluxes
    )
end

const _NO_STATE = SVector{0, Float64}()

# The strains of the two phases: ε̄₁ the unknown, ε̄₂ from the mixture rule.
_affine_strains(c::AffineComposite, E, ε₁) = (ε₁, (E - c.fractions[1] * ε₁) / c.fractions[2])

function _residual(c::AffineComposite, y, g, prev, Δt)
    ε₁, ε₂ = _affine_strains(c, g, y)
    # The affine linearization of each phase about its average strain.
    L₁, L₂ = map((law, ε) -> ForwardDiff.jacobian(e -> _fluxes(law, e, _NO_STATE), ε), c.phases, (ε₁, ε₂))
    T = promote_type(eltype(y), eltype(g), typeof(c.p.numeric), _numeric_type(c.p.phases))
    # Where a phase has no finite tangent (a power law at zero strain) the
    # linearization does not exist: outside the domain of the fixed point.
    all(isfinite, L₁) && all(isfinite, L₂) || return SVector{6, T}(ntuple(_ -> T(NaN), 6))
    τ₁, τ₂ = _fluxes(c.phases[1], ε₁, _NO_STATE) - L₁ * ε₁, _fluxes(c.phases[2], ε₂, _NO_STATE) - L₂ * ε₂
    # The linear comparison composite of the tangents.
    cell = foldl(zip(c.p.names, (L₁, L₂)); init = c.p.cell) do cell, (φ, L)
        MFH_Core.cell_set_property(cell, φ, :C, TensND.from_mandel(L))
    end
    A = Schemes.concentration_tensors(cell, c.p.scheme, :C; c.p.options...).strain
    A₁, A₂ = map(φ -> _mandel(TensND.change_tens_canon(A[φ])), c.p.names)
    f₁, f₂ = c.fractions
    τ̃ = f₁ * transpose(A₁) * τ₁ + f₂ * transpose(A₂) * τ₂
    τ̄ = f₁ * τ₁ + f₂ * τ₂
    return SVector{6, T}(f₁ * (ε₁ - A₁ * g) - (L₁ - L₂) \ (τ̃ - τ̄))::SVector{6, T}
end

# The stress of the comparison composite is that of the phases at their strains.
function _fluxes(c::AffineComposite, g, y)
    ε₁, ε₂ = _affine_strains(c, g, y)
    return c.fractions[1] * _fluxes(c.phases[1], ε₁, _NO_STATE) + c.fractions[2] * _fluxes(c.phases[2], ε₂, _NO_STATE)
end

# At rest the phases carry no strain: the linearization of a law may not exist
# there (a power law has no finite tangent at zero strain), and none is needed.
_frozen(::AffineComposite, g, prev, Δt) = iszero(g) && iszero(prev.y)

# The first phase starts from its strain of the last step plus the macroscopic
# increment.
_newton_start(c::AffineComposite, g, prev) = TensND.primal.(prev.y + (g - prev.g))

_with_external(c::AffineComposite, ::NamedTuple{()}) = c
_with_external(c::AffineComposite, ext::NamedTuple) = AffineComposite(
    c.name, map(law -> _with_external(law, ext), c.phases), c.fractions, c.y₀, c.p, c.layout, c.gradients, c.fluxes
)
