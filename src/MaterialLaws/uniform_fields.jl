# =============================================================================
#  uniform_fields.jl — the two composites of laws with uniform fields: every
#  phase at the macroscopic gradient (Voigt, Taylor) or at the macroscopic flux
#  (Reuss, Sachs).
#
#  Each is a law of MaterialLaws whose internal variables gather those of the
#  phases (and, for the uniform flux, the phase gradients and the macroscopic
#  flux), so that the local Newton iteration integrates every phase at once and
#  the consistent tangent of the composite comes from the implicit-function
#  theorem, the interaction of the phases included.
# =============================================================================

"""
    AbstractPhaseComposite

Supertype of the laws of `MaterialLaws` built from the laws of several
phases: [`UniformStrain`](@ref) and [`UniformStress`](@ref).
"""
abstract type AbstractPhaseComposite <: AbstractPointLaw end

# The phases of a composite: their laws and volume fractions, checked to share
# their fields and to fill the volume.
function _composite_phases(laws::NamedTuple, fractions::NamedTuple)
    keys(laws) == keys(fractions) || throw(
        ArgumentError("the laws $(keys(laws)) and the fractions $(keys(fractions)) must name the same phases, in the same order")
    )
    isempty(laws) && throw(ArgumentError("a composite needs at least one phase"))
    first_law = first(values(laws))
    for (φ, law) in pairs(laws)
        law isa AbstractPointLaw || throw(ArgumentError("phase :$φ: $law is not a law of MaterialLaws"))
        _sizes(law.gradients) == _sizes(first_law.gradients) && _sizes(law.fluxes) == _sizes(first_law.fluxes) || throw(
            ArgumentError("phase :$φ has the fields $(_sizes(law.gradients)) → $(_sizes(law.fluxes)); the first phase has $(_sizes(first_law.gradients)) → $(_sizes(first_law.fluxes))")
        )
    end
    TensND.is_hard_numeric(_numeric_type(fractions)) || throw(ArgumentError("the volume fractions must be numbers that can be compared; got $(values(fractions))"))
    all(>(0), values(fractions)) && isapprox(sum(values(fractions)), 1; rtol = 1.0e-12) || throw(
        ArgumentError("the volume fractions must be positive and add up to one; got $fractions")
    )
    return first_law
end

# The internal variables of a phase, renamed after it: εᵖ of phase :M is M_εᵖ.
_phase_sizes(φ::Symbol, law) = NamedTuple{map(v -> Symbol(φ, :_, v), keys(law.layout))}(values(_sizes(law.layout)))

# The static slices of consecutive blocks of the given sizes, starting after `start`.
_blocks(start::Int, sizes::Tuple{}) = ()
_blocks(start::Int, sizes::Tuple) = (StaticArrays.SUnitRange(start + 1, start + first(sizes)), _blocks(start + first(sizes), Base.tail(sizes))...)

# The residual of one phase, or the equation that keeps its internal variables
# when its step is elastic (then the residual of the law need not vanish there).
_phase_residual(law, y, g, prev, Δt) = _frozen(law, g, prev, Δt) ? y - prev.y : _residual(law, y, g, prev, Δt)
_phase_residual(law, y::SVector{0}, g, prev, Δt) = y

_phase_state(prev::PointState, ys, g) = PointState(prev.y[ys], g)

"""
    UniformStrain(laws, fractions)

The composite of the phase `laws` in which every phase undergoes the
macroscopic gradient (the strain): the Voigt bound for linear elastic phases,
the Taylor model for the polycrystals. `laws` and `fractions` are `NamedTuple`s
by phase name, e.g. `UniformStrain((M = matrix, I = inclusion), (M = 0.7, I = 0.3))`.
The phases may carry internal variables and several fields; they share their
fields.

```math
\\boldsymbol{\\varepsilon}_r = \\boldsymbol{E},
\\qquad
\\boldsymbol{\\Sigma} = \\sum_r f_r\\,\\boldsymbol{\\sigma}_r(\\boldsymbol{E}, \\mathsf{y}_r).
```

The internal variables are those of the phases, renamed after them (`M_εᵖ`,
`M_p`, …). See [the theory page](@ref ml-theory-uniform-fields).
"""
struct UniformStrain{P <: Tuple, F <: Tuple, S <: Tuple, N, Y, Q <: NamedTuple, L <: NamedTuple, G <: NamedTuple, FL <: NamedTuple} <: AbstractPhaseComposite
    name::Symbol
    phases::P
    fractions::F
    slices::S
    y₀::SVector{N, Y}
    p::Q
    layout::L
    gradients::G
    fluxes::FL
end

function UniformStrain(laws::NamedTuple, fractions::NamedTuple; name::Symbol = :UniformStrain)
    first_law = _composite_phases(laws, fractions)
    sizes = merge(map(φ -> _phase_sizes(φ, laws[φ]), keys(laws))...)
    slices = _blocks(0, map(law -> _length(law.layout), Tuple(values(laws))))
    y₀ = foldl(vcat, map(law -> law.y₀, Tuple(values(laws))); init = SVector{0, Bool}())
    p = (fractions = fractions, phases = map(law_parameters, laws))
    return UniformStrain(
        name, Tuple(values(laws)), Tuple(values(fractions)), slices, SVector(y₀), p,
        isempty(sizes) ? (;) : _layout(sizes), first_law.gradients, first_law.fluxes
    )
end

_fluxes(c::UniformStrain, g, y) = sum(map((law, f, ys) -> f * _fluxes(law, g, y[ys]), c.phases, c.fractions, c.slices))

_residual(c::UniformStrain, y, g, prev, Δt) = foldl(
    vcat, map((law, ys) -> _phase_residual(law, y[ys], g, _phase_state(prev, ys, prev.g), Δt), c.phases, c.slices);
    init = SVector{0, eltype(y)}()
)

_frozen(c::UniformStrain, g, prev, Δt) = all(map((law, ys) -> _frozen(law, g, _phase_state(prev, ys, prev.g), Δt), c.phases, c.slices))

"""
    UniformStress(laws, fractions)

The composite of the phase `laws` in which every phase carries the macroscopic
flux (the stress): the Reuss bound for linear elastic phases, the Sachs model
for the polycrystals. Same arguments as [`UniformStrain`](@ref).

```math
\\boldsymbol{\\sigma}_r(\\boldsymbol{\\varepsilon}_r, \\mathsf{y}_r) = \\boldsymbol{\\Sigma},
\\qquad
\\sum_r f_r\\,\\boldsymbol{\\varepsilon}_r = \\boldsymbol{E}.
```

The gradient of each phase and the macroscopic flux are unknowns of the step:
the internal variables are the macroscopic flux (`Σ_σ`, …), the gradients of
the phases (`M_ε`, …) and their internal variables (`M_εᵖ`, …). See
[the theory page](@ref ml-theory-uniform-fields).
"""
struct UniformStress{P <: Tuple, F <: Tuple, SG <: Tuple, SY <: Tuple, B, N, Y, Q <: NamedTuple, L <: NamedTuple, G <: NamedTuple, FL <: NamedTuple} <: AbstractPhaseComposite
    name::Symbol
    phases::P
    fractions::F
    gslices::SG      # the gradient of each phase
    yslices::SY      # the internal variables of each phase
    flux::B          # the macroscopic flux
    y₀::SVector{N, Y}
    p::Q
    layout::L
    gradients::G
    fluxes::FL
end

function UniformStress(laws::NamedTuple, fractions::NamedTuple; name::Symbol = :UniformStress)
    first_law = _composite_phases(laws, fractions)
    nG, nF = _length(first_law.gradients), _length(first_law.fluxes)
    nG == nF || throw(ArgumentError("a uniform flux needs as many fluxes as gradients; got $nG gradients and $nF fluxes"))
    R = length(laws)
    ny = map(law -> _length(law.layout), Tuple(values(laws)))
    flux = StaticArrays.SUnitRange(1, nF)
    gslices = _blocks(nF, ntuple(_ -> nG, R))
    yslices = _blocks(nF + R * nG, ny)
    sizes = merge(
        NamedTuple{map(f -> Symbol(:Σ_, f), keys(first_law.fluxes))}(values(_sizes(first_law.fluxes))),
        map(φ -> NamedTuple{map(v -> Symbol(φ, :_, v), keys(first_law.gradients))}(values(_sizes(first_law.gradients))), keys(laws))...,
        map(φ -> _phase_sizes(φ, laws[φ]), keys(laws))...
    )
    Y = promote_type(Bool, map(law -> eltype(law.y₀), Tuple(values(laws)))...)
    y₀ = vcat(zero(SVector{nF + R * nG, Y}), foldl(vcat, map(law -> SVector{length(law.y₀), Y}(law.y₀), Tuple(values(laws))); init = SVector{0, Y}()))
    p = (fractions = fractions, phases = map(law_parameters, laws))
    return UniformStress(
        name, Tuple(values(laws)), Tuple(values(fractions)), gslices, yslices, flux, y₀, p,
        _layout(sizes), first_law.gradients, first_law.fluxes
    )
end

_fluxes(c::UniformStress, g, y) = y[c.flux]

function _residual(c::UniformStress, y, g, prev, Δt)
    Σ = y[c.flux]
    balance = sum(map((f, gs) -> f * y[gs], c.fractions, c.gslices)) - g
    phases = map(c.phases, c.gslices, c.yslices) do law, gs, ys
        gr = y[gs]
        vcat(_fluxes(law, gr, y[ys]) - Σ, _phase_residual(law, y[ys], gr, _phase_state(prev, ys, prev.y[gs]), Δt))
    end
    return foldl(vcat, phases; init = balance)
end

# The phases' gradients are unknowns of the step: there is no predictor to test
# before solving.
_frozen(::UniformStress, g, prev, Δt) = false

# The iteration starts with every phase taking the macroscopic increment of the
# gradient (the uniform-gradient predictor), not from the gradients of the last
# step: a law may have no finite tangent there (a power law at zero strain).
function _newton_start(c::UniformStress, g, prev)
    Δg = TensND.primal.(g - prev.g)
    y = MVector(TensND.primal.(prev.y))
    foreach(gs -> (y[gs] = y[gs] + Δg), c.gslices)
    return SVector(y)
end

# External variables (a temperature, …) reach every phase.
_with_external(c::UniformStrain, ::NamedTuple{()}) = c
_with_external(c::UniformStress, ::NamedTuple{()}) = c
_with_external(c::UniformStrain, ext::NamedTuple) = UniformStrain(
    c.name, map(law -> _with_external(law, ext), c.phases), c.fractions, c.slices, c.y₀, c.p, c.layout, c.gradients, c.fluxes
)
_with_external(c::UniformStress, ext::NamedTuple) = UniformStress(
    c.name, map(law -> _with_external(law, ext), c.phases), c.fractions, c.gslices, c.yslices, c.flux, c.y₀, c.p, c.layout, c.gradients, c.fluxes
)

function Base.show(io::IO, c::AbstractPhaseComposite)
    print(io, c.name, " of ", length(c.phases), " phases (", join(keys(c.p.fractions), ", "), ")")
    return nothing
end
