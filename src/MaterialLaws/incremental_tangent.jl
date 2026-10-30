# =============================================================================
#  incremental_tangent.jl — a composite of elastoplastic (or any) phases by the
#  incremental tangent approach: over a step, every phase is linearized by its
#  consistent tangent at the end of the step, projected on the isotropic
#  tensors, and a linear scheme of the package concentrates the increment of
#  the macroscopic strain in the phases.
#
#  The unknowns of a step are the strains and the internal variables of the
#  phases. Their equations: the laws of the phases, the concentration of the
#  strain increment in every phase but the last, and the mixture rule of the
#  strains. The tangents depend on the unknowns, so that the concentration is a
#  fixed point, solved with the rest by the local Newton iteration.
# =============================================================================

"""
    IncrementalTangent(rve, laws; scheme = MoriTanaka())

The composite of the phase `laws` (a `NamedTuple` by phase name, one law per
phase of `rve`) by the incremental tangent approach of [hill1965](@citet): over
a step, every phase is replaced by a linear material of its consistent tangent
at the end of the step, projected on the isotropic tensors as
[doghriOuaar2003](@citet) do, and the linear `scheme` concentrates the increment
of the macroscopic strain:

```math
\\Delta\\boldsymbol{\\varepsilon}_r = \\mathbb{A}_r\\big(\\mathbb{C}^{t}_1, \\ldots, \\mathbb{C}^{t}_n\\big):\\Delta\\boldsymbol{E},
\\qquad
\\boldsymbol{\\Sigma} = \\sum_r f_r\\,\\boldsymbol{\\sigma}_r .
```

The geometry and the fractions come from `rve`; the stiffnesses it declares
are replaced. The internal variables are the strains of the phases (`M_ε`, …)
and their own internal variables (`M_εᵖ`, …). See
[the theory page](@ref ml-theory-incremental-tangent).
"""
struct IncrementalTangent{P <: Tuple, F <: Tuple, SG <: Tuple, SY <: Tuple, N, Y, Q <: NamedTuple, L <: NamedTuple, G <: NamedTuple, FL <: NamedTuple} <: AbstractPhaseComposite
    name::Symbol
    phases::P
    fractions::F
    gslices::SG
    yslices::SY
    y₀::SVector{N, Y}
    p::Q
    layout::L
    gradients::G
    fluxes::FL
end

function IncrementalTangent(rve::Schemes.RVE, laws::NamedTuple; scheme = Schemes.MoriTanaka(), name::Symbol = :IncrementalTangent)
    Set(keys(laws)) == Set(Schemes.phase_names(rve)) || throw(
        ArgumentError("one law per phase of the RVE: the RVE has $(Schemes.phase_names(rve)), the laws $(keys(laws))")
    )
    fractions = NamedTuple{keys(laws)}(map(φ -> Schemes.volume_fraction(rve, φ), keys(laws)))
    first_law = _composite_phases(laws, fractions)
    _sizes(first_law.gradients) == (ε = 6,) && _sizes(first_law.fluxes) == (σ = 6,) ||
        throw(ArgumentError("the incremental tangent approach takes phases of the strain and the stress"))
    for φ in keys(laws)
        Schemes.phase_property(rve, φ, :C) isa TensND.TensISO{4} ||
            throw(ArgumentError("phase :$φ must carry an isotropic stiffness :C (TensISO), replaced by its tangent"))
    end
    R = length(laws)
    ny = map(law -> _length(law.layout), Tuple(values(laws)))
    gslices = _blocks(0, ntuple(_ -> 6, R))
    yslices = _blocks(6R, ny)
    sizes = merge(
        map(φ -> NamedTuple{(Symbol(φ, :_ε),)}((6,)), keys(laws))...,
        map(φ -> _phase_sizes(φ, laws[φ]), keys(laws))...
    )
    Y = promote_type(Bool, map(law -> eltype(law.y₀), Tuple(values(laws)))...)
    y₀ = vcat(zero(SVector{6R, Y}), foldl(vcat, map(law -> SVector{length(law.y₀), Y}(law.y₀), Tuple(values(laws))); init = SVector{0, Y}()))
    # The element type of the cell's data, known here: the concentration tensors
    # of a cell are not inferred, and the residual is converted to the
    # promotion of this type with those of the step.
    # (A zero of that type is kept, not the type: a type in a field of a
    # NamedTuple is a DataType, which inference does not see through.)
    numeric = zero(mapreduce(φ -> eltype(Schemes.phase_property(rve, φ, :C)), promote_type, keys(laws); init = eltype(values(fractions))))
    p = (fractions = fractions, phases = map(law_parameters, laws), rve = rve, scheme = scheme, names = keys(laws), numeric = numeric)
    return IncrementalTangent(
        name, Tuple(values(laws)), Tuple(values(fractions)), gslices, yslices, y₀, p,
        _layout(sizes), first_law.gradients, first_law.fluxes
    )
end

# The consistent tangent of a phase at the strain ε and the internal variables
# y of an iterate, as the law would give it at the end of its step.
function _phase_tangent(law, ε, y, prev, Δt)
    frozen = _frozen(law, ε, prev, Δt)
    return last(_fluxes_and_tangent(law, y, ε, prev, Δt, frozen))
end
# A law without internal variable has nothing to solve: its tangent is that of
# its fluxes.
_phase_tangent(law, ε, y::SVector{0}, prev, Δt) = last(_fluxes_and_tangent(law, y, ε, prev, Δt, true))

function _residual(c::IncrementalTangent, y, g, prev, Δt)
    εs = map(gs -> y[gs], c.gslices)
    states = map((gs, ys) -> _phase_state(prev, ys, prev.y[gs]), c.gslices, c.yslices)
    # The linear comparison composite of the isotropic projections of the tangents.
    rve = foldl(zip(c.p.names, c.phases, εs, c.yslices, states); init = c.p.rve) do rve, (φ, law, ε, ys, st)
        # The rotation average of the tangent, its isotropic projection (TensND).
        k, μ = TensND.bulk_shear(TensND.isotropify(TensND.from_mandel(_phase_tangent(law, ε, y[ys], st, Δt))))
        rve = Schemes.set_param(rve, Schemes.PropertyParameter(φ, :C, :bulk), 3k)
        Schemes.set_param(rve, Schemes.PropertyParameter(φ, :C, :shear), 2μ)
    end
    A = Schemes.concentration_tensors(rve, c.p.scheme, :C).strain
    ΔE = g - prev.g
    # The concentration of the increment in every phase but the last, the
    # mixture rule for the last.
    R = length(c.phases)
    conc = map(1:(R - 1)) do r
        εs[r] - prev.y[c.gslices[r]] - _mandel(TensND.canonical(A[c.p.names[r]])) * ΔE
    end
    mixture = sum(map((f, ε) -> f * ε, c.fractions, εs)) - g
    laws = map((law, ε, ys, st) -> _phase_residual(law, y[ys], ε, st, Δt), c.phases, εs, c.yslices, states)
    T = promote_type(eltype(y), eltype(g), eltype(prev.y), typeof(Δt), typeof(c.p.numeric))
    return SVector{length(y), T}(foldl(vcat, (conc..., laws...); init = mixture))::SVector{length(y), T}
end

_fluxes(c::IncrementalTangent, g, y) = sum(map((law, f, gs, ys) -> f * _fluxes(law, y[gs], y[ys]), c.phases, c.fractions, c.gslices, c.yslices))
_frozen(::IncrementalTangent, g, prev, Δt) = false

# Every phase starts from its strain of the last step plus the macroscopic
# increment (a uniform-strain predictor).
function _newton_start(c::IncrementalTangent, g, prev)
    Δg = TensND.primal.(g - prev.g)
    y = MVector(TensND.primal.(prev.y))
    foreach(gs -> (y[gs] = y[gs] + Δg), c.gslices)
    return SVector(y)
end

_with_external(c::IncrementalTangent, ::NamedTuple{()}) = c
_with_external(c::IncrementalTangent, ext::NamedTuple) = IncrementalTangent(
    c.name, map(law -> _with_external(law, ext), c.phases), c.fractions, c.gslices, c.yslices, c.y₀, c.p, c.layout, c.gradients, c.fluxes
)
