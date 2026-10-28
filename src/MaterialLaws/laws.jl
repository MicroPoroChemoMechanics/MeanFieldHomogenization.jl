# =============================================================================
#  laws.jl — the two ways of writing a law, and the state a point carries.
#
#  A law is data: a few plain functions and a NamedTuple of parameters. The
#  functions say *what* the law is; how it is integrated over a step is the
#  business of the local algorithm (algorithms.jl), and where it is evaluated
#  that of the caller (a point driver, an FE element loop). Keeping the three
#  apart is what lets a new law be a dozen lines, a new algorithm serve every
#  law, and every law plug into every caller.
#
#  A law may couple several primal fields. It declares its gradients, say the
#  strain and the pore pressure (ε = 6, p = 1), and the fluxes conjugate to
#  them, the stress and the change of fluid content (σ = 6, φ = 1), and it may
#  carry internal variables with an evolution law (plastic strain, damage). The
#  functions of the law receive and return NamedTuples by those names; the
#  local algorithm flattens them into static vectors and differentiates the
#  whole, so that every block of the coupled tangent comes from one
#  implicit-function computation.
#
#  The parameters are explicit (`p`) so that a response can be differentiated
#  with respect to them: a law built with `ForwardDiff.Dual` parameters returns
#  the derivatives of its fluxes, tangent and state with respect to them.
# =============================================================================

"""
    AbstractPointLaw <: AbstractMFHMaterial

Supertype of the constitutive laws of
[`MaterialLaws`](@ref MeanFieldHomogenization.MaterialLaws): a law written
with [`ResidualLaw`](@ref) or [`PotentialLaw`](@ref), integrated over a step by
[`point_update`](@ref).

Being an [`AbstractMFHMaterial`](@ref), a law answers
[`material_response`](@ref) and plugs into any caller of that contract, a
finite-element element loop included, exactly as a homogenized material does.
"""
abstract type AbstractPointLaw <: AbstractMFHMaterial end

"""
    kinematics(law) -> SmallStrain()

The kinematics a law is written in. [`SmallStrain`](@ref) is the only one so far.
"""
kinematics(::AbstractPointLaw) = SmallStrain()

"""
    PointState(y, g) <: AbstractMaterialState

The state a material point carries from one step to the next: its internal
variables `y` and its gradients `g` (the strain, and any other primal field of
the law) at the end of the step, both flat `SVector`s laid out as the law's
[`state_layout`](@ref) and [`gradient_layout`](@ref) say.

The functions of a law read the state of the previous step by name, as
`prev.y.εᵖ` or `prev.g.ε`. A state is never mutated: [`point_update`](@ref)
returns a new one, so that a step a caller rejects is retried from the old one.
"""
struct PointState{N, M, T} <: AbstractMaterialState
    y::SVector{N, T}
    g::SVector{M, T}
end

function PointState(y::StaticVector{N, Y}, g::StaticVector{M, G}) where {N, M, Y, G}
    T = promote_type(Y, G)
    return PointState{N, M, T}(SVector{N, T}(y), SVector{M, T}(g))
end

# ── Layouts: names ↔ flat static vectors ─────────────────────────────────────

# `(εᵖ = 6, p = 1)` → `(εᵖ = SUnitRange(1, 6), p = 7)`: the slice of the flat
# vector each name owns, static, a size-one entry being read as a scalar.
function _layout(sizes::NamedTuple)
    all(n -> n isa Integer && n > 0, values(sizes)) || throw(
        ArgumentError("a layout gives a positive integer size per name; got $sizes")
    )
    stops = cumsum(collect(Int, values(sizes)))
    slices = map(Tuple(values(sizes)), Tuple(stops)) do n, s
        n == 1 ? s : StaticArrays.SUnitRange(s - n + 1, s)
    end
    return NamedTuple{keys(sizes)}(slices)
end

_width(::Integer) = 1
_width(::StaticArrays.SUnitRange{S, L}) where {S, L} = L
_length(layout::NamedTuple) = sum(map(_width, values(layout)); init = 0)
_sizes(layout::NamedTuple) = map(_width, layout)

# A flat vector read by name, and back.
_named(v::StaticVector, layout::NamedTuple) = map(r -> v[r], layout)

_flat(v::StaticVector, layout::NamedTuple) = _checked_length(v, layout)
function _flat(x::NamedTuple, layout::NamedTuple{K}) where {K}
    parts = map(_as_svector, values(NamedTuple{K}(x)))
    return _checked_length(foldl(vcat, parts; init = SVector{0, Bool}()), layout)
end

_as_svector(a::Number) = SVector(a)
_as_svector(v::StaticVector) = SVector(v)

function _checked_length(v::StaticVector, layout)
    length(v) == _length(layout) || throw(
        DimensionMismatch(
            "a law returned $(length(v)) values where its layout $(_sizes(layout)) " *
                "announces $(_length(layout))"
        )
    )
    return v
end

"""
    state_layout(law) -> NamedTuple

The names of a law's internal variables and the size of each, e.g.
`(εᵖ = 6, p = 1)` for von Mises plasticity: what a post-processing reads, and
what [`PointResponse`](@ref) indexes by name.
"""
state_layout(law::AbstractPointLaw) = _sizes(law.layout)

"""
    gradient_layout(law) -> NamedTuple

The primal fields a law consumes and the fluxes it returns, with their sizes:
`(ε = 6,)` and `(σ = 6,)` for a mechanical law, `(ε = 6, p = 1)` and
`(σ = 6, φ = 1)` for a poroelastic one. A size-6 entry is a symmetric tensor in
[`Mandel`](@ref) components, a size-1 entry a scalar.
"""
gradient_layout(law::AbstractPointLaw) = _sizes(law.gradients)

"""
    flux_layout(law) -> NamedTuple

The fluxes a law returns, with their sizes, in the order of the gradients they
are conjugate to (see [`gradient_layout`](@ref)).
"""
flux_layout(law::AbstractPointLaw) = _sizes(law.fluxes)

"""
    law_parameters(law) -> NamedTuple

The parameters `p` the law's functions receive. Building the law again with
`ForwardDiff.Dual` values in their place differentiates its response with
respect to them.
"""
law_parameters(law::AbstractPointLaw) = law.p

Constitutive.gradient_names(law::AbstractPointLaw) = keys(law.gradients)
Constitutive.flux_names(law::AbstractPointLaw) = keys(law.fluxes)
Constitutive.tangent_blocks(law::AbstractPointLaw) =
    Tuple(Symbol(f, g) for f in keys(law.fluxes) for g in keys(law.gradients))

# The fields of a law: one gradient and its flux by default, the strain and the
# stress. Coupled laws name theirs, conjugate pairs in the same order and of the
# same sizes.
function _field_layouts(gradients::NamedTuple, fluxes::NamedTuple, name)
    length(gradients) == length(fluxes) && all(values(gradients) .== values(fluxes)) || throw(
        ArgumentError(
            "law $name: gradients $gradients and fluxes $fluxes must be conjugate pairs, " *
                "in the same order and of the same sizes"
        )
    )
    return _layout(gradients), _layout(fluxes)
end

# ── The residual route ───────────────────────────────────────────────────────

"""
    ResidualLaw(; fluxes, residual, layout, gradients = (ε = 6,), flux_names = (σ = 6,),
                  p = (;), elastic = nothing, y₀ = zeros, name = :ResidualLaw)
    ResidualLaw(; stress, residual, layout, …)

A law given by the equations of its implicit integration over a step.

- `fluxes(g, y, p)` returns the fluxes, a `NamedTuple` by the flux names:
  `(σ = …,)` for a mechanical law, `(σ = …, φ = …)` for a poroelastic one.
  A mechanical law may give `stress(ε, y, p)` instead.
- `residual(y, g, prev, Δt, p)` returns the residual of the local equations,
  a `NamedTuple` by the names of the internal variables (or a flat `SVector`),
  whose root is the state at the end of the step, in any units.
- `layout` names the internal variables and their sizes, e.g.
  `(εᵖ = 6, p = 1)`, or `(d = 1,)` for a scalar damage.
- `gradients`, `flux_names`: the primal fields and their conjugate fluxes, with
  their sizes, in the same order. The default is the strain and the stress.
- `p` holds every parameter the functions read.
- `elastic(g, prev, Δt, p)`, optional, returns `true` when the step leaves the
  internal variables unchanged: the elastic predictor of a rate-independent
  law, or the threshold of a damage law. Without it every step is solved.
- `y₀`, optional, the initial internal variables (zero by default).

The functions receive `g`, `y` and the previous state `prev` (fields `prev.y`,
`prev.g`) as `NamedTuple`s: `g.ε`, `g.p`, `y.εᵖ`, `prev.y.d`. A size-6 entry
is a symmetric tensor in [`Mandel`](@ref) components, a size-1 entry a scalar.

[`point_update`](@ref) solves the residual by Newton's method with a Jacobian
by automatic differentiation, and returns the consistent tangent of the fluxes
``\\boldsymbol{F}`` with respect to the gradients ``\\boldsymbol{G}`` by the
implicit-function theorem,

```math
\\frac{\\mathrm{d}\\boldsymbol{F}}{\\mathrm{d}\\boldsymbol{G}}
= \\frac{\\partial\\boldsymbol{F}}{\\partial\\boldsymbol{G}}
- \\frac{\\partial\\boldsymbol{F}}{\\partial\\boldsymbol{y}}
  \\left(\\frac{\\partial\\boldsymbol{R}}{\\partial\\boldsymbol{y}}\\right)^{-1}
  \\frac{\\partial\\boldsymbol{R}}{\\partial\\boldsymbol{G}},
```

every coupling block included, so that no derivative is written by hand.

```julia
# A viscous shear flow in series with a spring: σ = 2μ (ε − εᵛ), ε̇ᵛ = dev σ / (2η)
law = ResidualLaw(;
    stress = (ε, y, p) -> 2p.μ * (ε - y.εᵛ),
    residual = (y, g, prev, Δt, p) -> (εᵛ = y.εᵛ - prev.y.εᵛ - Δt * deviator(2p.μ * (g.ε - y.εᵛ)) / (2p.η),),
    layout = (εᵛ = 6,), p = (μ = 1.0, η = 10.0),
)
```
"""
struct ResidualLaw{FF, FR, FE, N, Y, P <: NamedTuple, L <: NamedTuple, G <: NamedTuple, F <: NamedTuple} <: AbstractPointLaw
    name::Symbol
    flux_function::FF
    residual::FR
    elastic::FE
    y₀::SVector{N, Y}
    p::P
    layout::L
    gradients::G
    fluxes::F
end

function ResidualLaw(;
        residual, layout::NamedTuple, fluxes = nothing, stress = nothing,
        gradients::NamedTuple = (ε = 6,), flux_names::NamedTuple = (σ = 6,),
        p::NamedTuple = (;), elastic = nothing, y₀ = nothing, name::Symbol = :ResidualLaw
    )
    ff = _flux_function(fluxes, stress, gradients, flux_names, name)
    gl, fl = _field_layouts(gradients, flux_names, name)
    yl = _layout(layout)
    return ResidualLaw(name, ff, residual, elastic, _initial_y(yl, y₀), p, yl, gl, fl)
end

# `stress(ε, y, p)` is the mechanical shorthand of `fluxes(g, y, p) = (σ = …,)`.
function _flux_function(fluxes, stress, gradients, flux_names, name)
    (fluxes === nothing) == (stress === nothing) && throw(
        ArgumentError("law $name: give either `fluxes = (g, y, p) -> (σ = …,)` or `stress = (ε, y, p) -> σ`")
    )
    fluxes === nothing || return fluxes
    keys(gradients) == (:ε,) && keys(flux_names) == (:σ,) || throw(
        ArgumentError("law $name: `stress` is for a law of the strain alone; a coupled law gives `fluxes`")
    )
    return _StressFlux(stress)
end

struct _StressFlux{S}
    stress::S
end
(f::_StressFlux)(g, y, p) = (σ = f.stress(g.ε, y, p),)

# ── The potential route ──────────────────────────────────────────────────────

"""
    PotentialLaw(; energy, dissipation = nothing, layout = (;), gradients = (ε = 6,),
                   flux_names = (σ = 6,), signs = nothing, p = (;), y₀ = zeros,
                   name = :PotentialLaw)

A law given by two potentials, in the framework of generalized standard
materials: the free energy ``\\psi(\\boldsymbol{G}, \\boldsymbol{\\alpha})`` of the
gradients and the internal variables, and the dissipation potential
``\\varphi(\\dot{\\boldsymbol{\\alpha}})``.

- `energy(g, α, p)` returns ``\\psi``;
- `dissipation(α̇, α, p)` returns ``\\varphi``, `α` being the internal variables
  at the beginning of the step; it is required as soon as the law has internal
  variables;
- each flux is the derivative of ``\\psi`` with respect to its gradient, times
  the sign `signs` gives it (`+1` by default): a poroelastic potential written
  in strain and pore pressure gives ``\\boldsymbol\\sigma = \\partial\\psi/\\partial\\boldsymbol\\varepsilon``
  and ``\\phi = -\\partial\\psi/\\partial p``, hence `signs = (σ = 1, φ = -1)`;
- `layout`, `gradients`, `flux_names`, `p`, `y₀` and `name` as for
  [`ResidualLaw`](@ref).

Over a step of length ``\\Delta t`` the internal variables minimize the
incremental energy

```math
\\boldsymbol{\\alpha} = \\arg\\min_{\\boldsymbol{a}}\\;
\\psi(\\boldsymbol{G}, \\boldsymbol{a})
+ \\Delta t\\,\\varphi\\!\\left(\\frac{\\boldsymbol{a} - \\boldsymbol{\\alpha}_n}{\\Delta t}\\right),
```

the stationarity condition being solved as the residual of a
[`ResidualLaw`](@ref): the gradient and the Hessian come from automatic
differentiation, so a law is written as its two potentials and nothing else.
Newton's method finds a stationary point; with convex potentials it is the
minimum. A step of zero length leaves the internal variables unchanged.

```julia
# Nonlinear elasticity, no internal variable
law = PotentialLaw(; energy = (g, α, p) -> p.k / 2 * trace_mandel(g.ε)^2 + p.μ * sum(abs2, deviator(g.ε))^2,
                   p = (k = 10.0, μ = 3.0))
```
"""
struct PotentialLaw{FΨ, FΦ, N, Y, P <: NamedTuple, L <: NamedTuple, G <: NamedTuple, F <: NamedTuple, S} <: AbstractPointLaw
    name::Symbol
    energy::FΨ
    dissipation::FΦ
    y₀::SVector{N, Y}
    p::P
    layout::L
    gradients::G
    fluxes::F
    signs::S
end

function PotentialLaw(;
        energy, dissipation = nothing, layout::NamedTuple = (;),
        gradients::NamedTuple = (ε = 6,), flux_names::NamedTuple = (σ = 6,), signs = nothing,
        p::NamedTuple = (;), y₀ = nothing, name::Symbol = :PotentialLaw
    )
    gl, fl = _field_layouts(gradients, flux_names, name)
    yl = _layout(layout)
    y = _initial_y(yl, y₀)
    length(y) == 0 || dissipation !== nothing || throw(
        ArgumentError(
            "law $name has internal variables $(keys(layout)) but no dissipation potential: " *
                "give `dissipation = (α̇, α, p) -> …`"
        )
    )
    return PotentialLaw(name, energy, dissipation, y, p, yl, gl, fl, _flux_signs(signs, gl, fl, name))
end

# One sign per entry of the flat flux vector, or `nothing` when all are +1.
_flux_signs(::Nothing, gl, fl, name) = nothing
function _flux_signs(signs::NamedTuple, gl, fl, name)
    keys(signs) ⊆ keys(fl) || throw(ArgumentError("law $name: signs $signs name no flux among $(keys(fl))"))
    per_flux = map(k -> Float64(get(signs, k, 1)), keys(fl))
    all(s -> s == 1 || s == -1, per_flux) || throw(ArgumentError("law $name: a sign is +1 or -1; got $signs"))
    return SVector(foldl(vcat, map((s, r) -> fill(s, _width(r)), per_flux, Tuple(values(fl))); init = Float64[])...)
end

# The initial internal variables: zero by default, else the given values, of the
# length the layout announces.
function _initial_y(layout::NamedTuple, y₀)
    N = _length(layout)
    y₀ === nothing && return zero(SVector{N, Float64})
    y = y₀ isa NamedTuple ? _flat(map(float, y₀), layout) : SVector{N}(float.(y₀))
    length(y) == N || throw(DimensionMismatch("y₀ has $(length(y)) entries; the layout announces $N"))
    return SVector{N}(y)
end

"""
    initial_state(law::AbstractPointLaw) -> PointState

The state of a fresh point: the law's initial internal variables and zero
gradients.
"""
Constitutive.initial_state(law::AbstractPointLaw) =
    PointState(law.y₀, zero(SVector{_length(law.gradients), eltype(law.y₀)}))

# ── What the local algorithm sees ────────────────────────────────────────────
# Both routes reduce to three functions of flat vectors: the fluxes, the
# residual of the local equations, and whether the step leaves the internal
# variables unchanged. The potential route obtains the first two by
# differentiating its potentials.

_prev(law, prev::PointState) = (y = _named(prev.y, law.layout), g = _named(prev.g, law.gradients))

_fluxes(law::ResidualLaw, g, y) =
    _flat(law.flux_function(_named(g, law.gradients), _named(y, law.layout), law.p), law.fluxes)
_residual(law::ResidualLaw, y, g, prev, Δt) =
    _flat(law.residual(_named(y, law.layout), _named(g, law.gradients), _prev(law, prev), Δt, law.p), law.layout)
_frozen(::ResidualLaw{<:Any, <:Any, Nothing}, g, prev, Δt) = false
_frozen(law::ResidualLaw, g, prev, Δt) = law.elastic(_named(g, law.gradients), _prev(law, prev), Δt, law.p)::Bool

_energy(law::PotentialLaw, g, y) = law.energy(_named(g, law.gradients), _named(y, law.layout), law.p)
_fluxes(law::PotentialLaw, g, y) = _signed(law.signs, ForwardDiff.gradient(h -> _energy(law, h, y), g))
_residual(law::PotentialLaw, y, g, prev, Δt) =
    ForwardDiff.gradient(a -> _incremental_energy(law, a, g, prev, Δt), y)
# A step of zero length dissipates nothing: the internal variables keep their
# values (and the incremental energy, divided by Δt, would not be defined).
_frozen(law::PotentialLaw, g, prev, Δt) = iszero(Δt)

_signed(::Nothing, F) = F
_signed(s::StaticVector, F) = s .* F

function _incremental_energy(law::PotentialLaw, a, g, prev, Δt)
    rate = _named((a - prev.y) / Δt, law.layout)
    return _energy(law, g, a) + Δt * law.dissipation(rate, _named(prev.y, law.layout), law.p)
end

# ── External variables ───────────────────────────────────────────────────────
# A temperature, a time, any field the law reads without returning its flux: the
# caller gives them by name for the step, and the law reads them in `p` beside
# its parameters (a name already in `p` takes the given value, which makes a
# parameter vary along a path). A dual value is differentiated like a parameter.

_with_external(law, ::NamedTuple{()}) = law
_with_external(law, ext::NamedTuple) = _with_parameters(law, merge(law.p, ext))

# The same law with other parameters.
_with_parameters(law::ResidualLaw, p) = ResidualLaw(
    law.name, law.flux_function, law.residual, law.elastic, law.y₀, p,
    law.layout, law.gradients, law.fluxes
)
_with_parameters(law::PotentialLaw, p) = PotentialLaw(
    law.name, law.energy, law.dissipation, law.y₀, p, law.layout, law.gradients, law.fluxes, law.signs
)

# ── Display ──────────────────────────────────────────────────────────────────

function Base.show(io::IO, law::Union{ResidualLaw, PotentialLaw})
    route = law isa ResidualLaw ? "residual" : "potential"
    vars = isempty(law.layout) ? "no internal variable" : "internal variables " * join(keys(law.layout), ", ")
    fields = join(keys(law.gradients), ", ") * " → " * join(keys(law.fluxes), ", ")
    print(io, law.name, " (", route, " route; ", fields, "; ", vars, ")")
    return nothing
end
