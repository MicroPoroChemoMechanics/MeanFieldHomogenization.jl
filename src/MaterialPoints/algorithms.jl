# =============================================================================
#  algorithms.jl — integrating a law over one step.
#
#  The local equations R(y) = 0 are solved on the values, with the numbers the
#  caller passed: a law built with dual parameters, or driven by a dual strain,
#  runs the same Newton iterations as a plain one, the iterate being a plain
#  vector. The derivatives are attached afterwards by full Newton steps taken in
#  the dual type from the converged values. From an exact root, one Newton step
#  gives the first derivatives exactly (they are those of the implicit-function
#  theorem), and k steps give every derivative up to order 2ᵏ − 1, which is what
#  nested ForwardDiff calls need. No tolerance is ever applied to a derivative.
# =============================================================================

"""
    AbstractLocalAlgorithm

Supertype of the algorithms that integrate a law over a step. [`LocalNewton`](@ref)
is the default; a new algorithm subtypes this and adds a method of
`MaterialPoints._solve_local`.
"""
abstract type AbstractLocalAlgorithm end

"""
    LocalNewton(; abstol = 1e-12, reltol = 1e-12, maxiters = 50)

Newton's method on the local equations of a law, with the Jacobian by automatic
differentiation.

The step is converged when the Newton correction ``\\delta`` of the internal
variables ``\\boldsymbol{y}`` satisfies
``\\max_i|\\delta_i| \\le`` `abstol` ``+`` `reltol` ``\\max_i|y_i|``. The
correction has the units of the internal variables whatever the units of the
residual, so the residual of a law may be written in any units: the equations
of a [`ResidualLaw`](@ref) in strain units and the stationarity condition of a
[`PotentialLaw`](@ref), in units of stress, converge alike. After `maxiters`
iterations without convergence, or at a singular Jacobian, a
[`LocalIntegrationError`](@ref) is thrown, which a driver catches to cut the
step.
"""
struct LocalNewton{T <: Real} <: AbstractLocalAlgorithm
    abstol::T
    reltol::T
    maxiters::Int
    function LocalNewton{T}(abstol, reltol, maxiters) where {T}
        abstol > 0 || throw(ArgumentError("LocalNewton: abstol must be positive; got $abstol"))
        reltol ≥ 0 || throw(ArgumentError("LocalNewton: reltol must be nonnegative; got $reltol"))
        maxiters ≥ 1 || throw(ArgumentError("LocalNewton: maxiters must be at least 1; got $maxiters"))
        return new{T}(abstol, reltol, maxiters)
    end
end

function LocalNewton(; abstol::Real = 1.0e-12, reltol::Real = 1.0e-12, maxiters::Integer = 50)
    T = float(promote_type(typeof(abstol), typeof(reltol)))
    return LocalNewton{T}(abstol, reltol, Int(maxiters))
end

"""
    LocalIntegrationError(law, iterations, correction, reason)

Thrown by [`point_update`](@ref) when the local equations of a law could not be
solved: the law's name, the iterations done, the last Newton correction (largest
entry) and the reason. A driver that catches it can retry the step in smaller
pieces, as [`point_response`](@ref) does; any other error is a bug and is not
caught.
"""
struct LocalIntegrationError <: Exception
    law::Symbol
    iterations::Int
    correction::Float64
    reason::String
end

Base.showerror(io::IO, e::LocalIntegrationError) = print(
    io, "LocalIntegrationError: law ", e.law, ": ", e.reason, " after ", e.iterations,
    " iteration(s), last Newton correction ", e.correction, ". Cut the step into smaller ones."
)

# ── One step ─────────────────────────────────────────────────────────────────

"""
    point_update(law, g, prev, Δt; algorithm = LocalNewton(), external = (;)) -> (F, 𝕂, state)

Integrate `law` over a step of length `Δt` that brings its gradients to `g`,
from the [`PointState`](@ref) `prev`. Returns the fluxes `F`, the consistent
tangent `𝕂 = dF/dg` and the new state.

`g` and `F` are flat `SVector`s laid out as [`gradient_layout`](@ref) and
[`flux_layout`](@ref) say: for a mechanical law, the strain and the stress as
[`Mandel`](@ref) vectors and `𝕂` the 6×6 tangent stiffness; for a coupled law,
the fields one after the other and `𝕂` with every coupling block.

This is the function every caller ends up in: [`material_response`](@ref) wraps
it for the TensND contract, [`point_response`](@ref) drives it along a loading
path, an element loop calls it at each quadrature point. Its inputs and outputs
are static arrays and it allocates nothing.

`external` gives, by name, the variables the law reads without returning their
flux at the end of the step: a temperature `(T = 350.0,)`, say. The law reads
them in `p`, beside its parameters.

The numbers may be `ForwardDiff.Dual`, in `g`, in `prev`, in `Δt`, in the
law's parameters or in `external`, nested or not: the fluxes, the tangent and
the state then carry the derivatives, exactly.
"""
function point_update(
        law::AbstractPointLaw, g::StaticVector, prev::PointState, Δt::Real;
        algorithm::AbstractLocalAlgorithm = LocalNewton(), external::NamedTuple = (;)
    )
    return _point_update(_with_external(law, external), _checked_length(SVector(g), law.gradients), prev, Δt, algorithm)
end

function _point_update(law, g, prev, Δt, alg)
    y, frozen = _solve_local(alg, law, g, prev, Δt)
    F, 𝕂 = _fluxes_and_tangent(law, y, g, prev, Δt, frozen)
    return F, 𝕂, PointState(y, g)
end

# The element type of a step: that of every number entering it. The residual and
# the fluxes are evaluated once to see it, which also catches parameters a law
# keeps outside `p` (a closure over a dual, say), so that their derivatives are
# not lost either.
function _step_type(law, g, prev, Δt, y)
    F = _fluxes(law, g, y)
    return float(promote_type(eltype(g), eltype(prev.y), eltype(prev.g), typeof(Δt), _residual_type(law, y, g, prev, Δt), eltype(F)))
end

# A law without internal variable has no residual to evaluate.
_residual_type(law, ::StaticVector{0}, g, prev, Δt) = Union{}
_residual_type(law, y, g, prev, Δt) = eltype(_residual(law, y, g, prev, Δt))

# The number type of the parameters: that of every number in `p`, however nested.
_numeric_type(x::Number) = typeof(x)
_numeric_type(x::AbstractArray{<:Number}) = eltype(x)
_numeric_type(x::Union{Tuple, NamedTuple}) = mapreduce(_numeric_type, promote_type, values(x); init = Union{})
_numeric_type(x) = Union{}

# Newton steps that bring an exact root of the values to exact derivatives:
# none for plain numbers, k for derivatives up to order 2ᵏ − 1.
_derivative_steps(::Type{T}) where {T} = (d = TensND.dual_depth(T); d == 0 ? 0 : ceil(Int, log2(d + 1)))

"""
    _solve_local(alg, law, g, prev, Δt) -> (y, frozen)

The internal variables at the end of the step, and whether the step left them
unchanged (then they do not depend on `g`).
"""
function _solve_local(alg::LocalNewton, law, g, prev::PointState{N}, Δt) where {N}
    # A law without internal variable has nothing to solve.
    N == 0 && return SVector{0, _step_type(law, g, prev, Δt, prev.y)}(), true
    # The elastic predictor and the iterations below compare values; a symbolic
    # number has none to compare, and no closed form replaces an iterative solve.
    # Checked before anything of the law is evaluated.
    S = promote_type(eltype(g), eltype(prev.y), eltype(prev.g), typeof(Δt), _numeric_type(law.p))
    TensND.is_hard_numeric(S) || throw(
        ArgumentError(
            "law $(law.name): the local equations are solved iteratively, which needs " *
                "numbers that can be compared; got $S. A law without internal variable " *
                "(PotentialLaw with no dissipation) accepts symbolic numbers."
        )
    )
    T = _step_type(law, g, prev, Δt, prev.y)
    _frozen(law, g, prev, Δt) && return SVector{N, T}(prev.y), true
    R(z) = _residual(law, z, g, prev, Δt)
    y = _newton_values(R, TensND.primal.(prev.y), alg, law.name)
    yT = SVector{N, T}(y)
    for _ in 1:_derivative_steps(T)
        yT = yT - ForwardDiff.jacobian(R, yT) \ R(yT)
    end
    return yT, false
end

# Newton on the values only: the iterate stays a vector of plain numbers, the
# residual and its Jacobian being read through `primal` when the law carries
# derivatives. Converged on the size of the correction, in the units of y.
function _newton_values(R, y, alg::LocalNewton, name::Symbol)
    last = Inf
    for it in 1:alg.maxiters
        r = TensND.primal.(R(y))
        J = TensND.primal.(ForwardDiff.jacobian(R, y))
        δ = J \ r
        all(isfinite, δ) || throw(LocalIntegrationError(name, it, last, "singular or non-finite Jacobian"))
        y = y - δ
        last = maximum(abs, δ)
        last <= alg.abstol + alg.reltol * maximum(abs, y) && return y
    end
    throw(LocalIntegrationError(name, alg.maxiters, last, "no convergence"))
end

# Fluxes and consistent tangent at the end of the step. When the step leaves the
# internal variables unchanged they do not depend on g; otherwise the
# implicit-function theorem gives their derivative from the residual.
function _fluxes_and_tangent(law, y, g, prev, Δt, frozen::Bool)
    F = _fluxes(law, g, y)
    ∂F∂g = ForwardDiff.jacobian(h -> _fluxes(law, h, y), g)
    frozen && return F, ∂F∂g
    ∂F∂y = ForwardDiff.jacobian(z -> _fluxes(law, g, z), y)
    ∂R∂y = ForwardDiff.jacobian(z -> _residual(law, z, g, prev, Δt), y)
    ∂R∂g = ForwardDiff.jacobian(h -> _residual(law, y, h, prev, Δt), g)
    return F, ∂F∂g - ∂F∂y * (∂R∂y \ ∂R∂g)
end
