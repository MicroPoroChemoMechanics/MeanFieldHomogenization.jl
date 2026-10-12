# =============================================================================
#  counters.jl
#
#  Opt-in work counters used by the benchmark harness (`scripts/bench/`).
#
#  Rationale: measuring *time* and *allocations* alone cannot distinguish a
#  genuine speed-up from a change of adaptive-quadrature behavior (fewer
#  nodes evaluated ⇒ faster *and* less accurate).  These counters expose the
#  amount of **work** actually performed, so a benchmark diff can assert that
#  the node/iteration count is unchanged while the time drops.
#
#  Two classes of counter:
#
#    * **always-on** — placed only at O(1)…O(100) seams (`hill_tensor`,
#      `cod_tensor`, `_quadgk`, one polynomial solve per φ node, one SC
#      iteration, one layered recurrence).  An atomic increment against
#      such a seam is far below measurement noise, and keeps the counts exact
#      when the seams run on several threads.
#
#    * **opt-in** (`COUNT_INTEGRAND`) — the true per-node counter.  It sits in
#      the innermost loop, so it must never be a branch *inside* the
#      integrand.  Instead `_counted_quadgk` branches **once per `quadgk`
#      call**, each side passing a concretely typed callable; when counting is
#      off the call is a plain `quadgk` on the original function.
#
#  Counting is off by default and the harness enables it only during a
#  dedicated pass, separate from the timed pass.
# =============================================================================

"""
    HILL_CALLS, COD_CALLS, QUADGK_OUTER, INTEGRAND_EVALS,
    RESIDUE_SOLVES, SC_ITERATIONS, LAYER_RECURRENCES

Work counters incremented by the instrumented seams.  Read them through
[`read_counters`](@ref) and zero them with [`reset_counters!`](@ref).

The docstring is attached to `HILL_CALLS`, the six others carry none: a
reference to one of them must stay a plain code span.
"""
const HILL_CALLS = Threads.Atomic{Int}(0)
const COD_CALLS = Threads.Atomic{Int}(0)
const QUADGK_OUTER = Threads.Atomic{Int}(0)
const INTEGRAND_EVALS = Threads.Atomic{Int}(0)
const RESIDUE_SOLVES = Threads.Atomic{Int}(0)
const SC_ITERATIONS = Threads.Atomic{Int}(0)
const LAYER_RECURRENCES = Threads.Atomic{Int}(0)

"""
    COUNT_INTEGRAND

When `true`, `_maybe_count` wraps quadrature integrands so that every
evaluation bumps `INTEGRAND_EVALS`.  **Off by default** — the wrapper
lives in the innermost loop and must not be active during a timed run.

(A plain code span rather than a cross-reference: `_maybe_count` has no
definition in the package any more, so an `@ref` to it would fail the moment
this docstring were listed on an API page.)
"""
const COUNT_INTEGRAND = Ref(false)

const _ALL_COUNTERS = (
    HILL_CALLS, COD_CALLS, QUADGK_OUTER, INTEGRAND_EVALS,
    RESIDUE_SOLVES, SC_ITERATIONS, LAYER_RECURRENCES,
)

@inline _bump!(r::Threads.Atomic{Int}) = (Threads.atomic_add!(r, 1); nothing)

"""
    reset_counters!()

Zero every work counter.  Does **not** change [`COUNT_INTEGRAND`](@ref).
"""
function reset_counters!()
    for r in _ALL_COUNTERS
        r[] = 0
    end
    return nothing
end

"""
    read_counters() -> NamedTuple

Current value of every work counter.
"""
read_counters() = (;
    hill_calls = HILL_CALLS[],
    cod_calls = COD_CALLS[],
    quadgk_outer = QUADGK_OUTER[],
    integrand_evals = INTEGRAND_EVALS[],
    residue_solves = RESIDUE_SOLVES[],
    sc_iterations = SC_ITERATIONS[],
    layer_recurrences = LAYER_RECURRENCES[],
)

"""
    _CountingFn(f)

Callable wrapper bumping `INTEGRAND_EVALS` on each call before
forwarding to `f`.

A `struct` rather than a closure on purpose: a closure returned from a
branch would make the branch's return type a small `Union`, and passing a
non-concrete callable to `quadgk` costs a dynamic dispatch **per node** —
which would perturb precisely the hot loop the counters exist to measure.
"""
struct _CountingFn{F}
    f::F
end

@inline (c::_CountingFn)(x) = (_bump!(INTEGRAND_EVALS); c.f(x))

"""
    _quadgk_norm(x) -> Real

The error norm every cubature of the package hands to QuadGK: `LinearAlgebra.norm`
of the values, a plain number.

`norm` itself, applied to `ForwardDiff.Dual` numbers, rescales by the largest
entry and divides by it unless that entry `iszero` — and a dual of value zero
with non-zero partials is not `iszero`. On a segment where the Kronrod and Gauss
rules agree to the last bit, which a smooth integrand produces, the error
estimate came out `NaN` and QuadGK stopped with a `DomainError`: the derivative
of the anisotropic conduction Hill tensor of a sphere, for one. Taken on the
values, the norm is the one a plain evaluation computes, so the adaptive
refinement of a differentiated evaluation makes the same decisions as the plain
one, and its value is that evaluation itself; the derivatives ride on the same
nodes.
"""
@inline _quadgk_norm(x) = LinearAlgebra.norm(x)
@inline _quadgk_norm(x::ForwardDiff.Dual) = LinearAlgebra.norm(_primal(x))
@inline _quadgk_norm(x::AbstractArray{<:ForwardDiff.Dual}) = LinearAlgebra.norm(_primal.(x))

"""
    _counted_quadgk(f, a, b; atol, rtol, maxevals) -> (value, error)

`QuadGK.quadgk` with opt-in per-node counting, and the error norm
[`_quadgk_norm`](@ref).

The [`COUNT_INTEGRAND`](@ref) branch is taken **once per `quadgk` call**, and
each side passes a *concretely typed* callable, so the integrand loop is
compiled exactly as it would be without instrumentation.  When counting is
off (always, during a timed run) this is a plain `quadgk` call on the
original function.
"""
@inline function _counted_quadgk(f::F, a, b; kw...) where {F}
    COUNT_INTEGRAND[] || return QuadGK.quadgk(f, a, b; norm = _quadgk_norm, kw...)
    return QuadGK.quadgk(_CountingFn(f), a, b; norm = _quadgk_norm, kw...)
end

"""
    with_integrand_counting(f)

Run `f()` with [`COUNT_INTEGRAND`](@ref) temporarily enabled, restoring the
previous value afterwards (even on error).
"""
function with_integrand_counting(f)
    old = COUNT_INTEGRAND[]
    COUNT_INTEGRAND[] = true
    try
        return f()
    finally
        COUNT_INTEGRAND[] = old
    end
end
