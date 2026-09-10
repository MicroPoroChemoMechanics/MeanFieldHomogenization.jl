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
#      iteration, one layered recurrence).  A `Ref{Int}` increment against
#      such a seam is far below measurement noise.
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
"""
# The signature above lists all seven, but Julia attaches a docstring to the
# *next expression only*, so this one documents `HILL_CALLS` and the other six
# carry none.  They are internal and unexported, so nothing renders them and
# nothing breaks; a reference to one of them from another docstring, however,
# has nothing to resolve to and must stay a plain code span.
const HILL_CALLS = Ref(0)
const COD_CALLS = Ref(0)
const QUADGK_OUTER = Ref(0)
const INTEGRAND_EVALS = Ref(0)
const RESIDUE_SOLVES = Ref(0)
const SC_ITERATIONS = Ref(0)
const LAYER_RECURRENCES = Ref(0)

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

@inline _bump!(r::Base.RefValue{Int}) = (r[] += 1; nothing)

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

@inline (c::_CountingFn)(x) = (INTEGRAND_EVALS[] += 1; c.f(x))

"""
    _counted_quadgk(f, a, b; atol, rtol, maxevals) -> (value, error)

`QuadGK.quadgk` with opt-in per-node counting.

The [`COUNT_INTEGRAND`](@ref) branch is taken **once per `quadgk` call**, and
each side passes a *concretely typed* callable, so the integrand loop is
compiled exactly as it would be without instrumentation.  When counting is
off (always, during a timed run) this is a plain `quadgk` call on the
original function.
"""
@inline function _counted_quadgk(f::F, a, b; kw...) where {F}
    COUNT_INTEGRAND[] || return QuadGK.quadgk(f, a, b; kw...)
    return QuadGK.quadgk(_CountingFn(f), a, b; kw...)
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
