# =============================================================================
#  MeanFieldHomogenizationNonlinearSolveExt.jl
#
#  Weak extension activated when `NonlinearSolve.jl` is loaded together
#  with `MeanFieldHomogenization` (`using NonlinearSolve`). It adds a
#  `MeanFieldHomogenization.Schemes._solve_sc` method dispatching on
#  `NonlinearSolve.jl`'s algorithm hierarchy, so any SC / ASC scheme can
#  be solved with a SciML nonlinear solver:
#
#      homogenize(rve, SelfConsistent(algorithm = NewtonRaphson()), :C)
#      homogenize(rve, SelfConsistent(algorithm = TrustRegion()), :C)
#
#  Both `SelfConsistent` and `AsymmetricSelfConsistent` (stiffness *and*
#  compliance branches) route through the single core dispatcher
#  `_solve_sc` (see `src/Schemes/self_consistent.jl`), so this one method
#  covers all of them.
#
#  ── ForwardDiff compatibility (the crux) ───────────────────────────────
#
#  `sensitivities.jl` differentiates `homogenize` by seeding a
#  `ForwardDiff.Dual` into the RVE (via `set_param`); that Dual then
#  flows into `x0`/`step`/the residual here — it is *baked into the
#  closure*, not passed through `NonlinearProblem`'s `p` field. This
#  means NonlinearSolve's own built-in ForwardDiff-over-solve machinery
#  (which lifts Duals found in `prob.p`) never triggers for us, and
#  naively calling `solve` on a Dual-eltype problem would let
#  NonlinearSolve seed *its own* internal Dual on top for the Jacobian —
#  nested Duals, tag-ordering-fragile, and not what its `p`-centric
#  lifting code was designed to handle.
#
#  Instead we implement our own implicit-function-theorem (IFT) lift:
#
#    1. Solve the *primal* problem — `NonlinearSolve` itself only ever
#       sees `Float64` (`u0`, and the residual/Jacobian callbacks are
#       `Float64 → Float64`) — with an **explicit** finite-difference
#       Jacobian passed to `NonlinearFunction`, which stops NonlinearSolve
#       from seeding any Dual of its own, uniformly for any algorithm
#       (`NewtonRaphson`, `TrustRegion`, …). Internally, every candidate
#       `p` NonlinearSolve proposes is *embedded* back into the fully
#       promoted Dual type (as a zero-partial Dual, via `embed`) before
#       `step` ever sees it, and only the resulting residual's *value* is
#       stripped back to `Float64` for NonlinearSolve. This matters
#       because `step`'s closure can carry `Dual`s beyond just the
#       iterating tensor itself (other phases, fractions, …); building a
#       genuinely-`Float64` running estimate while those stay `Dual`
#       would mix element types within the same tensor computation,
#       which is unsupported for some tensor symmetry classes (hit by
#       non-spherical / symmetrized geometries — see the `_solve_sc`
#       method body for the concrete failure this embedding avoids).
#    2. Recover the outer (user) partials with a single linear-algebra
#       correction `p* - Jf⁻¹·F_dual(p*)` (`F_dual` evaluated once at the
#       primal root, `Jf` the same Float64 Jacobian): by the implicit
#       function theorem, `dp*/dθ = -(∂F/∂p)⁻¹ ∂F/∂θ`, exactly what this
#       one correction reconstructs. Only the user's own Dual tag is ever
#       present — no nesting, first-order exact, and cheaper than
#       differentiating through the iterations (no repeated user-partial
#       propagation every iteration).
#
#  See `docs/src/tutorials/12_nonlinear_solvers.md` for a worked
#  derivation and cross-check against the built-in solvers.
# =============================================================================

module MeanFieldHomogenizationNonlinearSolveExt

using MeanFieldHomogenization
using NonlinearSolve
using TensND
using ForwardDiff

const S = MeanFieldHomogenization.Schemes

# `NonlinearSolve` re-exports `SciMLBase` (its trigger is `using
# NonlinearSolve`; `SciMLBase` itself is not a weak/strong dep of
# MeanFieldHomogenization, so we reach the type through the already-loaded
# `NonlinearSolve` module rather than a separate `import SciMLBase`).
const _AbstractNLSAlg = NonlinearSolve.SciMLBase.AbstractNonlinearAlgorithm

# ── Shared residual (mirrors `_solve_sc(::NewtonDefault, …)` exactly,
#    including the positive-definite guard on both the rebuilt input and
#    the step's output) ────────────────────────────────────────────────

function _nls_residual(step, x0, ε_pos)
    rebuild = p -> S._tens_from_param_vec(x0, p)
    return function residual_vec(p)
        x_in = S._sc_pd_guard_apply(rebuild(p), ε_pos)
        x_out = step(x_in)
        return S._tens_to_param_vec(x_out) .- S._tens_to_param_vec(x_in)
    end
end

# Central finite-difference Jacobian, `Float64` only. Used exclusively on
# the Dual/IFT branch below, where an *explicit* `jac` is what prevents
# NonlinearSolve from AD-seeding `p` itself.
#
# Step size: `cbrt(eps)`, the standard optimum for a *central* difference
# (balances O(h²) truncation against O(eps/h) round-off; the tighter
# `sqrt(eps)` step optimal for *forward* differences leaves this
# round-off-dominated and materially degrades the IFT-lifted derivative
# on badly-scaled problems, e.g. the canonical (α,β) = (3K,2μ) pair for
# a high-contrast porous solid where α ~ O(10⁶) and β ~ O(1)).
function _fd_jacobian(f, p::AbstractVector{<:Real})
    f0 = f(p)
    n, m = length(p), length(f0)
    J = zeros(Float64, m, n)
    for j in 1:n
        h = cbrt(eps(Float64)) * max(one(Float64), abs(p[j]))
        pp = copy(p)
        pp[j] += h
        pm = copy(p)
        pm[j] -= h
        J[:, j] .= (f(pp) .- f(pm)) ./ (2h)
    end
    return J
end

"""
    _solve_sc(algo::NonlinearSolve.AbstractNonlinearAlgorithm, step, x0; kw...)

SC / ASC fixed-point solver backed by `NonlinearSolve.jl`. `algo` is any
concrete algorithm (`NewtonRaphson()`, `TrustRegion()`,
`LevenbergMarquardt()`, …). The fixed point `x = step(x)` is recast as
the root problem `F(p) = 0` on the canonical symmetry components `p`
(same `_tens_to_param_vec` / `_tens_from_param_vec` bridge as
[`NewtonDefault`](@ref)), with the positive-definite guard applied
before every `step` evaluation.

`abstol` and `reltol` are measured in the **Frobenius norm of the tensor**, as
they are for the two built-in solvers — the weights of `S._sc_param_weights`
turn the canonical components into that norm, and they are passed as the
`internalnorm` of a `NormTerminationMode` rather than applied to the unknowns
themselves. That distinction matters: rescaling the unknowns would also rescale
the trust region, which is not invariant under it, and in the percolated regime
that moves where the algorithm stops. Without either, the same `abstol` would
express a stricter requirement in one symmetry class than in another. SciML's
test is `‖Δ‖ ≤ abstol` or `‖Δ‖ ≤ reltol · ‖Δ + u‖`, the `max` form of the
additive convention the built-in solvers use.

`abstol`, `reltol`, `maxiters` are forwarded to `NonlinearSolve.solve`.
`damping` and `select_best` are Picard-only concepts (relaxation /
best-iterate tracking) with no SciML equivalent and are accepted but
ignored here; `verbose` is not forwarded (v4 expects a
`NonlinearVerbosity`, not a `Bool`).

When `eltype(x0) <: ForwardDiff.Dual` (i.e. this call happens inside an
outer `derivative`/`gradient`/`jacobian`), the fixed point is solved by
an implicit-function-theorem lift instead of naively handing Dual
numbers to NonlinearSolve — see the module docstring for the full
rationale. This keeps `derivative(rve, SelfConsistent(algorithm = algo),
p)` exact and free of nested `ForwardDiff.Dual`s for any `algo`.
"""
function S._solve_sc(
        algo::_AbstractNLSAlg, step, x0::TensND.AbstractTens;
        abstol::Real = 1.0e-12, reltol::Real = 1.0e-8, maxiters::Int = 100,
        damping::Real = 0.0, verbose::Bool = false, select_best::Bool = false,
        kw...
    )
    ε_pos = S._sc_pd_eps(x0)
    residual_vec = _nls_residual(step, x0, ε_pos)
    rebuild = p -> S._tens_from_param_vec(x0, p)
    p0 = S._tens_to_param_vec(x0)
    # `abstol` has to mean the same thing here as it does for the two built-in
    # solvers, namely a bound on the Frobenius norm of the tensor. The weights
    # of `S._sc_param_weights` convert one into the other, and they are applied
    # to the *norm* rather than to the unknowns: rescaling the unknowns would
    # also rescale the trust region, which is not invariant under it, and in
    # the percolated regime that moves where the algorithm stops — onto the
    # plateau of the positive-definiteness guard, where the residual is large
    # and its Jacobian identically zero. Measuring differently is safe;
    # searching differently is not.
    w_nls, _isometric = S._sc_param_weights(x0)
    _wnorm(v) = sqrt(sum(abs2, w_nls .* v))
    # `NormTerminationMode` keeps BOTH tolerances — `‖Δ‖ ≤ abstol` or
    # `‖Δ‖ ≤ reltol · ‖Δ + u‖` — which is `max` where the package's own contract
    # writes `+`, i.e. the same test to within a factor of two. `AbsNorm…` would
    # have measured in the right norm but made `reltol` inert here, and silently
    # dropping a documented knob on one solver is exactly the kind of divergence
    # between algorithms this release exists to remove.
    term = NonlinearSolve.NormTerminationMode(_wnorm)
    # `eltype(p0)` alone does not always detect Dual-ness: when the
    # differentiated parameter lives on a phase other than the one `x0`
    # is built from (e.g. an *inclusion* modulus or a volume fraction,
    # while `x0 = matrix property`), `step` promotes to `Dual`
    # internally even though `p0` itself is plain `Float64`. Check the
    # residual's own eltype too (single evaluation, reused as the
    # iteration-0 residual on the real branch below via NonlinearSolve's
    # own first callback).
    r0 = residual_vec(p0)
    # `T_full` reflects the Dual-ness of the WHOLE computation, not just
    # `x0` on its own: `step`'s closure can carry `Dual`s on phases other
    # than the one `x0` is built from (an inclusion modulus, a volume
    # fraction, …), in which case `x0` itself stays plain `Float64` while
    # `residual_vec(p0)` already comes back `Dual`-valued. Promoting
    # against both catches every case.
    T_full = promote_type(eltype(p0), eltype(r0))
    is_dual = T_full <: ForwardDiff.Dual

    if !is_dual
        # ── Real branch: ordinary root-find, NonlinearSolve's default
        #    autodiff (ForwardDiff, single layer) computes its own
        #    Jacobian — correct and fast since no user Dual is present.
        prob = NonlinearProblem((p, _) -> residual_vec(p), collect(float.(p0)))
        sol = NonlinearSolve.solve(
            prob, algo; abstol = abstol, reltol = reltol, maxiters = maxiters,
            termination_condition = term
        )
        NonlinearSolve.SciMLBase.successful_retcode(sol) ||
            @debug "NonlinearSolve SC solver did not report success" retcode = sol.retcode
        return rebuild(sol.u)
    else
        # ── Dual branch: implicit-function-theorem lift (see module
        #    docstring). NonlinearSolve itself only ever sees `Float64`
        #    (`u0`/`Fval`/`Jfd` are `Float64 → Float64`), so it never
        #    seeds a `Dual` of its own. Crucially, every call into
        #    `step` — even during the "primal" solve — embeds its
        #    candidate `p` back into `T_full` (as a zero-partial `Dual`)
        #    via `embed` *before* rebuilding the tensor: constructing a
        #    genuinely-`Float64` running estimate while `step`'s closure
        #    still carries `Dual`s elsewhere (other phases, fractions, …)
        #    would mix element types across the same tensor computation,
        #    which is unsupported for some tensor symmetry classes
        #    (e.g. a `TensTI` dcontract requires matching types on both
        #    operands — hit by non-spherical / symmetrized geometries).
        #    Embedding keeps every `step` call type-homogeneous; only the
        #    residual *value* is stripped to `Float64` for the primal
        #    solve.
        embed(p) = convert.(T_full, p)
        u0 = Float64[ForwardDiff.value(x) for x in p0]
        Fval(p) = ForwardDiff.value.(residual_vec(embed(p)))
        Jfd(p) = _fd_jacobian(Fval, p)
        nf = NonlinearFunction((p, _) -> Fval(p); jac = (p, _) -> Jfd(p))
        prob = NonlinearProblem(nf, u0)
        sol = NonlinearSolve.solve(
            prob, algo; abstol = abstol, reltol = reltol, maxiters = maxiters,
            termination_condition = term
        )
        NonlinearSolve.SciMLBase.successful_retcode(sol) ||
            @debug "NonlinearSolve SC solver (primal/IFT) did not report success" retcode = sol.retcode
        pstar = sol.u
        r_dual = residual_vec(embed(pstar))  # single-layer Dual{UserTag}: value ≈ 0
        Jf = Jfd(pstar)                      # Float64 ∂F/∂p at the primal root
        pstar_dual = embed(pstar) .- (Jf \ r_dual)  # IFT correction: dp*/dθ = -Jf⁻¹ ∂F/∂θ
        return rebuild(pstar_dual)
    end
end

# ── Auto-resolving default, called by `_solve_sc(::AutoNonlinear, …)` ───────

"""
    default_solve_sc(step, x0; kw...)

Entry point called by `MeanFieldHomogenization.Schemes._solve_sc(::AutoNonlinear,
…)` once this extension is loaded. Uses `NonlinearSolve.TrustRegion()`
— a globalized algorithm, more robust than a plain Newton step near the
self-consistent bifurcation — through the exact same dispatch (and
ForwardDiff IFT lift) as any explicitly-chosen `NonlinearSolve`
algorithm.
"""
default_solve_sc(step, x0; kw...) = S._solve_sc(NonlinearSolve.TrustRegion(), step, x0; kw...)

end # module MeanFieldHomogenizationNonlinearSolveExt
