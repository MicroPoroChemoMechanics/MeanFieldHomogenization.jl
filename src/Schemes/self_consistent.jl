# =============================================================================
#  self_consistent.jl — SelfConsistent + AsymmetricSelfConsistent.
#
#  Iterates `C^{n+1} = step(C^n)` where `step` is a Mori-Tanaka-like
#  evaluation that uses C^n as the reference matrix. The dispatcher
#  `_solve_sc` picks the non-linear solver:
#
#   * `AndersonDefault`            — built-in damped Picard fixed point.
#                                    Pure Julia, Dual-safe. Default.
#   * `NewtonDefault`              — built-in Newton-Raphson with a
#                                    ForwardDiff Jacobian on the canonical
#                                    symmetry components. Dependency-free;
#                                    ships with the package (no
#                                    NonlinearSolve.jl needed).
#   * `AutoNonlinear`              — auto-resolving marker: dispatches to
#                                    a globalized `NonlinearSolve.jl`
#                                    algorithm (`TrustRegion`) when the
#                                    weak extension
#                                    `MeanFieldHomogenizationNonlinearSolveExt` is
#                                    loaded, else falls back to
#                                    `NewtonDefault`. Not the default of
#                                    `SelfConsistent` — opt in explicitly.
#   * any algorithm from
#     `NonlinearSolve.jl`          — handled by the weak extension
#                                    (`MeanFieldHomogenizationNonlinearSolveExt`),
#                                    ForwardDiff-safe via an
#                                    implicit-function-theorem lift (see
#                                    the extension source for details).
#
#  A future native Anderson with memory > 1 will replace the current
#  `AndersonDefault` (currently Picard with relaxation, equivalent to
#  Anderson with memory = 1).
# =============================================================================

# ── Public _evaluate ────────────────────────────────────────────────────────

"""
    _evaluate(rve, sc::SelfConsistent, ::Val{p}; kw...) -> AbstractTens

Self-consistent scheme for property `:p`
[hill1965composites, budiansky1965](@cite). Iterates

```math
\\mathbb{C}^{(n+1)} = \\Big(\\sum_i f_i\\,\\mathbb{C}_i : \\mathbb{A}_i^{\\mathrm{dil}}(\\mathbb{C}^{(n)})\\Big)
                     :\\Big(\\sum_i f_i\\,\\mathbb{A}_i^{\\mathrm{dil}}(\\mathbb{C}^{(n)})\\Big)^{-1}
```

with the dilute concentration tensor evaluated against the current
estimate ``\\mathbb{C}^{(n)}`` itself (rather than the matrix property).

The solver algorithm is selected by `sc.algorithm`; convergence kwargs
in `sc.options` (`abstol`, `reltol`, `maxiters`, `damping`, `verbose`,
`select_best`) override their defaults. The stopping test is the additive
SciML convention ``\\|\\Delta x\\| \\le \\mathrm{abstol} + \\mathrm{reltol}\\,\\|x\\|``, so `reltol` (default `1e-8`)
is what binds on a stiffness of physical magnitude — see [`_solve_sc`](@ref)
for the full contract. External algorithms from `NonlinearSolve.jl` are
supported via the weak extension `MeanFieldHomogenizationNonlinearSolveExt`.

Crack families enter through their compliance contribution in the current
estimate: each adds ``\\kappa_c\\,\\varepsilon_c\\,\\mathbb{H}_c(\\mathbb{C}^{(n)}):\\mathbb{C}^{(n)}``
to the strain average (the limit of ``f_c\\,\\mathbb{A}_c``) and nothing to the stress
average, a traction-free crack carrying no stress, with ``\\kappa_c = 4\\pi/3`` for an
elliptic family and ``\\pi`` for a ribbon — see `_sc_crack_total`.
"""
function _evaluate(rve::RVE, sc::SelfConsistent, ::Val{p}; kw...) where {p}
    # Symmetric Hill / Budiansky SC iteration on the ECHOES `B · A^{-1}`
    # body : crack phases contribute their compliance contribution
    # `H_c(C_n)` to the denominator A_avg via `_phase_compliance_contribution`,
    # and their stiffness contribution `ΔC_c(C_n)` to the numerator
    # CA_avg (traction-free → no stress contribution from solid side).
    # The eigenvalue guard `_sc_pd_guard` prevents the iteration
    # from collapsing to the trivial percolated fixed point at moderate
    # density.
    P_init = _sc_initial(sc.init, rve, p; kw...)
    solver_kw, step_kw = _split_sc_kwargs(kw)
    step = C -> _sc_step(rve, C, p; step_kw...)
    return _solve_sc(sc.algorithm, step, P_init; sc.options..., solver_kw...)
end

# The Voigt average is infinite as soon as one phase is rigid or incompressible
# (an `iso_stiffness(Inf, μ)` layer, say), and an infinite seed makes the first
# step fail deep inside a linear solve with "matrix contains Infs or NaNs".
# Say what happened, and what to pass instead.
function _voigt_seed(rve::RVE, prop::Symbol; kw...)
    C = _evaluate(rve, Voigt(), Val(prop); kw...)
    a = TensND.get_array(C)
    if is_hard_numeric(eltype(a)) && !all(isfinite, a)
        throw(
            ArgumentError(
                "the self-consistent iteration starts from the Voigt average of " *
                    "the phases, which is infinite here: a phase is rigid or " *
                    "incompressible. Pass a finite starting point: " *
                    "`SelfConsistent(init = <tensor>)`, or the name of a phase " *
                    "with a finite property, as Echoes starts from its matrix."
            )
        )
    end
    return C
end

"""
    _sc_initial(init, rve, prop; kw...) -> AbstractTens

Starting iterate of the self-consistent fixed point.

The self-consistent morphology distinguishes no phase, so the seed cannot come
from "the matrix" the way it did while an RVE was required to have one. It is
stated on the scheme instead:

- `:voigt` — the Voigt average of the phases. Always defined, positive definite
  by construction, and it needs no phase to be singled out, which is what makes
  a matrix-free RVE solvable at all.
- `:reuss` — the Reuss average.
- a phase name — that phase's property: the start from the matrix, as Echoes
  makes it.
- a tensor — an explicit guess, the natural form when restarting from the
  previous step of a constitutive loop.

A phase named `:voigt` or `:reuss` wins over the keyword: a phase name is the
caller's data, a keyword is ours.
"""
function _sc_initial(init::Symbol, rve::RVE, prop::Symbol; kw...)
    haskey(rve.phases, init) && return phase_property(rve, init, prop)
    init === :voigt && return _voigt_seed(rve, prop; kw...)
    init === :reuss && return _evaluate(rve, Reuss(), Val(prop); kw...)
    throw(
        ArgumentError(
            "init = :$(init) names neither a phase of this RVE " *
                "($(rve.phase_names)) nor one of :voigt, :reuss"
        )
    )
end
_sc_initial(init::TensND.AbstractTens, ::RVE, ::Symbol; kw...) = init
# The default, and why it is what it is.
#
# `:voigt` is the seed that needs no phase to be singled out, and it is what
# makes a matrix-free RVE solvable at all — so it is the fallback. It is not the
# blanket default, because measuring it against the historical seed (the phase
# taking up the complement, which is what the matrix used to be) showed no
# advantage and one drawback: across a porous RVE at f ∈ {0.1, 0.3, 0.45, 0.6}
# and a stiff one at f ∈ {0.1, 0.3, 0.5, 0.7} the two agree to 1e-9 or better,
# and on a deeply percolated oblate configuration the Voigt start makes
# `TrustRegion` stop where its Jacobian is exactly
# singular, which the phase seed does not. Equivalent where it matters, worse in
# one corner: not a default worth taking.
function _sc_initial(::Nothing, rve::RVE, prop::Symbol; kw...)
    r = remainder_phase_name(rve)
    r === nothing && return _voigt_seed(rve, prop; kw...)
    return phase_property(rve, r, prop)
end

# Solver-only kwargs are intercepted at this level and never forwarded
# to the underlying `_sc_step` (which would otherwise leak them down to
# `hill_tensor` and trigger a `MethodError` on the unknown kwarg).
const _SC_SOLVER_KWARGS = (
    :abstol, :reltol, :maxiters, :damping,
    :verbose, :select_best,
)

function _split_sc_kwargs(kw)
    solver = Dict{Symbol, Any}()
    step = Dict{Symbol, Any}()
    for (k, v) in kw
        if k in _SC_SOLVER_KWARGS
            solver[k] = v
        else
            step[k] = v
        end
    end
    return pairs(NamedTuple(solver)), pairs(NamedTuple(step))
end

# ── SC step (one Mori-Tanaka-like iterate against current estimate) ─────────

function _sc_step(rve::RVE, C_n, prop::Symbol; kw...)
    return _sc_step_dispatch(rve, C_n, prop; kw...)
end

# ── The two accumulator loops of the SC body, factored out ───────────────────
#
# They are shared by the 4th- and 2nd-order `_sc_step_dispatch` methods above
# (only the contraction operator differs, and that stays in the caller) *and*
# by `crack_family_compliances` in `crack_families.jl`, which has to reproduce
# the very same averages at the converged estimate to expose the per-family
# compliance contributions. Keeping one definition is what stops the two from
# drifting apart: a change to the SC body is automatically reflected in the
# per-family decomposition, and the identity test in
# `test/Schemes/test_crack_families.jl` fails loudly if it is not.

"""
    _sc_solid_averages(rve, P_n, prop; kw...) -> (A_avg, CA_avg)

Volume-weighted dilute-concentration and stress-average accumulators over every
phase that carries volume, evaluated in the reference medium `P_n`:

```math
\\langle\\mathbb{A}\\rangle = \\sum_i f_i\\,\\langle\\mathbb{A}_i(\\mathbb{P}_n)\\rangle,
\\qquad
\\langle\\mathbb{C}:\\mathbb{A}\\rangle = \\sum_i f_i\\,\\langle\\mathbb{C}_i:\\mathbb{A}_i(\\mathbb{P}_n)\\rangle
```

``f_i`` is the *resolved* fraction: the RVE's fraction closure has already turned
a [`Remainder`](@ref) into ``1 - \\sum f`` and, under [`RescaledFractions`](@ref),
renormalized the declared ones. No phase is distinguished — which is the whole
point of the self-consistent morphology, and what lets it run on an RVE that
designates no matrix at all.

`CrackDensity` phases are skipped: their strain concentration is singular and
they are collected by [`_sc_crack_total`](@ref) instead.
"""
function _sc_solid_averages(
        rve::RVE, P_n::TensND.AbstractTens, prop::Symbol; kw...
    )
    A_avg = zero(P_n)
    CA_avg = zero(P_n)
    for name in _volume_phase_names(rve)
        f = volume_fraction(rve, name)
        A_dil, CA = _phase_dilute_and_stress_average(rve, name, prop, P_n; kw...)
        A_avg += f * A_dil
        CA_avg += f * CA
    end
    return A_avg, CA_avg
end

"""
    _major_symmetrize(P) -> AbstractTens

Project a property estimate onto the **major-symmetric** tensors:
``P_{ijkl} \\mapsto (P_{ijkl} + P_{klij})/2`` at 4th order, and
``P_{ij} \\mapsto (P_{ij} + P_{ji})/2`` at 2nd order.

An effective stiffness (or conductivity) derives from an energy, so it *is*
major-symmetric. The self-consistent body, however, assembles it as
``\\mathbb{B}:\\mathbb{A}^{-1}`` — a product of two tensors that do not commute — and that product
is only major-symmetric when the phases happen to share a common frame. With
non-coaxial crack families it is not: the asymmetry starts at roundoff and is
then **amplified by the fixed-point iteration** (measured: ``3\\times10^{-15}`` at the second
iterate, ``2\\times10^{-4}`` at the third). A stiffness that is not major-symmetric is not a
valid Eshelby reference medium, and the anisotropic crack cubature returns a
`NaN` integrand on it, which used to abort the whole solve with a `DomainError`.

Projecting at every step keeps the iteration inside the admissible set. It is a
no-op up to roundoff whenever the product is already symmetric, and it matches
the reference ECHOES implementation, whose self-consistent estimate comes out
exactly major-symmetric.

It is TensND's `Tensors.majorsymmetric` (order 4) and `Tensors.symmetric`
(order 2), which keep the symmetry class of a structured estimate:
`TensISO`, `TensCubic`, `TensOrtho` and the five-parameter Walpole
`TensTI{4,T,5}` are returned as they are, a six- or eight-parameter `TensTI` as
its five-parameter form, a three-parameter `TensTI{2}` without its skew part.
That matters: the Newton parameterization (`_sc_newton_seed`) reads the
structured components through `get_data`, and the symmetry-class dispatch keys
on the concrete type. Only the genuinely unstructured estimates — which is
exactly what a mix of non-coaxial families produces — are rebuilt, in their own
basis.
"""
_major_symmetrize(P::TensND.AbstractTens{4}) = Tensors.majorsymmetric(P)
_major_symmetrize(P::TensND.AbstractTens{2}) = Tensors.symmetric(P)

"""
    _sc_crack_total(rve, P_n, prop; kw...) -> (H_total, has_cracks)

Sum of the *density-scaled* compliance (resistivity) contributions of every
[`CrackDensity`](@ref) phase, evaluated in the reference medium `P_n`:

```math
\\mathbb{H}^{\\mathrm{tot}} = \\sum_i \\kappa_i\\,\\varepsilon_i\\,\\mathbb{H}_i(\\mathbb{P}_n) ,
```

with ``\\kappa_i = 4\\pi/3`` for an elliptic family and ``\\pi`` for a ribbon family
(`crack_density_factor`).

`has_cracks` reports whether the sum has any term at all, which is what selects
the crack branch of the SC body.
"""
function _sc_crack_total(
        rve::RVE, P_n::TensND.AbstractTens, prop::Symbol; kw...
    )
    H_total = zero(P_n)
    has_cracks = false
    for name in _crack_phase_names(rve)
        H_total += _phase_compliance_contribution(rve, name, prop, P_n; kw...)
        has_cracks = true
    end
    return H_total, has_cracks
end

# 4th-order — symmetric (Hill 1965 / Budiansky 1965) self-consistent
# iteration : all phases (matrix included) carry a non-trivial dilute strain
# concentration A_α = inv(I + P(C_α - C_n)) computed in the iterating
# effective medium C_n. This gives the textbook SC fixed point with
# Hashin-Shtrikman lower-bound percolation behavior for porous media.
function _sc_step_dispatch(
        rve::RVE, C_n::TensND.AbstractTens{4, 3}, prop::Symbol;
        kw...
    )
    # Self-consistent body :
    #   strain_Stress_α  = A_α(C_n) · S_n   (solid)
    #   strain_Stress_c  = sym(H_c(C_n))    (void crack — NO trailing S_n!)
    #   stress_Stress_α  = C_α · strain_Stress_α
    #   stress_Stress_c  = 0                 (traction-free)
    # Accumulators :  A_E = Σ f_α·sym(strain_Stress_α)
    #                 B_E = Σ f_α·sym(stress_Stress_α)
    # Result  : C_eff = sym(B_E · A_E^{-vol}).
    # The trailing `S_n` factor cancels between A_E and B_E for solid
    # phases — but NOT for cracks, whose `strain_Stress` is the bare
    # compliance contribution `H_c`.  This breaks the cancellation and
    # gives a different fixed point than the textbook
    # `(Σ f·C·A)·(Σ f·A)^{-1}` SC body when cracks are present.
    A_avg, CA_avg = _sc_solid_averages(rve, C_n, prop; kw...)
    H_total, has_cracks = _sc_crack_total(rve, C_n, prop; kw...)
    if has_cracks
        S_n = inv(C_n)
        A_E = (A_avg ⊡ S_n) + H_total
        B_E = CA_avg ⊡ S_n
        return _major_symmetrize(B_E ⊡ inv(A_E))
    else
        return _major_symmetrize(CA_avg ⊡ inv(A_avg))
    end
end

# 2nd-order — same symmetric SC structure for conductivity / diffusion.
function _sc_step_dispatch(
        rve::RVE, K_n::TensND.AbstractTens{2, 3}, prop::Symbol;
        kw...
    )
    # Conduction analog of the 4th-order ECHOES SC body :
    # solids have `gradient_Flux = A · R_n` (R_n = inv(K_n) — resistivity),
    # cracks contribute the bare resistivity contribution `R_c`.
    A_avg, KA_avg = _sc_solid_averages(rve, K_n, prop; kw...)
    R_total, has_cracks = _sc_crack_total(rve, K_n, prop; kw...)
    if has_cracks
        R_n = inv(K_n)
        A_E = (A_avg ⋅ R_n) + R_total
        B_E = KA_avg ⋅ R_n
        return _major_symmetrize(B_E ⋅ inv(A_E))
    else
        return _major_symmetrize(KA_avg ⋅ inv(A_avg))
    end
end

# ── Built-in solvers ────────────────────────────────────────────────────────

"""
    _solve_sc(algo, step, x0; abstol, reltol, maxiters, damping, verbose,
              select_best, kw...) -> AbstractTens

Generic solver dispatcher for SC fixed points. Built-in:

- [`AndersonDefault`](@ref) — Picard with relaxation
  (``x_{n+1} = (1-\\mathrm{damping})\\,\\mathrm{step}(x_n) + \\mathrm{damping}\\,x_n``). `damping = 0.0`
  default; raise to about ``0.5`` for high-contrast iterations that overshoot.
  Convergence near a bifurcation (e.g. SC at the porous-percolation
  threshold) is intrinsically slow because the Picard Jacobian
  eigenvalue approaches 1 there; in that regime, set
  `select_best = true` to return the best iterate observed during the
  loop, or load `NonlinearSolve.jl` and switch to Newton/Anderson via
  the `algorithm` keyword.
- [`NewtonDefault`](@ref) — built-in Newton-Raphson with a ForwardDiff
  Jacobian on the canonical symmetry components and an Armijo line
  search. Dependency-free; ships with the package.
- [`AutoNonlinear`](@ref) — auto-resolving marker: uses a globalized
  `NonlinearSolve.jl` algorithm (`TrustRegion`) when the weak extension
  `MeanFieldHomogenizationNonlinearSolveExt` is loaded (`using NonlinearSolve`),
  else falls back to `NewtonDefault`.

Any algorithm from `NonlinearSolve.jl` (`NewtonRaphson()`,
`TrustRegion()`, `LevenbergMarquardt()`, …) can also be passed directly
as `algorithm`; it is handled by the same weak extension, through a
ForwardDiff-safe implicit-function-theorem lift so that
`derivative`/`gradient`/`jacobian` (see `sensitivities.jl`) work
transparently regardless of which solver is selected.

Convergence is declared when ``\\|x_{\\mathrm{new}} - x_{\\mathrm{old}}\\| \\le \\mathrm{abstol} + \\mathrm{reltol}\\,\\|x_{\\mathrm{old}}\\|``
(absolute *and* relative tolerance, additive convention; pass
`abstol = 0` to require purely relative convergence). Default values:
`abstol = 1e-12`, `reltol = 1e-8`.

``\\|\\cdot\\|`` is the **Frobenius norm of the tensor**, for every solver, so that a
given `abstol` expresses one requirement regardless of `algorithm` and of the
symmetry class the fixed point lives in. The Picard loop measures it directly;
the Newton path reaches it by working in the isometric parametrization of
[`_sc_param_weights`](@ref); the `NonlinearSolve` path passes those same
weights to SciML as the `internalnorm` of its termination condition, leaving
the unknowns alone — a trust region is not invariant under a rescaling of
them, and moving its metric moves where it stops.

Because a stiffness carries a physical magnitude, ``\\mathrm{reltol}\\,\\|x\\|`` dominates the
sum at the defaults above: tightening `abstol` alone does not tighten the
iteration. Set both, or `abstol = 0`, when a converged value is read off
rather than plotted.

When `select_best = true`, the solver tracks the best iterate seen
during the loop (smallest residual on the value field) and returns it
at the end. Useful for high-contrast iterations where Picard
oscillates around the fixed point: the *last* iterate may be worse
than an earlier one. Default is `false` (return last iterate).

Non-convergence is reported via `@debug` (silent by default; set
`JULIA_DEBUG=MeanFieldHomogenization` to surface it) rather than `@warn`. Near
bifurcation points the Picard step intrinsically slows down (the
linearized step has a Jacobian eigenvalue close to ``1``) and the residual stalls
above `tol_eff` while still being negligibly small compared to the
matrix-property scale; the returned iterate is informative even when
the strict tolerance is not reached, so a default warning would be
noise.
"""
function _solve_sc(
        ::AndersonDefault, step, x0::TensND.AbstractTens;
        abstol::Real = 1.0e-12, reltol::Real = 1.0e-8,
        maxiters::Int = 100,
        damping::Real = 0.0, verbose::Bool = false,
        select_best::Bool = false,
        kw...
    )
    x = x0
    last_resid = _sc_residual_zero(x0)
    x_best = x0
    resid_best_val = typemax(_value_eltype(x0))
    ε_pos = _sc_pd_eps(x0)
    for k in 1:maxiters
        MFH_Core._bump!(MFH_Core.SC_ITERATIONS)
        x = _sc_pd_guard_apply(x, ε_pos)
        x_new = step(x)
        last_resid = _sc_residual_norm(x_new, x)
        norm_x = _sc_residual_norm(x, zero(x))
        tol_eff = abstol + reltol * _scalar_value(norm_x)
        verbose && @info "SC iter $k : ‖Δ‖ = $last_resid   tol = $tol_eff"
        if select_best
            v = _scalar_value(last_resid)
            if v < resid_best_val
                resid_best_val = v
                x_best = x_new
            end
        end
        _sc_converged(last_resid, tol_eff) && _sc_partials_converged(x_new, x; abstol, reltol) && return x_new
        x = (one(real(eltype(x))) - damping) * x_new + damping * x
    end
    # Non-convergence is reported as a `@debug` message rather than a
    # `@warn` so it stays out of normal output. Set
    # `JULIA_DEBUG=MeanFieldHomogenization` (or pass `verbose = true`) to surface
    # the diagnostics. Near bifurcation points (porous SC at
    # percolation, …) the Picard step intrinsically slows down and
    # `last_resid` may stall above the requested tolerance while
    # remaining negligible compared to the matrix-property scale; the
    # returned iterate is still informative.
    @debug "SC (AndersonDefault/Picard) did not reach tolerance" maxiters last_resid abstol reltol
    return select_best ? x_best : x
end

"""
    _solve_sc(::NewtonDefault, step, x0::AbstractTens; …) -> AbstractTens

Built-in Newton-Raphson SC solver, parameterizing the iterating
estimate by its symmetry-class **canonical components**
(`TensND.get_data` → `(α, β)` for iso, `(ℓ₁, …, ℓ₆)` for TI / Walpole,
9 components for ortho).  At each Newton step:

1. Build the residual ``\\boldsymbol{F}(p) = \\mathrm{canonical}(\\mathrm{step}(\\mathrm{rebuild}(p))) - p``,
2. Compute the Jacobian ``\\mathbf{J} = \\partial\\boldsymbol{F}/\\partial p`` via `ForwardDiff.jacobian`,
3. Take the Newton step ``\\Delta p = -\\mathbf{J}^{-1}\\boldsymbol{F}(p)`` with backtracking line
   search (Armijo with shrinking factor 1/2, minimum step 1e-6).
4. Fall back to a single Picard step when the line search fails.

Compared to the SciML weak-extension path, this is dependency-free and
specialized to the small parameter spaces of `MeanFieldHomogenization` symmetry
classes (at most ``21`` components for the most general aniso 4-tensor); the
Jacobian is computed once per iteration through the same `step`
function the AndersonDefault loop calls.
"""
function _solve_sc(
        ::NewtonDefault, step, x0::TensND.AbstractTens;
        abstol::Real = 1.0e-12, reltol::Real = 1.0e-8,
        maxiters::Int = 50,
        damping::Real = 0.0, verbose::Bool = false,
        select_best::Bool = false,
        kw...
    )
    # The parameter space is the symmetry class of the ITERATE, and that class
    # is not always the one of `x0`: `x0` defaults to a phase property (often
    # `TensISO`, 2 components) while one application of the scheme can land in
    # a richer class — e.g. `TISymmetrize` phases make `step` return a
    # `TensTI{4,T,8}` (8 components).  Parameterizing on `x0` then made the
    # residual subtract an 8-vector from a 2-vector (`DimensionMismatch`), so
    # `NewtonDefault` could not solve any problem whose fixed point is richer
    # than its starting guess — while `AndersonDefault` coped, because Picard
    # simply propagates whatever `step` returns.
    #
    # Seed the parametrization from `step(x0)` instead: one application
    # reaches the class the fixed point lives in (the scheme's symmetrization
    # is idempotent), and the starting point is then a strictly better iterate
    # than `x0` at no extra cost.
    x_seed = _sc_newton_seed(x0, step)
    # Work in the *isometric* parametrization, so that `‖F‖` and `‖p‖` below
    # are the Frobenius norms of the tensors they stand for — the same quantity
    # the Picard loop tests, rather than an unweighted sum over canonical
    # components that means something different in every symmetry class.
    w = _sc_param_weights(x_seed)
    p0 = w .* TensND.to_vector(x_seed)
    L = length(p0)
    rebuild = p -> TensND.from_vector(x_seed, p ./ w)
    ε_pos = _sc_pd_eps(x_seed)
    residual_vec = function (p)
        x_in = _sc_pd_guard_apply(rebuild(p), ε_pos)
        x_out = step(x_in)
        return w .* (TensND.to_vector(x_out) .- TensND.to_vector(x_in))
    end
    # The class basis is orthogonal, so the Euclidean norm of the scaled vector
    # *is* the Frobenius norm of the tensor it encodes.
    # `eltype(p0)` alone is not always enough to know whether `Dual`s are
    # in play: when the differentiated parameter lives on a phase other
    # than the one `x0` is built from (e.g. differentiating w.r.t. an
    # *inclusion* modulus or a volume fraction while `x0 = matrix
    # property`), `step` promotes to `Dual` internally even though `p0`
    # itself is plain `Float64`. Promote `Tref` against the residual's
    # own eltype too — evaluated once, no cost beyond the iteration the
    # loop would run anyway.
    # One residual evaluation is a FULL RVE pass — one `hill_tensor` per phase,
    # the dominant cost of the whole solve.  `r0` is therefore computed once
    # and reused for two purposes: fixing `Tref` (the residual's own eltype
    # matters, because `step` can promote to `Dual` internally even when `p0`
    # is plain `Float64` — e.g. differentiating w.r.t. an inclusion modulus
    # while `x0` is the matrix property), and seeding the loop.  Previously the
    # value was computed for `Tref` and then thrown away.
    r0 = residual_vec(p0)
    Tref = float(promote_type(eltype(p0), eltype(r0)))
    p = collect(Tref, p0)
    r = collect(Tref, r0)
    p_best = copy(p); resid_best = Inf
    for iter in 1:maxiters
        MFH_Core._bump!(MFH_Core.SC_ITERATIONS)
        norm_r = LinearAlgebra.norm(r)
        norm_p = LinearAlgebra.norm(p)
        tol_eff = abstol + reltol * _scalar_value(norm_p)
        verbose && @info "SC-Newton iter $iter : ‖F‖ = $norm_r   tol = $tol_eff"
        if select_best && _scalar_value(norm_r) < resid_best
            resid_best = _scalar_value(norm_r)
            p_best .= p
        end
        if norm_r ≤ tol_eff
            _sc_partials_converged(r, p; abstol, reltol) && return rebuild(p)
            # The value has converged and the derivatives have not: one full
            # Newton step gives them, by the implicit function theorem. A line
            # search on the value would only be judging its rounding.
            p .+= ForwardDiff.jacobian(residual_vec, p) \ (-r)
            r .= residual_vec(p)
            continue
        end
        # Jacobian via ForwardDiff (strong dependency).
        J = ForwardDiff.jacobian(residual_vec, p)
        # Solve J·δ = -r with a regularizing fallback if J is singular.
        δ = try
            J \ (-r)
        catch err
            # Only a singular (or numerically singular) Jacobian is regularized;
            # any other error is a bug and must surface.
            err isa Union{LinearAlgebra.SingularException, LinearAlgebra.LAPACKException} || rethrow()
            @debug "SC-Newton: singular Jacobian, applying tiny Tikhonov" err
            (J + 1.0e-10 * sqrt(sum(abs2, J)) * LinearAlgebra.I) \ (-r)
        end
        # Backtracking line search (Armijo).  The accepted `r_new` IS the
        # residual at the new point, so it is carried into the next iteration
        # instead of being recomputed there — one full RVE pass saved per
        # Newton step.
        α_step = one(Tref)
        accepted = false
        for _ in 1:30
            p_new = p .+ α_step .* δ
            r_new = residual_vec(p_new)
            if LinearAlgebra.norm(r_new) ≤ (1 - 1.0e-4 * α_step) * norm_r
                p .= p_new
                r .= r_new
                accepted = true
                break
            end
            α_step /= 2
            α_step < 1.0e-6 && break
        end
        if !accepted
            # Fall back to a damped Picard step.  `r` is already `F(p)`, and a
            # Picard step is `p ← F(p) + p`, so no new evaluation is needed
            # here either; the residual at the updated point is unknown, so it
            # is refreshed at the top of the next iteration.
            verbose && @info "SC-Newton: line search failed, taking Picard step"
            p .= r .+ p
            r .= residual_vec(p)
        end
    end
    @debug "SC-Newton: maxiters reached without convergence" maxiters
    return rebuild(select_best ? p_best : p)
end

# ── Auto-resolving nonlinear solver ─────────────────────────────────────────

"""
    _solve_sc(::AutoNonlinear, step, x0::AbstractTens; kw...) -> AbstractTens

Resolver for [`AutoNonlinear`](@ref): checks at runtime whether the weak
extension `MeanFieldHomogenizationNonlinearSolveExt` is loaded
(`Base.get_extension`) and, if so, delegates to
`ext.default_solve_sc(step, x0; kw...)` — a globalized SciML algorithm
(`NonlinearSolve.TrustRegion()`) run through the same
ForwardDiff-safe (implicit-function-theorem) path as any other
`NonlinearSolve.jl` algorithm. Falls back to the built-in
[`NewtonDefault`](@ref) solver when the extension is not loaded, so
`SelfConsistent(algorithm = AutoNonlinear())` always works — with or
without `using NonlinearSolve`.
"""
function _solve_sc(::AutoNonlinear, step, x0::TensND.AbstractTens; kw...)
    ext = Base.get_extension(parentmodule(@__MODULE__), :MeanFieldHomogenizationNonlinearSolveExt)
    return _solve_sc_auto(ext, step, x0; kw...)
end

# The extension when it is loaded, the built-in Newton solver otherwise.
_solve_sc_auto(ext::Module, step, x0; kw...) = ext.default_solve_sc(step, x0; kw...)
_solve_sc_auto(::Nothing, step, x0; kw...) = _solve_sc(NewtonDefault(), step, x0; kw...)

# ── Positive-definite guard for the SC running estimate ───────────────────
#
# At high contrast (porous SC near the percolation threshold, cracks at
# moderate density), the SC iteration map can have a stable fixed point
# at the trivial null tensor (`C = 0`).  A Picard iteration starting
# from `C_M` may drift into this percolation fixed point even when a
# physically meaningful finite fixed point exists nearby.  The reference
# implementation mitigates this by detecting a
# negative-definite running estimate and resetting it to a tiny
# positive identity (`mX = EPSILON · I`) before each step — this
# prevents the iteration from collapsing to zero and lets it find the
# physical finite-modulus branch when it exists.  We mirror the same
# guard here, eigenvalue by eigenvalue: every eigenvalue of the running
# estimate that falls below a threshold relative to the starting point is
# raised to that threshold, and the eigenvectors are kept.
#
# The structured classes are direct sums of blocks of their stored
# coefficients, and the spectrum is read block by block.  The coefficients of
# an isotropic, a cubic and a symmetric second-order TI tensor are its
# eigenvalues; those of a fourth-order TI tensor (Walpole) and of an
# orthotropic one hold a 2×2 or 3×3 symmetric block whose off-diagonal
# couplings are legitimately negative or zero — clamping each coefficient on
# its own moved the fixed point of every auxetic or uncoupled material.  A
# class with no such structure is passed through.

# `x0` is fixed for the entire SC iteration (Picard or Newton), so the threshold
# is computed once. It is a tolerance and carries no derivative.
function _sc_pd_eps(x0)
    α0_max = _max_canonical_value(x0)
    return max(α0_max * sqrt(eps(real(_value_eltype(x0)))), 1.0e-12)
end

const _SCGuardedTens = Union{TensND.TensISO, TensND.TensCubic, TensND.TensTI, TensND.TensOrtho}

# Scale of the threshold: the largest stored coefficient, and 1 for a class the
# guard passes through (the threshold is then never used).
_max_canonical_value(x::_SCGuardedTens) =
    maximum(v -> _scalar_value(abs(real(v))), TensND.get_data(x))
_max_canonical_value(::TensND.AbstractTens) = 1.0

_sc_pd_guard_apply(x::TensND.AbstractTens, ε_pos) = x

# Every stored coefficient is an eigenvalue: on (𝕁, 𝕂), on (𝕁, 𝔼, 𝕋), on (nT, nₙ).
function _sc_pd_guard_apply(
        x::Union{TensND.TensISO, TensND.TensCubic, TensND.TensTI{2, <:Any, 2}}, ε_pos
    )
    d = TensND.get_data(x)
    all(v -> _sc_above(v, ε_pos), d) && return x
    return TensND._rebuild(x, map(v -> _sc_clamp(v, ε_pos), d))
end

# a nT + b nₙ + c w: the in-plane rotation c w is antisymmetric and stores no
# energy, a and b are the eigenvalues of the symmetric part.
function _sc_pd_guard_apply(x::TensND.TensTI{2, <:Any, 3}, ε_pos)
    a, b, c = TensND.get_data(x)
    _sc_above(a, ε_pos) && _sc_above(b, ε_pos) && return x
    return TensND._rebuild(x, (_sc_clamp(a, ε_pos), _sc_clamp(b, ε_pos), c))
end

# Walpole coefficients ([[ℓ₁, ℓ₃], [ℓ₃, ℓ₂]], ℓ₅, ℓ₆) of a major-symmetric TI
# tensor: ℓ₅ and ℓ₆ are eigenvalues, the 2×2 block holds the other two.
function _sc_pd_guard_apply(x::TensND.TensTI{4, <:Any, 5}, ε_pos)
    ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆ = TensND.get_data(x)
    B = SMatrix{2, 2}(ℓ₁, ℓ₃, ℓ₃, ℓ₂)
    _sc_above(B, ε_pos) && _sc_above(ℓ₅, ε_pos) && _sc_above(ℓ₆, ε_pos) && return x
    C = _sc_clamp(B, ε_pos)
    return TensND._rebuild(x, (C[1, 1], C[2, 2], C[1, 2], _sc_clamp(ℓ₅, ε_pos), _sc_clamp(ℓ₆, ε_pos)))
end

# A TI tensor without major symmetry (N = 6), or with the in-plane rotation
# couplings ℓ₇ and ℓ₈ (N = 8, both antisymmetric): the guard acts on the
# symmetric part, and the antisymmetric part (ℓ₃ - ℓ₄)/2, ℓ₇, ℓ₈ is kept.
function _sc_pd_guard_apply(x::TensND.TensTI{4, <:Any, N}, ε_pos) where {N}
    d = TensND.get_data(x)
    ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆ = d
    s, a = (ℓ₃ + ℓ₄) / 2, (ℓ₃ - ℓ₄) / 2
    B = SMatrix{2, 2}(ℓ₁, s, s, ℓ₂)
    _sc_above(B, ε_pos) && _sc_above(ℓ₅, ε_pos) && _sc_above(ℓ₆, ε_pos) && return x
    C = _sc_clamp(B, ε_pos)
    return TensND._rebuild(
        x, (
            C[1, 1], C[2, 2], C[1, 2] + a, C[1, 2] - a,
            _sc_clamp(ℓ₅, ε_pos), _sc_clamp(ℓ₆, ε_pos), ntuple(i -> d[6 + i], Val(N - 6))...,
        )
    )
end

# (C₁₁, C₂₂, C₃₃, C₁₂, C₁₃, C₂₃, C₄₄, C₅₅, C₆₆): a 3×3 block in the material
# frame, and the eigenvalues 2C₄₄, 2C₅₅, 2C₆₆.
function _sc_pd_guard_apply(x::TensND.TensOrtho, ε_pos)
    c₁₁, c₂₂, c₃₃, c₁₂, c₁₃, c₂₃, c₄₄, c₅₅, c₆₆ = TensND.get_data(x)
    B = SMatrix{3, 3}(c₁₁, c₁₂, c₁₃, c₁₂, c₂₂, c₂₃, c₁₃, c₂₃, c₃₃)
    shear = (c₄₄, c₅₅, c₆₆)
    _sc_above(B, ε_pos) && all(c -> _sc_above(2c, ε_pos), shear) && return x
    C = _sc_clamp(B, ε_pos)
    return TensND._rebuild(
        x, (C[1, 1], C[2, 2], C[3, 3], C[1, 2], C[1, 3], C[2, 3], map(c -> _sc_clamp(2c, ε_pos) / 2, shear)...)
    )
end

# Whether an eigenvalue, or every eigenvalue of a symmetric block, exceeds ε.
# Decided on the real values, which is what a comparison of duals reads anyway.
_sc_above(v::Number, ε) = real(v) > ε
_sc_above(B::SMatrix, ε) = minimum(eigvals(Symmetric(_sc_values(B)))) > ε

# Unchanged above ε; otherwise ε, or V max(Λ, ε) Vᵀ for a block, whose
# eigenvectors are kept. A clamped iterate is a reset and carries no derivative.
_sc_clamp(v::Number, ε) = _sc_above(v, ε) ? v : oftype(v, ε)
function _sc_clamp(B::SMatrix{n, n, T}, ε) where {n, T}
    _sc_above(B, ε) && return B
    λ, V = eigen(Symmetric(_sc_values(B)))
    return T.(V * Diagonal(max.(λ, ε)) * V')
end

_sc_values(B::SMatrix) = map(v -> _scalar_value(real(v)), B)

# ── Tens ↔ canonical parameter vector helpers ──────────────────────────────
#
# `TensND.get_data(t)` already returns the canonical tuple of an iso /
# TI / ortho / Walpole tensor (`(α, β)` for iso, `(ℓ₁, …, ℓ₆)` for
# Walpole TI, etc.).  We `collect` it into a `Vector` for the
# ForwardDiff-friendly residual function, and rebuild the same
# concrete type via the canonical constructor inferred from the
# prototype.
"""
    _sc_newton_seed(x0, step) -> AbstractTens

Prototype whose symmetry class defines the Newton parameter space.

Returns `x0` when one application of `step` stays in the same class, and
`step(x0)` when it lands in a richer one (e.g. `TensISO` → `TensTI{4,T,8}`
for `TISymmetrize` phases).  Parameterizing on `x0` alone is what used to
make the residual subtract vectors of different lengths.

The extra `step` call is not wasted work: its result is also a strictly
better starting iterate than `x0`.
"""
function _sc_newton_seed(x0::TensND.AbstractTens, step)
    x1 = step(x0)        # an error here is the one the main loop would raise
    return length(TensND.get_data(x1)) == length(TensND.get_data(x0)) ? x0 : x1
end

# The Newton unknowns are the coordinates of TensND's `to_vector` (the
# coefficients of a structured class, the canonical Kelvin-Mandel components of a
# generic tensor), weighted by TensND's `param_norms` so that their Euclidean
# norm is the Frobenius norm of the tensor they encode: `abstol` and `reltol`
# then mean the same whatever the `algorithm` and the symmetry class. Newton is
# invariant under that diagonal rescaling (the root and the iterates are
# untouched); only the norm of the stopping test and of the Armijo condition
# changes, and a trust region is put in a physical metric.
_sc_param_weights(prototype::TensND.AbstractTens) =
    [float(TensND.primal(w)) for w in TensND.param_norms(prototype)]

# The residual of two iterates, written in one basis by TensND's subtraction.
_sc_residual_norm(a::TensND.AbstractTens, b::TensND.AbstractTens) = LinearAlgebra.norm(a - b)

# Components whose sums of squares, channel by channel, are Frobenius norms: the
# stored array in an orthonormal basis; in any other, where the convergence of
# the derivatives needs components rather than a norm, the canonical ones.
_comparable(t, ::TensND.OrthonormalBasis) = get_array(t)
_comparable(t, ::TensND.AbstractBasis) = TensND.components_canon(t)

# Initial zero residual with eltype matching `x0` (Dual-preserving).
_sc_residual_zero(x0::TensND.AbstractTens) = zero(real(eltype(x0)))

# Float64 view of a residual or a norm, for tolerances, comparisons and
# best-iterate tracking: a `ForwardDiff.Dual` is a `Real`, so `float` alone
# would keep its partials.
_scalar_value(r::Real) = float(TensND.primal(r))

# The plain float type under the duals of `x0`, for a `typemax` and a tolerance.
_value_eltype(x0::TensND.AbstractTens) = real(TensND.primal_type(eltype(x0)))

# Convergence of the value: `‖Δ‖ ≤ abstol + reltol ‖x‖`, compared on values.
_sc_converged(r::Real, abstol::Real) = r ≤ abstol

# Convergence of the derivatives. With ForwardDiff duals they are iterated
# along with the values, and converge only as fast, one step behind: started
# from the solution itself (`init`), the value has converged at the first step
# while the derivatives have not moved. Each partial — at every level of
# nesting — is therefore tested on its own and against its own scale, as the
# value is: ‖∂Δ‖ ≤ abstol + reltol ‖∂x‖.
function _sc_partials_converged(x_new::TensND.AbstractTens, x::TensND.AbstractTens; kw...)
    a, b = TensND.same_basis(x_new, x)          # one basis, as for the residual
    ℬ = TensND.get_basis(b)
    xb = _comparable(b, ℬ)
    return _sc_partials_converged(_comparable(a, ℬ) .- xb, xb; kw...)
end

function _sc_partials_converged(Δ::AbstractArray, x::AbstractArray; abstol::Real, reltol::Real)
    T = promote_type(eltype(Δ), eltype(x))
    T <: ForwardDiff.Dual || return true
    sΔ, sx = _channel_sqnorms(Δ, T), _channel_sqnorms(x, T)
    # Channel 1 is the value, which the caller has tested.
    return all(sqrt(sΔ[c]) ≤ abstol + reltol * sqrt(sx[c]) for c in 2:length(sΔ))
end

# Σ over `a` of the squares of every coefficient of an element: the value, then
# each partial, recursively through the nesting (TensND's `dual_channels`).
function _channel_sqnorms(a::AbstractArray, ::Type{T}) where {T}
    acc = map(abs2, TensND.dual_channels(zero(T)))
    for v in a
        acc = acc .+ map(abs2, TensND.dual_channels(convert(T, v)))
    end
    return acc
end

# =============================================================================
#  AsymmetricSelfConsistent — Mori-Tanaka-style iteration with self-reference.
#
#  At each iteration, the dilute concentration tensor is computed in the
#  current effective medium (stiffness `C^n` or compliance `S^n`), but
#  the resulting dilute correction is added to the matrix property
#  (`C_m` or `S_m`) — not to the iterating estimate. This is the
#  "asymmetric" formulation: the matrix retains its privileged role
#  even in the SC iteration.
#
#  WHEN the two fixed points coincide — they do NOT in general. Write
#  `A_r = A_r^dil(C)` at the fixed point. The Hill-symmetric SC solves
#
#      Σ_r f_r (C_r − C) : A_r = 0        (sum over ALL phases, matrix included)
#
#  while the asymmetric form solves
#
#      C − C_m = Σ_i f_i (C_i − C_m) : A_i     (sum over INCLUSIONS only).
#
#  Splitting `C_i − C_m = (C_i − C) + (C − C_m)` in the second and injecting
#  the first turns the pair into
#
#      (C − C_m) : [𝟙 − Σ_r f_r A_r] = 0 ,
#
#  so the two agree exactly when `Σ_r f_r A_r = 𝟙`. That identity is not
#  free: it holds when every phase shares ONE Hill tensor `P` (same shape,
#  same orientation, same reference), because then `C_r − C = P⁻¹ : (A_r⁻¹ − 𝟙)`
#  and `Σ_r f_r (C_r − C) : A_r = P⁻¹ : (𝟙 − Σ_r f_r A_r)`, making the SC
#  condition *equivalent* to it. With unequal shapes the `P`s differ, the
#  equivalence breaks, and the two schemes converge to genuinely different
#  effective media — 2.5 % apart on `k` for an oblate ω = 0.2 inclusion in a
#  spherical matrix, 6 % at ω = 0.05, the gap tracking `‖Σ f_r A_r − 𝟙‖`.
#  Sphericity is only the common special case: a matrix and an inclusion that
#  are BOTH oblate ω = 0.05 satisfy the identity and agree again.
#
#  An orientation average does not rescue it: `⟨A⟩` of an averaged spheroid is
#  not the `A` of a sphere.
#
#  The iteration dynamics differ in every case (different basins of
#  attraction), which is the point of offering both.
#
#  Stiffness form  : C^{n+1} = C_m + Σ_i f_i (C_i − C_m) A_{εε,i}^{(C^n)}
#  Compliance form : S^{n+1} = S_m + (⟨A_{εε}⟩ − S_m ⟨C_i A_{εε}⟩) S^n
#
#  The branch is picked by the matrix-vs-Voigt-bound contrast: when the
#  matrix is much stiffer than the Voigt upper bound (matrix-stiff /
#  inclusions-soft, e.g. a porous medium in elasticity), the compliance
#  form is contractive; otherwise stiffness form. Selection follows the
#  C++ reference's squared-Frobenius-norm criterion.
# =============================================================================

"""
    _evaluate(rve, asc::AsymmetricSelfConsistent, ::Val{p}; kw...) -> AbstractTens

Asymmetric self-consistent scheme. Iterates a Mori-Tanaka-like update
where the dilute concentration tensors use the *current effective
medium* as reference but the dilute correction is added to the
*matrix property*. For porous and crack RVEs the asymmetric form
converges to a different physical branch (the matrix-distinguished
branch).

!!! warning "This is a different scheme, not just a different iteration"
    The asymmetric fixed point coincides with the Hill-symmetric
    [`SelfConsistent`](@ref) one **only when every phase shares one Hill
    tensor** — the same shape, orientation and reference — which makes
    ``\\sum_i f_i\\,\\mathbb{A}_i = \\mathbb{I}`` (see the derivation at the top of
    `self_consistent.jl`). All-spherical phases are the usual such case.
    With unequal shapes the two converge to genuinely different effective
    media: about 2.5 % apart on ``k`` for an oblate ``\\omega = 0.2`` inclusion in a
    spherical matrix, 6 % at ``\\omega = 0.05``. An orientation average does not
    restore the identity. Pick the scheme on physical grounds, not on the
    assumption that it is the cheaper route to the same answer.

Compliance form is selected when the matrix property is "stiffer"
(in squared Frobenius norm) than the Voigt upper bound — the
contractivity argument otherwise breaks down for high-contrast
matrix-stiff systems. When an internally heterogeneous phase exposes no
layer-wise average, so that no bound can be evaluated, the dilute estimate
plays the same role: the branch heuristic is the only place a bound ever
entered, and the iteration itself needs nothing beyond the concentration
tensors that `SelfConsistent` also requires.
"""
function _evaluate(rve::RVE, asc::AsymmetricSelfConsistent, ::Val{p}; kw...) where {p}
    m = matrix_name(asc, rve)
    if _asc_use_stiffness(rve, p, m; kw...)
        return _asc_iterate_stiffness(rve, asc, m, Val(p); kw...)
    else
        return _asc_iterate_compliance(rve, asc, m, Val(p); kw...)
    end
end

# Squared Frobenius norm — matches the C++ reference selection.
# Crack phases (`CrackDensity`) carry no volume but still soften the
# composite ; the Voigt heuristic above ignores them and may wrongly
# select the stiffness form (which skips cracks → no degradation).
# Force the compliance form whenever an RVE has at least one crack.
#
# The heuristic only has to answer "is the composite stiffer than the matrix?".
# The Voigt bound is the C++ reference's answer and is kept whenever it can be
# evaluated — but it needs the internal volume fractions of every heterogeneous
# inclusion, which the *iteration* does not. When a phase cannot supply them
# the dilute estimate answers the same question from the very tensors ASC
# already requires, so the scheme stays available: nothing in the asymmetric
# self-consistent algorithm itself depends on a bound.
function _asc_use_stiffness(rve::RVE, prop::Symbol, m::Symbol; kw...)
    any(_is_crack_family, values(rve.amounts)) && return false
    P₀ = phase_property(rve, m, prop)
    probe = _bounds_available(rve) ? Voigt() : Dilute()
    return TensND.sqnorm(_evaluate(rve, probe, Val(prop); kw...)) ≥ TensND.sqnorm(P₀)
end


# ── Stiffness-form iteration ────────────────────────────────────────────────
#
# C^{n+1} = C_m + Σ_i f_i (C_i − C_m) A_{εε,i}^{(C^n)}
#
# Sum runs over INCLUSIONS only (matrix excluded, treated implicitly via
# C_m). This is the C++ reference's `evaluate_dilute(X)` body.

function _asc_iterate_stiffness(
        rve::RVE, asc::AsymmetricSelfConsistent, m::Symbol,
        ::Val{p}; kw...
    ) where {p}
    C_m = phase_property(rve, m, p)
    solver_kw, step_kw = _split_sc_kwargs(kw)
    step = C_n -> _asc_step_stiffness(rve, p, m, C_m, C_n; step_kw...)
    x0 = asc.init === nothing ? C_m : _sc_initial(asc.init, rve, p; kw...)
    return _solve_sc(asc.algorithm, step, x0; asc.options..., solver_kw...)
end

function _asc_step_stiffness(rve::RVE, prop::Symbol, m::Symbol, C_m, C_n; kw...)
    return _asc_step_stiffness_dispatch(rve, prop, m, C_m, C_n; kw...)
end

function _asc_step_stiffness_dispatch(
        rve::RVE, prop::Symbol, m::Symbol,
        C_m::TensND.AbstractTens{4, 3},
        C_n::TensND.AbstractTens{4, 3}; kw...
    )
    Δ = zero(C_n)
    for name in _volume_phase_names(rve, m)   # cracks not supported in stiffness form
        f = volume_fraction(rve, name)
        # The asymmetry that defines the ASC: concentration is computed in the
        # CURRENT reference `C_n`, but the contribution is measured against the
        # MATRIX `C_m`.  `_phase_stiffness_contribution` cannot be used here —
        # it uses one and the same reference for both.  The B tensor (⟨C:A⟩)
        # handles heterogeneous inclusions and symmetrizes the product; the
        # `C_m` term reuses ⟨A⟩.
        A_dil, CA = _phase_dilute_and_stress_average(rve, name, prop, C_n; kw...)
        Δ += f * (CA - C_m ⊡ A_dil)
    end
    return C_m + Δ
end

function _asc_step_stiffness_dispatch(
        rve::RVE, prop::Symbol, m::Symbol,
        K_m::TensND.AbstractTens{2, 3},
        K_n::TensND.AbstractTens{2, 3}; kw...
    )
    Δ = zero(K_n)
    for name in _volume_phase_names(rve, m)
        f = volume_fraction(rve, name)
        A_dil, KA = _phase_dilute_and_stress_average(rve, name, prop, K_n; kw...)
        Δ += f * (KA - K_m ⋅ A_dil)
    end
    return K_m + Δ
end

# ── Compliance-form iteration ───────────────────────────────────────────────
#
# Iterating estimate is the *stiffness* `C^n` (we invert at the end);
# the formula reads
#
#   S^{n+1} = S_m + ⟨A_{εε}^{(C^n)}⟩ S^n − S_m ⟨C_i A_{εε}^{(C^n)}⟩ S^n
#
# where S_n = inv(C^n), S_m = inv(C_m), and the sum is over INCLUSIONS
# only. We then invert `S^{n+1}` to recover `C^{n+1}` so the iteration
# stays in stiffness space and the same `_solve_sc` driver is reused.

function _asc_iterate_compliance(
        rve::RVE, asc::AsymmetricSelfConsistent, m::Symbol,
        ::Val{p}; kw...
    ) where {p}
    C_m = phase_property(rve, m, p)
    S_m = inv(C_m)
    solver_kw, step_kw = _split_sc_kwargs(kw)
    step = C_n -> _asc_step_compliance(rve, p, m, C_m, S_m, C_n; step_kw...)
    x0 = asc.init === nothing ? C_m : _sc_initial(asc.init, rve, p; kw...)
    return _solve_sc(asc.algorithm, step, x0; asc.options..., solver_kw...)
end

function _asc_step_compliance(rve::RVE, prop::Symbol, m::Symbol, C_m, S_m, C_n; kw...)
    return _asc_step_compliance_dispatch(rve, prop, m, C_m, S_m, C_n; kw...)
end

function _asc_step_compliance_dispatch(
        rve::RVE, prop::Symbol, m::Symbol,
        C_m::TensND.AbstractTens{4, 3},
        S_m::TensND.AbstractTens{4, 3},
        C_n::TensND.AbstractTens{4, 3}; kw...
    )
    S_n = inv(C_n)
    A_avg = zero(C_n)
    CA_avg = zero(C_n)
    # Volume-fraction inclusions: standard A_εε. Crack densities: the strain
    # concentration is singular; their compliance contribution is the natural
    # object instead and is added directly to S^{n+1} below.
    for name in _volume_phase_names(rve, m)
        f = volume_fraction(rve, name)
        # `⟨C:A⟩`, NOT `C ⊡ ⟨A⟩`: the orientation average does not commute
        # with the tensor product, so the product has to be formed on the RAW
        # concentration tensor and averaged afterwards, which is what the bundle
        # does: formed from the averaged `A_dil` and the raw `C_i`, an
        # iso-symmetrized anisotropic phase would come back transversely
        # isotropic, and not even major-symmetric. The bundle also routes
        # heterogeneous inclusions through `concentration_and_flux_average`, whose
        # stress average the declared (placeholder) property of a
        # `LayeredSphere` cannot give.
        A_dil, CA = _phase_dilute_and_stress_average(rve, name, prop, C_n; kw...)
        A_avg += f * A_dil
        CA_avg += f * CA
    end
    S_new = S_m + (A_avg - S_m ⊡ CA_avg) ⊡ S_n
    # Crack contributions add directly to the compliance.
    for name in _crack_phase_names(rve, m)
        S_new += _phase_compliance_contribution(rve, name, prop, C_n; kw...)
    end
    return inv(S_new)
end

function _asc_step_compliance_dispatch(
        rve::RVE, prop::Symbol, m::Symbol,
        K_m::TensND.AbstractTens{2, 3},
        R_m::TensND.AbstractTens{2, 3},
        K_n::TensND.AbstractTens{2, 3}; kw...
    )
    R_n = inv(K_n)
    A_avg = zero(K_n)
    KA_avg = zero(K_n)
    for name in _volume_phase_names(rve, m)
        f = volume_fraction(rve, name)
        # `⟨K·A⟩`, not `K ⋅ ⟨A⟩` — see the order-4 dispatch above.
        A_dil, KA = _phase_dilute_and_stress_average(rve, name, prop, K_n; kw...)
        A_avg += f * A_dil
        KA_avg += f * KA
    end
    R_new = R_m + (A_avg - R_m ⋅ KA_avg) ⋅ R_n
    # Crack contributions add directly to the resistivity, as at order 4.
    for name in _crack_phase_names(rve, m)
        R_new += _phase_compliance_contribution(rve, name, prop, K_n; kw...)
    end
    return inv(R_new)
end
