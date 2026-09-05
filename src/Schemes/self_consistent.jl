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
([mclaughlin1977](@cite)). Iterates

```math
\\mathbb C^{(n+1)} = \\Big(\\sum_i f_i\\,\\mathbb C_i \\!:\\! \\mathbb A_\\mathrm{dil}^{(i)}(\\mathbb C^{(n)})\\Big)
                     :\\Big(\\sum_i f_i\\,\\mathbb A_\\mathrm{dil}^{(i)}(\\mathbb C^{(n)})\\Big)^{-1}
```

with the dilute concentration tensor evaluated against the current
estimate `C^{(n)}` itself (rather than the matrix property).

The solver algorithm is selected by `sc.algorithm`; convergence kwargs
in `sc.options` (`abstol`, `reltol`, `maxiters`, `damping`, `verbose`,
`select_best`) override their defaults. The stopping test is the additive
SciML convention `‖Δx‖ ≤ abstol + reltol · ‖x‖`, so `reltol` (default `1e-8`)
is what binds on a stiffness of physical magnitude — see [`_solve_sc`](@ref)
for the full contract. External algorithms from `NonlinearSolve.jl` are
supported via the weak extension `MeanFieldHomogenizationNonlinearSolveExt`.

Cracks are not natively handled by this stiffness-form SC (the strain
concentration tensor is singular). For mixed RVEs (solid + crack) use
[`AsymmetricSelfConsistent`](@ref) instead.
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
- a phase name — that phase's property. This is what versions up to 0.7 used
  implicitly through the matrix; pass it to reproduce an older number exactly.
- a tensor — an explicit guess, the natural form when restarting from the
  previous step of a constitutive loop.

A phase named `:voigt` or `:reuss` wins over the keyword: a phase name is the
caller's data, a keyword is ours.
"""
function _sc_initial(init::Symbol, rve::RVE, prop::Symbol; kw...)
    haskey(rve.phases, init) && return phase_property(rve, init, prop)
    init === :voigt && return _evaluate(rve, Voigt(), Val(prop); kw...)
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
# `TrustRegion` stop where the finite-difference IFT Jacobian is exactly
# singular, which the phase seed does not. Equivalent where it matters, worse in
# one corner: not a default worth taking.
function _sc_initial(::Nothing, rve::RVE, prop::Symbol; kw...)
    r = remainder_phase_name(rve)
    r === nothing && return _evaluate(rve, Voigt(), Val(prop); kw...)
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

```
A_avg  = Σ_α f_α ⟨A_α(P_n)⟩          CA_avg = Σ_α f_α ⟨C_α : A_α(P_n)⟩
```

`f_α` is the *resolved* fraction: the RVE's fraction closure has already turned
a [`Remainder`](@ref) into `1 - Σ f` and, under [`RescaledFractions`](@ref),
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
    for name in rve.phase_names
        rve.amounts[name] isa CrackDensity && continue
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
`𝔹_E : 𝔸_E⁻¹` — a product of two tensors that do not commute — and that product
is only major-symmetric when the phases happen to share a common frame. With
non-coaxial crack families it is not: the asymmetry starts at roundoff and is
then **amplified by the fixed-point iteration** (measured: 3·10⁻¹⁵ at the second
iterate, 2·10⁻⁴ at the third). A stiffness that is not major-symmetric is not a
valid Eshelby reference medium, and the anisotropic crack cubature returns a
`NaN` integrand on it, which used to abort the whole solve with a `DomainError`.

Projecting at every step keeps the iteration inside the admissible set. It is a
no-op up to roundoff whenever the product is already symmetric, and it matches
the reference ECHOES implementation, whose self-consistent estimate comes out
exactly major-symmetric.

The components are taken and rebuilt in the tensor's **own** basis, so the
projection commutes with the orientation of the running estimate.

!!! note "Structured types are returned untouched"
    `TensISO`, `TensOrtho` and the five-parameter Walpole `TensTI{4,T,5}` are
    major-symmetric *by construction*, so there is nothing to project — and
    rebuilding them as a generic `Tens` would be actively harmful: the Newton
    parameterization (`_sc_newton_seed`) reads the structured components through
    `get_data`, and the symmetry-class dispatch keys on the concrete type. Only
    the genuinely unstructured estimates — which is exactly what a mix of
    non-coaxial families produces — are rebuilt.
"""
_major_symmetrize(P::TensND.TensISO{4, 3}) = P
_major_symmetrize(P::TensND.TensISO{2, 3}) = P
_major_symmetrize(P::TensND.TensOrtho) = P
_major_symmetrize(P::TensND.TensTI{4, T, 5}) where {T} = P
_major_symmetrize(P::TensND.TensTI{2}) = P

function _major_symmetrize(P::TensND.AbstractTens{4, 3})
    A = get_array(P)
    T = eltype(A)
    sym = Tensors.SymmetricTensor{4, 3}(
        (i, j, k, l) -> (A[i, j, k, l] + A[k, l, i, j]) / T(2)
    )
    return TensND.Tens(sym, TensND.get_basis(P))
end

function _major_symmetrize(P::TensND.AbstractTens{2, 3})
    A = get_array(P)
    T = eltype(A)
    sym = Tensors.SymmetricTensor{2, 3}((i, j) -> (A[i, j] + A[j, i]) / T(2))
    return TensND.Tens(sym, TensND.get_basis(P))
end

"""
    _sc_crack_total(rve, P_n, prop; kw...) -> (H_total, has_cracks)

Sum of the *density-scaled* compliance (resistivity) contributions of every
[`CrackDensity`](@ref) phase, evaluated in the reference medium `P_n`:

```
H_total = Σ_i (4π/3) d_i ℍ_i(P_n)
```

`has_cracks` reports whether the sum has any term at all, which is what selects
the crack branch of the SC body.
"""
function _sc_crack_total(
        rve::RVE, P_n::TensND.AbstractTens, prop::Symbol; kw...
    )
    H_total = zero(P_n)
    has_cracks = false
    for name in rve.phase_names
        rve.amounts[name] isa CrackDensity || continue
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
  (`x_{n+1} = (1-damping)·step(x_n) + damping·x_n`). `damping = 0.0`
  default; raise to ≈ 0.5 for high-contrast iterations that overshoot.
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

Convergence is declared when `‖x_new − x_old‖ ≤ abstol + reltol · ‖x_old‖`
(absolute *and* relative tolerance, additive convention; pass
`abstol = 0` to require purely relative convergence). Default values:
`abstol = 1e-12`, `reltol = 1e-8`.

`‖·‖` is the **Frobenius norm of the tensor**, for every solver, so that a
given `abstol` expresses one requirement regardless of `algorithm` and of the
symmetry class the fixed point lives in. The Picard loop measures it directly;
the Newton path reaches it by working in the isometric parametrization of
[`_sc_param_weights`](@ref); the `NonlinearSolve` path passes those same
weights to SciML as the `internalnorm` of its termination condition, leaving
the unknowns alone — a trust region is not invariant under a rescaling of
them, and moving its metric moves where it stops.

Because a stiffness carries a physical magnitude, `reltol · ‖x‖` dominates the
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
linearized step has a Jacobian eigenvalue ≈ 1) and the residual stalls
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
        _sc_converged(last_resid, tol_eff) && return x_new
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

1. Build the residual `F(p) = canonical(step(rebuild(p))) − p`,
2. Compute the Jacobian `J = ∂F/∂p` via `ForwardDiff.jacobian`,
3. Take the Newton step `Δp = −J⁻¹·F(p)` with backtracking line
   search (Armijo with shrinking factor 1/2, minimum step 1e-6).
4. Fall back to a single Picard step when the line search fails.

Compared to the SciML weak-extension path, this is dependency-free and
specialized to the small parameter spaces of `MeanFieldHomogenization` symmetry
classes (≤ 21 components for the most general aniso 4-tensor); the
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
    w, isometric = _sc_param_weights(x_seed)
    p0 = w .* _tens_to_param_vec(x_seed)
    L = length(p0)
    rebuild = p -> _tens_from_param_vec(x_seed, p ./ w)
    ε_pos = _sc_pd_eps(x_seed)
    residual_vec = function (p)
        x_in = _sc_pd_guard_apply(rebuild(p), ε_pos)
        x_out = step(x_in)
        return w .* (_tens_to_param_vec(x_out) .- _tens_to_param_vec(x_in))
    end
    # With an orthogonal class basis the Euclidean norm of the scaled vector
    # *is* the Frobenius norm. Where the probe failed (unit weights) fall back
    # to measuring the tensors themselves: exact too, only more costly.
    _nrm_r = isometric ? (r, p) -> sqrt(sum(abs2, r)) :
        (r, p) -> _sc_residual_norm(rebuild(p .+ r), rebuild(p))
    _nrm_p = isometric ? p -> sqrt(sum(abs2, p)) : p -> _sc_frobenius(rebuild(p))
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
        norm_r = _nrm_r(r, p)
        norm_p = _nrm_p(p)
        tol_eff = abstol + reltol * _scalar_value(norm_p)
        verbose && @info "SC-Newton iter $iter : ‖F‖ = $norm_r   tol = $tol_eff"
        if select_best && _scalar_value(norm_r) < resid_best
            resid_best = _scalar_value(norm_r)
            p_best .= p
        end
        if norm_r ≤ tol_eff
            return rebuild(p)
        end
        # Jacobian via ForwardDiff (strong dependency).
        J = ForwardDiff.jacobian(residual_vec, p)
        # Solve J·δ = -r with a regularizing fallback if J is singular.
        δ = try
            J \ (-r)
        catch err
            @debug "SC-Newton: linear solve failed, applying tiny Tikhonov" err
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
            if _nrm_r(r_new, p_new) ≤ (1 - 1.0e-4 * α_step) * norm_r
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
    if ext === nothing
        return _solve_sc(NewtonDefault(), step, x0; kw...)
    else
        return ext.default_solve_sc(step, x0; kw...)
    end
end

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
# guard here: when any canonical eigenvalue of the running estimate
# falls below a relative threshold, we reset it to `ε · α_M_init`
# (matrix scale).

# `x0` is fixed for the entire SC iteration (Picard or Newton) — computing
# this once and passing `ε_pos` into `_sc_pd_guard_apply` directly avoids
# recomputing `_max_canonical_value(x0)` (which itself does a `try`/`catch`,
# blocking inlining) on every single guard call within the loop.
function _sc_pd_eps(x0)
    α0_max = _max_canonical_value(x0)
    return max(α0_max * sqrt(eps(real(_value_eltype(x0)))), 1.0e-12)
end

# Iso 4-tensor: check (α, β); reset to ε·𝕁 + ε·𝕂 if either component
# is non-positive.
function _sc_pd_guard_apply(x::TensND.TensISO{3}, ε_pos)
    α, β = TensND.get_data(x)
    if real(α) ≤ ε_pos || real(β) ≤ ε_pos
        return TensND.TensISO{3}(max(real(α), ε_pos), max(real(β), ε_pos))
    end
    return x
end

# Default: try to detect the smallest canonical component value.  If
# any canonical component is non-positive, reset to a tiny positive
# baseline of the same shape.  Falls back to passthrough for tensors
# whose `get_data` does not give a meaningful "smallest eigenvalue"
# notion.
function _sc_pd_guard_apply(x::TensND.AbstractTens, ε_pos)
    try
        d = TensND.get_data(x)
        for v in d
            real(v) ≤ ε_pos && return _rebuild_min_eps(x, ε_pos)
        end
    catch
    end
    return x
end

function _rebuild_min_eps(x::TensND.AbstractTens, ε_pos)
    d = TensND.get_data(x)
    new_d = Tuple(real(v) ≤ ε_pos ? ε_pos : v for v in d)
    return TensND._rebuild(x, new_d)
end

# Used to scale the eigenvalue tolerance.
_max_canonical_value(x::TensND.AbstractTens) = begin
    try
        d = TensND.get_data(x)
        return maximum(abs(real(v)) for v in d)
    catch
        return 1.0
    end
end

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
    x1 = try
        step(x0)
    catch
        return x0            # let the main loop surface the real failure
    end
    return length(TensND.get_data(x1)) == length(TensND.get_data(x0)) ? x0 : x1
end

_tens_to_param_vec(t::TensND.AbstractTens) = collect(TensND.get_data(t))
_tens_from_param_vec(prototype::TensND.AbstractTens, p::AbstractVector) =
    _rebuild_from_data(prototype, p)

# Default rebuild: `_rebuild` is TensND-internal; for known types we use
# the public constructor.
_rebuild_from_data(::TensND.TensISO{3}, p) = TensND.TensISO{3}(p[1], p[2])
_rebuild_from_data(t::TensND.AbstractTens, p) =
    TensND._rebuild(t, ntuple(i -> p[i], length(p)))

# Residual norm operating on stored components (works for any tensor type).
_sc_residual_norm(a::TensND.AbstractTens, b::TensND.AbstractTens) =
    sqrt(sum(abs2, get_array(a) .- get_array(b)))

# Frobenius norm of a tensor, from its stored array. Basis-independent for an
# orthonormal frame, so a rotated tensor's local components give the same value
# as its canonical ones.
_sc_frobenius(t::TensND.AbstractTens) = sqrt(sum(abs2, get_array(t)))

"""
    _sc_param_weights(prototype) -> (w::Vector{Float64}, isometric::Bool)

Per-component weights making the Euclidean norm of the Newton / SciML parameter
vector equal the **Frobenius norm of the tensor it encodes**, so that `abstol`
and `reltol` express the same requirement whatever the `algorithm` and whatever
the symmetry class.

The canonical components are coordinates in a basis of the symmetry class
(`(α, β)` on `(𝕁, 𝕂)` for `TensISO`, the Walpole coefficients for `TensTI`, …).
Those bases are *orthogonal* but not orthonormal — `‖𝕁‖_F = 1` while
`‖𝕂‖_F = √5` — so the plain Euclidean norm of the component vector understates
the tensor norm, and understates it by a class-dependent factor. Scaling
component `i` by `wᵢ = ‖rebuild(eᵢ)‖_F` restores the isometry exactly:

```math
\\lVert w \\odot p \\rVert_2 = \\lVert \\mathrm{rebuild}(p) \\rVert_F .
```

Newton is invariant under a diagonal rescaling of its unknowns — `J` becomes
`W J W⁻¹`, the step `W δ`, and the iterate `W(p + δ)` — so the root and the
sequence of iterates are untouched; only the norm the stopping test and the
Armijo condition are measured in changes. A trust-region method is *not*
invariant, and there the rescaling additionally puts its region in a
physically meaningful metric.

The isometry needs the class basis to be orthogonal, which is checked rather
than assumed: `isometric = false` (with unit weights) is returned when the
probe fails — the general `Mandel66` fallback among others — and the caller
then measures the tensor norm directly instead.
"""
# `‖rebuild(prototype, v)‖_F`, or `nothing` when the class cannot be rebuilt
# from those components at all. Not every symmetry class accepts every
# coordinate vector — a rebuild that validates its argument rejects the
# canonical unit vectors, which are semi-definite — so this is a normal answer
# rather than an error, and one guarded call site serves both probes below.
function _sc_basis_norm(prototype::TensND.AbstractTens, v::AbstractVector)
    return try
        _scalar_value(_sc_frobenius(_rebuild_from_data(prototype, v)))
    catch
        return nothing
    end
end

function _sc_param_weights(prototype::TensND.AbstractTens)
    L = length(TensND.get_data(prototype))
    ones_w = ones(Float64, L)
    w = Vector{Float64}(undef, L)
    for i in 1:L
        e = zeros(Float64, L)
        e[i] = 1.0
        wi = _sc_basis_norm(prototype, e)
        (wi === nothing || !isfinite(wi) || wi ≤ 0) && return (ones_w, false)
        w[i] = wi
    end
    # Orthogonality probe: a vector with pairwise-distinct entries, so that a
    # non-zero cross term `2 qᵢ qⱼ ⟨Eᵢ, Eⱼ⟩` cannot cancel out by symmetry.
    q = Float64[i for i in 1:L]
    nq = _sc_basis_norm(prototype, q)
    nq === nothing && return (ones_w, false)
    isapprox(nq^2, sum(abs2, q .* w); rtol = 1.0e-10) || return (ones_w, false)
    return (w, true)
end

# Initial zero residual with eltype matching `x0` (Dual-preserving).
_sc_residual_zero(x0::TensND.AbstractTens) = zero(real(eltype(x0)))

# Scalar Float64 view of a residual: extract the `.value` field for
# `ForwardDiff.Dual` (via duck-typing — no hard dependency on ForwardDiff)
# so that comparisons and best-iterate tracking work uniformly.
_scalar_value(r::Real) = float(r)
_scalar_value(r) = hasfield(typeof(r), :value) ? float(_scalar_value(getfield(r, :value))) :
    throw(ArgumentError("cannot reduce residual of type $(typeof(r)) to a Float64"))

# Float64 type of `_scalar_value(zero_like_eltype(x0))`. Used to pick a
# safe `typemax` for `resid_best_val` regardless of whether x0 is real
# or Dual. ForwardDiff.Dual is `<: Real` so we can't dispatch on `T<:Real`
# alone — we must inspect the `:value` field first, only falling through
# to `Real` for plain numeric eltypes.
_value_eltype(::TensND.AbstractTens{<:Any, <:Any, T}) where {T} = _value_eltype(T)
function _value_eltype(::Type{T}) where {T}
    hasfield(T, :value) && return _value_eltype(fieldtype(T, :value))
    T <: Real && return float(T)
    return Float64
end

# Convergence criterion. For Real residuals it is the obvious `r < abstol`.
# For Dual residuals (ForwardDiff), require both the value AND every
# partial derivative to be below `abstol`: a fixed-point iteration where
# ‖value‖ has converged but ‖partials‖ has not gives a derivative that
# is numerically wrong (each Picard step propagates a contraction
# coefficient of order ‖∂step/∂x‖ — partials converge as fast as the
# value only if the contraction is well-behaved, which is not guaranteed
# for ill-conditioned SC iterations).
_sc_converged(r::Real, abstol::Real) = r ≤ abstol
function _sc_converged(r, abstol::Real)
    hasfield(typeof(r), :value) || return float(r) ≤ abstol
    abs(_scalar_value(getfield(r, :value))) ≤ abstol || return false
    if hasfield(typeof(r), :partials)
        p = getfield(r, :partials)
        # `partials` is a `ForwardDiff.Partials{N,T}` wrapper; iterate values.
        for v in p
            _sc_converged(v, abstol) || return false
        end
    end
    return true
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
    `Σ_r f_r A_r = 𝟙` (see the derivation at the top of
    `self_consistent.jl`). All-spherical phases are the usual such case.
    With unequal shapes the two converge to genuinely different effective
    media: about 2.5 % apart on `k` for an oblate `ω = 0.2` inclusion in a
    spherical matrix, 6 % at `ω = 0.05`. An orientation average does not
    restore the identity. Pick the scheme on physical grounds, not on the
    assumption that it is the cheaper route to the same answer.

Compliance form is selected when the matrix property is "stiffer"
(in squared Frobenius norm) than the Voigt upper bound — the
contractivity argument otherwise breaks down for high-contrast
matrix-stiff systems. When an internally heterogeneous phase exposes no
layer-wise average, so that no bound can be evaluated, the dilute estimate
plays the same role: the branch heuristic is the only place a bound ever
entered, and the iteration itself needs nothing beyond the localization
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
    if any(a isa CrackDensity for a in values(rve.amounts))
        return false
    end
    P₀ = phase_property(rve, m, prop)
    probe = _bounds_available(rve) ? Voigt() : Dilute()
    return _frob_sq(_evaluate(rve, probe, Val(prop); kw...)) ≥ _frob_sq(P₀)
end

_frob_sq(t::TensND.AbstractTens) = sum(abs2, get_array(t))

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
    for name in inclusion_phase_names(rve, m)
        a = rve.amounts[name]
        a isa CrackDensity && continue   # cracks not supported in stiffness form
        f = volume_fraction(rve, name)
        # The asymmetry that defines the ASC: localization is computed in the
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
    for name in inclusion_phase_names(rve, m)
        a = rve.amounts[name]
        a isa CrackDensity && continue
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
    for name in inclusion_phase_names(rve, m)
        a = rve.amounts[name]
        # Volume-fraction inclusions: standard A_εε.
        # Crack densities: the strain concentration is singular; their
        # "compliance contribution" is the natural object instead and is
        # added directly to S^{n+1} via `compliance_contribution`.
        if !(a isa CrackDensity)
            f = volume_fraction(rve, name)
            # `⟨C:A⟩`, NOT `C ⊡ ⟨A⟩`: the orientation average does not commute
            # with the tensor product, so the product has to be formed on the
            # RAW localization tensor and averaged afterwards — which is what
            # the bundle does.  Building it here from the already-averaged
            # `A_dil` and the raw `C_i` was wrong whenever the phase property
            # is anisotropic and `symmetrize` is on (an ISO `C_i` commutes with
            # the average, which is why the stiffness branch and every
            # isotropic-phase RVE agreed anyway): an iso-symmetrized phase came
            # back transversely isotropic, and not even major-symmetric.
            # The bundle also routes heterogeneous inclusions through
            # `loc_and_stress_average`, whose stress average the declared
            # (placeholder) property of a `LayeredSphere` cannot give.
            A_dil, CA = _phase_dilute_and_stress_average(rve, name, prop, C_n; kw...)
            A_avg += f * A_dil
            CA_avg += f * CA
        end
    end
    S_new = S_m + (A_avg - S_m ⊡ CA_avg) ⊡ S_n
    # Crack contributions add directly to the compliance.
    for name in inclusion_phase_names(rve, m)
        a = rve.amounts[name]
        a isa CrackDensity || continue
        H = _phase_compliance_contribution(rve, name, prop, C_n; kw...)
        S_new += H
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
    for name in inclusion_phase_names(rve, m)
        a = rve.amounts[name]
        a isa CrackDensity && continue
        f = volume_fraction(rve, name)
        # `⟨K·A⟩`, not `K ⋅ ⟨A⟩` — see the order-4 dispatch above.
        A_dil, KA = _phase_dilute_and_stress_average(rve, name, prop, K_n; kw...)
        A_avg += f * A_dil
        KA_avg += f * KA
    end
    R_new = R_m + (A_avg - R_m ⋅ KA_avg) ⋅ R_n
    return inv(R_new)
end
