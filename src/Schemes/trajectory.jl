# =============================================================================
#  trajectory.jl — DifferentialTrajectory hierarchy and per-step
#  fraction-path resolution.
#
#  Each trajectory describes how the *target* per-phase volume fractions
#  (or crack densities) are reached cumulatively along the fictitious
#  incorporation time `τ ∈ [0, 1]` :
#
#       f_α(τ) ∈ [0, 1],  f_α(0) = 0,  f_α(1) = 1.
#
#  `_resolve_paths` returns, for every phase other than `matrix`, the pair
#  `(f, df)` of callables (`f(τ)` is the effective amount ratio,
#  `df(τ)` its derivative).  The differential ODE then assembles the
#  Norris RHS from `dφ_α/dτ` (Sherman-Morrison-inverted from the
#  per-phase `df_α/dτ`) for solids and `dε_α/dτ = df(τ) · ε^∞` for
#  cracks.
# =============================================================================

"""
    _resolve_paths(traj::DifferentialTrajectory, rve::RVE, nsteps::Int, matrix::Symbol)
        -> Dict{Symbol, NamedTuple{(:f, :df), Tuple{Function, Function}}}

Build per-phase callables `f_α(τ)` and `df_α(τ)` for each non-matrix
phase of `rve`.  Each `f_α : [0, 1] → [0, 1]` is monotone non-decreasing
with `f_α(0) = 0`, `f_α(1) = 1` ; `df_α` is the analytical derivative
(or `ForwardDiff.derivative` for [`Path`](@ref)).
"""
function _resolve_paths end

# ── Proportional ────────────────────────────────────────────────────────────

function _resolve_paths(::Proportional, rve::RVE, nsteps::Int, matrix::Symbol)
    out = Dict{Symbol, @NamedTuple{f::Function, df::Function}}()
    for name in inclusion_phase_names(rve, matrix)
        out[name] = (; f = τ -> τ, df = τ -> 1.0)
    end
    return out
end

# ── Sequential ──────────────────────────────────────────────────────────────

function _resolve_paths(s::Sequential, rve::RVE, nsteps::Int, matrix::Symbol)
    inc_names = inclusion_phase_names(rve, matrix)
    Set(s.order) == Set(inc_names) ||
        throw(
        ArgumentError(
            "Sequential.order = $(s.order) must list every " *
                "inclusion phase exactly once; got phases $(inc_names)"
        )
    )
    out = Dict{Symbol, @NamedTuple{f::Function, df::Function}}()
    n_phases = length(s.order)
    n_phases == 0 && return out
    Δτ = 1.0 / n_phases
    for (i, name) in enumerate(s.order)
        τ_lo = (i - 1) * Δτ
        τ_hi = i * Δτ
        # f_i ramps 0 → 1 over (τ_lo, τ_hi) and is otherwise saturated.
        f = let τ_lo = τ_lo, τ_hi = τ_hi, Δτ = Δτ
            τ -> τ ≤ τ_lo ? zero(τ) :
                τ ≥ τ_hi ? one(τ) :
                (τ - τ_lo) / Δτ
        end
        df = let τ_lo = τ_lo, τ_hi = τ_hi, Δτ = Δτ
            τ -> (τ_lo < τ < τ_hi) ? one(τ) / Δτ : zero(τ)
        end
        out[name] = (; f, df)
    end
    return out
end

# ── CustomPath ──────────────────────────────────────────────────────────────
#
# Discrete vector → piecewise-linear callable on `range(0, 1; length = N+1)`.

function _resolve_paths(c::CustomPath, rve::RVE, nsteps::Int, matrix::Symbol)
    inc_names = inclusion_phase_names(rve, matrix)
    out = Dict{Symbol, @NamedTuple{f::Function, df::Function}}()
    for name in inc_names
        haskey(c.path, name) ||
            throw(ArgumentError("CustomPath: missing trajectory for phase :$(name)"))
        traj = c.path[name]
        N = length(traj) - 1
        N ≥ 1 ||
            throw(ArgumentError("CustomPath[:$(name)] needs at least 2 entries"))
        traj[1] == 0 ||
            throw(ArgumentError("CustomPath[:$(name)][1] must be 0; got $(traj[1])"))
        traj[end] == 1 ||
            throw(ArgumentError("CustomPath[:$(name)][end] must be 1; got $(traj[end])"))
        for k in 1:N
            traj[k + 1] - traj[k] ≥ -1.0e-12 ||
                throw(ArgumentError("CustomPath[:$(name)] is not monotone non-decreasing"))
        end
        traj_f = collect(Float64.(traj))
        f = let traj_f = traj_f, N = N
            τ -> _piecewise_linear(traj_f, N, τ)
        end
        df = let traj_f = traj_f, N = N
            τ -> _piecewise_linear_deriv(traj_f, N, τ)
        end
        out[name] = (; f, df)
    end
    return out
end

# Helpers : piecewise linear interpolation on uniform grid `0, 1/N, ..., 1`.
function _piecewise_linear(traj::AbstractVector, N::Int, τ)
    τ ≤ 0 && return traj[1]
    τ ≥ 1 && return traj[end]
    s = τ * N
    k = floor(Int, s)        # 0 ≤ k ≤ N-1
    α = s - k
    return traj[k + 1] * (1 - α) + traj[k + 2] * α
end

function _piecewise_linear_deriv(traj::AbstractVector, N::Int, τ)
    (τ ≤ 0 || τ ≥ 1) && return zero(τ)
    s = τ * N
    k = floor(Int, s)
    return (traj[k + 2] - traj[k + 1]) * N
end

# ── Path (functional) ───────────────────────────────────────────────────────

function _resolve_paths(p::Path, rve::RVE, nsteps::Int, matrix::Symbol)
    inc_names = inclusion_phase_names(rve, matrix)
    out = Dict{Symbol, @NamedTuple{f::Function, df::Function}}()
    for name in inc_names
        haskey(p.path, name) ||
            throw(ArgumentError("Path: missing trajectory for phase :$(name)"))
        f_user = p.path[name]
        # Light validation : `f(0) ≈ 0`, `f(1) ≈ 1`, no NaN, monotone on
        # 32 sample points (full monotonicity of a black-box callable is
        # undecidable but a uniform sample catches obvious mistakes).
        _validate_functional_path(name, f_user)
        df_user = let f = f_user
            τ -> ForwardDiff.derivative(f, τ)
        end
        out[name] = (; f = f_user, df = df_user)
    end
    return out
end

function _validate_functional_path(name::Symbol, f::Function)
    # Endpoints
    f0 = f(0.0)
    f1 = f(1.0)
    isnan(f0) || isnan(f1) &&
        throw(ArgumentError("Path[:$(name)] returned NaN at τ ∈ {0, 1}"))
    isapprox(f0, 0.0; atol = 1.0e-10) ||
        throw(ArgumentError("Path[:$(name)](0) = $f0, expected 0"))
    isapprox(f1, 1.0; atol = 1.0e-10) ||
        throw(ArgumentError("Path[:$(name)](1) = $f1, expected 1"))
    # Sampled monotonicity : 32 uniform points.
    n = 32
    prev = f0
    for k in 1:n
        τ = k / n
        v = f(τ)
        isnan(v) &&
            throw(ArgumentError("Path[:$(name)] returned NaN at τ = $τ"))
        v ≥ prev - 1.0e-10 ||
            throw(
            ArgumentError(
                "Path[:$(name)] is not monotone non-decreasing " *
                    "(detected drop at τ = $τ : $prev → $v)"
            )
        )
        prev = v
    end
    return nothing
end
