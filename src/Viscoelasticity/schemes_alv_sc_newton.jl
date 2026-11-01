# =============================================================================
#  schemes_alv_sc_newton.jl — row-by-row Newton-Raphson for ALV SC.
#
#  The ALV SC equation `X = step(X)` is causal: the step at row i depends on
#  the input at rows ≤ i only.  This makes the Jacobian block-lower-triangular
#  (in row-block partition) and lets us **march** row by row: with rows
#  1..i-1 already at the SC fixed point, the residual at row i is a small
#  nonlinear system in the row-i parameters of the isotropic estimate,
#     order 4: `(α(t_i, t_1..t_i), β(t_i, t_1..t_i))`  (3k and 2μ, 2i unknowns),
#     order 2: `λ(t_i, t_1..t_i)`                       (the conductivity, i unknowns),
#  solved by Newton-Raphson with `ForwardDiff.jacobian`, the line search and
#  the fallbacks of the elastic Newton solver (`Schemes._sc_newton_update!`).
#
#  It solves the same equation as the Picard iteration of
#  `homogenize_alv(…, SelfConsistent())` (the step is the same function), and
#  reaches the same estimate where the equation has one physical root.  Near
#  percolation the equation has several roots: the march starts from the
#  Picard estimate, which reaches the physical one, and a root with a negative
#  instantaneous modulus is refused.  An iterative solve: no symbolic route.
# =============================================================================

const _SC_NEWTON_WHO = "self_consistent_alv_newton"

"""
    self_consistent_alv_newton(rve, prop; times, matrix = nothing,
                               abstol = 1e-10, reltol = 1e-8,
                               maxiters_per_row = 30, init = :picard,
                               verbose = false) -> Matrix

Row-by-row Newton-Raphson solver of the ALV self-consistent equation, the equation
the Picard iteration of [`homogenize_alv`](@ref) with [`SelfConsistent`](@ref) solves,
for an elastic property (`:C`, order 4) or a conductivity (`:K`, order 2). Reached
through `homogenize_alv(rve, SelfConsistent(algorithm = NewtonDefault()), prop; times)`.

The equation is causal: the estimate at instant ``t_i`` depends on the estimate at
the instants ``t_j \\le t_i`` only. The solver therefore marches through the time
grid and solves, at each instant ``t_i``, the small nonlinear system whose unknowns
are the row ``i`` of the isotropic estimate (``2i`` unknowns at order 4, the bulk
and shear kernels; ``i`` at order 2), the rows before it being fixed. Each system is
solved by Newton-Raphson with a `ForwardDiff` Jacobian, a backtracking Armijo line
search and a Picard step when the line search fails.

The phases are those of the Picard iteration: phases with a Hill kernel, layered
spheres, and, at order 4, crack families. Each must keep the estimate isotropic: a
sphere of isotropic behavior, a layered sphere, or an isotropic orientation average
(`symmetrize = :iso`), which a crack family always needs.

# Keyword arguments

  * `abstol`, `reltol`  — convergence of each row, ``\\|F_i\\| \\le`` `abstol` ``+`` `reltol` ``\\|p_i\\|``.
  * `maxiters_per_row`  — Newton iteration cap per row.
  * `init`              — starting point of the march: `:picard` (default), the
                          estimate of the Picard iteration loosely converged, or
                          `:matrix`, the matrix kernel.
  * `verbose`           — log per-row residuals.

# Several roots near percolation

The self-consistent equation may have several roots near percolation (a high
fraction of stiff or of compliant phases, or a high crack density). Marching from
the matrix kernel (`init = :matrix`), the Newton solver may then reach a non-physical
root, with a negative stiffness beyond the percolation of a stiff phase; started from
the Picard estimate, it refines the root the Picard iteration reaches, which is the
elastic one in the elastic limit. A root with a negative instantaneous modulus is
refused, and a row that does not converge within `maxiters_per_row` iterations is
reported with a warning.
"""
function self_consistent_alv_newton(
        rve::RVE, prop::Symbol;
        times::AbstractVector{<:Real}, matrix::Union{Nothing, Symbol} = nothing, kw...
    )
    m = host_phase_name(rve, matrix, _SC_NEWTON_WHO)
    law = phase_property(rve, m, prop)
    law isa ALVProperty ||
        throw(ArgumentError("$(_SC_NEWTON_WHO): matrix property is neither a viscoelastic law, a relaxation operator nor an elastic tensor"))
    return _alv_newton(Val(:sc), _alv_order(law, times), rve, prop; times, matrix = m, kw...)
end

# The march of either equation (`kind`: `Val(:sc)`, the symmetric self-consistent one,
# `Val(:asc)`, the asymmetric one) and either order: the phases read as the Picard
# iteration reads them, the isotropic parameters of the matrix kernel, of the start, and
# the solved estimate.
function _alv_newton(
        kind::Val, order::Val, rve::RVE, prop::Symbol;
        times::AbstractVector{<:Real},
        matrix::Symbol,
        abstol::Real = 1.0e-10,
        reltol::Real = 1.0e-8,
        maxiters_per_row::Int = 30,
        init::Symbol = :picard,
        verbose::Bool = false
    )
    # A run-time choice of the starting point, read once.
    who = _alv_newton_who(kind)
    init in (:picard, :matrix) ||
        throw(ArgumentError("$who: init must be :picard or :matrix; got :$(init)"))
    data = _sc_newton_data(order, rve, prop, times, matrix, who)
    θ_M = _sc_newton_params(order, _sc_newton_matrix_kernel(order, data))
    # The parameters carry the promoted eltype of every input (kernels, fractions,
    # geometry tensors, crack data, layers), so that outer `ForwardDiff.Dual`
    # parameters propagate through the root-find.
    Tp = _sc_newton_eltype(order, data, times)
    θ = map(Θ -> Tp.(Θ), _sc_newton_start(Val(init), kind, order, rve, prop, times, matrix, θ_M))
    for i in eachindex(times)
        # A march: row i is solved with the rows before it fixed.
        F = _sc_newton_row_residual(kind, order, θ, data, times, i)
        p = reduce(vcat, (Θ[i, 1:i] for Θ in θ))
        _sc_newton_row!(p, F, i; abstol, reltol, maxiters = maxiters_per_row, verbose) ||
            @warn "$who: row $i not converged in $maxiters_per_row iterations"
        for (k, Θ) in enumerate(θ)
            Θ[i, 1:i] .= view(p, ((k - 1) * i + 1):(k * i))
        end
        _sc_newton_check_root(θ, θ_M, i, who)
    end
    return _sc_newton_blocks(order, θ)
end

# The name the messages of each equation carry.
_alv_newton_who(::Val{:sc}) = _SC_NEWTON_WHO
_alv_newton_who(::Val{:asc}) = "asymmetric_self_consistent_alv (Newton)"

# The phases of each order, on full matrices, the matrix first.
_sc_newton_data(::Val{4}, rve, prop, times, m, who) = _sc_alv_dense_data(_sc_alv_phases(rve, prop, times, m, who))
_sc_newton_data(::Val{2}, rve, prop, times, m, who) = _sc2_phases(rve, prop, times, m, who)

_sc_newton_matrix_kernel(::Val{4}, dense) = dense.C_0
_sc_newton_matrix_kernel(::Val{2}, (K_M, _, _)) = K_M

_sc_newton_eltype(::Val{4}, dense, times) = _sc_alv_dense_eltype(dense, times)
_sc_newton_eltype(::Val{2}, (K_M, data, cracks), times) = _sc2_eltype(K_M, data, cracks, times)

# The isotropic estimate and its parameters, lower-triangular n × n matrices.
_sc_newton_params(::Val{4}, C) = iso_params_from_blocks(C)
_sc_newton_params(::Val{2}, K) = (iso_order2_params_from_blocks(K),)
_sc_newton_blocks(::Val{4}, (α, β)) = iso_blocks_from_params(α, β)
_sc_newton_blocks(::Val{2}, (λ,)) = iso_order2_blocks_from_params(λ)

# The starting point of the Newton march: the Picard estimate of the same equation,
# loosely converged, or the matrix kernel.
_sc_newton_start(::Val{:matrix}, kind, order, rve, prop, times, m, θ_M) = θ_M
_sc_newton_start(::Val{:picard}, kind, order, rve, prop, times, m, θ_M) =
    _sc_newton_params(order, Matrix(_alv_picard(kind, order, rve, prop; times, matrix = m, reltol = 1.0e-6, maxiters = 500)))

# The residual of row i, p ↦ step(estimate with row i = p)[row i] − p, on the leading
# sub-grid t_1..t_i, which the causal step only reads.
function _sc_newton_row_residual(kind, order::Val{4}, θ, dense, times, i)
    sub = _sc_alv_leading(dense, i)
    t_i = times[1:i]
    return p -> begin
        C_new = _alv_dense_step(kind, _sc_newton_blocks(order, _sc_newton_with_row(θ, p, i)), sub, t_i)
        α_new, β_new = _sc_newton_params(order, C_new)
        [view(α_new, i, :); view(β_new, i, :)] .- p
    end
end
function _sc_newton_row_residual(kind, order::Val{2}, θ, (K_M, data, cracks), times, i)
    sub = [merge(d, (K_r = d.K_r[1:3i, 1:3i],)) for d in data]
    sub_cracks = [merge(c, (A_int = _leading_block(c.A_int, i),)) for c in cracks]
    K_M_i = K_M[1:3i, 1:3i]
    t_i = times[1:i]
    return p -> begin
        K_new = _alv2_dense_step(kind, _sc_newton_blocks(order, _sc_newton_with_row(θ, p, i)), K_M_i, sub, sub_cracks, t_i)
        (λ_new,) = _sc_newton_params(order, K_new)
        λ_new[i, :] .- p
    end
end

# The phases of the order-4 step restricted to the leading sub-grid t_1..t_i.
_sc_alv_leading(dense, i) = merge(
    dense, (
        C_0 = dense.C_0[1:6i, 1:6i],
        C_phases = [C[1:6i, 1:6i] for C in dense.C_phases],
        crack_data = [(g, ε, sym, _leading_block(Rn, i), _leading_block(Rt, i)) for (g, ε, sym, Rn, Rt) in dense.crack_data],
    )
)

# The step on full matrices of each equation.
_alv_dense_step(::Val{:sc}, C, dense, times) = _sc_alv_dense_step(C, dense, times)
_alv_dense_step(::Val{:asc}, C, dense, times) = _asc_alv_dense_step(C, dense, times)
_alv2_dense_step(::Val{:sc}, K, K_M, data, cracks, times) = _sc2_dense_step(K, data, cracks, times)
_alv2_dense_step(::Val{:asc}, K, K_M, data, cracks, times) = _asc2_dense_step(K, K_M, data, cracks, times)

# The leading i × i block of an interface stiffness, none for a traction-free direction.
_leading_block(::Nothing, i) = nothing
_leading_block(R::AbstractMatrix, i) = R[1:i, 1:i]

# The leading i × i blocks of the parameters, in the element type of `p`, whose
# row i is `p` (one chunk of length i per parameter matrix).
_sc_newton_with_row(θ, p, i) = ntuple(length(θ)) do k
    Θ = similar(p, i, i)
    Θ .= view(θ[k], 1:i, 1:i)
    Θ[i, :] .= view(p, ((k - 1) * i + 1):(k * i))
    Θ
end

# Newton iterations on one row, globalized as the elastic solver is; returns whether
# the row converged. The partials of a dual `p` are converged by one more step.
function _sc_newton_row!(p, F, i; abstol, reltol, maxiters, verbose)
    r = F(p)
    for iter in 1:maxiters
        norm_r = norm(r)
        verbose && @info "  SC-Newton row $i iter $iter : ‖F‖ = $norm_r"
        if norm_r ≤ abstol + reltol * norm(p)
            _sc_newton_converge_partials!(p, F, r)
            return true
        end
        J = ForwardDiff.jacobian(F, p)
        Schemes._sc_newton_update!(p, r, Schemes._sc_newton_direction(J, r), F, norm_r; verbose)
    end
    return norm(r) ≤ abstol + reltol * norm(p)
end

# The convergence test reads the values only: under Newton, once the value has
# converged, one full step gives the derivatives a dual carries.
_sc_newton_converge_partials!(p, F, r) = p
_sc_newton_converge_partials!(p::AbstractVector{<:ForwardDiff.Dual}, F, r) =
    (p .+= Schemes._sc_newton_direction(ForwardDiff.jacobian(F, p), r))

# A root whose instantaneous modulus (3k or 2μ at order 4, the conductivity at order
# 2) is negative is not physical: the self-consistent equation has several roots near
# percolation, and the march reached another one than the estimate (the scale of the
# test is the matrix's).
function _sc_newton_check_root(θ, θ_M, i, who)
    tol = sqrt(eps(float(real(eltype(first(θ_M))))))
    all(((Θ, Θ_M),) -> Θ[i, i] ≥ -tol * abs(Θ_M[i, i]), zip(θ, θ_M)) || throw(
        ErrorException(
            "$who: the root reached at instant $i has a negative instantaneous modulus " *
                "$(Tuple(Θ[i, i] for Θ in θ)): another root of the self-consistent equation than the physical one; " *
                "start from the Picard estimate (init = :picard) or use homogenize_alv"
        )
    )
    return nothing
end
