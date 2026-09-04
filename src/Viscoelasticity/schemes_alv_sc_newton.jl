# =============================================================================
#  schemes_alv_sc_newton.jl — row-by-row Newton-Raphson for ALV SC.
#
#  The ALV SC equation `C_eff = step(C_eff)` is causal: the step at row i
#  depends on the input at rows ≤ i only.  This makes the Jacobian
#  block-lower-triangular (in row-block partition) and lets us **march**
#  row by row : with rows 1..i-1 already at the SC fixed point, the
#  residual at row i is a 2i-D nonlinear system in
#     `p_i = (α(t_i, t_1..t_i), β(t_i, t_1..t_i))`
#  that we solve by Newton-Raphson with `ForwardDiff.jacobian` and a
#  backtracking line search.
#
#  This avoids the multiple-fixed-point pitfall of vanilla Picard on the
#  ECHOES SC body (which from `C_m = C_M` converges to a different
#  basin than ECHOES does at finite crack density), and matches ECHOES
#  SC numerically while keeping the same `B · A^{-vol}` body equation.
# =============================================================================

"""
    self_consistent_alv_newton(rve, prop; times,
                                abstol = 1e-10, reltol = 1e-8,
                                maxiters_per_row = 30, verbose = false)

Row-by-row Newton-Raphson SC for ALV with iso phases.  Solves
`C_eff = ECHOES_step(C_eff)` (the `B · A^{-vol}` body) by marching
through the time grid, solving a 2i-dimensional Newton problem at each
row `i` with `ForwardDiff.jacobian` and a backtracking Armijo line
search.

# Arguments

  * `rve`               — RVE with iso matrix and iso (or crack) phases.
  * `prop`              — property symbol (`:C`).
  * `times`             — increasing time grid.

# Keyword arguments

  * `abstol`, `reltol`  — convergence on `‖F_i‖`.
  * `maxiters_per_row`  — Newton iteration cap per row.
  * `verbose`           — log per-row residuals.

# Notes

Currently the iso path only.  For non-iso phases (TI, ortho), use
[`self_consistent_alv`](@ref) (Anderson-Picard with optional damping).
"""
function self_consistent_alv_newton(
        rve::RVE, prop::Symbol;
        times::AbstractVector{<:Real},
        matrix::Union{Nothing, Symbol} = nothing,
        abstol::Real = 1.0e-10,
        reltol::Real = 1.0e-8,
        maxiters_per_row::Int = 30,
        verbose::Bool = false
    )
    m = host_phase_name(rve, matrix, "self_consistent_alv_newton")
    C_M_law = phase_property(rve, m, prop)
    C_M_law isa ViscoLaw ||
        throw(ArgumentError("self_consistent_alv_newton: matrix property is not a ViscoLaw"))

    n = length(times)
    f_M = volume_fraction(rve, m)
    incl_names = inclusion_phase_names(rve, m)

    C_M_full = _trapezoidal_relaxation(C_M_law, times, 6)

    # Eltype-generic containers (`ForwardDiff.Dual` moduli, fractions or
    # geometry parameters must flow through — no `Float64` hard-coding).
    C_phases_full = Matrix[C_M_full]
    geometries = Any[rve.phases[m].geometry]
    fractions = [f_M]
    symmetrizes = AbstractSymmetrize[NoSymmetrize()]

    crack_data_full = Tuple{
        Any, Any, AbstractSymmetrize,
        Union{Nothing, AbstractMatrix},
        Union{Nothing, AbstractMatrix},
    }[]

    for name in incl_names
        ph = rve.phases[name]
        a = rve.amounts[name]
        if a isa CrackDensity
            ph.geometry isa MFH_Core.AbstractCrack ||
                throw(ArgumentError("self_consistent_alv_newton: phase $name has CrackDensity but geometry is not a crack"))
            Rn_mat = haskey(ph.properties, :Rn) ?
                _trapezoidal_relaxation_scalar(ph.properties[:Rn], times) : nothing
            Rt_mat = haskey(ph.properties, :Rt) ?
                _trapezoidal_relaxation_scalar(ph.properties[:Rt], times) : nothing
            push!(
                crack_data_full, (
                    ph.geometry, a.value,
                    phase_symmetrize(rve, name), Rn_mat, Rt_mat,
                )
            )
            continue
        end
        C_r_law = phase_property(rve, name, prop)
        C_r_law isa ViscoLaw ||
            throw(ArgumentError("self_consistent_alv_newton: phase $name property is not a ViscoLaw"))
        push!(C_phases_full, _trapezoidal_relaxation(C_r_law, times, 6))
        push!(geometries, ph.geometry)
        push!(fractions, _amount_value(rve, name))
        push!(symmetrizes, phase_symmetrize(rve, name))
    end

    U_M_phases = Matrix[_tens_to_mandel66(tens_UA(g)) for g in geometries]
    V_M_phases = Matrix[_tens_to_mandel66(tens_VA(g)) for g in geometries]

    # Promoted scalar type of every input the Newton state must carry
    # (matrix/phase kernels, fractions, geometry tensors, crack data) so
    # outer `ForwardDiff.Dual` parameters propagate through the root-find.
    Tp = promote_type(
        eltype(C_M_full), eltype(fractions),
        (eltype(C) for C in C_phases_full)...,
        (eltype(U) for U in U_M_phases)...,
    )
    for (_, ε, _, Rn, Rt) in crack_data_full
        Tp = promote_type(Tp, typeof(ε))
        Rn === nothing || (Tp = promote_type(Tp, eltype(Rn)))
        Rt === nothing || (Tp = promote_type(Tp, eltype(Rt)))
    end

    # State : lower-triangular (α[k,l], β[k,l]) for l ≤ k ≤ n.
    # Initial guess from the matrix kernel — the SC root is the SC
    # extension of `C_M` driven by the inclusions, so `C_M` is the
    # natural starting point at every time row.
    α_M, β_M = iso_params_from_blocks(C_M_full)
    α_full = Tp.(α_M)
    β_full = Tp.(β_M)

    for i in 1:n
        # Slice phase data to the (1..i) sub-grid.
        C_phases_i = [view(C, 1:6i, 1:6i) for C in C_phases_full]
        crack_data_i = [
            (
                g, ε, sym,
                Rn === nothing ? nothing : Rn[1:i, 1:i],
                Rt === nothing ? nothing : Rt[1:i, 1:i],
            )
                for (g, ε, sym, Rn, Rt) in crack_data_full
        ]
        Id_i = _identity_alv(i, Tp)

        residual_row_i = function (p)
            T = eltype(p)
            α_local = zeros(T, i, i)
            β_local = zeros(T, i, i)
            @inbounds for k in 1:(i - 1), l in 1:k
                α_local[k, l] = T(α_full[k, l])
                β_local[k, l] = T(β_full[k, l])
            end
            @inbounds for j in 1:i
                α_local[i, j] = p[j]
                β_local[i, j] = p[i + j]
            end
            C_m_i = iso_blocks_from_params(α_local, β_local)
            extra_A_i = isempty(crack_data_i) ? nothing :
                _build_sc_crack_extra_J(C_m_i, crack_data_i)
            # `_sc_alv_step` is dispatched on `Id` element type, so we
            # promote the identity to the input element type to keep
            # the pipeline `Dual`-friendly.
            Id_T = T === Tp ? Id_i : T.(Id_i)
            C_m_new_i = _sc_alv_step(
                C_m_i, C_phases_i, U_M_phases, V_M_phases,
                fractions, i, Id_T, symmetrizes;
                extra_A = extra_A_i
            )
            α_new, β_new = iso_params_from_blocks(C_m_new_i)
            r = Vector{T}(undef, 2i)
            @inbounds for j in 1:i
                r[j] = α_new[i, j] - p[j]
                r[i + j] = β_new[i, j] - p[i + j]
            end
            return r
        end

        # Initial guess at row i : reuse the matrix-driven values
        # (or the previously-extrapolated row if i > 1).
        p = Tp[α_full[i, j] for j in 1:i]
        append!(p, Tp[β_full[i, j] for j in 1:i])

        for iter in 1:maxiters_per_row
            r = residual_row_i(p)
            norm_r = sqrt(sum(abs2, r))
            tol_eff = abstol + reltol * sqrt(sum(abs2, p))
            verbose && @info "  SC-Newton row $i iter $iter : ‖F‖ = $norm_r"
            if norm_r ≤ tol_eff
                break
            end
            J = ForwardDiff.jacobian(residual_row_i, p)
            δ = J \ -r
            # Backtracking line search.
            α_step = 1.0
            accepted = false
            for _ in 1:30
                p_new = p .+ α_step .* δ
                r_new = residual_row_i(p_new)
                if sqrt(sum(abs2, r_new)) ≤ (1 - 1.0e-4 * α_step) * norm_r
                    p .= p_new
                    accepted = true
                    break
                end
                α_step /= 2
                α_step < 1.0e-8 && break
            end
            if !accepted
                # Fallback : pure Picard step on row i.
                r_new = residual_row_i(p)
                p .+= r_new   # since residual = step(p) - p, p ← p + r = step(p)
            end
        end

        @inbounds for j in 1:i
            α_full[i, j] = p[j]
            β_full[i, j] = p[i + j]
        end
    end

    return iso_blocks_from_params(α_full, β_full)
end
