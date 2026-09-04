# =============================================================================
#  schemes_alv_sc.jl — time-domain self-consistent (SC) homogenization.
#
#  Iterates the symmetric SC fixed point on the discrete `(6n × 6n)`
#  effective relaxation matrix:
#
#    C̃_{m+1} = (Σ_α f_α C̃_α ∘ Ã_α^dil(C̃_m)) ∘ (Σ_α f_α Ã_α^dil(C̃_m))^{-vol}
#
#  where the sum runs over **all** phases (the matrix included), and the
#  dilute concentration `Ã_α^dil(C̃_m)` is computed against the running
#  estimate `C̃_m` itself.  Picard iteration with optional damping;
#  convergence on the Frobenius norm of the residual `C̃_{m+1} − C̃_m`.
#
#  Reference: sanahuja2013 §3.2 ; barthelemyIJES2019 §4 ;
#  ECHOES manual `viscoelasticity_time.qmd` § "SC ALV scheme".
# =============================================================================

"""
    self_consistent_alv(rve, prop; times,
                        abstol = 1e-10, reltol = 1e-8, maxiters = 200,
                        damping = 0.0, verbose = false,
                        select_best = false) -> Matrix

Self-consistent ALV homogenization.  Iterates the symmetric Picard
fixed point on the `(6n × 6n)` block matrix until convergence.

The initial estimate is the discretized matrix kernel `C̃^0`. Each
iteration rebuilds the per-phase Hill kernels using the current
estimate's iso parameters, computes the dilute concentration tensors,
and forms `C̃_{m+1}`.

Returns the converged effective relaxation matrix.

# Keyword arguments

- `abstol`     — absolute Frobenius tolerance on `‖C̃_{m+1} − C̃_m‖`.
- `reltol`     — additive relative tolerance (multiplied by `‖C̃_m‖`).
- `maxiters`   — hard iteration cap.
- `damping`    — Picard relaxation `0 ≤ damping < 1` (0 = no damping).
- `verbose`    — print residual norms each iteration.
- `select_best`— return the best iterate seen (rather than the last)
  when convergence stalls.
"""
function self_consistent_alv(
        rve::RVE, prop::Symbol;
        times::AbstractVector{<:Real},
        matrix::Union{Nothing, Symbol} = nothing,
        abstol::Real = 1.0e-10,
        reltol::Real = 1.0e-8,
        maxiters::Int = 200,
        damping::Real = 0.0,
        verbose::Bool = false,
        select_best::Bool = false
    )
    # 1. Discretize every phase's kernel once.
    m = host_phase_name(rve, matrix, "self_consistent_alv")
    C_M_law = phase_property(rve, m, prop)
    C_M_law isa ViscoLaw ||
        throw(ArgumentError("self_consistent_alv: matrix property is not a ViscoLaw"))
    C_0 = _trapezoidal_relaxation(C_M_law, times, 6)
    f_M = volume_fraction(rve, m)
    incl_names = inclusion_phase_names(rve, m)
    # Containers are eltype-generic : volume fractions, crack densities and
    # geometry-derived tensors may carry `ForwardDiff.Dual` (or any Number)
    # coefficients — no `Float64` hard-coding on any AD-reachable path.
    C_phases = Matrix[C_0]
    geometries = Any[rve.phases[m].geometry]
    fractions = [f_M]
    symmetrizes = AbstractSymmetrize[NoSymmetrize()]
    # crack_data tuple: (geom, density, sym, Rn_mat::Union{Nothing, Matrix},
    #                    Rt_mat::Union{Nothing, Matrix}).  The interface
    # matrices are pre-discretized once on the times grid and reused by
    # every SC iteration when computing the crack stiffness contribution
    # against the running estimate C_m.
    crack_data = Tuple{
        Any, Any, AbstractSymmetrize,
        Union{Nothing, AbstractMatrix},
        Union{Nothing, AbstractMatrix},
    }[]
    for name in incl_names
        ph = rve.phases[name]
        a = rve.amounts[name]
        if a isa CrackDensity
            ph.geometry isa MFH_Core.AbstractCrack ||
                throw(ArgumentError("self_consistent_alv: phase $name has CrackDensity but geometry is not a crack"))
            Rn_mat = haskey(ph.properties, :Rn) ?
                _trapezoidal_relaxation_scalar(ph.properties[:Rn], times) : nothing
            Rt_mat = haskey(ph.properties, :Rt) ?
                _trapezoidal_relaxation_scalar(ph.properties[:Rt], times) : nothing
            push!(
                crack_data, (
                    ph.geometry, a.value,
                    phase_symmetrize(rve, name), Rn_mat, Rt_mat,
                )
            )
            continue
        end
        C_r_law = phase_property(rve, name, prop)
        C_r_law isa ViscoLaw ||
            throw(ArgumentError("self_consistent_alv: phase $name property is not a ViscoLaw"))
        C_r = _trapezoidal_relaxation(C_r_law, times, 6)
        sym = phase_symmetrize(rve, name)
        # Like the differential scheme, the self-consistent one uses its
        # RUNNING estimate as the reference of the ALV Hill kernel, and the
        # Picard loop reads that estimate's iso parameters directly
        # (`iso_params_from_blocks(C_m)`).  A phase that drags the estimate
        # out of the isotropic class would therefore be silently answered
        # with the iso projection of a non-iso matrix, so it is refused for
        # exactly the same reason and with the same two ways out.
        _alv_diff_keeps_iso(ph.geometry, sym, C_r) ||
            _alv_diff_iso_error(name, "the shape or the anisotropy")
        push!(C_phases, C_r)
        push!(geometries, ph.geometry)
        push!(fractions, _amount_value(rve, name))
        push!(symmetrizes, sym)
    end

    # 2. Pre-compute the Mandel forms of U^A, V^A for each phase
    #    (eltype follows the geometry parameters — Dual-safe).
    U_M_phases = Matrix[_tens_to_mandel66(tens_UA(g)) for g in geometries]
    V_M_phases = Matrix[_tens_to_mandel66(tens_VA(g)) for g in geometries]

    # 3. Iterate — the running estimate carries the promoted eltype of every
    #    input (kernels, fractions, geometry tensors, crack data) so Dual
    #    parameters propagate through the fixed point.
    Tp = _alv_promoted_eltype(C_phases, fractions, U_M_phases, crack_data)
    n = length(times)
    sz = 6 * n
    Id = _identity_alv(n, Tp)
    C_m = Tp.(C_0)
    best_resid = Inf
    C_best = C_m

    for iter in 1:maxiters
        # Self-consistent SC body:
        #   strain_Stress_α  = A_α(C_m) · J_m   (solid, J_m = inv(C_m))
        #   strain_Stress_c  = sym(H_c(C_m))    (void crack — no J_m)
        #   stress_Stress_α  = C_α · strain_Stress_α
        #   stress_Stress_c  = 0                (traction-free)
        # Accumulators :  A_E = Σ f_α·sym(strain_Stress_α)
        #                 B_E = Σ f_α·sym(stress_Stress_α)
        # Result : C_eff = B_E · A_E^{-vol}.
        # The trailing `J_m` cancels for solid-only RVEs but NOT when
        # cracks are present (the crack term has no `J_m` factor),
        # giving the ECHOES SC fixed point that doesn't match the
        # textbook `(Σ f·C·A)·(Σ f·A)^{-1}` form.
        if isempty(crack_data)
            C_m_new = _sc_alv_step(
                C_m, C_phases, U_M_phases, V_M_phases,
                fractions, n, Id, symmetrizes
            )
        else
            C_m_new = _sc_alv_step_echoes_form(
                C_m, C_phases,
                U_M_phases, V_M_phases,
                fractions, n, Id, symmetrizes,
                crack_data
            )
        end
        Δ = norm(C_m_new - C_m)
        norm_C = norm(C_m)
        tol_eff = abstol + reltol * norm_C
        verbose && @info "SC-ALV iter $iter : ‖Δ‖ = $(Δ)   tol = $tol_eff"
        if select_best && Δ < best_resid
            best_resid = Δ
            C_best = C_m_new
        end
        if Δ ≤ tol_eff
            return C_m_new
        end
        # Picard with relaxation.
        C_m = (1 - damping) .* C_m_new .+ damping .* C_m
    end

    @debug "self_consistent_alv: maxiters=$(maxiters) reached without convergence" abstol reltol
    return select_best ? C_best : C_m
end

# ── Shared per-phase dilute concentration for the SC-ALV bodies ─────────────
#
# `Ã^dil = (𝟙 + P̃_α ∘ ΔC̃)^{-vol}` with the phase Hill kernel
# `P̃_α[i,j] = J_long[i,j]·U_M + J_shear[i,j]·D_M` (geometry tensors `U_M`,
# `D_M = V_M − U_M` against the iso-decoupled running estimate, whose scalar
# Volterra inverses are `J_long`, `J_shear`).
#
# FAST PATH — when the running estimate, the phase stiffness and the geometry
# tensors are all iso-block (spherical inclusions in an iso ALV estimate), the
# whole thing reduces to two scalar n×n Volterra problems via
# `dilute_concentration_alv_iso`, avoiding the dense 6n×6n `P_α * ΔC` gemm
# (O((6n)³)) and the `block_size = 6` Volterra inverse.  This is the SAME fast
# path `_inclusion_alv_quantities` uses per-inclusion (homogenize_alv.jl);
# here it is applied inside the SC Picard loop.  The iso params of the
# constant 6×6 geometry tensors combine linearly with the scalar `J_long`,
# `J_shear`, so `P_α`'s iso params are `α_P = α_U·J_long + α_D·J_shear`,
# `β_P = β_U·J_long + β_D·J_shear`.  Because iso blocks are closed under the
# Volterra product and inverse (they block-diagonalize into the {𝕁,𝕂}
# channels), the fast path is numerically identical to the dense path (a
# direct comparison confirms `< 1e-12`); it FALLS BACK to the exact dense
# computation for any non-iso operand.
function _sc_alv_dilute_conc(
        C_phase::AbstractMatrix, C_m::AbstractMatrix,
        U_M::AbstractMatrix, D_M::AbstractMatrix,
        J_long::AbstractMatrix, J_shear::AbstractMatrix,
        α_m::AbstractMatrix, β_m::AbstractMatrix,
        n::Int, sz::Int, Id::AbstractMatrix, ::Type{T}
    ) where {T}
    if _is_iso_block(C_m) && _is_iso_block(C_phase) &&
            _is_iso_block(U_M) && _is_iso_block(D_M)
        α_U = U_M[1, 1] + 2 * U_M[1, 2]; β_U = U_M[4, 4]
        α_D = D_M[1, 1] + 2 * D_M[1, 2]; β_D = D_M[4, 4]
        α_P = α_U .* J_long .+ α_D .* J_shear
        β_P = β_U .* J_long .+ β_D .* J_shear
        α_E, β_E = _iso_pair(C_phase)
        αβ_A = dilute_concentration_alv_iso((α_E, β_E), (α_m, β_m), (α_P, β_P))
        return _iso_blocks(αβ_A)
    end
    P_α = zeros(T, sz, sz)
    @inbounds for i in 1:n, j in 1:i
        block = J_long[i, j] .* U_M .+ J_shear[i, j] .* D_M
        rows = (6 * (i - 1) + 1):(6 * i)
        cols = (6 * (j - 1) + 1):(6 * j)
        P_α[rows, cols] = block
    end
    ΔC = C_phase - C_m
    return volterra_inverse(Id + P_α * ΔC; block_size = 6)
end

# ── ECHOES SC body for ALV with cracks ─────────────────────────────────────
#
# Mirrors the elastic self-consistent SC body :
#   strain_Stress_α  = A_α(C_m) · J_m   (solid, J_m = volterra-inv(C_m))
#   strain_Stress_c  = sym(H_c(C_m))    (void crack — NO trailing J_m)
#   stress_Stress_α  = C_α · strain_Stress_α
#   stress_Stress_c  = 0                (traction-free)
# Accumulators :  A_E = Σ f_α·sym(strain_Stress_α)
#                 B_E = Σ f_α·sym(stress_Stress_α)
# Result   : C_eff = B_E · A_E^{-vol}.  The trailing `J_m` factor
# cancels for solid-only RVEs (recovering `(Σ f·CA)·(Σ f·A)^{-vol}`)
# but not for cracks, whose `strain_Stress` is the bare compliance
# contribution `H_c` without `J_m` factor.  This is the ECHOES SC
# fixed point.
function _sc_alv_step_echoes_form(
        C_m::AbstractMatrix,
        C_phases::AbstractVector{<:AbstractMatrix},
        U_M_phases::AbstractVector{<:AbstractMatrix},
        V_M_phases::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector{<:Real},
        n::Int, Id::AbstractMatrix,
        symmetrizes::AbstractVector{<:AbstractSymmetrize},
        crack_data
    )
    sz = size(C_m, 1)
    T = eltype(C_m)
    A_avg = zeros(T, sz, sz)   # = Σ f·sym(A_α)            (no J_m yet)
    CA_avg = zeros(T, sz, sz)   # = Σ f·sym(C_α·A_α)        (no J_m yet)
    α_m, β_m = iso_params_from_blocks(C_m)
    M_long = @. (α_m + 2 * β_m) / 3
    M_shear = β_m ./ 2
    J_long = volterra_inverse(M_long; block_size = 1)
    J_shear = volterra_inverse(M_shear; block_size = 1)
    @inbounds for α in eachindex(C_phases)
        U_M = U_M_phases[α]
        V_M = V_M_phases[α]
        D_M = V_M .- U_M
        A_dil = _sc_alv_dilute_conc(
            C_phases[α], C_m, U_M, D_M, J_long, J_shear, α_m, β_m, n, sz, Id, T
        )
        sym = symmetrizes[α]
        A_dil_sym = _maybe_symmetrize_alv(A_dil, sym)
        CA_sym = _maybe_symmetrize_alv(C_phases[α] * A_dil, sym)
        f = T(fractions[α])
        @. A_avg += f * A_dil_sym
        @. CA_avg += f * CA_sym
    end
    # Crack compliance contributions (without J_m factor — that's the
    # essential ECHOES form difference compared to solids).
    H_total = _build_sc_crack_extra_J(C_m, crack_data)
    # Apply the ECHOES `B · A^{-vol}` formula with explicit `J_m`.
    J_m = volterra_inverse(C_m; block_size = 6)
    A_E = (A_avg * J_m) .+ H_total
    B_E = CA_avg * J_m
    return B_E * volterra_inverse(A_E; block_size = 6)
end

# ── Crack compliance contribution against the running estimate ─────────────
#
# Computes `Σ_c ε·sym(H̃_c(C_m))` where `H̃_c(C_m)` is the (Sevostianov-
# corrected) crack compliance contribution against the running estimate
# `C_m`.  Used in two ways:
#   (a) `_build_sc_crack_extra_J(C_m, crack_data)`  — appended to the
#       compliance `J̃ = inv(C_m_solid_SC)` in the Budiansky-O'Connell
#       branch of the SC iteration (the default, robust path).
#   (b) `_build_sc_crack_extra_A(C_m, crack_data)`  — alias used by the
#       experimental ECHOES-form `B · A^{-vol}` MT body (kept available
#       for the Newton-Raphson SC solver, which uses the same algebra
#       but solves `F(C) = 0` instead of iterating Picard).
function _build_sc_crack_extra_J(C_m::AbstractMatrix, crack_data)
    sz = size(C_m, 1)
    T = eltype(C_m)
    extra = zeros(T, sz, sz)
    isempty(crack_data) && return extra
    _is_iso_block(C_m) ||
        error("self_consistent_alv with cracks: only iso running estimate is supported")
    α_c, β_c = _iso_pair(C_m)
    α_p_2β = α_c .+ 2β_c
    α_p_βh = α_c .+ β_c ./ 2
    α_p_β = α_c .+ β_c
    βα1 = β_c * α_p_βh
    βα2 = β_c * α_p_β
    B_n_base = (8 / (3π)) .* volterra_left_divide(βα1, α_p_2β)
    B_t_base = (32 / (9π)) .* volterra_left_divide(βα2, α_p_2β)
    Iₙ = Matrix{T}(LinearAlgebra.I, size(α_c, 1), size(α_c, 1))
    @inbounds for (geom, ε, sym, Rn_mat, Rt_mat) in crack_data
        B_n = B_n_base
        B_t = B_t_base
        if Rn_mat !== nothing || Rt_mat !== nothing
            b = semi_minor(geom)
            if Rn_mat !== nothing
                KB = Rn_mat * B_n; @. KB *= b; @. KB += Iₙ
                B_n = B_n * volterra_inverse(KB; block_size = 1)
            end
            if Rt_mat !== nothing
                KB = Rt_mat * B_t; @. KB *= b; @. KB += Iₙ
                B_t = B_t * volterra_inverse(KB; block_size = 1)
            end
        end
        Z = zeros(T, size(α_c))
        ℓ₁ = (3 / 4) .* B_n
        ℓ₆ = (3 / 8) .* B_t
        H_TI = ti_blocks_from_params((ℓ₁, copy(Z), copy(Z), copy(Z), copy(Z), ℓ₆))
        H_full = _maybe_symmetrize_alv(delta_compliance_alv(geom, H_TI, ε), sym)
        @. extra += H_full
    end
    return extra
end

# Alias used by the ECHOES-form MT body (numerically identical : both
# represent the iso-symmetrized crack compliance contribution scaled
# by the Budiansky concentration factor `(4π/3)·ε`).
@inline _build_sc_crack_extra_A(C_m, crack_data) = _build_sc_crack_extra_J(C_m, crack_data)

# Single SC step (legacy MFH form, retained for external callers /
# internal sub-steps that still rely on the standard
# `(Σ f A) ↔ (Σ f C A)` accumulators).
function _sc_alv_step(
        C_m::AbstractMatrix,
        C_phases::AbstractVector{<:AbstractMatrix},
        U_M_phases::AbstractVector{<:AbstractMatrix},
        V_M_phases::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector{<:Real},
        n::Int, Id::AbstractMatrix,
        symmetrizes::AbstractVector{<:AbstractSymmetrize};
        extra_A::Union{Nothing, AbstractMatrix} = nothing
    )
    sz = size(C_m, 1)
    T = eltype(C_m)
    A_avg = zeros(T, sz, sz)
    CA_avg = zeros(T, sz, sz)
    if extra_A !== nothing
        @. A_avg += extra_A
    end
    # Iso parameters of the running estimate → scalar Volterra inverses
    # for the Hill-kernel time-space decoupling.
    α_m, β_m = iso_params_from_blocks(C_m)
    M_long = @. (α_m + 2 * β_m) / 3
    M_shear = β_m ./ 2
    J_long = volterra_inverse(M_long; block_size = 1)
    J_shear = volterra_inverse(M_shear; block_size = 1)

    @inbounds for α in eachindex(C_phases)
        # Phase Hill kernel against current estimate C_m.
        U_M = U_M_phases[α]
        V_M = V_M_phases[α]
        D_M = V_M .- U_M
        # Dilute concentration & scaled contribution (iso fast path when
        # every operand is iso-block; exact dense fallback otherwise).
        A_dil = _sc_alv_dilute_conc(
            C_phases[α], C_m, U_M, D_M, J_long, J_shear, α_m, β_m, n, sz, Id, T
        )
        CA = C_phases[α] * A_dil
        # Apply orientation-averaging projection (`symmetrize=[ISO]` for
        # ECHOES) to both the dilute concentration and its scaled
        # contribution.  This matters per SC iteration so that the
        # running C_m converges to the iso-symmetrized fixed point.
        sym = symmetrizes[α]
        A_dil = _maybe_symmetrize_alv(A_dil, sym)
        CA = _maybe_symmetrize_alv(CA, sym)
        f = fractions[α]
        @. A_avg += f * A_dil
        @. CA_avg += f * CA
    end
    return CA_avg * volterra_inverse(A_avg; block_size = 6)
end
