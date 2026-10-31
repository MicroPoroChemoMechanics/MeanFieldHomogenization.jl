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

# A crack family as the SC schemes consume it: its geometry, its density, its
# orientation projection and the Volterra matrices of its optional normal and
# tangential interface laws (`:Rn`, `:Rt`).
function _alv_crack_record(rve, name, a::CrackDensity, times, who)
    ph = rve.phases[name]
    ph.geometry isa MFH_Core.AbstractCrack ||
        throw(ArgumentError("$who: phase $name has CrackDensity but geometry is not a crack"))
    Rn_mat = _trapezoidal_relaxation_scalar(get(ph.properties, :Rn, nothing), times)
    Rt_mat = _trapezoidal_relaxation_scalar(get(ph.properties, :Rt, nothing), times)
    return (ph.geometry, a.value, phase_symmetrize(rve, name), Rn_mat, Rt_mat)
end

"""
    self_consistent_alv(rve, prop; times,
                        abstol = 1e-10, reltol = 1e-8, maxiters = 200,
                        damping = 0.0, verbose = false,
                        select_best = false) -> Matrix

Self-consistent ALV homogenization.  Iterates the symmetric Picard
fixed point on the ``6n\\times 6n`` block matrix until convergence.

The initial estimate is the discretized matrix kernel ``\\widetilde{\\mathbb{C}}_0``. Each
iteration rebuilds the per-phase Hill kernels using the current
estimate's iso parameters, computes the dilute concentration tensors,
and forms ``\\widetilde{\\mathbb{C}}^{(m+1)}``.

Returns the converged effective relaxation matrix.

# Keyword arguments

- `abstol`     — absolute Frobenius tolerance on ``\\|\\widetilde{\\mathbb{C}}^{(m+1)} - \\widetilde{\\mathbb{C}}^{(m)}\\|``.
- `reltol`     — additive relative tolerance (multiplied by ``\\|\\widetilde{\\mathbb{C}}^{(m)}\\|``).
- `maxiters`   — hard iteration cap.
- `damping`    — Picard relaxation ``0 \\le`` `damping` ``< 1`` (0 = no damping).
- `verbose`    — print residual norms each iteration.
- `select_best`— return the best iterate seen (rather than the last)
  when convergence stalls.
"""
self_consistent_alv(rve::RVE, prop::Symbol; kw...) = Matrix(_self_consistent_alv(rve, prop; kw...))

# The same, returning a `VolterraOperator` when it iterates in the classes of the
# operators (what `homogenize_alv` keeps), a full matrix otherwise.
function _self_consistent_alv(
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
    C_M_law isa ALVProperty ||
        throw(ArgumentError("self_consistent_alv: matrix property is neither a viscoelastic law nor a relaxation operator"))
    grid = TimeGrid(times)
    C_0 = _relaxation_operator(C_M_law, grid)
    f_M = volume_fraction(rve, m)
    incl_names = inclusion_phase_names(rve, m)
    # Containers are eltype-generic : volume fractions, crack densities and
    # geometry-derived tensors may carry `ForwardDiff.Dual` (or any Number)
    # coefficients — no `Float64` hard-coding on any AD-reachable path.
    C_phases = AbstractMatrix[C_0]
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
    # Layered spheres have no Hill kernel: their concentration and stress come
    # from the recurrences, against the running estimate, at every iteration.
    layered_data = NamedTuple[]
    for name in incl_names
        ph = rve.phases[name]
        a = rve.amounts[name]
        # The phases of a cell are of many types, known at run time: a branch.
        if ph.geometry isa LayeredSphere
            push!(layered_data, (geom = ph.geometry, f = _amount_value(rve, name), sym = phase_symmetrize(rve, name)))
            continue
        end
        if _is_crack_family(a)
            push!(crack_data, _alv_crack_record(rve, name, a, times, "self_consistent_alv"))
            continue
        end
        C_r_law = phase_property(rve, name, prop)
        C_r_law isa ALVProperty ||
            throw(ArgumentError("self_consistent_alv: phase $name property is neither a viscoelastic law nor a relaxation operator"))
        C_r = _relaxation_operator(C_r_law, grid)
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

    # Without cracks or layered spheres, the iteration runs in the classes of the
    # operators; otherwise on full matrices.
    if isempty(crack_data) && isempty(layered_data)
        Tp = _alv_promoted_eltype(C_phases, fractions, Matrix{Float64}[], crack_data)
        Tp = mapreduce(g -> eltype(tens_UA(g)), promote_type, geometries; init = Tp)
        return _sc_alv_in_classes(
            C_0, C_phases, geometries, fractions, symmetrizes, Tp;
            abstol, reltol, maxiters, damping, verbose, select_best
        )
    end
    C_0 = Matrix(C_0)
    C_phases = Matrix[Matrix(C) for C in C_phases]

    # 2. Pre-compute the Mandel forms of U^A, V^A for each phase
    #    (eltype follows the geometry parameters — Dual-safe).
    U_M_phases = Matrix[TensND.mandel(TensND.canonical_components(tens_UA(g)); symmetrize = true) for g in geometries]
    V_M_phases = Matrix[TensND.mandel(TensND.canonical_components(tens_VA(g)); symmetrize = true) for g in geometries]

    # 3. Iterate — the running estimate carries the promoted eltype of every
    #    input (kernels, fractions, geometry tensors, crack data) so Dual
    #    parameters propagate through the fixed point.
    Tp = _alv_promoted_eltype(C_phases, fractions, U_M_phases, crack_data)
    Tp = _alv_layered_eltype(Tp, layered_data, C_0, times)
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
        extra_A, extra_CA = _sc_alv_layered_terms(layered_data, C_m, times)
        if isempty(crack_data)
            C_m_new = _sc_alv_step(
                C_m, C_phases, U_M_phases, V_M_phases,
                fractions, symmetrizes; extra_A, extra_CA
            )
        else
            C_m_new = _sc_alv_step_echoes_form(
                C_m, C_phases,
                U_M_phases, V_M_phases,
                fractions, symmetrizes,
                crack_data; extra_A, extra_CA
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

# ── The iteration in the classes of the operators ──────────────────────────
#
# The running estimate is isotropic (the guards above refuse a phase that would
# drag it out of the class): two n×n blocks. Each phase's Hill kernel is built in
# the class of its shape from the two scalar inverses of the estimate, computed
# once per iteration; the concentrations are averaged and divided in the block
# algebra, so that no 6n×6n matrix is written out.
function _sc_alv_in_classes(
        C_0, C_phases, geometries, fractions, symmetrizes, ::Type{Tp};
        abstol, reltol, maxiters, damping, verbose, select_best
    ) where {Tp}
    C_m = _as_iso_estimate(_with_eltype(Tp, C_0))
    best_resid = Inf
    C_best = C_m
    for iter in 1:maxiters
        C_new = _sc_alv_step_in_classes(C_m, C_phases, geometries, fractions, symmetrizes)
        Δ = norm(C_new - C_m)
        tol_eff = abstol + reltol * norm(C_m)
        verbose && @info "SC-ALV iter $iter : ‖Δ‖ = $(Δ)   tol = $tol_eff"
        if select_best && Δ < best_resid
            best_resid = Δ
            C_best = C_new
        end
        Δ ≤ tol_eff && return C_new
        C_m = (1 - damping) * C_new + damping * C_m
    end
    @debug "self_consistent_alv: maxiters=$(maxiters) reached without convergence" abstol reltol
    return select_best ? C_best : C_m
end

function _sc_alv_step_in_classes(C_m, C_phases, geometries, fractions, symmetrizes)
    J_long, J_shear = _alv_iso_inverses(C_m)
    terms = map(C_phases, geometries, fractions, symmetrizes) do C, g, f, sym
        P = _hill_kernel_op(g, J_long, J_shear, C_m.grid)
        A = inv(I + P * (C - C_m))
        CA = C * A
        f * _maybe_symmetrize_alv(A, sym), f * _maybe_symmetrize_alv(CA, sym)
    end
    A_avg = reduce(+, first.(terms))
    CA_avg = reduce(+, last.(terms))
    return _as_iso_estimate(CA_avg * inv(A_avg))
end

# The running estimate in the isotropic class: kept as is when it is there,
# converted when its class is wider, refused when its values are off it.
_as_iso_estimate(R::VolterraOperator{<:TensISO{4, 3}}) = R
_as_iso_estimate(R::VolterraOperator) = convert_class(R, _ISO4; check = true)

# The concentration `A` and the average stress `B = Ñ + C ∘ A` of a layered
# sphere against the running estimate `C`, from its recurrences: what a phase
# with a Hill kernel gets from its dilute concentration.
function _alv_layered_AB(s::LayeredSphere, C::AbstractMatrix, times)
    A = gradient_gradient_concentration_alv(s, C, times)
    return A, property_contribution_alv(s, C, times) .+ C * A
end

# Their volume-weighted, symmetrized sums over the layered phases.
function _sc_alv_layered_terms(layered_data, C_m::AbstractMatrix, times)
    isempty(layered_data) && return nothing, nothing
    A_sum = zero(C_m)
    B_sum = zero(C_m)
    for d in layered_data
        A, B = _alv_layered_AB(d.geom, C_m, times)
        A_sum .+= d.f .* _maybe_symmetrize_alv(A, d.sym)
        B_sum .+= d.f .* _maybe_symmetrize_alv(B, d.sym)
    end
    return A_sum, B_sum
end

# The element type a layered phase brings to the running estimate (a dual
# number in one of its layers, for instance), probed once against `C_0`.
function _alv_layered_eltype(Tp, layered_data, C_0, times)
    for d in layered_data
        Tp = promote_type(Tp, eltype(gradient_gradient_concentration_alv(d.geom, C_0, times)), typeof(d.f))
    end
    return Tp
end

# The Newton self-consistent body builds the Hill kernel of each phase from
# the shape of an ellipsoid and has no layered-sphere branch. Refused by name
# rather than through a `MethodError` of `tens_UA`.
_alv_sc_refuse_layered(::Any, ::Symbol, ::AbstractString) = nothing
_alv_sc_refuse_layered(::LayeredSphere, name::Symbol, scheme::AbstractString) = throw(
    ArgumentError(
        "$scheme: phase $name is a LayeredSphere, which this solver does not " *
            "support; `homogenize_alv(rve, SelfConsistent(), prop; times)` does."
    )
)

# ── Shared per-phase dilute concentration for the SC-ALV bodies ─────────────
#
# `Ã^dil = (𝟙 + P̃_α ∘ ΔC̃)^{-vol}` with the phase Hill kernel
# `P̃_α[i,j] = J_long[i,j]·U_M + J_shear[i,j]·D_M` (geometry tensors `U_M`,
# `D_M = V_M − U_M` against the iso-decoupled running estimate, whose scalar
# Volterra inverses are `J_long`, `J_shear`), evaluated in the cheapest algebra
# the operands share: two scalar n×n Volterra problems for a sphere in an iso
# estimate, the dense 6n×6n ones only for an operand that needs them.
_sc_alv_dilute_conc(C_phase, U_M, V_M, J_long, J_shear, C_m) =
    _alv_dilute_concentration(C_phase, C_m, _alv_hill_blocks(J_long, J_shear, U_M, V_M .- U_M))

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
        symmetrizes::AbstractVector{<:AbstractSymmetrize},
        crack_data;
        extra_A::Union{Nothing, AbstractMatrix} = nothing,
        extra_CA::Union{Nothing, AbstractMatrix} = nothing
    )
    T = eltype(C_m)
    A_avg = zero(C_m)    # = Σ f·sym(A_α)            (no J_m yet)
    CA_avg = zero(C_m)   # = Σ f·sym(C_α·A_α)        (no J_m yet)
    extra_A === nothing || (A_avg .+= extra_A)
    extra_CA === nothing || (CA_avg .+= extra_CA)
    J_long, J_shear = _alv_iso_inverses(C_m)
    for α in eachindex(C_phases)
        A_dil = _sc_alv_dilute_conc(C_phases[α], U_M_phases[α], V_M_phases[α], J_long, J_shear, C_m)
        sym = symmetrizes[α]
        A_dil_sym = _maybe_symmetrize_alv(A_dil, sym)
        CA_sym = _maybe_symmetrize_alv(_alv_prod(C_phases[α], A_dil), sym)
        f = T(fractions[α])
        @. A_avg += f * A_dil_sym
        @. CA_avg += f * CA_sym
    end
    # Crack compliance contributions (without J_m factor — that's the
    # essential ECHOES form difference compared to solids).
    H_total = _build_sc_crack_extra_J(C_m, crack_data)
    # Apply the ECHOES `B · A^{-vol}` formula with explicit `J_m`.
    J_m = _alv_inv(C_m)
    A_E = _alv_prod(A_avg, J_m) .+ H_total
    B_E = _alv_prod(CA_avg, J_m)
    return _alv_right_divide(B_E, A_E)
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
    extra = zero(C_m)
    isempty(crack_data) && return extra
    _is_iso_block(C_m) ||
        error("self_consistent_alv with cracks: only iso running estimate is supported")
    α_c, β_c = iso_params_from_blocks(C_m)
    B_n_base, B_t_base = _penny_cod_alv(α_c, β_c)
    @inbounds for (geom, ε, sym, Rn_mat, Rt_mat) in crack_data
        B_n = _interface_corrected(B_n_base, Rn_mat, semi_minor(geom))
        B_t = _interface_corrected(B_t_base, Rt_mat, semi_minor(geom))
        H_TI = _penny_compliance_blocks(B_n, B_t)
        H_full = _maybe_symmetrize_alv(delta_inverse_property_alv(geom, H_TI, ε), sym)
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
        symmetrizes::AbstractVector{<:AbstractSymmetrize};
        extra_A::Union{Nothing, AbstractMatrix} = nothing,
        extra_CA::Union{Nothing, AbstractMatrix} = nothing
    )
    A_avg = zero(C_m)
    CA_avg = zero(C_m)
    extra_A === nothing || (A_avg .+= extra_A)
    extra_CA === nothing || (CA_avg .+= extra_CA)
    # Iso parameters of the running estimate → scalar Volterra inverses
    # for the Hill-kernel time-space decoupling.
    J_long, J_shear = _alv_iso_inverses(C_m)
    for α in eachindex(C_phases)
        # Phase Hill kernel against the current estimate, dilute concentration
        # and its scaled contribution.
        A_dil = _sc_alv_dilute_conc(C_phases[α], U_M_phases[α], V_M_phases[α], J_long, J_shear, C_m)
        CA = _alv_prod(C_phases[α], A_dil)
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
    return _alv_right_divide(CA_avg, A_avg)
end
