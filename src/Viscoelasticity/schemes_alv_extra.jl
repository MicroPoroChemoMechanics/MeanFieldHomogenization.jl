# =============================================================================
#  schemes_alv_extra.jl — Ponte-Castañeda & Willis (PCW),
#  Asymmetric Self-Consistent (ASC) and Differential (DIFF) schemes
#  in ageing linear viscoelasticity.
#
#  All operate on the discrete `(6n × 6n)` block matrices produced by
#  `trapezoidal_matrix` (or its `_trapezoidal_relaxation` wrapper for
#  `:creep`-mode laws).  The Volterra product is the regular matrix
#  product (`*`); the Volterra inverse is [`volterra_inverse`](@ref).
#
#  References:
#    * PCW : Ponte-Castañeda & Willis 1995 — coincides with Maxwell in
#      the single-shape case.
#    * ASC : the iteration update is anchored on the matrix property
#      `C_M` rather than on the running estimate.  Its fixed point
#      coincides with `self_consistent_alv`'s only when every phase
#      shares one Hill tensor (all-spherical being the usual case) —
#      see the derivation atop `Schemes/self_consistent.jl`; with
#      unequal shapes the two are different schemes, not two routes to
#      one answer.
#    * DIFF : Norris 1985 ; user's hand-written DEM N-component note.
#      Solved as a SciML ODE on the fictitious incorporation time
#      `τ ∈ [0, 1]` (`Tsit5` default ; user-overridable via the
#      `alg = …` kwarg of [`DifferentialScheme`](@ref)).
# =============================================================================

# ── PCW : single-shape distribution ⇒ algebraically identical to Maxwell ───

"""
    pcw_alv(C_0, contribs, fractions; H_dist) -> Matrix

Ponte-Castañeda & Willis (1995) viscoelastic homogenization in
single-distribution-shape form.  The formula is algebraically
identical to [`maxwell_alv`](@ref) ; the only difference is that the
Hill kernel `H_dist` is computed against the **distribution shape**
stored in `rve.distribution_shape`, not against any individual phase.

Use [`homogenize_alv`](@ref) with `scheme = PonteCastanedaWillis()` —
the dispatcher reads the RVE's distribution shape and forwards here.
"""
function pcw_alv(
        C_0::AbstractMatrix,
        contribs::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector;
        H_dist::AbstractMatrix
    )
    return maxwell_alv(C_0, contribs, fractions; H_0 = H_dist)
end

# ── Asymmetric Self-Consistent (ASC) — stiffness form ──────────────────────

"""
    asymmetric_self_consistent_alv(rve::RVE, prop::Symbol; times,
                                    abstol = 1e-10, reltol = 1e-8,
                                    maxiters = 200, damping = 0.0,
                                    verbose = false, select_best = false)
        -> Matrix{T}

Asymmetric self-consistent viscoelastic homogenization.  The
iteration update reads

    `C^{n+1} = C_M + Σ_i f_i (C_i − C_M) ∘ A^{dil,i}(C^n)`,

mirroring the reference ASC form.  Returns the
`(6n × 6n)` effective relaxation matrix once the residual
`‖C^{n+1} − C^n‖_F` falls below `abstol + reltol · ‖C^n‖_F` (or after
`maxiters` Picard steps).
"""
function asymmetric_self_consistent_alv(
        rve::RVE, prop::Symbol;
        times::AbstractVector{<:Real},
        abstol::Real = 1.0e-10,
        reltol::Real = 1.0e-8,
        maxiters::Int = 200,
        damping::Real = 0.0,
        verbose::Bool = false,
        select_best::Bool = false
    )
    C_M_law = matrix_property(rve, prop)
    C_M_law isa ViscoLaw ||
        throw(ArgumentError("asymmetric_self_consistent_alv: matrix property is not a ViscoLaw"))
    C_M = _trapezoidal_relaxation(C_M_law, times, 6)
    incl_names = inclusion_phase_names(rve)
    # Eltype-generic containers (Dual-safe — see schemes_alv_sc.jl).
    C_phases = Matrix[]
    geometries = Any[]
    fractions = typeof(matrix_volume_fraction(rve))[]   # eltype of the RVE amounts
    symmetrizes = AbstractSymmetrize[]
    crack_data = Tuple{
        Any, Any, AbstractSymmetrize,
        Union{Nothing, AbstractMatrix},
        Union{Nothing, AbstractMatrix},
    }[]
    for name in incl_names
        ph = rve.phases[name]
        a = rve.amounts[name]
        if a isa CrackDensity
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
            throw(ArgumentError("asymmetric_self_consistent_alv: phase $name property is not a ViscoLaw"))
        push!(C_phases, _trapezoidal_relaxation(C_r_law, times, 6))
        push!(geometries, ph.geometry)
        push!(fractions, _amount_value(rve, name))
        push!(symmetrizes, phase_symmetrize(rve, name))
    end

    U_M_phases = Matrix[_tens_to_mandel66(tens_UA(g)) for g in geometries]
    V_M_phases = Matrix[_tens_to_mandel66(tens_VA(g)) for g in geometries]

    Tp = _alv_promoted_eltype(
        vcat(Matrix[C_M], C_phases), fractions, U_M_phases, crack_data
    )
    n = length(times)
    Id = _identity_alv(n, Tp)
    C_n = Tp.(C_M)
    best_resid = Inf
    C_best = C_n

    for iter in 1:maxiters
        C_n_new = _asc_alv_step(
            C_M, C_n, C_phases, U_M_phases, V_M_phases,
            fractions, symmetrizes, n, Id
        )
        # Crack contribution (Budiansky-O'Connell SC):
        # `ΔJ̃_cracks(C_n)` against the running estimate, added to the
        # compliance side of the ASC solid update.
        if !isempty(crack_data)
            ΔJ = zeros(eltype(C_n), size(C_n)...)
            J_n = volterra_inverse(C_n; block_size = 6)
            @inbounds for (geom, ε, sym, Rn_mat, Rt_mat) in crack_data
                Ñ = stiffness_contribution_alv_at(
                    geom, C_n;
                    Rn_mat = Rn_mat,
                    Rt_mat = Rt_mat
                )
                ΔC = delta_stiffness_alv(geom, Ñ, ε)
                ΔJ_block = -(J_n * ΔC * J_n)
                ΔJ_block = _maybe_symmetrize_alv(ΔJ_block, sym)
                ΔJ .+= ΔJ_block
            end
            J_solid_new = volterra_inverse(C_n_new; block_size = 6)
            C_n_new = volterra_inverse(J_solid_new .+ ΔJ; block_size = 6)
        end
        Δ = norm(C_n_new - C_n)
        norm_C = norm(C_n)
        tol_eff = abstol + reltol * norm_C
        verbose && @info "ASC-ALV iter $iter : ‖Δ‖ = $(Δ)   tol = $tol_eff"
        if select_best && Δ < best_resid
            best_resid = Δ
            C_best = C_n_new
        end
        if Δ ≤ tol_eff
            return C_n_new
        end
        C_n = (1 - damping) .* C_n_new .+ damping .* C_n
    end

    @debug "asymmetric_self_consistent_alv: maxiters=$(maxiters) reached without convergence" abstol reltol
    return select_best ? C_best : C_n
end

# Single ASC step — increment is anchored on C_M.
function _asc_alv_step(
        C_M::AbstractMatrix,
        C_n::AbstractMatrix,
        C_phases::AbstractVector{<:AbstractMatrix},
        U_M_phases::AbstractVector{<:AbstractMatrix},
        V_M_phases::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector{<:Real},
        symmetrizes::AbstractVector{<:AbstractSymmetrize},
        n::Int, Id::AbstractMatrix
    )
    sz = size(C_n, 1)
    T = eltype(C_n)
    Δ = zeros(T, sz, sz)
    α_n, β_n = iso_params_from_blocks(C_n)
    M_long = @. (α_n + 2 * β_n) / 3
    M_shear = β_n ./ 2
    J_long = volterra_inverse(M_long; block_size = 1)
    J_shear = volterra_inverse(M_shear; block_size = 1)

    @inbounds for r in eachindex(C_phases)
        U_M = U_M_phases[r]
        V_M = V_M_phases[r]
        D_M = V_M .- U_M
        P_r = zeros(T, sz, sz)
        for i in 1:n, j in 1:i
            block = J_long[i, j] .* U_M .+ J_shear[i, j] .* D_M
            rows = (6 * (i - 1) + 1):(6 * i)
            cols = (6 * (j - 1) + 1):(6 * j)
            P_r[rows, cols] = block
        end
        ΔCr = C_phases[r] - C_n
        A_dil = volterra_inverse(Id + P_r * ΔCr; block_size = 6)
        contrib = (C_phases[r] - C_M) * A_dil
        sym = symmetrizes[r]
        contrib = _maybe_symmetrize_alv(contrib, sym)
        f = fractions[r]
        @. Δ += f * contrib
    end
    return C_M .+ Δ
end

# =============================================================================
#  Differential ALV — SciML ODE on the fictitious incorporation time τ.
#
#  State : `vec(C̃::Matrix{Float64})` of length `(6n)²`.
#  RHS    : reshape `u → C̃`, build the per-phase Hill kernel against
#           `C̃`, assemble Norris terms via Sherman-Morrison and crack
#           density terms, return the flattened tensor derivative.
#  Solver : `OrdinaryDiffEq.solve` with `Tsit5` default.
# =============================================================================

"""
    _alv_diff_keeps_iso(geom, sym, C_r) -> Bool

Whether a solid phase leaves the **running** effective medium of the
differential ALV ODE inside the isotropic class.

This matters only for the differential scheme.  The ALV Hill kernel
[`hill_kernel`](@ref) exists for an isotropic reference **only**, and
where Mori-Tanaka or the dilute scheme evaluate it against the (fixed,
isotropic) matrix, the differential scheme evaluates it against the
running estimate `C̃(τ)` — which an aligned, non-spherical inclusion
progressively takes out of the iso class.  Rather than silently reading
`(α, β)` off a matrix that is no longer isotropic, the ODE refuses to
start (see [`differential_alv`](@ref)).

An isotropic orientation average (`symmetrize = :iso`) restores the
property for any shape, which is what randomly oriented inclusions or
cracks mean physically.
"""
function _alv_diff_keeps_iso(geom, sym::AbstractSymmetrize, C_r)
    sym isa IsoSymmetrize && return true
    geom isa LayeredSphere && return true    # iso contribution by construction
    return _alv_geom_is_spherical(geom) && (C_r === nothing || _is_iso_block(C_r))
end

_alv_geom_is_spherical(geom) = false
_alv_geom_is_spherical(::Ellipsoid{3, Elasticity.Spherical}) = true

# Error message shared by the two ALV differential drivers (order 4 and
# order 2), naming the offending phase and the two ways out.
function _alv_diff_iso_error(name::Symbol, what::AbstractString)
    return throw(
        ArgumentError(
            "differential_alv: $what of phase :$(name) would take the " *
                "running effective medium out of the isotropic class, for which " *
                "no ALV Hill kernel exists. Either give the phase an isotropic " *
                "orientation average (`symmetrize = :iso` in `add_phase!`), or " *
                "use a scheme whose reference medium stays the (isotropic) " *
                "matrix — Mori-Tanaka, dilute, Maxwell, PCW."
        )
    )
end

"""
    differential_alv(rve::RVE, prop::Symbol; times,
                      nsteps = 100, trajectory = nothing,
                      abstol = 1e-8, reltol = 1e-6, alg = nothing,
                      formulation = :stiffness) -> Matrix{T}

Differential homogenization in ageing linear viscoelasticity, solved
as a SciML ODE on the fictitious incorporation time `τ ∈ [0, 1]`
([norris1985](@cite); user's hand-written DEM note) :

```math
\\frac{\\mathrm d \\tilde{\\mathbb C}}{\\mathrm d \\tau}
  = \\sum_\\alpha \\frac{\\mathrm d \\varphi_\\alpha}{\\mathrm d \\tau}
                  (\\tilde{\\mathbb C}_\\alpha - \\tilde{\\mathbb C})
                  \\circ \\tilde{\\mathbb A}_\\alpha^{dil}(\\tilde{\\mathbb C})
   + \\sum_c \\frac{\\mathrm d \\varepsilon_c}{\\mathrm d \\tau}
              \\Delta\\tilde{\\mathbb C}^{crack}_c(\\tilde{\\mathbb C})
```

with the volume balance `df = (𝟙 − f ⊗ 𝐔)·dφ` inverted by Sherman-
Morrison for solid phases (cracks contribute their density derivative
directly).  `nsteps` is the density of save points along τ ; the
integration step is controlled by `abstol` / `reltol`.

`formulation = :compliance` integrates the dual ODE on the creep
function `J̃ = C̃^{-vol}` instead, through `H̃_α = −J̃ ∘ Ñ_α ∘ J̃`, and
inverts the result — the same choice as the elastic
[`DifferentialScheme`](@ref).

Supported inclusion geometries: ellipsoids / spheroids (through the ALV
Hill kernel), `LayeredSphere` (through the bulk + shear ALV
recurrences) and flat cracks (`CrackDensity`).  Because the reference of
the ODE is the *running* medium and the ALV Hill kernel is isotropic
only, every phase must keep that medium isotropic — see
[`_alv_diff_keeps_iso`](@ref); the ODE throws an explicit `ArgumentError`
otherwise instead of returning a wrong answer.
"""
function differential_alv(
        rve::RVE, prop::Symbol;
        times::AbstractVector{<:Real},
        nsteps::Int = 100,
        trajectory = nothing,
        abstol::Real = 1.0e-8,
        reltol::Real = 1.0e-6,
        alg = nothing,
        formulation::Symbol = :stiffness,
        solver_kwargs::NamedTuple = NamedTuple()
    )
    formulation in (:stiffness, :compliance) ||
        throw(
        ArgumentError(
            "differential_alv: formulation must be :stiffness or " *
                ":compliance; got :$(formulation)"
        )
    )
    C_M_law = matrix_property(rve, prop)
    C_M_law isa ViscoLaw ||
        throw(ArgumentError("differential_alv: matrix property is not a ViscoLaw"))
    C_M_full = _trapezoidal_relaxation(C_M_law, times, 6)
    _is_iso_block(C_M_full) ||
        throw(
        ArgumentError(
            "differential_alv: the ALV matrix must be isotropic (the " *
                "differential ODE evaluates the ALV Hill kernel against its " *
                "running effective medium, and that kernel exists for an " *
                "isotropic reference only)."
        )
    )
    n = length(times)

    # Per-phase data : split solids vs cracks.
    solid_data = NamedTuple[]
    crack_data = NamedTuple[]
    for name in inclusion_phase_names(rve)
        ph = rve.phases[name]
        amt = rve.amounts[name]
        if amt isa Schemes.VolumeFraction
            sym = phase_symmetrize(rve, name)
            geom = ph.geometry
            if geom isa LayeredSphere
                # Heterogeneous inclusion: the per-layer moduli carried by the
                # geometry are the material data; the declared phase property
                # is a placeholder, exactly as in the elastic pipeline.
                push!(
                    solid_data, (
                        name = name, kind = :layered_sphere,
                        C_r = nothing, geom = geom,
                        target = amt.value, sym = sym,
                        U_M = nothing, V_M = nothing,
                    )
                )
                continue
            end
            C_r_law = phase_property(rve, name, prop)
            C_r_law isa ViscoLaw ||
                throw(ArgumentError("differential_alv: phase $name property is not a ViscoLaw"))
            C_r = _trapezoidal_relaxation(C_r_law, times, 6)
            _alv_diff_keeps_iso(geom, sym, C_r) ||
                _alv_diff_iso_error(name, "the shape or the anisotropy")
            push!(
                solid_data, (
                    name = name, kind = :ellipsoid,
                    C_r = C_r,
                    geom = geom,
                    target = amt.value,
                    sym = sym,
                    U_M = _tens_to_mandel66(tens_UA(geom)),
                    V_M = _tens_to_mandel66(tens_VA(geom)),
                )
            )
        else  # CrackDensity
            ph.geometry isa MFH_Core.AbstractCrack ||
                throw(ArgumentError("differential_alv: phase $name has CrackDensity but geometry $(typeof(ph.geometry)) is not a crack"))
            sym = phase_symmetrize(rve, name)
            # A crack contributes a TI tensor in its own frame; only an
            # isotropic orientation average keeps the running medium iso.
            sym isa IsoSymmetrize ||
                _alv_diff_iso_error(name, "the transversely isotropic crack contribution")
            Rn_mat = haskey(ph.properties, :Rn) ?
                _trapezoidal_relaxation_scalar(ph.properties[:Rn], times) : nothing
            Rt_mat = haskey(ph.properties, :Rt) ?
                _trapezoidal_relaxation_scalar(ph.properties[:Rt], times) : nothing
            push!(
                crack_data, (
                    name = name,
                    geom = ph.geometry,
                    target = amt.value,
                    sym = sym,
                    Rn_mat = Rn_mat,
                    Rt_mat = Rt_mat,
                )
            )
        end
    end

    # Trajectory : default proportional path.
    paths = trajectory === nothing ?
        _resolve_paths_alv(Schemes.Proportional(), rve, nsteps) :
        _resolve_paths_alv(trajectory, rve, nsteps)

    # ODE state and parameters — the state carries the promoted eltype of
    # every input (Dual-safe, cf. `_alv_promoted_eltype`).
    Tp = eltype(C_M_full)
    for sd in solid_data
        Tp = promote_type(Tp, typeof(sd.target))
        sd.C_r === nothing || (Tp = promote_type(Tp, eltype(sd.C_r)))
        sd.U_M === nothing || (Tp = promote_type(Tp, eltype(sd.U_M)))
    end
    for cd in crack_data
        Tp = promote_type(Tp, typeof(cd.target))
        cd.Rn_mat === nothing || (Tp = promote_type(Tp, eltype(cd.Rn_mat)))
        cd.Rt_mat === nothing || (Tp = promote_type(Tp, eltype(cd.Rt_mat)))
    end
    sz = 6 * n
    dual = formulation === :compliance
    # The dual form integrates the creep function J̃ = C̃^{-vol}.
    x0 = vec(Tp.(dual ? volterra_inverse(C_M_full; block_size = 6) : C_M_full))
    ode_p = (
        n = n, sz = sz,
        solid_data = solid_data,
        crack_data = crack_data,
        paths = paths,
        times = times,
        dual = dual,
    )
    rhs! = (du, u, p, τ) -> _diff_alv_ode_rhs!(du, u, p, τ)
    prob = ODEProblem(rhs!, x0, (0.0, 1.0), ode_p)
    sol = solve(
        prob,
        alg === nothing ? Tsit5() : alg;
        abstol, reltol,
        saveat = range(0.0, 1.0; length = max(nsteps, 1) + 1),
        dense = false,
        solver_kwargs...
    )
    P_end = reshape(sol.u[end], sz, sz)
    return dual ? volterra_inverse(P_end; block_size = 6) : P_end
end

# ── ALV ODE RHS ─────────────────────────────────────────────────────────────

function _diff_alv_ode_rhs!(du, u, p, τ)
    n, sz = p.n, p.sz
    # In the dual form the state carries J̃ and every phase kernel is still
    # evaluated against the running relaxation C̃ = J̃^{-vol}; each stiffness
    # contribution Ñ is mapped to its compliance counterpart by the exact
    # identity H̃ = − J̃ ∘ Ñ ∘ J̃ (same relation as
    # `stiffness_contribution_alv(crack, …)`, read backwards).
    P_curr = reshape(u, sz, sz)
    C_curr = p.dual ? volterra_inverse(P_curr; block_size = 6) : P_curr
    J_curr = p.dual ? P_curr : nothing
    Δ = zeros(eltype(u), sz, sz)
    Id = _identity_alv(n, eltype(u))
    push_contrib! = (contrib, weight) -> begin
        term = p.dual ? -(J_curr * contrib * J_curr) : contrib
        @. Δ += weight * term
        nothing
    end

    n_solid = length(p.solid_data)
    # Sherman-Morrison : dφ_α/dτ = df_α/dτ + (f_α / f_0) · sum(df).  Only
    # solid phases enter the volume balance; the sum is needed by the crack
    # loop below as well, so it is computed unconditionally.
    f = Vector{eltype(u)}(undef, n_solid)
    df = Vector{eltype(u)}(undef, n_solid)
    @inbounds for (i, r) in enumerate(p.solid_data)
        nt = p.paths[r.name]
        f[i] = nt.f(τ) * r.target
        df[i] = nt.df(τ) * r.target
    end
    f0 = one(eltype(u)) - sum(f; init = zero(eltype(u)))
    sum_df = sum(df; init = zero(eltype(u)))

    if n_solid > 0
        # Pre-compute Volterra inverses against C_curr (shared across solid phases).
        α_c, β_c = iso_params_from_blocks(C_curr)
        M_long = @. (α_c + 2 * β_c) / 3
        M_shear = β_c ./ 2
        J_long = volterra_inverse(M_long; block_size = 1)
        J_shear = volterra_inverse(M_shear; block_size = 1)

        @inbounds for (i, r) in enumerate(p.solid_data)
            dφᵢ = df[i] + (f[i] / f0) * sum_df
            iszero(dφᵢ) && continue
            contrib = if r.kind === :layered_sphere
                # No Hill kernel: the bulk + shear ALV recurrences give the
                # contribution of the whole composite sphere directly.
                stiffness_contribution_alv_at(r.geom, C_curr, p.times)
            else
                # Per-phase Hill kernel against C_curr.
                D_M = r.V_M .- r.U_M
                P_r = zeros(eltype(u), sz, sz)
                for ii in 1:n, jj in 1:ii
                    block = J_long[ii, jj] .* r.U_M .+ J_shear[ii, jj] .* D_M
                    rows = (6 * (ii - 1) + 1):(6 * ii)
                    cols = (6 * (jj - 1) + 1):(6 * jj)
                    P_r[rows, cols] = block
                end
                ΔC = r.C_r - C_curr
                A_dil = volterra_inverse(Id + P_r * ΔC; block_size = 6)
                ΔC * A_dil
            end
            push_contrib!(_maybe_symmetrize_alv(contrib, r.sym), dφᵢ)
        end
    end

    # Crack contributions : dε_α/dτ · ΔC̃^crack_α(C_curr) where the
    # crack ΔC̃^crack is the dilute stiffness contribution evaluated
    # against the running matrix `C_curr` (with optional Sevostianov
    # interface stiffness correction).
    # As in the elastic scheme, a crack family carries no volume but is
    # still diluted by the solid increments:
    #     dφ_c^ε = dε_c + (ε_c / f_0) Σ_{j solid} df_j .
    @inbounds for r in p.crack_data
        nt = p.paths[r.name]
        εᶜ = nt.f(τ) * r.target
        dφᶜ = nt.df(τ) * r.target + (εᶜ / f0) * sum_df
        iszero(dφᶜ) && continue
        Ñ = stiffness_contribution_alv_at(
            r.geom, C_curr;
            Rn_mat = r.Rn_mat, Rt_mat = r.Rt_mat
        )
        ΔC = delta_stiffness_alv(r.geom, Ñ, 1.0)
        push_contrib!(_maybe_symmetrize_alv(ΔC, r.sym), dφᶜ)
    end

    du .= vec(Δ)
    return nothing
end

# ── Trajectory path resolution for ALV — share the elastic
#    `_resolve_paths` (which now returns callables, not vectors) ──────────────
function _resolve_paths_alv(
        trajectory::Schemes.DifferentialTrajectory,
        rve::RVE, nsteps::Int
    )
    return Schemes._resolve_paths(trajectory, rve, nsteps)
end
