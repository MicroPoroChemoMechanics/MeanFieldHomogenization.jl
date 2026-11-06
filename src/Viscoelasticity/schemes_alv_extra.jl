# =============================================================================
#  schemes_alv_extra.jl — Ponte-Castañeda & Willis (PCW),
#  Asymmetric Self-Consistent (ASC) and Differential (DIFF) schemes
#  in aging linear viscoelasticity.
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

The scheme of [ponte1995](@citet) for viscoelastic homogenization, in
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

# ── Asymmetric self-consistent (ASC) ───────────────────────────────────────

"""
    asymmetric_self_consistent_alv(rve::RVE, prop::Symbol; times, matrix = nothing,
                                    abstol = 1e-10, reltol = 1e-8,
                                    maxiters = 200, damping = 0.0,
                                    verbose = false, select_best = false) -> Matrix

Asymmetric self-consistent estimate in aging viscoelasticity, at order 4 (a stiffness,
``\\widetilde{\\mathbb{C}}``) or 2 (a conductivity, ``\\widetilde{\\boldsymbol{K}}``, the
products ``\\stackrel{\\circ}{:}`` becoming ``\\stackrel{\\circ}{\\cdot}``): the fixed point of

```math
\\widetilde{\\mathbb{C}} = \\widetilde{\\mathbb{C}}_{\\mathrm{M}}
  + \\sum_{i \\neq \\mathrm{M}} f_i\\,(\\widetilde{\\mathbb{C}}_i - \\widetilde{\\mathbb{C}}_{\\mathrm{M}})
    \\stackrel{\\circ}{:}\\widetilde{\\mathbb{A}}_i^{\\mathrm{dil}}(\\widetilde{\\mathbb{C}}),
```

every dilute concentration computed in the estimate itself and every contribution
measured against the matrix ``\\mathrm{M}`` (the phase `matrix`, or the one with
`fraction = :rest`). With crack families it is the compliance form of the elastic
scheme, the cracks adding their compliance against the estimate,

```math
\\widetilde{\\mathbb{S}} = \\widetilde{\\mathbb{S}}_{\\mathrm{M}}
  + \\bigl(\\langle\\widetilde{\\mathbb{A}}\\rangle
    - \\widetilde{\\mathbb{S}}_{\\mathrm{M}}\\stackrel{\\circ}{:}\\langle\\widetilde{\\mathbb{C}}\\stackrel{\\circ}{:}\\widetilde{\\mathbb{A}}\\rangle\\bigr)
    \\stackrel{\\circ}{:}\\widetilde{\\mathbb{S}}
  + \\sum_c \\Delta\\widetilde{\\mathbb{H}}_c(\\widetilde{\\mathbb{C}}),
\\qquad \\widetilde{\\mathbb{S}} = \\widetilde{\\mathbb{C}}^{-1^{\\circ}},
```

``\\langle\\cdot\\rangle`` the sums over the inclusion phases weighted by their fractions;
without cracks the two forms have the same fixed point. Solved by Picard iteration
from the matrix kernel, in the classes of the operators (on full matrices with layered
spheres); `homogenize_alv(rve, AsymmetricSelfConsistent(algorithm = NewtonDefault()), prop; times)`
solves the same equation by a Newton march through the instants. The matrix kernel must
be isotropic, and every phase must keep the estimate isotropic, as the ALV Hill kernel
against it requires: a sphere of isotropic behavior, a layered sphere, or
`symmetrize = :iso`, which a crack family always needs.
"""
asymmetric_self_consistent_alv(rve::RVE, prop::Symbol; kw...) = Matrix(_asymmetric_self_consistent_alv(rve, prop; kw...))

# The same, returning a `VolterraOperator` when it iterates in the classes of the
# operators (what `homogenize_alv` keeps), a full matrix otherwise; at the order of the
# matrix kernel.
function _asymmetric_self_consistent_alv(
        rve::RVE, prop::Symbol; times::AbstractVector{<:Real}, matrix::Union{Nothing, Symbol} = nothing, kw...
    )
    m = host_phase_name(rve, matrix, _ASC_WHO)
    law = phase_property(rve, m, prop)
    law isa ALVProperty ||
        throw(ArgumentError("$(_ASC_WHO): matrix property is neither a viscoelastic law, a relaxation operator nor an elastic tensor"))
    return _asc_alv(_alv_order(law, times), rve, prop; times, matrix = m, kw...)
end

const _ASC_WHO = "asymmetric_self_consistent_alv"

function _asc_alv(
        ::Val{4}, rve::RVE, prop::Symbol;
        times::AbstractVector{<:Real}, matrix::Symbol,
        abstol::Real = 1.0e-10, reltol::Real = 1.0e-8, maxiters::Int = 200,
        damping::Real = 0.0, verbose::Bool = false, select_best::Bool = false
    )
    d = _sc_alv_phases(rve, prop, times, matrix, _ASC_WHO)
    opts = (; abstol, reltol, maxiters, damping, verbose, select_best)
    if isempty(d.layered_data)
        Tp = _alv_promoted_eltype(d.C_phases, d.fractions, Matrix{Float64}[], d.crack_data)
        Tp = mapreduce(g -> eltype(tens_UA(g)), promote_type, d.geometries; init = Tp)
        return _fixed_point_in_classes(C_n -> _asc_alv_step_in_classes(C_n, d), d.C_0, Tp, "ASC-ALV"; opts...)
    end
    dense = _sc_alv_dense_data(d)
    return _fixed_point_dense(C_n -> _asc_alv_dense_step(C_n, dense, times), _sc_alv_dense_eltype(dense, times).(dense.C_0), "ASC-ALV"; opts...)
end

function _asc_alv(
        ::Val{2}, rve::RVE, prop::Symbol;
        times::AbstractVector{<:Real}, matrix::Symbol,
        abstol::Real = 1.0e-10, reltol::Real = 1.0e-8, maxiters::Int = 200,
        damping::Real = 0.0, verbose::Bool = false, select_best::Bool = false
    )
    K_M, data, cracks = _sc2_phases(rve, prop, times, matrix, _ASC_WHO)
    opts = (; abstol, reltol, maxiters, damping, verbose, select_best)
    if !any(d -> d.geom isa LayeredSphere, data)
        phases = _sc2_class_phases(data, times)
        K_M_op = first(phases).K_r
        return _fixed_point_in_classes(
            K -> _asc2_step_in_classes(K, K_M_op, phases, cracks), K_M_op, _sc2_eltype(K_M, data, cracks, times), "ASC-ALV order 2"; opts...
        )
    end
    return _fixed_point_dense(K -> _asc2_dense_step(K, K_M, data, cracks, times), _sc2_eltype(K_M, data, cracks, times).(K_M), "ASC-ALV order 2"; opts...)
end

# The ASC update from the sums ⟨A⟩ and ⟨X ∘ A⟩ of the inclusion phases against the
# running estimate X_n and the compliance ΔH of the crack families against it: the
# stiffness form X_M + ⟨X ∘ A⟩ − X_M ∘ ⟨A⟩ without cracks, the compliance form of the
# elastic scheme with them (the same fixed point without cracks). `vinv` is the Volterra
# inverse of the representation: `inv` for an operator, the block inverse of the order
# for a full matrix.
_asc_update(X_M, X_n, A_sum, B_sum, ::Nothing, vinv) = X_M + B_sum - X_M * A_sum
function _asc_update(X_M, X_n, A_sum, B_sum, ΔH, vinv)
    S_M = vinv(X_M)
    return vinv(S_M + (A_sum - S_M * B_sum) * vinv(X_n) + ΔH)
end

# The sums over the inclusion phases (`terms`, pairs (f A, f X ∘ A)), zero without any.
_asc_sums(terms, X) = isempty(terms) ? (zero(X), zero(X)) : (reduce(+, first.(terms)), reduce(+, last.(terms)))

# One ASC step in the classes of the operators, order 4: the concentrations of the
# inclusion phases (the matrix, first in `d`, excluded) against the isotropic estimate.
function _asc_alv_step_in_classes(C_n, d)
    J_long, J_shear = _alv_iso_inverses(C_n)
    terms = map(Iterators.drop(zip(d.C_phases, d.geometries, d.fractions, d.symmetrizes), 1)) do (C, g, f, sym)
        A = _dilute_in_classes(C, g, C_n, J_long, J_shear)
        f * _maybe_symmetrize_alv(A, sym), f * _maybe_symmetrize_alv(C * A, sym)
    end
    ΔH = isempty(d.crack_data) ? nothing : _crack_compliance_in_classes(C_n, d.crack_data)
    return _as_iso_estimate(_asc_update(d.C_0, C_n, _asc_sums(terms, C_n)..., ΔH, inv))
end

# The same on full matrices, the layered spheres through their recurrences.
function _asc_alv_dense_step(C_n, dense, times)
    J_long, J_shear = _alv_iso_inverses(C_n)
    terms = map(Iterators.drop(zip(dense.C_phases, dense.U_M_phases, dense.V_M_phases, dense.fractions, dense.symmetrizes), 1)) do (C, U, V, f, sym)
        A = _sc_alv_dilute_conc(C, U, V, J_long, J_shear, C_n)
        f * _maybe_symmetrize_alv(A, sym), f * _maybe_symmetrize_alv(_alv_prod(C, A), sym)
    end
    A_sum, B_sum = _asc_sums(terms, C_n)
    A_lay, B_lay = _sc_alv_layered_terms(dense.layered_data, C_n, times)
    A_lay === nothing || (A_sum = A_sum + A_lay; B_sum = B_sum + B_lay)
    ΔH = isempty(dense.crack_data) ? nothing : _build_sc_crack_extra_J(C_n, dense.crack_data)
    return _asc_update(dense.C_0, C_n, A_sum, B_sum, ΔH, X -> volterra_inverse(X; block_size = 6))
end

# Order 2, in the classes of the operators: `phases` the matrix first, then the
# inclusions; the insulating crack families add their resistivity.
function _asc2_step_in_classes(K, K_M, phases, cracks)
    terms = map(Iterators.drop(phases, 1)) do d
        A = _dilute2_in_classes(d.K_r, d.geom, K)
        d.f * _maybe_symmetrize_alv2(A, d.sym), d.f * _maybe_symmetrize_alv2(d.K_r * A, d.sym)
    end
    ΔR = isempty(cracks) ? nothing : _sc2_crack_resistivity(K, cracks)
    return _as_iso_estimate(_asc_update(K_M, K, _asc_sums(terms, K)..., ΔR, inv))
end

# The same on full matrices, the layered spheres through their recurrences.
function _asc2_dense_step(K, K_M, data, cracks, times)
    terms = map(Iterators.drop(data, 1)) do d
        A, B = _sc2_phase_AB(d.geom, d.K_r, K, d.sym, times)
        d.f * A, d.f * B
    end
    ΔR = isempty(cracks) ? nothing : Matrix(_sc2_crack_resistivity(K, cracks))
    return _asc_update(K_M, K, _asc_sums(terms, K)..., ΔR, X -> volterra_inverse(X; block_size = 3))
end

# =============================================================================
#  Differential ALV — SciML ODE on the fictitious incorporation time τ.
#
#  State : `vec(C̃::Matrix{Float64})` of length `(6n)²`.
#  RHS    : reshape `u → C̃`, build the per-phase Hill kernel against
#           `C̃`, assemble Norris terms via Sherman-Morrison and crack
#           density terms, return the flattened tensor derivative.
#  Solver : the SciML `solve` with `Tsit5` default.
# =============================================================================

"""
    _alv_diff_keeps_iso(geom, sym, C_r) -> Bool

Whether a solid phase leaves the **running** effective medium of the
differential ALV ODE inside the isotropic class.

This matters only for the differential scheme.  The ALV Hill kernel
[`hill_kernel`](@ref) exists for an isotropic reference **only**, and
where Mori-Tanaka or the dilute scheme evaluate it against the (fixed,
isotropic) matrix, the differential scheme evaluates it against the
running estimate ``\\widetilde{\\mathbb{C}}^{\\mathrm{hom}}(\\tau)`` — which an aligned, non-spherical inclusion
progressively takes out of the iso class.  Rather than silently reading
``(\\alpha, \\beta)`` off a matrix that is no longer isotropic, the ODE refuses to
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

# Error message shared by the ALV schemes whose reference is the running estimate
# (differential and self-consistent, orders 4 and 2), naming the scheme, the
# offending phase and the two ways out.
function _alv_iso_error(who::AbstractString, name::Symbol, what::AbstractString)
    return throw(
        ArgumentError(
            "$who: $what of phase :$(name) would take the " *
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

Differential homogenization in aging linear viscoelasticity, solved
as a SciML ODE on the fictitious incorporation time ``\\tau \\in [0, 1]``
[norris1985](@cite):

```math
\\frac{\\mathrm{d}\\widetilde{\\mathbb{C}}^{\\mathrm{hom}}}{\\mathrm{d}\\tau}
  = \\sum_i \\frac{\\mathrm{d}\\varphi_i}{\\mathrm{d}\\tau}
                  (\\widetilde{\\mathbb{C}}_i - \\widetilde{\\mathbb{C}}^{\\mathrm{hom}})
                  \\stackrel{\\circ}{:}\\widetilde{\\mathbb{A}}_i^{\\mathrm{dil}}(\\widetilde{\\mathbb{C}}^{\\mathrm{hom}})
   + \\sum_c \\frac{\\mathrm{d}\\varepsilon_c}{\\mathrm{d}\\tau}
              \\Delta\\widetilde{\\mathbb{C}}^{\\mathrm{crack}}_c(\\widetilde{\\mathbb{C}}^{\\mathrm{hom}})
```

with the volume balance ``[\\mathrm{d}f] = (\\mathbb 1 - [f]\\,\\mathbf{U}^{\\!T})\\,[\\mathrm{d}\\varphi]`` inverted by Sherman-
Morrison for solid phases (cracks contribute their density derivative
directly).  Only the end state is kept: `nsteps`, the density of save points
of the elastic [`DifferentialScheme`](@ref), has no effect here; the
integration step is controlled by `abstol` / `reltol`.

`formulation = :compliance` integrates the dual ODE on the creep
function ``\\widetilde{\\mathbb{L}}^{\\mathrm{hom}} = (\\widetilde{\\mathbb{C}}^{\\mathrm{hom}})^{-1^{\\circ}}`` instead, through
``\\widetilde{\\mathbb{H}}_i = -\\widetilde{\\mathbb{L}}^{\\mathrm{hom}}\\stackrel{\\circ}{:}\\widetilde{\\mathbb{N}}_i\\stackrel{\\circ}{:}\\widetilde{\\mathbb{L}}^{\\mathrm{hom}}``, and
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
        matrix::Union{Nothing, Symbol} = nothing,
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
    m = host_phase_name(rve, matrix, "differential_alv")
    C_M_law = phase_property(rve, m, prop)
    C_M_law isa ALVProperty ||
        throw(ArgumentError("differential_alv: matrix property is neither a viscoelastic law, a relaxation operator nor an elastic tensor"))
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
    for name in inclusion_phase_names(rve, m)
        ph = rve.phases[name]
        amt = rve.amounts[name]
        amt isa Schemes.Remainder &&
            throw(
            ArgumentError(
                "differential_alv: phase :$(name) carries the volume " *
                    "complement and so has no target to integrate up to; name it " *
                    "as the matrix, or give it an explicit fraction."
            )
        )
        if !(amt isa Schemes.CrackDensity)   # volume: the remainder is refused above
            sym = phase_symmetrize(rve, name)
            push!(
                solid_data, (
                    name = name,
                    C_r = _diff_alv_solid_operator(ph.geometry, rve, name, prop, sym, times),
                    geom = ph.geometry,
                    target = amt.value,
                    sym = sym,
                )
            )
        else  # CrackDensity
            ph.geometry isa MFH_Core.AbstractCrack ||
                throw(ArgumentError("differential_alv: phase $name has CrackDensity but geometry $(typeof(ph.geometry)) is not a crack"))
            sym = phase_symmetrize(rve, name)
            # A crack contributes a TI tensor in its own frame; only an
            # isotropic orientation average keeps the running medium iso.
            sym isa IsoSymmetrize ||
                _alv_iso_error("differential_alv", name, "the transversely isotropic crack contribution")
            Rn_mat = _trapezoidal_relaxation_scalar(get(ph.properties, :Rn, nothing), times)
            Rt_mat = _trapezoidal_relaxation_scalar(get(ph.properties, :Rt, nothing), times)
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
    paths = _resolve_paths_alv(something(trajectory, Schemes.Proportional()), rve, nsteps, m)

    # ODE state and parameters — the state carries the promoted eltype of
    # every input (Dual-safe, cf. `_alv_promoted_eltype`).
    Tp = eltype(C_M_full)
    for sd in solid_data
        Tp = promote_type(Tp, typeof(sd.target))
        sd.C_r === nothing || (Tp = promote_type(Tp, eltype(sd.C_r)))
        sd.C_r === nothing || (Tp = promote_type(Tp, eltype(tens_UA(sd.geom))))
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
    # Tuples of concrete records, each carrying its own trajectory: the
    # right-hand side runs over them without a dynamic dispatch per phase.
    ode_p = (
        sz = sz,
        solid_data = Tuple(merge(r, (path = paths[r.name],)) for r in solid_data),
        crack_data = Tuple(merge(r, (path = paths[r.name],)) for r in crack_data),
        times = times,
        grid = TimeGrid(times),
        dual = dual,
    )
    rhs! = (du, u, p, τ) -> _diff_alv_ode_rhs!(du, u, p, τ)
    prob = ODEProblem(rhs!, x0, (0.0, 1.0), ode_p)
    sol = solve(
        prob,
        something(alg, Tsit5());
        abstol, reltol,
        # Only the end state is returned: one copy of the dense state, not one per point.
        saveat = (1.0,),
        dense = false,
        solver_kwargs...
    )
    P_end = reshape(sol.u[end], sz, sz)
    return dual ? volterra_inverse(P_end; block_size = 6) : P_end
end

# A layered sphere: the moduli of its layers, carried by the geometry, are the
# material data; the declared phase property is a placeholder, as in the elastic
# pipeline.
_diff_alv_solid_operator(::LayeredSphere, rve, name, prop, sym, times) = nothing
# Any other inclusion: its relaxation operator, stored by the blocks of its class,
# in which the right-hand side computes its contribution.
function _diff_alv_solid_operator(geom, rve, name, prop, sym, times)
    C_r_law = phase_property(rve, name, prop)
    C_r_law isa ALVProperty ||
        throw(ArgumentError("differential_alv: phase $name property is neither a viscoelastic law, a relaxation operator nor an elastic tensor"))
    C_r = _relaxation_operator(C_r_law, TimeGrid(times))
    _alv_diff_keeps_iso(geom, sym, C_r) ||
        _alv_iso_error("differential_alv", name, "the shape or the anisotropy")
    return C_r
end

# ── ALV ODE RHS ─────────────────────────────────────────────────────────────

function _diff_alv_ode_rhs!(du, u, p, τ)
    T = eltype(u)
    # In the dual form the state carries J̃ and every phase kernel is still
    # evaluated against the running relaxation C̃ = J̃^{-vol}; each stiffness
    # contribution Ñ is mapped to its compliance counterpart by the exact
    # identity H̃ = − J̃ ∘ Ñ ∘ J̃ (same relation as
    # `property_contribution_alv(crack, …)`, read backwards).
    P_curr = reshape(u, p.sz, p.sz)
    C_curr = p.dual ? _alv_inv(P_curr) : P_curr
    Δ = reshape(du, p.sz, p.sz)
    fill!(Δ, zero(T))

    # Sherman-Morrison : dφ_α/dτ = df_α/dτ + (f_α / f_0) · sum(df).  Only
    # solid phases enter the volume balance; the sum is needed by the crack
    # loop below as well, so it is computed unconditionally.
    f = map(r -> r.path.f(τ) * r.target, p.solid_data)
    df = map(r -> r.path.df(τ) * r.target, p.solid_data)
    f0 = one(T) - sum(f; init = zero(T))
    sum_df = sum(df; init = zero(T))

    # The running medium is isotropic (the setup refuses what would take it out
    # of the class): its two blocks, and their Volterra inverses, shared across the
    # solid phases.
    C_op = _iso_operator(C_curr, p.grid, Val(4))
    J_long, J_shear = _alv_iso_inverses(C_op)
    map(p.solid_data, f, df) do r, fᵢ, dfᵢ
        dφᵢ = dfᵢ + (fᵢ / f0) * sum_df
        iszero(dφᵢ) || _diff_alv_add!(
            Δ, _maybe_symmetrize_alv(_diff_alv_solid_contribution(r, r.geom, C_curr, C_op, J_long, J_shear, p.times), r.sym),
            dφᵢ, p.dual, P_curr, p.grid
        )
        return nothing
    end

    # Crack contributions : dε_α/dτ · ΔC̃^crack_α(C̃) where the
    # crack ΔC̃^crack is the dilute stiffness contribution evaluated
    # against the running medium (with optional Sevostianov
    # interface stiffness correction).
    # As in the elastic scheme, a crack family carries no volume but is
    # still diluted by the solid increments:
    #     dφ_c^ε = dε_c + (ε_c / f_0) Σ_{j solid} df_j .
    map(p.crack_data) do r
        εᶜ = r.path.f(τ) * r.target
        dφᶜ = r.path.df(τ) * r.target + (εᶜ / f0) * sum_df
        iszero(dφᶜ) && return nothing
        # In the classes of its operators, against the isotropic operator of the
        # running medium.
        Ñ = property_contribution_alv_at(r.geom, C_op; Rn_mat = r.Rn_mat, Rt_mat = r.Rt_mat)
        _diff_alv_add!(Δ, _maybe_symmetrize_alv(delta_property_alv(r.geom, Ñ, 1.0), r.sym), dφᶜ, p.dual, P_curr, p.grid)
        return nothing
    end
    return nothing
end

# The contribution of a solid phase against the running medium: from its
# Hill kernel for an ellipsoid, from the bulk and shear ALV recurrences of the
# whole composite sphere for a layered one, which has no Hill kernel.
_diff_alv_solid_contribution(r, ::LayeredSphere, C_curr, C_op, J_long, J_shear, times) =
    property_contribution_alv_at(r.geom, C_curr, times)
# An ellipsoid: its dilute contribution in the classes of its operators.
_diff_alv_solid_contribution(r, geom, C_curr, C_op, J_long, J_shear, times) =
    (r.C_r - C_op) * _dilute_in_classes(r.C_r, geom, C_op, J_long, J_shear)

# The isotropic operator of a full matrix the running medium is known to be.
_iso_operator(M::AbstractMatrix, grid, ::Val{4}) = VolterraOperator(_ISO4, iso_params_from_blocks(M), grid)
_iso_operator(M::AbstractMatrix, grid, ::Val{2}) = VolterraOperator(_ISO2, (iso_order2_params_from_blocks(M),), grid)

# Adds `weight` times a stiffness contribution to the right-hand side, mapped to
# its compliance counterpart −J̃ ∘ Ñ ∘ J̃ in the dual form.
function _diff_alv_add!(Δ, contrib::AbstractMatrix, weight, dual::Bool, J_curr, grid)
    term = dual ? -_alv_prod(J_curr, _alv_prod(contrib, J_curr)) : contrib
    @. Δ += weight * term
    return nothing
end
# Stored by blocks: computed in the classes, added into the full derivative in
# place. The setup keeps every term isotropic (`_alv_diff_keeps_iso`); the
# conversion checks it for an operator stored in a wider class.
function _diff_alv_add!(Δ, contrib::VolterraOperator, weight, dual::Bool, J_curr, grid)
    term = dual ? (J = _iso_operator(J_curr, grid, Val(_order(contrib.class))); -(J * contrib * J)) : contrib
    _add_iso!(Δ, weight, _as_iso_estimate(term))
    return nothing
end
_add_iso!(Δ, w, R::VolterraOperator{<:TensISO{4, 3}}) = _add_iso_blocks!(Δ, w, R.blocks...)
_add_iso!(Δ, w, R::VolterraOperator{<:TensISO{2, 3}}) = _add_iso_order2_blocks!(Δ, w, R.blocks[1])

# ── Trajectory path resolution for ALV — share the elastic
#    `_resolve_paths` (which now returns callables, not vectors) ──────────────
function _resolve_paths_alv(
        trajectory::Schemes.DifferentialTrajectory,
        rve::RVE, nsteps::Int, matrix::Symbol
    )
    return Schemes._resolve_paths(trajectory, rve, nsteps, matrix)
end
