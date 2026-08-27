# =============================================================================
#  differential.jl — Differential homogenization scheme as a SciML ODE.
#
#  Integrates the multi-phase Norris ODE on the fictitious incorporation
#  time `τ ∈ [0, 1]` (cf. user's hand-written DEM note ; @norris1985) :
#
#      dC^hom / dτ = Σ_α dφ_α/dτ · (C_α - C^hom) ⊡ A_α^dil(C^hom)
#                  + Σ_c dε_c/dτ · ΔC^crack_c(C^hom)
#
#  with the volumetric balance `df = (𝟙 − f ⊗ 𝐔) · dφ` inverted by
#  Sherman-Morrison so the user supplies effective volume fractions
#  `f_α(τ)` along the chosen `trajectory`.  The ODE is solved by
#  `OrdinaryDiffEq.solve` with an adaptive RK (default `Tsit5`) and
#  the result returned at `τ = 1`.
#
#  Cracks have negligible volume — their density `ε_c(τ)` enters the
#  RHS directly without going through Sherman-Morrison.
#
#  DUAL (COMPLIANCE) FORM — `formulation = :compliance`.  The same
#  incorporation process written on the compliance,
#
#      dS^hom / dτ = Σ_α dφ_α/dτ · ℍ_α(S^hom) + Σ_c dε_c/dτ · ΔS^crack_c ,
#
#  is the exact image of the stiffness ODE under `S = C⁻¹` (the two are
#  related term by term by `ℍ = −𝕊 : 𝐍 : 𝕊`), so both integrate the same
#  trajectory — they differ only in which variable carries the solver's
#  error control.  The compliance form is the better-conditioned one when
#  the effective medium softens towards percolation (`C → 0`), the
#  stiffness form when it stiffens.  Whichever is integrated, the value
#  returned to the caller is the same quantity as the declared property.
# =============================================================================

# Solver options are read out of `scheme.options`; every OTHER keyword
# stored there is forwarded verbatim to `OrdinaryDiffEq.solve` (`maxiters`,
# `dtmax`, `dt`, `callback`, …).
const _DIFF_RESERVED_OPTIONS = (:nsteps, :abstol, :reltol, :alg, :formulation)

_diff_solver_kwargs(options::NamedTuple) = Base.structdiff(
    options, NamedTuple{_DIFF_RESERVED_OPTIONS}
)

function _diff_formulation(scheme::DifferentialScheme)
    form = get(scheme.options, :formulation, :stiffness)
    form in (:stiffness, :compliance) ||
        throw(
        ArgumentError(
            "DifferentialScheme: formulation must be :stiffness or " *
                ":compliance; got :$(form)"
        )
    )
    return form
end

"""
    _evaluate(rve, scheme::DifferentialScheme, ::Val{p}; kw...) -> AbstractTens

Differential scheme for property `:p` ([norris1985](@cite)).
Integrates the multi-phase incorporation-sequence ODE on `τ ∈ [0, 1]`
with the SciML `OrdinaryDiffEq.solve` driver (default `Tsit5`), in the
stiffness or the compliance variable according to the scheme's
`formulation`.
"""
function _evaluate(rve::RVE, scheme::DifferentialScheme, ::Val{p}; kw...) where {p}
    return last(_differential_states(rve, scheme, p; kw...)[2])
end

"""
    differential_path(rve, scheme::DifferentialScheme, property::Symbol; kw...)
        -> (τ::Vector, states::Vector{<:AbstractTens})

Effective property of `rve` **all along** the differential scheme's
fictitious incorporation time `τ ∈ [0, 1]`, instead of only at `τ = 1`
as [`homogenize`](@ref) returns.

`τ` carries `nsteps + 1` uniformly spaced save points (the `saveat` of
the underlying ODE solve — the integration step itself remains adaptive
and controlled by `abstol` / `reltol`), and `states[k]` is the effective
property after the fraction of each phase has been grown to
`f_α(τ[k])` along the scheme's `trajectory`.  `states[1]` is the matrix
property and `states[end]` is exactly what `homogenize` returns.

Useful to plot the construction history of a composite, and to compare
incorporation trajectories at equal target fractions.

```julia
τ, Cs = differential_path(rve, DifferentialScheme(; nsteps = 200), :C)
ks = [k_mu(C)[1] for C in Cs]
```
"""
function differential_path(
        rve::RVE, scheme::DifferentialScheme, property::Symbol; kw...
    )
    validate_rve(rve)
    return _differential_states(rve, scheme, property; kw...)
end

function _differential_states(rve::RVE, scheme::DifferentialScheme, prop::Symbol; kw...)
    nsteps = get(scheme.options, :nsteps, 100)
    abstol = get(scheme.options, :abstol, 1.0e-8)
    reltol = get(scheme.options, :reltol, 1.0e-6)
    alg = get(scheme.options, :alg, nothing)
    formulation = _diff_formulation(scheme)
    paths = _resolve_paths(scheme.trajectory, rve, nsteps)
    P_matrix = matrix_property(rve, prop)
    # The compliance form integrates S = C⁻¹ and inverts back on the way out,
    # so the caller always receives the declared property.
    P_init = formulation === :compliance ? inv(P_matrix) : P_matrix
    τ, states = _diff_integrate_ode(
        rve, paths, prop, P_init;
        nsteps, abstol, reltol, alg, formulation,
        solver_kwargs = _diff_solver_kwargs(scheme.options), kw...
    )
    formulation === :compliance && (states = map(inv, states))
    return τ, states
end

# ── ODE integrator ──────────────────────────────────────────────────────────

function _diff_integrate_ode(
        rve::RVE,
        paths::AbstractDict{Symbol},
        prop::Symbol,
        P_init::TensND.AbstractTens;
        nsteps::Int = 100,
        abstol::Real = 1.0e-8,
        reltol::Real = 1.0e-6,
        alg = nothing,
        formulation::Symbol = :stiffness,
        solver_kwargs::NamedTuple = NamedTuple(),
        kw...
    )
    # Split inclusion phases between solids and cracks.  Targets are the
    # final values reached at `τ = 1` (volume fractions for solids,
    # densities for cracks).  The target eltype is preserved (it can be
    # `ForwardDiff.Dual` when the user differentiates `homogenize`
    # through its scalar inputs).
    solid_names = Symbol[]
    crack_names = Symbol[]
    incl = inclusion_phase_names(rve)
    T_target = isempty(incl) ? Float64 :
        promote_type((typeof(amount_value(rve.amounts[name])) for name in incl)...)
    targets = Dict{Symbol, T_target}()
    for name in incl
        a = rve.amounts[name]
        if a isa VolumeFraction
            push!(solid_names, name)
        else  # CrackDensity
            push!(crack_names, name)
        end
        targets[name] = T_target(amount_value(a))
    end

    # State : canonical components of the running estimate (length
    # 2 / 5 / 9 for iso/TI/ortho 4-tensors, 1 / 3 / 6 for the
    # corresponding 2-tensors, 36 / 9 for the fully-anisotropic Mandel
    # fallback).  `proto` carries the matrix values in the smallest
    # symmetry class the running estimate can stay in — the matrix's own
    # class when every phase preserves it, a wider one otherwise (see
    # `_diff_state_proto`).
    #
    # For ForwardDiff sensitivity the state eltype must accommodate every
    # input the integration touches.  The matrix property (through `x0`) and
    # the per-phase targets are only two of them: differentiating with
    # respect to an INCLUSION property, an inclusion geometry, an interface
    # stiffness or anything buried in a nested cell leaves both of those
    # `Float64` while the right-hand side returns `Dual`s — and the solver's
    # `du` buffer, whose eltype is fixed once and for all by `x0`, would then
    # reject them.  `T_contrib` closes that gap: it is the eltype the phase
    # kernels actually produce, read off the same probe that fixes the state
    # layout.
    dual = formulation === :compliance
    proto, T_contrib = _diff_state_proto(rve, P_init, prop, dual; kw...)
    sym_tag = _symmetry_tag(proto)
    x0 = _diff_initial_state(sym_tag, P_init, proto)
    T_state = promote_type(eltype(x0), T_target, T_contrib)
    x0 = T_state.(x0)
    ode_kw = (
        rve = rve,
        prop = prop,
        paths = paths,
        solid_names = solid_names,
        crack_names = crack_names,
        targets = targets,
        sym_tag = sym_tag,
        proto = proto,
        dual = dual,
        kw = kw,
    )
    rhs! = (du, u, p, τ) -> _diff_ode_rhs!(du, u, p, τ)
    prob = ODEProblem(rhs!, x0, (0.0, 1.0), ode_kw)
    τ_save = range(0.0, 1.0; length = max(nsteps, 1) + 1)
    sol = solve(
        prob,
        alg === nothing ? Tsit5() : alg;
        abstol, reltol,
        saveat = τ_save,
        dense = false,
        solver_kwargs...
    )
    states = [_reconstruct_tens(sym_tag, proto, u) for u in sol.u]
    return collect(sol.t), states
end

# ── RHS ─────────────────────────────────────────────────────────────────────

function _diff_ode_rhs!(du, u, p, τ)
    # In the dual form the state carries S and the phase kernels are
    # evaluated against C = S⁻¹ (closed-form inverse for ISO / TI / ortho).
    Pτ = _reconstruct_tens(p.sym_tag, p.proto, u)
    Cτ = p.dual ? inv(Pτ) : Pτ
    # Sherman-Morrison inversion of dφ = (𝟙 − f ⊗ 𝐔)^{-1} · df.
    n_solid = length(p.solid_names)
    f = Vector{eltype(u)}(undef, n_solid)
    df = Vector{eltype(u)}(undef, n_solid)
    @inbounds for (i, name) in enumerate(p.solid_names)
        nt = p.paths[name]
        f[i] = nt.f(τ) * p.targets[name]
        df[i] = nt.df(τ) * p.targets[name]
    end
    f0 = one(eltype(u)) - sum(f; init = zero(eltype(u)))
    sum_df = sum(df; init = zero(eltype(u)))
    Δ = zero(Pτ)
    @inbounds for (i, name) in enumerate(p.solid_names)
        # dφ_i / dτ = df_i + (f_i / f_0) · sum(df)   (Sherman-Morrison)
        dφᵢ = df[i] + (f[i] / f0) * sum_df
        iszero(dφᵢ) && continue
        Δ += dφᵢ * (
            p.dual ?
                _diff_compliance_correction(p.rve, name, p.prop, Cτ; p.kw...) :
                _diff_dilute_correction(p.rve, name, p.prop, Cτ; p.kw...)
        )
    end
    # Crack families carry no volume, so they contribute nothing to the sum
    # inverted above.  They are still diluted by it: replacing dφ of the
    # current medium by solid material destroys the cracks that piece of
    # medium contained, hence
    #     dε_c = dφ_c^ε − ε_c Σ_{j solid} dφ_j ,
    # which inverts with the very same Sherman-Morrison factor as the
    # solid fractions, the density playing the role of the fraction:
    #     dφ_c^ε = dε_c + (ε_c / f_0) Σ_{j solid} df_j .
    # The correction vanishes for a crack-only RVE and whenever no solid
    # phase grows at the same τ (e.g. `Sequential`).
    @inbounds for name in p.crack_names
        nt = p.paths[name]
        εᶜ = nt.f(τ) * p.targets[name]
        dφᶜ = nt.df(τ) * p.targets[name] + (εᶜ / f0) * sum_df
        iszero(dφᶜ) && continue
        Δ += dφᶜ * (
            p.dual ?
                _diff_crack_density_compliance_kernel(p.rve, name, p.prop, Cτ; p.kw...) :
                _diff_crack_density_kernel(p.rve, name, p.prop, Cτ; p.kw...)
        )
    end
    _set_state_layout!(du, p.sym_tag, p.proto, Δ)
    return nothing
end

# Write `Δ` into the state layout fixed at setup.  `Δ` normally lives in
# exactly that class, but it can come out NARROWER — every increment is
# zero at a `Sequential` window boundary, and `zero(::TensTI)` is a
# `TensISO` — in which case its canonical components are re-expressed in
# the layout's basis by the same linear embedding as `_diff_embed`.
function _set_state_layout!(du, tag, proto, Δ)
    _diff_same_layout(Δ, proto) && return _set_state!(du, tag, Δ)
    du .= _get_state(tag, Δ + proto) .- _get_state(tag, proto)
    return nothing
end

# ── Symmetry tag + reconstruction helpers ──────────────────────────────────

# We use a small sum type stored in `p.sym_tag` to dispatch the
# reconstruction (and the initial flatten) without a dynamic
# constructor lookup at every RHS step.
#
# Which state representation must the ODE use?  The canonical components
# of the matrix property are the cheapest state, but they are only
# usable while the running estimate STAYS in the matrix's symmetry
# class — and a phase whose contribution is of a lower symmetry drags it
# out at the very first step.  Crack families are the obvious case (a TI
# contribution in the crack's own frame), but so is any aligned
# non-spherical solid inclusion: `A_dil` of a spheroid is TI even when
# both the matrix and the inclusion are isotropic.
#
# Rather than enumerate the symmetry algebra of every geometry ×
# property × `symmetrize` combination, we ask the kernels themselves.
# Summing the matrix property and one unit contribution per phase
# produces, through TensND's own promotion, exactly the smallest class
# that can hold the running estimate (`TensISO + TensTI → TensTI`,
# two misaligned `TensTI` → the full anisotropic type, …).  Evaluating
# the contributions against a richer medium can widen the class again,
# so the probe is iterated; the class chain ISO ⊂ TI ⊂ ortho ⊂ full is
# short, so it converges in a couple of passes.
#
# Using the *join* rather than the full Mandel fallback matters twice
# over: the state stays small, and the running estimate keeps a
# structured TensND type, so the Hill-tensor backends keep dispatching
# to their closed-form paths instead of the numerical residue
# algorithm.
#
# The probe costs a few kernel evaluations, against the hundreds the
# integration itself performs.
#
# The same probe also reports the eltype the kernels produce (second
# return value), which is what the ODE state must be able to hold — see
# `T_contrib` in `_diff_integrate_ode`.
function _diff_state_proto(rve::RVE, P_init::TensND.AbstractTens, prop::Symbol, dual::Bool; kw...)
    incl = inclusion_phase_names(rve)
    T_contrib = eltype(P_init)
    isempty(incl) && return P_init, T_contrib
    proto = P_init
    for pass in 1:4
        # The kernels always take the stiffness-like tensor; in the dual
        # formulation the state — and hence `proto` — is its inverse.
        ref = dual ? inv(proto) : proto
        acc = proto
        for name in incl
            contribution = try
                _diff_probe_contribution(rve, name, prop, ref, dual; kw...)
            catch err
                # The widened reference is a fully anisotropic *type* holding,
                # at τ = 0, isotropic *values* — the case on which the residue
                # algorithm's acoustic polynomial degenerates.  `:auto` routes
                # around it (it selects a cubature, see `Core/dispatch.jl`), so
                # this is only reachable when the caller asked for
                # `method = :residues` explicitly.
                pass == 1 && rethrow()
                throw(
                    ArgumentError(
                        "DifferentialScheme: the running effective medium of " *
                            "this RVE is fully anisotropic from the first step, " *
                            "and the Hill-tensor backend selected for phase " *
                            ":$(name) cannot be evaluated against it while its " *
                            "values are still isotropic " *
                            "($(sprint(showerror, err))). The residue algorithm " *
                            "degenerates there: drop `method = :residues` and let " *
                            "`:auto` pick a cubature, or give the phase an " *
                            "orientation average (`symmetrize = :iso`), which " *
                            "keeps the running medium isotropic."
                    )
                )
            end
            T_contrib = promote_type(T_contrib, eltype(contribution))
            acc += contribution
        end
        _diff_same_layout(acc, proto) && break
        proto = _diff_embed(P_init, acc)
    end
    return proto, T_contrib
end

# `P_init`, re-expressed in the symmetry class of `acc` (a wider class,
# so the embedding is exact).  Widening by arithmetic does not work —
# `zero(::TensTI)` simplifies back to `TensISO`, and TensND rebuilds the
# narrowest class its values allow.  The canonical components are linear
# in the tensor, so reading them off `P_init + acc` and subtracting
# those of `acc` yields `P_init`'s components in `acc`'s basis, which
# `_reconstruct_tens` turns back into a tensor of that class (`acc` also
# carries the axis for the TI / ortho classes).
function _diff_embed(P_init, acc)
    tag = _symmetry_tag(acc)
    comps = _get_state(tag, P_init + acc) .- _get_state(tag, acc)
    return _reconstruct_tens(tag, acc, comps)
end

# Initial state vector.  The Mandel layouts read any tensor class, so the
# matrix property is flattened DIRECTLY there — going through the embedded
# `proto` would cost the last bits (`comps(P + acc) − comps(acc)` is a
# subtraction of nearly equal numbers), and an adaptive solver turns a
# perturbed initial condition into a different step sequence.  The
# structured layouts have no such shortcut: `_get_state` needs a tensor
# already in the class, i.e. `proto`.
_diff_initial_state(tag::Val{:full_4}, P_init, proto) = _get_state(tag, P_init)
_diff_initial_state(tag::Val{:full_2}, P_init, proto) = _get_state(tag, P_init)
_diff_initial_state(tag::Val, P_init, proto) = _get_state(tag, proto)

# Unit contribution of one phase, in the formulation the ODE integrates.
function _diff_probe_contribution(
        rve::RVE, name::Symbol, prop::Symbol, ref, dual::Bool; kw...
    )
    is_crack = rve.amounts[name] isa CrackDensity
    return if dual
        is_crack ?
            _diff_crack_density_compliance_kernel(rve, name, prop, ref; kw...) :
            _diff_compliance_correction(rve, name, prop, ref; kw...)
    else
        is_crack ?
            _diff_crack_density_kernel(rve, name, prop, ref; kw...) :
            _diff_dilute_correction(rve, name, prop, ref; kw...)
    end
end

# Same symmetry class *and* same number of canonical components — the
# two conditions for one state vector to be usable for the other tensor
# (`TensTI` comes in a 5- and an 8-component flavor).  Called once per
# RHS evaluation, so it reads the component count off the (statically
# sized) canonical data rather than materializing a state vector.
function _diff_same_layout(a, b)
    tag = _symmetry_tag(a)
    tag === _symmetry_tag(b) || return false
    return _state_length(tag, a) == _state_length(tag, b)
end

_state_length(::Val, t) = length(TensND.get_data(t))
_state_length(::Val{:full_4}, t) = 36
_state_length(::Val{:full_2}, t) = 9

_symmetry_tag(::TensND.TensISO{4}) = Val(:iso_4)
_symmetry_tag(::TensND.TensISO{2}) = Val(:iso_2)
_symmetry_tag(::TensND.TensTI) = Val(:ti)
_symmetry_tag(::TensND.TensOrtho) = Val(:ortho)
# Fully-anisotropic 4-tensor (`TensCanonical`, `Tens`, …) — flatten
# the full 6×6 Mandel matrix as the ODE state (36 components ; not
# minimal but generic).
_symmetry_tag(::TensND.AbstractTens{4, 3}) = Val(:full_4)
_symmetry_tag(::TensND.AbstractTens{2, 3}) = Val(:full_2)

# Flatten / reconstruct helpers : the iso / TI / ortho cases use the
# canonical components ; the full-aniso fallback uses the Mandel
# array.

# Initial state vector for the ODE solver.
_get_state(::Val{:iso_4}, t) = collect(TensND.get_data(t))
_get_state(::Val{:iso_2}, t) = collect(TensND.get_data(t))
_get_state(::Val{:ti}, t) = collect(TensND.get_data(t))
_get_state(::Val{:ortho}, t) = collect(TensND.get_data(t))
_get_state(::Val{:full_4}, t) = vec(collect(TensND.KM(t)))
_get_state(::Val{:full_2}, t) = vec(collect(TensND.KM(t)))

# Push a tensor into a flat vector for `du`.
_set_state!(du, ::Val{:iso_4}, Δ) = (du .= TensND.get_data(Δ); nothing)
_set_state!(du, ::Val{:iso_2}, Δ) = (du .= TensND.get_data(Δ); nothing)
_set_state!(du, ::Val{:ti}, Δ) = (du .= TensND.get_data(Δ); nothing)
_set_state!(du, ::Val{:ortho}, Δ) = (du .= TensND.get_data(Δ); nothing)
_set_state!(du, ::Val{:full_4}, Δ) = (du .= vec(TensND.KM(Δ)); nothing)
_set_state!(du, ::Val{:full_2}, Δ) = (du .= vec(TensND.KM(Δ)); nothing)

# Reconstruct the tensor from a state vector.
_reconstruct_tens(::Val{:iso_4}, ::TensND.AbstractTens, u) =
    TensND.TensISO{3}(u[1], u[2])
_reconstruct_tens(::Val{:iso_2}, ::TensND.AbstractTens, u) =
    TensND.TensISO{3}(u[1])
_reconstruct_tens(::Val{:ti}, proto::TensND.AbstractTens, u) =
    TensND._rebuild(proto, ntuple(i -> u[i], length(u)))
_reconstruct_tens(::Val{:ortho}, proto::TensND.AbstractTens, u) =
    TensND._rebuild(proto, ntuple(i -> u[i], length(u)))
# Fully-anisotropic fallback : rebuild from the Mandel 6×6 form.
_reconstruct_tens(::Val{:full_4}, ::TensND.AbstractTens, u) =
    TensND.inv_KM(reshape(u, 6, 6))
_reconstruct_tens(::Val{:full_2}, ::TensND.AbstractTens, u) =
    TensND.inv_KM(reshape(u, 3, 3))

# ── Per-phase contribution helpers ──────────────────────────────────────────

# Dilute correction `(C_i − C) ⊡ A_dil(C)` for a solid inclusion phase
# (symmetrize honored through `_phase_dilute_concentration`).
function _diff_dilute_correction(
        rve::RVE, name::Symbol, prop::Symbol,
        P_curr::TensND.AbstractTens{4, 3}; kw...
    )
    geom = rve.phases[name].geometry
    P_i = phase_property(rve, name, prop)
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P_curr, sym)
    N = MFH_Core.stiffness_contribution(geom, P_i, P₀_proj; kw...)
    return _apply_symmetrize(N, sym)
end

function _diff_dilute_correction(
        rve::RVE, name::Symbol, prop::Symbol,
        P_curr::TensND.AbstractTens{2, 3}; kw...
    )
    geom = rve.phases[name].geometry
    P_i = phase_property(rve, name, prop)
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P_curr, sym)
    N = MFH_Core.conductivity_contribution(geom, P_i, P₀_proj; kw...)
    return _apply_symmetrize(N, sym)
end

# Crack contribution kernel **per unit density** : returns
# `delta_stiffness(geom, N, 1.0)` (or `delta_conductivity` for 2-tensor)
# so the RHS multiplies by `dε/dτ` directly.
function _diff_crack_density_kernel(
        rve::RVE, name::Symbol, prop::Symbol,
        P_curr::TensND.AbstractTens{4, 3}; kw...
    )
    geom = rve.phases[name].geometry
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P_curr, sym)
    K_int = _crack_interface_K4(rve, name)
    N = MFH_Core.stiffness_contribution(
        geom, P₀_proj;
        K_interface = K_int, kw...
    )
    return _apply_symmetrize(MFH_Core.delta_stiffness(geom, N, 1.0), sym)
end

function _diff_crack_density_kernel(
        rve::RVE, name::Symbol, prop::Symbol,
        P_curr::TensND.AbstractTens{2, 3}; kw...
    )
    geom = rve.phases[name].geometry
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P_curr, sym)
    α_int = _crack_interface_α(rve, name)
    N = MFH_Core.conductivity_contribution(
        geom, P₀_proj;
        α_interface = α_int, kw...
    )
    return _apply_symmetrize(MFH_Core.delta_conductivity(geom, N, 1.0), sym)
end

# ── Dual (compliance-side) per-phase kernels ────────────────────────────────
#
# Mirror images of the four kernels above, evaluated against the same
# running STIFFNESS `P_curr` (the caller inverts the state once per RHS
# step) but returning the compliance / resistivity contribution, so the
# ODE integrates S rather than C.  Heterogeneous inclusions route through
# the `ℍ = −𝕊 : 𝐍 : 𝕊` fallback of `_phase_compliance_contribution`.

# Unit-amount compliance contribution `ℍ_i(C)` of a solid inclusion phase.
function _diff_compliance_correction(
        rve::RVE, name::Symbol, prop::Symbol,
        P_curr::TensND.AbstractTens{4, 3}; kw...
    )
    geom = rve.phases[name].geometry
    P_i = phase_property(rve, name, prop)
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P_curr, sym)
    H = if MFH_Core.is_homogeneous_inclusion(geom)
        compliance_contribution(geom, P_i, P₀_proj; kw...)
    else
        S₀ = inv(P₀_proj)
        -(S₀ ⊡ MFH_Core.stiffness_contribution(geom, P_i, P₀_proj; kw...) ⊡ S₀)
    end
    return _apply_symmetrize(H, sym)
end

function _diff_compliance_correction(
        rve::RVE, name::Symbol, prop::Symbol,
        P_curr::TensND.AbstractTens{2, 3}; kw...
    )
    geom = rve.phases[name].geometry
    P_i = phase_property(rve, name, prop)
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P_curr, sym)
    R = if MFH_Core.is_homogeneous_inclusion(geom)
        MFH_Core.resistivity_contribution(geom, P_i, P₀_proj; kw...)
    else
        R₀ = inv(P₀_proj)
        -(R₀ ⋅ MFH_Core.conductivity_contribution(geom, P_i, P₀_proj; kw...) ⋅ R₀)
    end
    return _apply_symmetrize(R, sym)
end

# Crack compliance kernel **per unit density** — the natural form for a
# crack, whose compliance contribution is finite by construction.
function _diff_crack_density_compliance_kernel(
        rve::RVE, name::Symbol, prop::Symbol,
        P_curr::TensND.AbstractTens{4, 3}; kw...
    )
    geom = rve.phases[name].geometry
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P_curr, sym)
    K_int = _crack_interface_K4(rve, name)
    H = compliance_contribution(geom, P₀_proj; K_interface = K_int, kw...)
    return _apply_symmetrize(delta_compliance(geom, H, 1.0), sym)
end

function _diff_crack_density_compliance_kernel(
        rve::RVE, name::Symbol, prop::Symbol,
        P_curr::TensND.AbstractTens{2, 3}; kw...
    )
    geom = rve.phases[name].geometry
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P_curr, sym)
    α_int = _crack_interface_α(rve, name)
    R = compliance_contribution(geom, P₀_proj; α_interface = α_int, kw...)
    return _apply_symmetrize(delta_resistivity(geom, R, 1.0), sym)
end
