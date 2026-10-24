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

Differential scheme for property `:p` [norris1985](@cite).
Integrates the multi-phase incorporation-sequence ODE on ``\\tau \\in [0, 1]``
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
fictitious incorporation time ``\\tau \\in [0, 1]``, instead of only at ``\\tau = 1``
as [`homogenize`](@ref) returns.

``\\tau`` carries `nsteps + 1` uniformly spaced save points (the `saveat` of
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
    validate_cell(rve, scheme)
    return _differential_states(rve, scheme, property; kw...)
end

function _differential_states(rve::RVE, scheme::DifferentialScheme, prop::Symbol; kw...)
    nsteps = get(scheme.options, :nsteps, 100)
    abstol = get(scheme.options, :abstol, 1.0e-8)
    reltol = get(scheme.options, :reltol, 1.0e-6)
    alg = get(scheme.options, :alg, nothing)
    formulation = _diff_formulation(scheme)
    m = matrix_name(scheme, rve)
    paths = _resolve_paths(scheme.trajectory, rve, nsteps, m)
    P_matrix = phase_property(rve, m, prop)
    # The compliance form integrates S = C⁻¹ and inverts back on the way out,
    # so the caller always receives the declared property.
    P_init = formulation === :compliance ? inv(P_matrix) : P_matrix
    τ, states = _diff_integrate_ode(
        rve, paths, prop, m, P_init;
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
        matrix::Symbol,
        P_init::TensND.AbstractTens;
        nsteps::Int = 100,
        abstol::Real = 1.0e-8,
        reltol::Real = 1.0e-6,
        alg = nothing,
        formulation::Symbol = :stiffness,
        solver_kwargs::NamedTuple = NamedTuple(),
        kw...
    )
    ode_kw, x0 = _diff_setup(rve, paths, prop, matrix, P_init, formulation; kw...)
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
    states = [_reconstruct_tens(ode_kw.sym_tag, ode_kw.proto, u) for u in sol.u]
    return collect(sol.t), states
end

# Everything the right-hand side needs, and the initial state: the phases split
# between solids and cracks with their targets, and the state layout.
function _diff_setup(
        rve::RVE, paths::AbstractDict{Symbol}, prop::Symbol, matrix::Symbol,
        P_init::TensND.AbstractTens, formulation::Symbol; kw...
    )
    # Split inclusion phases between solids and cracks.  Targets are the
    # final values reached at `τ = 1` (volume fractions for solids,
    # densities for cracks).  The target eltype is preserved (it can be
    # `ForwardDiff.Dual` when the user differentiates `homogenize`
    # through its scalar inputs).
    solid_names = Symbol[]
    crack_names = Symbol[]
    incl = inclusion_phase_names(rve, matrix)
    # The scheme integrates every non-matrix phase from zero up to its declared
    # amount, so each of them needs one. The phase taking up the volume
    # complement has none — it is whatever the others leave — and is only
    # meaningful here as the matrix itself.
    for name in incl
        rve.amounts[name] isa Remainder && throw(
            ArgumentError(
                "DifferentialScheme integrates each phase up to its declared amount, " *
                    "and phase :$(name) carries the volume complement instead of one. " *
                    "Either name it as the scheme's matrix — " *
                    "DifferentialScheme(:$(name)) — or give it an explicit fraction."
            )
        )
    end
    T_target = isempty(incl) ? Float64 :
        promote_type((typeof(amount_value(rve.amounts[name])) for name in incl)...)
    targets = Dict{Symbol, T_target}()
    for name in incl
        a = rve.amounts[name]
        push!(_is_crack_family(a) ? crack_names : solid_names, name)
        targets[name] = T_target(amount_value(a))
    end

    # State : canonical components of the running estimate (length
    # 2 / 5 / 9 for iso/TI/ortho 4-tensors, 1 / 3 / 6 for the
    # corresponding 2-tensors, 36 / 6 for the fully-anisotropic Mandel
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
    proto, T_contrib = _diff_state_proto(rve, P_init, prop, matrix, dual; kw...)
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
    return ode_kw, x0
end

# ── RHS ─────────────────────────────────────────────────────────────────────

function _diff_ode_rhs!(du, u, p, τ)
    # In the dual form the state carries S and the phase kernels are
    # evaluated against C = S⁻¹ (closed-form inverse for ISO / TI / ortho).
    Pτ = _reconstruct_tens(p.sym_tag, p.proto, u)
    Cτ = p.dual ? inv(Pτ) : Pτ
    # The rates take the type of τ as well as the state's: an implicit solver
    # differentiates the right-hand side in τ with dual numbers.
    dφ, dφᶜ = _diff_rates(p, τ, promote_type(eltype(u), typeof(τ)))
    Δ = zero(Pτ)
    @inbounds for (i, name) in enumerate(p.solid_names)
        dφᵢ = dφ[i]
        iszero(dφᵢ) && continue
        Δ += dφᵢ * (
            p.dual ?
                _diff_compliance_correction(p.rve, name, p.prop, Cτ; p.kw...) :
                _diff_dilute_correction(p.rve, name, p.prop, Cτ; p.kw...)
        )
    end
    @inbounds for (c, name) in enumerate(p.crack_names)
        dφᶜ_c = dφᶜ[c]
        iszero(dφᶜ_c) && continue
        Δ += dφᶜ_c * (
            p.dual ?
                _diff_crack_density_compliance_kernel(p.rve, name, p.prop, Cτ; p.kw...) :
                _diff_crack_density_kernel(p.rve, name, p.prop, Cτ; p.kw...)
        )
    end
    _set_state_layout!(du, p.sym_tag, p.proto, Δ)
    return nothing
end

# Rates at which the solid phases and the crack families are incorporated into
# the current medium, dφ_i/dτ and dφ_c^ε/dτ, from the prescribed df_i/dτ.
# Sherman-Morrison inversion of dφ = (𝟙 − f ⊗ 𝐔)⁻¹ · df for the solids.
# Crack families carry no volume, so they contribute nothing to the sum
# inverted.  They are still diluted by it: replacing dφ of the current medium
# by solid material destroys the cracks that piece of medium contained, hence
#     dε_c = dφ_c^ε − ε_c Σ_{j solid} dφ_j ,
# which inverts with the very same Sherman-Morrison factor as the solid
# fractions, the density playing the role of the fraction:
#     dφ_c^ε = dε_c + (ε_c / f_0) Σ_{j solid} df_j .
# The correction vanishes for a crack-only RVE and whenever no solid phase
# grows at the same τ (e.g. `Sequential`).
function _diff_rates(p, τ, ::Type{T}) where {T}
    f, df, f0, sum_df = _diff_amounts(p, τ, T)
    dφ = T[df[i] + (f[i] / f0) * sum_df for i in eachindex(f)]
    dφᶜ = T[
        p.paths[name].df(τ) * p.targets[name] + (p.paths[name].f(τ) * p.targets[name] / f0) * sum_df
            for name in p.crack_names
    ]
    return dφ, dφᶜ
end

# Fractions of the solid phases at `τ`, their rates, the matrix fraction and the
# total rate.
function _diff_amounts(p, τ, ::Type{T}) where {T}
    f = T[p.paths[name].f(τ) * p.targets[name] for name in p.solid_names]
    df = T[p.paths[name].df(τ) * p.targets[name] for name in p.solid_names]
    return f, df, one(T) - sum(f; init = zero(T)), sum(df; init = zero(T))
end

# The same rates per unit target amount, solids then cracks: finite when a
# target vanishes, they feed the concentration tensor of a phase rather than
# its weighted tensor (see `concentrations.jl`).
function _diff_unit_rates(p, τ, ::Type{T}) where {T}
    _, _, f0, sum_df = _diff_amounts(p, τ, T)
    return T[
        p.paths[name].df(τ) + (p.paths[name].f(τ) / f0) * sum_df
            for name in vcat(p.solid_names, p.crack_names)
    ]
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
function _diff_state_proto(
        rve::RVE, P_init::TensND.AbstractTens, prop::Symbol, matrix::Symbol, dual::Bool; kw...
    )
    incl = inclusion_phase_names(rve, matrix)
    T_contrib = eltype(P_init)
    isempty(incl) && return P_init, T_contrib
    proto = P_init
    for _ in 1:4
        # The kernels always take the stiffness-like tensor; in the dual
        # formulation the state — and hence `proto` — is its inverse.
        ref = dual ? inv(proto) : proto
        acc = proto
        for name in incl
            contribution = _diff_probe_contribution(rve, name, prop, ref, dual; kw...)
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
    is_crack = _is_crack_family(rve.amounts[name])
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
    return _state_length(tag, a) == _state_length(tag, b) && _same_orientation(tag, a, b)
end

# A structured state holds coefficients about an axis or a frame: one tensor's
# coefficients stand for the other's only about the same one. `isequal`, not
# `==`: it answers a `Bool` on symbolic axes too.
_same_orientation(::Val{:ti}, a, b) = isequal(TensND.axis(a), TensND.axis(b))
_same_orientation(::Val{:ortho}, a, b) = isequal(TensND.frame(a), TensND.frame(b))
_same_orientation(::Val, a, b) = true

_state_length(::Val, t) = length(TensND.get_data(t))
_state_length(::Val{:full_4}, t) = 36
# A conductivity is symmetric: its Mandel vector has 6 components.
_state_length(::Val{:full_2}, t) = 6

_symmetry_tag(::TensND.TensISO{4}) = Val(:iso_4)
_symmetry_tag(::TensND.TensISO{2}) = Val(:iso_2)
_symmetry_tag(::TensND.TensTI) = Val(:ti)
_symmetry_tag(::TensND.TensOrtho) = Val(:ortho)
# Fully-anisotropic 4-tensor (`TensCanonical`, `Tens`, …) — flatten
# the full 6×6 Mandel matrix as the ODE state (36 components ; not
# minimal but generic).
_symmetry_tag(::TensND.AbstractTens{4, 3}) = Val(:full_4)
_symmetry_tag(::TensND.AbstractTens{2, 3}) = Val(:full_2)

# Flatten / reconstruct: TensND's `to_vector` / `from_vector`, the coefficients
# of a structured class (with its axis or frame), the canonical Kelvin-Mandel
# components of a generic tensor (an anisotropic Hill tensor is written in its
# inclusion's basis, and `from_vector` rebuilds it canonical).
# The full layouts hold canonical Kelvin-Mandel components whatever the class of
# the tensor read into them (the matrix property may be isotropic while the state
# is anisotropic): `generic_tens` first.
const _FullLayout = Union{Val{:full_4}, Val{:full_2}}
_get_state(::Val, t) = TensND.to_vector(t)
_get_state(::_FullLayout, t) = TensND.to_vector(TensND.generic_tens(t))
_set_state!(du, tag::Val, Δ) = (du .= _get_state(tag, Δ); nothing)
_reconstruct_tens(::Val, proto::TensND.AbstractTens, u) = TensND.from_vector(proto, u)

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
    N = MFH_Core.property_contribution(geom, P_i, P₀_proj; kw...)
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
    N = MFH_Core.property_contribution(geom, P_i, P₀_proj; kw...)
    return _apply_symmetrize(N, sym)
end

# Crack contribution kernel **per unit density** : returns
# `delta_property(geom, N, 1.0)`, of either order,
# so the RHS multiplies by `dε/dτ` directly.
function _diff_crack_density_kernel(
        rve::RVE, name::Symbol, prop::Symbol,
        P_curr::TensND.AbstractTens{4, 3}; kw...
    )
    geom = rve.phases[name].geometry
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P_curr, sym)
    K_int = _crack_interface_K4(rve, name)
    N = MFH_Core.property_contribution(
        geom, P₀_proj;
        K_interface = K_int, kw...
    )
    return _apply_symmetrize(MFH_Core.delta_property(geom, N, 1.0), sym)
end

function _diff_crack_density_kernel(
        rve::RVE, name::Symbol, prop::Symbol,
        P_curr::TensND.AbstractTens{2, 3}; kw...
    )
    geom = rve.phases[name].geometry
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P_curr, sym)
    α_int = _crack_interface_α(rve, name)
    N = MFH_Core.property_contribution(
        geom, P₀_proj;
        α_interface = α_int, kw...
    )
    return _apply_symmetrize(MFH_Core.delta_property(geom, N, 1.0), sym)
end

# ── Dual (compliance-side) per-phase kernels ────────────────────────────────
#
# Mirror images of the four kernels above, evaluated against the same
# running STIFFNESS `P_curr` (the caller inverts the state once per RHS
# step) but returning the compliance / resistivity contribution, so the
# ODE integrates S rather than C.  `inverse_property_contribution` evaluates
# `ℍ = −𝕊 : 𝐍 : 𝕊`, which holds for heterogeneous inclusions and voids alike.

# Unit-amount compliance contribution `ℍ_i(C)` of a solid inclusion phase.
function _diff_compliance_correction(
        rve::RVE, name::Symbol, prop::Symbol,
        P_curr::TensND.AbstractTens{4, 3}; kw...
    )
    geom = rve.phases[name].geometry
    P_i = phase_property(rve, name, prop)
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P_curr, sym)
    H = inverse_property_contribution(geom, P_i, P₀_proj; kw...)
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
    R = MFH_Core.inverse_property_contribution(geom, P_i, P₀_proj; kw...)
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
    H = inverse_property_contribution(geom, P₀_proj; K_interface = K_int, kw...)
    return _apply_symmetrize(delta_inverse_property(geom, H, 1.0), sym)
end

function _diff_crack_density_compliance_kernel(
        rve::RVE, name::Symbol, prop::Symbol,
        P_curr::TensND.AbstractTens{2, 3}; kw...
    )
    geom = rve.phases[name].geometry
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P_curr, sym)
    α_int = _crack_interface_α(rve, name)
    R = inverse_property_contribution(geom, P₀_proj; α_interface = α_int, kw...)
    return _apply_symmetrize(delta_inverse_property(geom, R, 1.0), sym)
end
