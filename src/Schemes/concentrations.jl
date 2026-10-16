# =============================================================================
#  concentrations.jl — concentration and contribution tensors of every phase,
#  for every scheme on an `RVE`, and Levin's theorem.
#
#  A scheme estimates the mean strain and stress of each phase under the
#  macroscopic strain E,
#
#      ⟨ε⟩_r = 𝔸_r : E ,        ⟨σ⟩_r = 𝔸_σε,r : E ,
#
#  with Σ_r w_r 𝔸_r = 𝕀 and Σ_r w_r 𝔸_σε,r = ℂ^hom, w_r being the volume
#  fraction of a phase, or the density of a crack family, whose tensors are then
#  given per unit density.  The one-site schemes share one construction: each
#  phase solves its dilute problem in a reference medium, loaded at infinity by
#  a field 𝔼 : E that the scheme fixes,
#
#      𝔸_r = 𝔸_r^dil : 𝔼 ,       𝔸_σε,r = 𝔸_σε,r^dil : 𝔼 ,
#
#  and, when the scheme singles out a matrix, the matrix takes the rest of both
#  averages.  The theory pages derive 𝔼 scheme by scheme.
# =============================================================================

"""
    ConcentrationTensors

Concentration tensors of every phase of a cell under one scheme, as returned by
[`concentration_tensors`](@ref).

For each phase `r`, the strain concentration tensor ``\\mathbb{A}_r`` and the
stress–strain one ``\\mathbb{A}_{\\sigma\\varepsilon,r}`` map the macroscopic strain onto the
mean strain and the mean stress of the phase,
``\\langle\\boldsymbol{\\varepsilon}\\rangle_r = \\mathbb{A}_r:\\boldsymbol{E}`` and
``\\langle\\boldsymbol{\\sigma}\\rangle_r = \\mathbb{A}_{\\sigma\\varepsilon,r}:\\boldsymbol{E}``. They satisfy

```math
\\sum_r w_r\\,\\mathbb{A}_r = \\mathbb{I} ,
\\qquad
\\sum_r w_r\\,\\mathbb{A}_{\\sigma\\varepsilon,r} = \\mathbb{C}^{\\mathrm{hom}} ,
```

``w_r`` being the volume fraction of the phase. A crack family has no volume: its
weight is its density, its strain tensor is the strain carried by its openings
per unit density, and its stress tensor is zero. In conduction the same holds
with gradients and fluxes.

Read the tensors with [`strain_strain_concentration`](@ref),
[`stress_strain_concentration`](@ref), [`strain_stress_concentration`](@ref) and
[`stress_stress_concentration`](@ref) (or their conduction twins), the
contributions with [`stiffness_contribution`](@ref) and
[`compliance_contribution`](@ref), and apply Levin's theorem with
[`levin_free_strain`](@ref) and [`levin_prestress`](@ref).

# Fields

- `effective` — ``\\mathbb{C}^{\\mathrm{hom}}`` (or ``\\boldsymbol{K}^{\\mathrm{hom}}``);
- `compliance` — its inverse;
- `names` — the phases, in declaration order;
- `strain`, `stress` — the tensors of each phase;
- `weights` — the volume fraction of each phase, or the density of a crack family;
- `cracks` — the crack families among the phases;
- `properties` — the property of each homogeneous phase, usable as a reference;
- `reference` — the default reference phase of the contributions, `nothing` when
  the scheme has none.
"""
struct ConcentrationTensors{T <: TensND.AbstractTens, S <: TensND.AbstractTens, D <: AbstractDict{Symbol}, W <: AbstractDict{Symbol}, Q <: AbstractDict{Symbol}}
    effective::T
    compliance::S
    names::Vector{Symbol}
    # Heterogeneous by nature: each phase keeps its own symmetry class (an
    # isotropic matrix next to a transversely isotropic inclusion), so the
    # values are abstract tensors. Nothing here is on a hot path.
    strain::D
    stress::D
    weights::W
    cracks::Vector{Symbol}
    properties::Q
    reference::Union{Nothing, Symbol}
end

Base.keys(ct::ConcentrationTensors) = ct.names

function Base.show(io::IO, ct::ConcentrationTensors)
    print(io, "ConcentrationTensors(", join(ct.names, ", "), ")")
    return nothing
end

# Order-agnostic composition: double contraction at order 4, single at order 2.
_compose(A::TensND.AbstractTens{4, 3}, B) = A ⊡ B
_compose(A::TensND.AbstractTens{2, 3}, B) = A ⋅ B

const _ConcentrationTensors4 = ConcentrationTensors{<:TensND.AbstractTens{4, 3}}
const _ConcentrationTensors2 = ConcentrationTensors{<:TensND.AbstractTens{2, 3}}

# ── Entry points ────────────────────────────────────────────────────────────

"""
    concentration_tensors(cell, scheme, property::Symbol; kw...) -> ConcentrationTensors
    concentration_tensors(cell, scheme; property = :C, kw...)

Concentration tensors of every phase of `cell` under `scheme`, for the property
`property` (`:C` in elasticity, `:K` in conduction), together with the effective
tensor they are consistent with. The keyword arguments are those of
[`homogenize`](@ref).

Every scheme on an [`RVE`](@ref) is supported. For a [`CrackDensity`](@ref)
phase the tensors are given per unit density; the bounds ignore cracks, as they
do in [`homogenize`](@ref).

```julia
ct = concentration_tensors(rve, MoriTanaka(:M), :C)
A = strain_strain_concentration(ct, :I)        # ⟨ε⟩_I = A : E
N = stiffness_contribution(ct, :I)             # ⟨σ − C_M : ε⟩_I = N : E
```
"""
function concentration_tensors(
        cell::AbstractHomogenizationCell, scheme::HomogenizationScheme, property::Symbol; kw...
    )
    validate_cell(cell, scheme)
    return MFH_Core._with_nested_cache() do
        _concentration_tensors(cell, scheme, Val(property); kw...)
    end
end

function concentration_tensors(
        cell::AbstractHomogenizationCell, scheme::HomogenizationScheme;
        property::Symbol = :C, kw...
    )
    return concentration_tensors(cell, scheme, property; kw...)
end

# The schemes whose effective tensor comes from `homogenize` itself; the
# differential scheme integrates its concentration tensors along the path and
# overrides this method.
function _concentration_tensors(rve::RVE, scheme::HomogenizationScheme, ::Val{p}; kw...) where {p}
    P_hom = _evaluate(rve, scheme, Val(p); kw...)
    names, strain, stress = _scheme_concentrations(rve, scheme, Val(p), P_hom; kw...)
    return _concentration_result(rve, scheme, p, P_hom, names, strain, stress)
end

function _concentration_result(rve::RVE, scheme, prop::Symbol, P_hom, names, strain, stress)
    weights = Dict(n => _phase_weight(rve, n) for n in names)
    cracks = Symbol[n for n in names if _is_crack_family(rve.amounts[n])]
    properties = Dict{Symbol, TensND.AbstractTens}(
        n => phase_property(rve, n, prop) for n in rve.phase_names
            if !_is_crack_family(rve.amounts[n]) &&
            MFH_Core.is_homogeneous_inclusion(rve.phases[n].geometry) &&
            haskey(rve.phases[n].properties, prop)
    )
    reference = requires_matrix(scheme) ? matrix_name(scheme, rve) : nothing
    return ConcentrationTensors(
        P_hom, inv(P_hom), collect(names), strain, stress, weights, cracks, properties, reference
    )
end

# Assembly of the result for a cell that is not an `RVE`: a laminate or a
# particle assembly, whose phases all carry a volume.
function _concentration_result(P_hom, names, strain, stress, weights, properties, reference)
    return ConcentrationTensors(
        P_hom, inv(P_hom), collect(names), strain, stress, weights, Symbol[], properties, reference
    )
end

function _concentration_tensors(cell::AbstractHomogenizationCell, scheme::HomogenizationScheme, ::Val; kw...)
    throw(
        ArgumentError(
            "concentration tensors are not available for $(nameof(typeof(scheme))) " *
                "on a $(nameof(typeof(cell)))"
        )
    )
end

# ── Phase-level ingredients ─────────────────────────────────────────────────

# Weight of a phase in the averages: its volume fraction, or the density of a
# crack family, whose tensors are per unit density.
_phase_weight(rve::RVE, name::Symbol) = _phase_weight(rve.amounts[name], rve, name)
_phase_weight(a::CrackDensity, ::RVE, ::Symbol) = amount_value(a)
_phase_weight(::AbstractAmount, rve::RVE, name::Symbol) = volume_fraction(rve, name)

# Compliance (resistivity) contribution of a crack family per unit density,
# geometric prefactor included.
function _crack_compliance_per_density(
        rve::RVE, name::Symbol, P_ref::TensND.AbstractTens{4, 3}; kw...
    )
    geom = rve.phases[name].geometry
    sym = phase_symmetrize(rve, name)
    H = compliance_contribution(
        geom, _project_matrix(P_ref, sym); K_interface = _crack_interface_K4(rve, name), kw...
    )
    return _apply_symmetrize(delta_compliance(geom, H, one(eltype(H))), sym)
end

function _crack_compliance_per_density(
        rve::RVE, name::Symbol, P_ref::TensND.AbstractTens{2, 3}; kw...
    )
    geom = rve.phases[name].geometry
    sym = phase_symmetrize(rve, name)
    R = compliance_contribution(
        geom, _project_matrix(P_ref, sym); α_interface = _crack_interface_α(rve, name), kw...
    )
    return _apply_symmetrize(delta_resistivity(geom, R, one(eltype(R))), sym)
end

# Dilute strain and stress concentration tensors of a phase in the reference
# `P_ref`. A crack family carries, per unit density, the strain κ ℍ_c : P_ref of
# its openings and no stress.
_dilute_pair(rve::RVE, name::Symbol, prop::Symbol, P_ref; kw...) =
    _dilute_pair(rve.amounts[name], rve, name, prop, P_ref; kw...)

_dilute_pair(::AbstractAmount, rve::RVE, name::Symbol, prop::Symbol, P_ref; kw...) =
    _phase_dilute_and_stress_average(rve, name, prop, P_ref; kw...)

function _dilute_pair(::CrackDensity, rve::RVE, name::Symbol, ::Symbol, P_ref; kw...)
    H = _crack_compliance_per_density(rve, name, P_ref; kw...)
    return (_compose(H, P_ref), zero(P_ref))
end

# One-site assembly around a matrix `m`: every other phase is loaded by
# `𝔼 : E`, and the matrix takes the rest of the two averages.
function _with_matrix(rve::RVE, m::Symbol, names, pairs, 𝔼, P_hom)
    strain = Dict{Symbol, TensND.AbstractTens}(
        n => _compose(A, 𝔼) for (n, (A, _)) in zip(names, pairs)
    )
    stress = Dict{Symbol, TensND.AbstractTens}(
        n => _compose(CA, 𝔼) for (n, (_, CA)) in zip(names, pairs)
    )
    f_m = volume_fraction(rve, m)
    strain[m] = _per_weight(_identity_like(P_hom) - _weighted_sum(rve, names, strain, P_hom), f_m)
    stress[m] = _per_weight(P_hom - _weighted_sum(rve, names, stress, P_hom), f_m)
    return vcat(m, names), strain, stress
end

_weighted_sum(rve::RVE, names, tensors, proto) =
    sum((_phase_weight(rve, n) * tensors[n] for n in names); init = zero(proto))

# A tensor divided by a weight, written as a product: `TensISO / Num` is
# ambiguous between Symbolics and TensND before 0.6.2, which the compat bound
# still admits; `Num * TensISO` never was.
_per_weight(t, w) = inv(w) * t

# ── Scheme by scheme ────────────────────────────────────────────────────────

# Voigt: a uniform strain. The bounds read no shape and ignore cracks.
function _scheme_concentrations(rve::RVE, ::Voigt, ::Val{p}, P_hom; kw...) where {p}
    names = _volume_phase_names(rve)
    I = _identity_like(P_hom)
    strain = Dict{Symbol, TensND.AbstractTens}(n => I for n in names)
    stress = Dict{Symbol, TensND.AbstractTens}(
        n => _phase_voigt_property(rve, n, p, P_hom) for n in names
    )
    return names, strain, stress
end

# Reuss: a uniform stress, the macroscopic one, Σ = ℂ^R : E.
function _scheme_concentrations(rve::RVE, ::Reuss, ::Val{p}, P_hom; kw...) where {p}
    names = _volume_phase_names(rve)
    strain = Dict{Symbol, TensND.AbstractTens}(
        n => _compose(_phase_reuss_property(rve, n, p, P_hom), P_hom) for n in names
    )
    stress = Dict{Symbol, TensND.AbstractTens}(n => P_hom for n in names)
    return names, strain, stress
end

# The schemes whose inclusions solve a dilute problem in the matrix, each under
# the remote field 𝔼 : E that `_remote_field` gives for the scheme; the matrix
# takes the rest of the two averages.
const _MatrixReferenceScheme = Union{Dilute, DiluteDual, MoriTanaka, Maxwell, PonteCastanedaWillis}

function _scheme_concentrations(rve::RVE, scheme::_MatrixReferenceScheme, prop::Val{p}, P_hom; kw...) where {p}
    m = matrix_name(scheme, rve)
    P₀ = phase_property(rve, m, p)
    names = inclusion_phase_names(rve, m)
    pairs = [_dilute_pair(rve, n, p, P₀; kw...) for n in names]
    𝔼 = _remote_field(scheme, rve, prop, m, names, pairs, P₀, P_hom; kw...)
    return _with_matrix(rve, m, names, pairs, 𝔼, P_hom)
end

# Dilute: the macroscopic strain itself.
_remote_field(::Dilute, rve, prop, m, names, pairs, P₀, P_hom; kw...) = _identity_like(P₀)

# Dual dilute: the strain of the matrix under the macroscopic stress, 𝕊₀ : ℂ^hom.
_remote_field(::DiluteDual, rve, prop, m, names, pairs, P₀, P_hom; kw...) =
    _compose(inv(P₀), P_hom)

# Mori–Tanaka: the mean strain of the matrix, (f₀ 𝕀 + Σ w 𝔸^dil)⁻¹, cracks
# included through their openings.
function _remote_field(::MoriTanaka, rve, prop, m, names, pairs, P₀, P_hom; kw...)
    A_avg = volume_fraction(rve, m) * _identity_like(P₀) +
        sum((_phase_weight(rve, n) * A for (n, (A, _)) in zip(names, pairs)); init = zero(P₀))
    return inv(A_avg)
end

# Maxwell and PCW (one distribution ellipsoid): the remote strain of the
# cluster, (𝕀 − ℙ_d : ℕ_Σ)⁻¹, with the contributions the estimate itself sums.
function _remote_field(
        scheme::Union{Maxwell, PonteCastanedaWillis}, rve, ::Val{p}, m, names, pairs, P₀, P_hom; kw...
    ) where {p}
    N_Σ = _accumulate_contributions(rve, P₀, p, m; kw...)
    P_d = hill_tensor(distribution_shape_of(distribution_shape(rve, scheme)), P₀; kw...)
    return inv(_identity_like(P₀) - _compose(P_d, N_Σ))
end

# Asymmetric self-consistent: the inclusions solve their dilute problem in the
# effective medium, under the macroscopic strain; the matrix takes the rest.
function _scheme_concentrations(
        rve::RVE, scheme::AsymmetricSelfConsistent, ::Val{p}, P_hom; kw...
    ) where {p}
    m = matrix_name(scheme, rve)
    names = inclusion_phase_names(rve, m)
    pairs = [_dilute_pair(rve, n, p, P_hom; kw...) for n in names]
    return _with_matrix(rve, m, names, pairs, _identity_like(P_hom), P_hom)
end

# Self-consistent: every phase solves its dilute problem in the effective
# medium, under the remote strain (Σ_r w_r 𝔸_r^dil)⁻¹ : E fixed by the average.
# Iterative, hence numeric only.
function _scheme_concentrations(rve::RVE, ::SelfConsistent, ::Val{p}, P_hom; kw...) where {p}
    names = copy(rve.phase_names)
    pairs = [_dilute_pair(rve, n, p, P_hom; kw...) for n in names]
    𝔼 = inv(sum((_phase_weight(rve, n) * A for (n, (A, _)) in zip(names, pairs)); init = zero(P_hom)))
    strain = Dict{Symbol, TensND.AbstractTens}(
        n => _compose(A, 𝔼) for (n, (A, _)) in zip(names, pairs)
    )
    stress = Dict{Symbol, TensND.AbstractTens}(
        n => _compose(CA, 𝔼) for (n, (_, CA)) in zip(names, pairs)
    )
    return names, strain, stress
end

# Differential: the concentration tensors evolve with the medium along the
# path. Weighted by the amounts, g_r = w_r 𝔸_r and h_r = w_r 𝔸_σε,r obey
#
#     dg_r/dτ = φ̇_r 𝔸_r^dil(ℂ) − g_r : X ,     dh_r/dτ = φ̇_r 𝔸_σε,r^dil(ℂ) − h_r : X ,
#     X = Σ_j φ̇_j 𝔸_j^dil(ℂ) + Σ_c φ̇_c^ε κ ℍ_c(ℂ) : ℂ ,
#
# from zero: an increment replaces part of the current medium, whose mean strain
# maps onto the new macroscopic one by 𝕀 − dτ X, and adds its own dilute
# response. The state holds them per unit target amount, g_r / w_r(1), fed by
# the rates per unit target: that is the concentration tensor itself at τ = 1,
# and it stays finite for a phase whose amount vanishes. They are integrated
# with the effective medium, in Kelvin–Mandel form, and the matrix takes the
# rest of the two averages at τ = 1. Iterative, hence numeric only.
function _concentration_tensors(rve::RVE, scheme::DifferentialScheme, ::Val{p}; kw...) where {p}
    opts = scheme.options
    nsteps = get(opts, :nsteps, 100)
    formulation = _diff_formulation(scheme)
    m = matrix_name(scheme, rve)
    paths = _resolve_paths(scheme.trajectory, rve, nsteps, m)
    P_matrix = phase_property(rve, m, p)
    P_init = formulation === :compliance ? inv(P_matrix) : P_matrix
    params, x0 = _diff_setup(rve, paths, p, m, P_init, formulation; kw...)
    names = vcat(params.solid_names, params.crack_names)
    probes = [_dilute_pair(rve, n, p, P_matrix; kw...) for n in names]
    T = promote_type(eltype(x0), (eltype(_flat_matrix(a)) for pr in probes for a in pr)...)
    L = length(_flat_matrix(P_matrix))
    y0 = vcat(T.(x0), zeros(T, 2 * L * length(names)))
    q = (; params, nu = length(x0), names, L, prop = p)
    prob = ODEProblem(_diff_concentration_rhs!, y0, (0.0, 1.0), q)
    alg = get(opts, :alg, nothing)
    sol = solve(
        prob, alg === nothing ? Tsit5() : alg;
        abstol = get(opts, :abstol, 1.0e-8), reltol = get(opts, :reltol, 1.0e-6),
        saveat = (1.0,), _diff_solver_kwargs(opts)...
    )
    y = last(sol.u)
    P_end = _reconstruct_tens(params.sym_tag, params.proto, view(y, 1:q.nu))
    P_hom = formulation === :compliance ? inv(P_end) : P_end
    strain = Dict{Symbol, TensND.AbstractTens}(
        n => _from_flat(_flat_block(y, q, k, 0), P_matrix) for (k, n) in enumerate(names)
    )
    stress = Dict{Symbol, TensND.AbstractTens}(
        n => _from_flat(_flat_block(y, q, k, 1), P_matrix) for (k, n) in enumerate(names)
    )
    f_m = volume_fraction(rve, m)
    strain[m] = _per_weight(_identity_like(P_hom) - _weighted_sum(rve, names, strain, P_hom), f_m)
    stress[m] = _per_weight(P_hom - _weighted_sum(rve, names, stress, P_hom), f_m)
    return _concentration_result(rve, scheme, p, P_hom, vcat(m, names), strain, stress)
end

function _diff_concentration_rhs!(dy, y, q, τ)
    p = q.params
    Pτ = _reconstruct_tens(p.sym_tag, p.proto, view(y, 1:q.nu))
    Cτ = p.dual ? inv(Pτ) : Pτ
    dφ, dφᶜ = _diff_rates(p, τ, eltype(y))
    rates = vcat(dφ, dφᶜ)
    unit = _diff_unit_rates(p, τ, eltype(y))
    active = findall(!iszero, unit)
    bundles = Dict(
        k => _diff_phase_bundle(p.rve, q.names[k], q.prop, Cτ, p.dual; p.kw...) for k in active
    )
    # The effective medium advances as in `_diff_ode_rhs!`, from the same kernels.
    Δ = sum((rates[k] * first(bundles[k]) for k in active); init = zero(Pτ))
    _set_state_layout!(view(dy, 1:q.nu), p.sym_tag, p.proto, Δ)
    flat = Dict(k => (_flat_matrix(bundles[k][2]), _flat_matrix(bundles[k][3])) for k in active)
    r = isqrt(q.L)
    X = sum((rates[k] * first(flat[k]) for k in active); init = zeros(eltype(y), r, r))
    for k in eachindex(q.names), part in (0, 1)
        dG = _flat_block(dy, q, k, part)
        dG .= -(_flat_block(y, q, k, part) * X)
        k in active && (dG .+= unit[k] .* flat[k][part + 1])
    end
    return nothing
end

# One dilute solve per phase and per evaluation of the right-hand side: the
# kernel of the effective ODE — the contribution that `_diff_ode_rhs!` sums, or
# its compliance form — together with the dilute strain and stress
# concentration tensors. The shared solves `concentration_and_stiffness` and
# `compliance_and_stiffness_contribution` give the contribution bitwise as the
# kernels of `differential.jl` do, and the rest follows by the exact identities
# ℍ = −𝕊₀ : ℕ : 𝕊₀ and 𝔸_σε = ℕ + ℂ₀ : 𝔸.
_diff_phase_bundle(rve::RVE, name::Symbol, prop::Symbol, Cτ, dual::Bool; kw...) =
    _diff_phase_bundle(rve.amounts[name], rve, name, prop, Cτ, dual; kw...)

function _diff_phase_bundle(
        ::AbstractAmount, rve::RVE, name::Symbol, prop::Symbol, Cτ, dual::Bool; kw...
    )
    geom = rve.phases[name].geometry
    sym = phase_symmetrize(rve, name)
    P₀ = _project_matrix(Cτ, sym)
    A_raw, N_raw = MFH_Core.concentration_and_stiffness(geom, phase_property(rve, name, prop), P₀; kw...)
    kernel = _apply_symmetrize(dual ? _dual_contribution(N_raw, P₀) : N_raw, sym)
    return kernel, _apply_symmetrize(A_raw, sym), _apply_symmetrize(N_raw + _compose(P₀, A_raw), sym)
end

# A crack family, per unit density: the strain of its openings and no stress.
function _diff_phase_bundle(
        ::CrackDensity, rve::RVE, name::Symbol, prop::Symbol, Cτ, dual::Bool; kw...
    )
    geom = rve.phases[name].geometry
    sym = phase_symmetrize(rve, name)
    P₀ = _project_matrix(Cτ, sym)
    H_raw, N_raw = MFH_Core.compliance_and_stiffness_contribution(
        geom, P₀; _crack_interface_kw(rve, name, P₀)..., kw...
    )
    H = _apply_symmetrize(_delta_dual(geom, H_raw), sym)
    kernel = dual ? H : _apply_symmetrize(_delta_primal(geom, N_raw), sym)
    return kernel, _compose(H, Cτ), zero(Cτ)
end

# The compliance form of a contribution: ℍ = −𝕊₀ : ℕ : 𝕊₀, 𝑹 = −𝑲₀⁻¹ · 𝑵 · 𝑲₀⁻¹.
function _dual_contribution(N, P₀)
    S₀ = inv(P₀)
    return -_compose(_compose(S₀, N), S₀)
end

_crack_interface_kw(rve::RVE, name::Symbol, ::TensND.AbstractTens{4, 3}) =
    (; K_interface = _crack_interface_K4(rve, name))
_crack_interface_kw(rve::RVE, name::Symbol, ::TensND.AbstractTens{2, 3}) =
    (; α_interface = _crack_interface_α(rve, name))

# A crack contribution per unit density, geometric prefactor included.
_delta_primal(geom, N::TensND.AbstractTens{4, 3}) = MFH_Core.delta_stiffness(geom, N, one(eltype(N)))
_delta_primal(geom, N::TensND.AbstractTens{2, 3}) = MFH_Core.delta_conductivity(geom, N, one(eltype(N)))
_delta_dual(geom, H::TensND.AbstractTens{4, 3}) = delta_compliance(geom, H, one(eltype(H)))
_delta_dual(geom, H::TensND.AbstractTens{2, 3}) = delta_resistivity(geom, H, one(eltype(H)))

# The block of phase `k` in the state: its strain (part 0) or stress (part 1)
# concentration tensor, weighted by its amount per unit target amount, as a
# square matrix.
function _flat_block(y, q, k, part)
    first_index = q.nu + (2 * (k - 1) + part) * q.L
    r = isqrt(q.L)
    return reshape(view(y, (first_index + 1):(first_index + q.L)), r, r)
end

# Kelvin–Mandel matrix in the canonical basis at order 4, components at order 2.
_flat_matrix(t::TensND.AbstractTens{4, 3}) =
    Matrix(TensND.KM(t, TensND.CanonicalBasis{3, Float64}()))
_flat_matrix(t::TensND.AbstractTens{2, 3}) = Matrix(TensND.components_canon(t))

_from_flat(M::AbstractMatrix, ::TensND.AbstractTens{4, 3}) = TensND.inv_KM(Matrix(M))
_from_flat(M::AbstractMatrix, ::TensND.AbstractTens{2, 3}) = TensND.TensCanonical(Matrix(M))

# ── Readers ─────────────────────────────────────────────────────────────────

"""
    strain_strain_concentration(ct::ConcentrationTensors, name) -> AbstractTens

Strain concentration tensor ``\\mathbb{A}_r`` of phase `name`:
``\\langle\\boldsymbol{\\varepsilon}\\rangle_r = \\mathbb{A}_r:\\boldsymbol{E}``. Per unit density for a crack family.
"""
MFH_Core.strain_strain_concentration(ct::_ConcentrationTensors4, name::Symbol) = _strain(ct, name)

"""
    stress_strain_concentration(ct::ConcentrationTensors, name) -> AbstractTens

Stress concentration tensor of phase `name` per unit macroscopic strain:
``\\langle\\boldsymbol{\\sigma}\\rangle_r = \\mathbb{A}_{\\sigma\\varepsilon,r}:\\boldsymbol{E}``. Zero for a crack family.
"""
MFH_Core.stress_strain_concentration(ct::_ConcentrationTensors4, name::Symbol) = _stress(ct, name)

"""
    strain_stress_concentration(ct::ConcentrationTensors, name) -> AbstractTens

Strain concentration tensor of phase `name` per unit macroscopic stress:
``\\langle\\boldsymbol{\\varepsilon}\\rangle_r = \\mathbb{A}_r:\\mathbb{S}^{\\mathrm{hom}}:\\boldsymbol{\\Sigma}``.
"""
MFH_Core.strain_stress_concentration(ct::_ConcentrationTensors4, name::Symbol) =
    _compose(_strain(ct, name), ct.compliance)

"""
    stress_stress_concentration(ct::ConcentrationTensors, name) -> AbstractTens

Stress concentration tensor ``\\mathbb{B}_r`` of phase `name`:
``\\langle\\boldsymbol{\\sigma}\\rangle_r = \\mathbb{B}_r:\\boldsymbol{\\Sigma}``, with
``\\mathbb{B}_r = \\mathbb{A}_{\\sigma\\varepsilon,r}:\\mathbb{S}^{\\mathrm{hom}}``.
"""
MFH_Core.stress_stress_concentration(ct::_ConcentrationTensors4, name::Symbol) =
    _compose(_stress(ct, name), ct.compliance)

"""
    gradient_gradient_concentration(ct::ConcentrationTensors, name) -> AbstractTens

Transport twin of [`strain_strain_concentration`](@ref):
``\\langle\\nabla T\\rangle_r = \\boldsymbol{A}_r\\cdot\\nabla T``.
"""
MFH_Core.gradient_gradient_concentration(ct::_ConcentrationTensors2, name::Symbol) = _strain(ct, name)

"""
    flux_gradient_concentration(ct::ConcentrationTensors, name) -> AbstractTens

Transport twin of [`stress_strain_concentration`](@ref), with
``\\boldsymbol{\\sigma}\\equiv-\\underline{q}``.
"""
MFH_Core.flux_gradient_concentration(ct::_ConcentrationTensors2, name::Symbol) = _stress(ct, name)

"""
    gradient_flux_concentration(ct::ConcentrationTensors, name) -> AbstractTens

Transport twin of [`strain_stress_concentration`](@ref).
"""
MFH_Core.gradient_flux_concentration(ct::_ConcentrationTensors2, name::Symbol) =
    _compose(_strain(ct, name), ct.compliance)

"""
    flux_flux_concentration(ct::ConcentrationTensors, name) -> AbstractTens

Transport twin of [`stress_stress_concentration`](@ref).
"""
MFH_Core.flux_flux_concentration(ct::_ConcentrationTensors2, name::Symbol) =
    _compose(_stress(ct, name), ct.compliance)

_strain(ct::ConcentrationTensors, name::Symbol) = ct.strain[_check_phase(ct, name)]
_stress(ct::ConcentrationTensors, name::Symbol) = ct.stress[_check_phase(ct, name)]

function _check_phase(ct::ConcentrationTensors, name::Symbol)
    name in ct.names || throw(
        ArgumentError(
            "no phase :$(name) among the concentration tensors; the phases are " *
                join((":" * string(n) for n in ct.names), ", ")
        )
    )
    return name
end

# ── Contributions of a scheme ───────────────────────────────────────────────

"""
    stiffness_contribution(ct::ConcentrationTensors, name; reference = ct.reference)

Contribution of phase `name` to the effective stiffness under the scheme of
`ct`, relative to the reference stiffness ``\\mathbb{C}_{\\mathrm{ref}}``: the tensor
``\\mathbb{N}_r = \\mathbb{A}_{\\sigma\\varepsilon,r} - \\mathbb{C}_{\\mathrm{ref}}:\\mathbb{A}_r`` that maps the macroscopic strain
onto the mean polarization of the phase,
``\\langle\\boldsymbol{\\sigma} - \\mathbb{C}_{\\mathrm{ref}}:\\boldsymbol{\\varepsilon}\\rangle_r = \\mathbb{N}_r:\\boldsymbol{E}``, so that

```math
\\mathbb{C}^{\\mathrm{hom}} = \\mathbb{C}_{\\mathrm{ref}} + \\sum_r w_r\\,\\mathbb{N}_r .
```

`reference` is a phase name (a homogeneous phase) or a stiffness tensor; it
defaults to the matrix of the schemes that have one. Per unit density for a crack
family. In conduction, see [`conductivity_contribution`](@ref).
"""
function MFH_Core.stiffness_contribution(
        ct::_ConcentrationTensors4, name::Symbol; reference = ct.reference
    )
    return _stress(ct, name) - _compose(_reference(ct, reference), _strain(ct, name))
end

"""
    compliance_contribution(ct::ConcentrationTensors, name; reference = ct.reference)

Contribution of phase `name` to the effective compliance under the scheme of
`ct`, relative to the reference compliance ``\\mathbb{S}_{\\mathrm{ref}}``: the tensor
``\\mathbb{H}_r`` that maps the macroscopic stress onto
``\\langle\\boldsymbol{\\varepsilon} - \\mathbb{S}_{\\mathrm{ref}}:\\boldsymbol{\\sigma}\\rangle_r``, so that

```math
\\mathbb{S}^{\\mathrm{hom}} = \\mathbb{S}_{\\mathrm{ref}} + \\sum_r w_r\\,\\mathbb{H}_r .
```

`reference` is a phase name or a stiffness tensor (whose inverse is taken), as
for [`stiffness_contribution`](@ref). In conduction, see
[`resistivity_contribution`](@ref).
"""
function MFH_Core.compliance_contribution(
        ct::_ConcentrationTensors4, name::Symbol; reference = ct.reference
    )
    return _compliance_contribution(ct, name, reference)
end

"""
    conductivity_contribution(ct::ConcentrationTensors, name; reference = ct.reference)

Transport twin of [`stiffness_contribution`](@ref): the tensor mapping the
macroscopic gradient onto ``\\langle-\\underline{q} - \\boldsymbol{K}_{\\mathrm{ref}}\\cdot\\nabla T\\rangle_r``.
"""
function MFH_Core.conductivity_contribution(
        ct::_ConcentrationTensors2, name::Symbol; reference = ct.reference
    )
    return _stress(ct, name) - _compose(_reference(ct, reference), _strain(ct, name))
end

"""
    resistivity_contribution(ct::ConcentrationTensors, name; reference = ct.reference)

Transport twin of [`compliance_contribution`](@ref).
"""
function MFH_Core.resistivity_contribution(
        ct::_ConcentrationTensors2, name::Symbol; reference = ct.reference
    )
    return _compliance_contribution(ct, name, reference)
end

function _compliance_contribution(ct::ConcentrationTensors, name::Symbol, reference)
    S_ref = inv(_reference(ct, reference))
    return _compose(_strain(ct, name) - _compose(S_ref, _stress(ct, name)), ct.compliance)
end

_reference(::ConcentrationTensors, P::TensND.AbstractTens) = P

function _reference(ct::ConcentrationTensors, name::Symbol)
    haskey(ct.properties, name) || throw(
        ArgumentError(
            "phase :$(name) cannot serve as a reference: it is not a homogeneous " *
                "phase of the cell. Pass a property tensor as `reference` instead."
        )
    )
    return ct.properties[name]
end

_reference(::ConcentrationTensors, ::Nothing) = throw(
    ArgumentError(
        "this scheme has no reference phase: pass `reference = <phase name>` or a " *
            "property tensor to define the polarization the contribution measures."
    )
)

# ── Levin's theorem ─────────────────────────────────────────────────────────
#
# One kernel for both tensor orders: a load of each phase — a second-order
# tensor in elasticity, a vector in conduction — contracted with the
# concentration tensor of the phase from the left, weighted and summed.

"""
    levin_free_strain(ct::ConcentrationTensors, eigenstrains) -> AbstractTens{2}

Macroscopic free strain of a composite whose phases carry uniform eigenstrains
(free strains) ``\\boldsymbol{\\varepsilon}^\\ast_r``, by Levin's theorem: with
``\\boldsymbol{\\sigma} = \\mathbb{C}:(\\boldsymbol{\\varepsilon} - \\boldsymbol{\\varepsilon}^\\ast)`` in each phase,
``\\boldsymbol{\\Sigma} = \\mathbb{C}^{\\mathrm{hom}}:(\\boldsymbol{E} - \\boldsymbol{E}^\\ast)`` with

```math
\\boldsymbol{E}^\\ast = \\sum_r f_r\\,\\boldsymbol{\\varepsilon}^\\ast_r:\\mathbb{B}_r ,
```

``\\mathbb{B}_r`` being the stress concentration tensor of the phase
([`stress_stress_concentration`](@ref)). `eigenstrains` maps phase names to
second-order tensors; phases left out carry none. A crack family holds no
material, so an eigenstrain in it has no effect and is refused. In conduction,
see [`levin_free_gradient`](@ref).
"""
function levin_free_strain(ct::_ConcentrationTensors4, eigenstrains::AbstractDict{Symbol})
    return _levin_free(ct, eigenstrains)
end

"""
    levin_prestress(ct::ConcentrationTensors, prestresses) -> AbstractTens{2}

Macroscopic prestress of a composite whose phases carry uniform prestresses
``\\boldsymbol{\\sigma}^\\ast_r``, by Levin's theorem: with
``\\boldsymbol{\\sigma} = \\mathbb{C}:\\boldsymbol{\\varepsilon} + \\boldsymbol{\\sigma}^\\ast`` in each phase,
``\\boldsymbol{\\Sigma} = \\mathbb{C}^{\\mathrm{hom}}:\\boldsymbol{E} + \\boldsymbol{\\Sigma}^\\ast`` with

```math
\\boldsymbol{\\Sigma}^\\ast = \\sum_r f_r\\,\\boldsymbol{\\sigma}^\\ast_r:\\mathbb{A}_r ,
```

``\\mathbb{A}_r`` being the strain concentration tensor of the phase. A crack
family enters with its density as weight and its tensor per unit density, and
only the traction ``\\boldsymbol{\\sigma}^\\ast_c\\cdot\\underline{n}`` on its faces matters. A
pore pressure ``p`` is the prestress ``-p\\,\\boldsymbol{1}`` of the pores and of the
cracks it fills, and the Biot tensor follows as
``\\boldsymbol{B} = \\sum_{\\mathrm{pores}} f_p\\,\\boldsymbol{1}:\\mathbb{A}_p +
\\sum_{\\mathrm{cracks}} \\varepsilon_c\\,\\boldsymbol{1}:\\mathbb{A}_c``. In conduction, see
[`levin_preflux`](@ref).
"""
function levin_prestress(ct::_ConcentrationTensors4, prestresses::AbstractDict{Symbol})
    return _levin_polarization(ct, prestresses)
end

"""
    levin_free_gradient(ct::ConcentrationTensors, eigengradients) -> AbstractTens{1}

Transport twin of [`levin_free_strain`](@ref). Each phase carries a uniform
eigengradient ``\\underline{g}^\\ast_r``, the gradient at which it conducts no flux,
``\\underline{q} = -\\boldsymbol{K}_r\\cdot(\\nabla T - \\underline{g}^\\ast_r)``, and the composite
obeys ``\\underline{Q} = -\\boldsymbol{K}^{\\mathrm{hom}}\\cdot(\\underline{G} - \\underline{G}^\\ast)`` with

```math
\\underline{G}^\\ast = \\sum_r f_r\\,\\underline{g}^\\ast_r\\cdot\\boldsymbol{B}_r ,
```

``\\boldsymbol{B}_r`` being the flux concentration tensor of the phase
([`flux_flux_concentration`](@ref)). `eigengradients` maps phase names to
vectors (first-order tensors); a crack family is refused, as in elasticity.
"""
function levin_free_gradient(ct::_ConcentrationTensors2, eigengradients::AbstractDict{Symbol})
    return _levin_free(ct, eigengradients)
end

"""
    levin_preflux(ct::ConcentrationTensors, prefluxes) -> AbstractTens{1}

Transport twin of [`levin_prestress`](@ref). Each phase carries a uniform
pre-flux ``\\underline{q}^\\ast_r``, the flux it conducts under a vanishing gradient,
``\\underline{q} = -\\boldsymbol{K}_r\\cdot\\nabla T + \\underline{q}^\\ast_r``, and the composite obeys
``\\underline{Q} = -\\boldsymbol{K}^{\\mathrm{hom}}\\cdot\\underline{G} + \\underline{Q}^\\ast`` with

```math
\\underline{Q}^\\ast = \\sum_r f_r\\,\\underline{q}^\\ast_r\\cdot\\boldsymbol{A}_r ,
```

``\\boldsymbol{A}_r`` being the gradient concentration tensor of the phase
([`gradient_gradient_concentration`](@ref)). A crack family enters with its
density as weight and its tensor per unit density.
"""
function levin_preflux(ct::_ConcentrationTensors2, prefluxes::AbstractDict{Symbol})
    return _levin_polarization(ct, prefluxes)
end

# The free field E* = Σ f ε*:𝔹, contracted with the stress-driven tensors.
function _levin_free(ct::ConcentrationTensors, loads::AbstractDict{Symbol})
    foreach(n -> _check_material_phase(ct, n), keys(loads))
    return _levin_sum(ct, loads) do name
        _compose(_stress(ct, name), ct.compliance)
    end
end

# The polarization Σ* = Σ f σ*:𝔸, contracted with the strain-driven tensors.
_levin_polarization(ct::ConcentrationTensors, loads::AbstractDict{Symbol}) =
    _levin_sum(name -> _strain(ct, name), ct, loads)

function _levin_sum(concentration, ct::ConcentrationTensors, loads::AbstractDict{Symbol})
    return sum(
        (ct.weights[_check_phase(ct, n)] * _load_contract(x, concentration(n)) for (n, x) in loads);
        init = _zero_load(ct)
    )
end

# A load of each phase contracts the concentration tensor from the left.
_load_contract(x::TensND.AbstractTens{2, 3}, A::TensND.AbstractTens{4, 3}) = x ⊡ A
_load_contract(x::TensND.AbstractTens{1, 3}, A::TensND.AbstractTens{2, 3}) = x ⋅ A

# The zero of the macroscopic load, in the element type of the estimate.
_zero_load(ct::_ConcentrationTensors4) = zero(TensND.tens_Id2(Val(3), Val(eltype(ct.effective))))
_zero_load(ct::_ConcentrationTensors2) = TensND.Tens(zeros(eltype(ct.effective), 3))

function _check_material_phase(ct::ConcentrationTensors, name::Symbol)
    _check_phase(ct, name) in ct.cracks && throw(
        ArgumentError(
            "crack family :$(name) holds no material, so an eigenfield in it has no " *
                "effect; a load on its faces is a polarization (`$(_polarization_reader(ct))`)"
        )
    )
    return name
end

_polarization_reader(::_ConcentrationTensors4) = "levin_prestress"
_polarization_reader(::_ConcentrationTensors2) = "levin_preflux"
