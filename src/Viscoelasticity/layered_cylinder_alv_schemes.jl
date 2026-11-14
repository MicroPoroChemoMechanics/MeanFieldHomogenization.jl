# =============================================================================
#  layered_cylinder_alv_schemes.jl — `LayeredCylinder` phases in the ALV schemes
#  whose reference is the running estimate (self-consistent, asymmetric
#  self-consistent, differential), orders 4 and 2.
#
#  Two situations:
#
#   * cylinders among phases with a Hill kernel. The ALV Hill kernel exists for
#     an isotropic reference only, so the estimate stays isotropic and a
#     cylinder, transversely isotropic, enters like a layered sphere but under an
#     isotropic orientation average only (the methods of the first section);
#
#   * every phase a cylinder, all of one axis, without orientation average. No
#     phase needs a Hill kernel, the radial problems accept a reference
#     transversely isotropic about that axis, and the estimate runs in that
#     class, on full matrices (the rest of the file).
#
#  Volterra operators do not commute. With E₀ the history of the remote strain
#  of the auxiliary problem of each phase, ⟨ε⟩ᵢ = Ãᵢ ∘ E₀ and ⟨σ⟩ᵢ = B̃ᵢ ∘ E₀,
#  so that a concentration always acts first, on the right:
#
#   * self-consistent: E = ⟨ε⟩ = (Σ fᵢ Ãᵢ) ∘ E₀ and Σ = (Σ fᵢ B̃ᵢ) ∘ E₀ give
#         C̃ = (Σ fᵢ B̃ᵢ) ∘ (Σ fᵢ Ãᵢ)⁻¹,                 the inverse on the right;
#   * asymmetric: E₀ = E, and Σ = C̃_M ∘ E + Σ fᵢ (⟨σ⟩ᵢ − C̃_M ∘ ⟨ε⟩ᵢ) gives
#         C̃ = C̃_M + Σ_{i≠M} fᵢ (B̃ᵢ − C̃_M ∘ Ãᵢ),         C̃_M on the left;
#   * differential: an increment dφᵢ of phase i in the running medium adds
#         dC̃ = dφᵢ Ñᵢ(C̃),  Ñᵢ = B̃ᵢ − C̃ ∘ Ãᵢ,
#     and in the dual form, from d(C̃⁻¹) = −C̃⁻¹ ∘ dC̃ ∘ C̃⁻¹,
#         dJ̃ = −dφᵢ J̃ ∘ Ñᵢ ∘ J̃.
#
#  Ãᵢ, B̃ᵢ against the estimate come from `_concentration_and_flux_alv`.
# =============================================================================

# ── Cylinders among phases with a Hill kernel ────────────────────────────────

# The concentration and the average flux of a cylinder against the running
# estimate, from one solution of its radial problems.
_alv_layered_AB(cyl::LayeredCylinder, C::AbstractMatrix, times) =
    _concentration_and_flux_alv(_alv_order(C, times), cyl, C, times)

# The host of the order-4 self-consistent schemes enters through the Hill kernel
# of its shape.
_alv_refuse_layered_host(who::AbstractString, name::Symbol, geom) = nothing
_alv_refuse_layered_host(who::AbstractString, name::Symbol, geom::_LayeredALV) = throw(
    ArgumentError(
        "$who: the host phase :$(name) is a $(nameof(typeof(geom))), which has no ALV Hill " *
            "kernel. A layered host is admitted in a cell whose every phase is a " *
            "LayeredCylinder, all of one axis, without orientation average, through `homogenize_alv`."
    )
)

# The differential scheme: no kernel of its own (the layers carry the moduli),
# and a contribution that only an isotropic orientation average makes isotropic.
function _diff_alv_solid_operator(geom::LayeredCylinder, rve, name, prop, sym, times)
    _alv_diff_keeps_iso(geom, sym, nothing) ||
        _alv_iso_error("differential_alv", name, "the transversely isotropic contribution")
    return nothing
end
_diff_alv_solid_contribution(r, ::LayeredCylinder, C_curr, C_op, J_long, J_shear, times) =
    property_contribution_alv(r.geom, C_curr, times)

# ── Every phase a cylinder, all of one axis ──────────────────────────────────

# The schemes whose reference is the running estimate.
const _ALVRunningScheme = Union{SelfConsistent, AsymmetricSelfConsistent, DifferentialScheme}

_is_aligned_cylinder(::LayeredCylinder, ::NoSymmetrize) = true
_is_aligned_cylinder(geom, ::AbstractSymmetrize) = false

"""
    _alv_all_cylinders(scheme, rve) -> Bool

Whether `homogenize_alv` runs `scheme` with an estimate transversely isotropic
about the axis of the cylinders: a scheme whose reference is the running
estimate, and a cell whose every phase, the host included, is a
[`LayeredCylinder`](@ref) without orientation average. That they share one axis
is checked by `_homogenize_alv_cylinders`.
"""
_alv_all_cylinders(::HomogenizationScheme, ::RVE) = false
_alv_all_cylinders(::_ALVRunningScheme, rve::RVE) = !isempty(rve.phase_names) && all(
    name -> _is_aligned_cylinder(rve.phases[name].geometry, phase_symmetrize(rve, name)), rve.phase_names
)

# The cylinder written in its own frame, the block matrices of its layers on the
# grid for moduli: discretized once, not at every evaluation against the running
# estimate, which lives in that frame too. Its axis is then e₃.
_discretized_cylinder(o::Val, cyl::LayeredCylinder{T, N}, times) where {T, N} = LayeredCylinder(
    cyl.radii, ntuple(k -> _cylinder_frame_blocks(o, cyl, layer_modulus(cyl, k), times, "layer $k"), Val(N)),
    cyl.interfaces, _E₃
)

# The phases `names` as the steps read them: the discretized cylinder and the
# volume fraction of each, in a tuple of concrete records.
_cylinder_phases(o::Val, rve::RVE, names, times) = Tuple(
    (geom = _discretized_cylinder(o, rve.phases[name].geometry, times), f = volume_fraction(rve, name))
        for name in names
)

# The kernel of the host of the schemes that single it out (asymmetric
# self-consistent, differential): the modulus of its only layer, in the frame
# of the cylinders.
function _cylinder_host_kernel(who::AbstractString, o::Val, rve::RVE, m::Symbol, times)
    host = rve.phases[m].geometry
    return _cylinder_frame_blocks(o, host, _host_modulus(who, m, host), times, "the host layer")
end
_host_modulus(who, m, host::LayeredCylinder{<:Number, 1, <:Any, <:Tuple{PerfectInterface}}) = layer_modulus(host, 1)
_host_modulus(who, m, host::LayeredCylinder) = throw(
    ArgumentError(
        "$who: the host phase :$(m) must be a one-layer LayeredCylinder with a perfect " *
            "interface, its layer carrying the kernel of the matrix; got $(host)"
    )
)

# Σ fᵢ Ãᵢ and Σ fᵢ B̃ᵢ over `phases` against the running estimate `X` (zero
# without any phase).
_cylinder_sums(order::Val, phases, X::AbstractMatrix, times) = _asc_sums(
    map(phases) do p
        A, B = _concentration_and_flux_alv(order, p.geom, X, times)
        p.f * A, p.f * B
    end, X
)

# One self-consistent step: (Σ fᵢ B̃ᵢ) ∘ (Σ fᵢ Ãᵢ)⁻¹ over every phase.
function _sc_cylinders_step(order::Val, phases, X::AbstractMatrix, times)
    A, B = _cylinder_sums(order, phases, X, times)
    return B * volterra_inverse(A; block_size = _alv_block_size(order))
end

# One asymmetric step: C̃_M + Σ fᵢ (B̃ᵢ − C̃_M ∘ Ãᵢ) over the inclusion phases.
_asc_cylinders_step(order::Val, X_M, phases, X::AbstractMatrix, times) =
    _asc_update(X_M, X, _cylinder_sums(order, phases, X, times)..., nothing, identity)

# The Picard iteration of both self-consistent schemes. Their Newton march is
# written on the parameters of an isotropic estimate.
_cylinders_picard(::AndersonDefault, step::F, X_0, name; kw...) where {F} =
    _cylinders_fixed_point(step, X_0, name; kw...)
_cylinders_picard(algorithm, step, X_0, name; kw...) = throw(
    ArgumentError(
        "homogenize_alv: $name(algorithm = $(algorithm)) is not available for a cell of " *
            "LayeredCylinder phases, whose transversely isotropic estimate is iterated by " *
            "AndersonDefault() (Picard iteration, the default)"
    )
)
_cylinders_fixed_point(
    step::F, X_0, name;
    abstol::Real = 1.0e-10, reltol::Real = 1.0e-8, maxiters::Int = 200,
    damping::Real = 0.0, verbose::Bool = false, select_best::Bool = false,
) where {F} = _fixed_point_dense(
    step, X_0, "$name-ALV (cylinders)"; abstol, reltol, maxiters, damping, verbose, select_best
)

"""
    _homogenize_alv_cylinders(rve, scheme, prop; times, kw...) -> Matrix

Aging viscoelastic estimate of a cell whose every phase is a
[`LayeredCylinder`](@ref), all of one axis ``\\underline{n}`` (any direction), by
a scheme whose reference is the running estimate, in elasticity or in conduction
(the order of the first layer of the first phase). The estimate is transversely
isotropic about ``\\underline{n}``. The moduli are those of the layers; the
property declared on a phase is a placeholder.

- [`SelfConsistent`](@ref): the fixed point of
  ``\\widetilde{\\mathbb{C}} = \\bigl(\\sum_i f_i\\,\\widetilde{\\mathbb{B}}_i\\bigr)\\stackrel{\\circ}{:}\\bigl(\\sum_i f_i\\,\\widetilde{\\mathbb{A}}_i\\bigr)^{-1^{\\circ}}``
  over every phase, from the Voigt average of the layers.
- [`AsymmetricSelfConsistent`](@ref): the fixed point of
  ``\\widetilde{\\mathbb{C}} = \\widetilde{\\mathbb{C}}_{\\mathrm{M}} + \\sum_{i\\neq\\mathrm{M}} f_i\\,\\bigl(\\widetilde{\\mathbb{B}}_i - \\widetilde{\\mathbb{C}}_{\\mathrm{M}}\\stackrel{\\circ}{:}\\widetilde{\\mathbb{A}}_i\\bigr)``,
  from the kernel of the host.
- [`DifferentialScheme`](@ref): the equation of [`differential_alv`](@ref) with
  the contributions ``\\widetilde{\\mathbb{N}}_i = \\widetilde{\\mathbb{B}}_i - \\widetilde{\\mathbb{C}}\\stackrel{\\circ}{:}\\widetilde{\\mathbb{A}}_i``
  of the cylinders against the running medium.

``\\widetilde{\\mathbb{A}}_i`` and ``\\widetilde{\\mathbb{B}}_i`` are the concentration and
the average flux of cylinder ``i`` in the estimate. The host of the last two
schemes is a one-layer cylinder with a perfect interface, whose layer carries
the kernel of the matrix.
"""
function _homogenize_alv_cylinders(rve::RVE, scheme::_ALVRunningScheme, prop::Symbol; times::AbstractVector{<:Real}, kw...)
    # One axis for all: the estimate is transversely isotropic about it.
    n = rve.phases[first(rve.phase_names)].geometry.axis
    for name in rve.phase_names
        a = rve.phases[name].geometry.axis
        TensND.isparallel(a, n) || throw(
            ArgumentError(
                "homogenize_alv: the cylinders of a cell solved with a transversely isotropic " *
                    "estimate must share one axis; the phase :$(name) has the axis $(a), the " *
                    "phase :$(first(rve.phase_names)) $(n)"
            )
        )
    end
    # The order is that of the property homogenized; every layer must share it.
    order = _alv_order(phase_property(rve, first(rve.phase_names), prop), times)
    for name in rve.phase_names, (k, P) in enumerate(rve.phases[name].geometry.moduli)
        _alv_order(P, times) === order || throw(
            ArgumentError(
                "homogenize_alv: layer $k of the phase :$(name) is not of the order of the " *
                    "property :$(prop) (order $(only(typeof(order).parameters))): the layers of " *
                    "every phase must belong to the physics homogenized"
            )
        )
    end
    # Computed in the frame of the cylinders, brought back at the end.
    X = _alv_cylinders(scheme, order, rve, times; kw...)
    return _canonical_frame_blocks(order, rve.phases[first(rve.phase_names)].geometry, X, times)
end

# The ALV counterpart of `Schemes._check_layered_host`: a host given as a
# one-layer cylinder declares the kernel of its layer, compared on the grid.
function _check_layered_host_alv(rve::RVE, scheme::HomogenizationScheme, prop::Symbol, times)
    r = Schemes._layered_host_name(rve, scheme)
    (r === nothing || !haskey(rve.phases, r)) && return nothing  # an absent phase is reported by the scheme
    geom = rve.phases[r].geometry
    geom isa LayeredCylinder && layer_count(geom) == 1 || return nothing
    declared = phase_property(rve, r, prop)
    b = _block_rows(_alv_order(declared, times))
    Schemes._same_property(_alv_blocks(declared, times, b), _alv_blocks(layer_modulus(geom, 1), times, b)) || throw(
        ArgumentError(
            "homogenize_alv: the host phase :$(r) is a one-layer LayeredCylinder whose layer " *
                "differs from its declared property :$(prop) on the grid: the schemes that take " *
                "the host as their reference read the declared property, the others its layer. " *
                "Declare the kernel of the layer"
        )
    )
    return nothing
end

function _alv_cylinders(sc::SelfConsistent, order::Val, rve::RVE, times; kw...)
    _alv_refuse_seed("SelfConsistent", sc.init)
    phases = _cylinder_phases(order, rve, rve.phase_names, times)
    # The Voigt average of the layers of every phase.
    X_0 = sum(
        p.f * layer_volume_fraction(p.geom, k) * layer_modulus(p.geom, k)
            for p in phases for k in 1:layer_count(p.geom)
    )
    return _cylinders_picard(
        sc.algorithm, X -> _sc_cylinders_step(order, phases, X, times), X_0, "SelfConsistent";
        sc.options..., kw...
    )
end

function _alv_cylinders(asc::AsymmetricSelfConsistent, order::Val, rve::RVE, times; kw...)
    _alv_refuse_seed("AsymmetricSelfConsistent", asc.init)
    m = matrix_name(asc, rve)
    X_M = _cylinder_host_kernel(_ASC_WHO, order, rve, m, times)
    phases = _cylinder_phases(order, rve, inclusion_phase_names(rve, m), times)
    return _cylinders_picard(
        asc.algorithm, X -> _asc_cylinders_step(order, X_M, phases, X, times), X_M, "AsymmetricSelfConsistent";
        asc.options..., kw...
    )
end

function _alv_cylinders(sch::DifferentialScheme, order::Val, rve::RVE, times; kw...)
    who = "differential_alv"
    opts = _diff_alv_options(sch)
    _diff_alv_check_formulation(who, opts.formulation)
    b = _alv_block_size(order)
    m = matrix_name(sch, rve)
    X_M = _cylinder_host_kernel(who, order, rve, m, times)
    names = inclusion_phase_names(rve, m)
    foreach(name -> _diff_alv_refuse_remainder(who, name, rve.amounts[name]), names)
    paths = _resolve_paths_alv(something(opts.trajectory, Schemes.Proportional()), rve, opts.nsteps, m)
    solid_data = Tuple(
        (geom = _discretized_cylinder(order, rve.phases[name].geometry, times), target = _amount_value(rve, name), path = paths[name])
            for name in names
    )
    # The state carries the element type of every input: the kernel of the host,
    # the targets, and what each cylinder brings (its layers, radii, interfaces).
    Tp = mapreduce(
        r -> promote_type(typeof(r.target), eltype(property_contribution_alv(r.geom, X_M, times))),
        promote_type, solid_data; init = eltype(X_M)
    )
    p = (; sz = size(X_M, 1), block_size = b, solid_data, times, dual = opts.formulation === :compliance)
    return _diff_alv_integrate(
        _diff_cylinders_rhs!, X_M, Tp, p, b;
        alg = opts.alg, abstol = opts.abstol, reltol = opts.reltol, solver_kwargs = opts.solver_kwargs
    )
end

# dC̃/dτ = Σ dφᵢ/dτ Ñᵢ(C̃), every contribution against the running medium; in the
# dual form the state is J̃ = C̃⁻¹ and each term −J̃ ∘ Ñᵢ ∘ J̃.
function _diff_cylinders_rhs!(du, u, p, τ)
    T = eltype(u)
    P_curr = reshape(u, p.sz, p.sz)
    X_curr = p.dual ? volterra_inverse(P_curr; block_size = p.block_size) : P_curr
    Δ = reshape(du, p.sz, p.sz)
    fill!(Δ, zero(T))
    dφ, _, _ = _diff_alv_increments(p.solid_data, τ, T)
    map(p.solid_data, dφ) do r, dφᵢ
        # The plain matrix products of `_diff_alv2_add!` serve both block sizes.
        iszero(dφᵢ) || _diff_alv2_add!(Δ, property_contribution_alv(r.geom, X_curr, p.times), dφᵢ, p.dual, P_curr, nothing)
        return nothing
    end
    return nothing
end
