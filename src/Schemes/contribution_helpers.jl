# =============================================================================
#  contribution_helpers.jl — phase-level dilute contribution helpers shared
#  by Dilute / DiluteDual / Mori-Tanaka / Maxwell / PCW / SC / Diff schemes.
#
#  Each phase carries either a `VolumeFraction` (solid inclusion) or a
#  `CrackDensity` (flat crack); the contribution tensors and their
#  geometry-specific prefactors differ accordingly. The helpers below pick
#  the right combination at dispatch time.
# =============================================================================

# ── Stiffness / conductivity contribution ────────────────────────────────────
#
#  Solid inclusion (VolumeFraction)
#    elasticity   : Δ_solid = f · stiffness_contribution(geom, C_i, C₀)
#    conductivity : Δ_solid = f · conductivity_contribution(geom, K_i, K₀)
#
#  Flat crack (CrackDensity)
#    elasticity   : Δ_crack = (4π/3 or π) · ε · stiffness_contribution(crack, C₀)
#    conductivity : Δ_crack = (4π/3 or π) · ε · conductivity_contribution(crack, K₀)
#
#  Dispatch on the tensor order (`AbstractTens{4,3}` vs `AbstractTens{2,3}`)
#  picks the right `stiffness_contribution` / `conductivity_contribution`.

"""
    _phase_stress_strain_average(rve, name, prop, P₀, A_dil; kw...)

Average stress per unit remote strain of phase `name`, `⟨C:ε⟩_r` — the **B**
concentration tensor of [echoes](@cite), as opposed to the
strain concentration tensor **A** (`A_dil`).

The two must be kept distinct. `⟨C:ε⟩_r = C_r : ⟨ε⟩_r` requires **both**:

  * `C_r` uniform inside the phase — false for a `LayeredSphere`, where the
    average is assembled layer by layer through `stress_strain_loc`
    (`Σ_k f_k C_k : A_k`);
  * the orientation average to commute with `C_r` — false for an anisotropic
    `C_r` carrying an ISO/TI orientation distribution, since
    `⟨R(C_r : A_r)⟩ ≠ C_r : ⟨R(A_r)⟩`.  The product must then be formed on the
    **un-symmetrized** `A_r` and averaged afterwards.

The shortcut `P_i ⊡ A_dil` is therefore taken only when the orientation average
is trivial or `C_r` is isotropic (an isotropic tensor commutes with every
rotation, so the two orders coincide exactly).
"""
function _phase_stress_strain_average(
        rve::RVE, name::Symbol, prop::Symbol,
        P₀::TensND.AbstractTens{4, 3},
        A_dil::TensND.AbstractTens{4, 3}; kw...
    )
    geom = rve.phases[name].geometry
    P_i = phase_property(rve, name, prop)
    sym = phase_symmetrize(rve, name)
    if MFH_Core.is_homogeneous_inclusion(geom)
        (sym isa NoSymmetrize || P_i isa TensND.TensISO) && return P_i ⊡ A_dil
        P₀_proj = _project_matrix(P₀, sym)
        A_raw = MFH_Core.strain_strain_loc(geom, P_i, P₀_proj; kw...)
        return _apply_symmetrize(P_i ⊡ A_raw, sym)
    end
    P₀_proj = _project_matrix(P₀, sym)
    return _apply_symmetrize(
        MFH_Core.stress_strain_loc(geom, P_i, P₀_proj; kw...), sym
    )
end

function _phase_stress_strain_average(
        rve::RVE, name::Symbol, prop::Symbol,
        P₀::TensND.AbstractTens{2, 3},
        A_dil::TensND.AbstractTens{2, 3}; kw...
    )
    geom = rve.phases[name].geometry
    P_i = phase_property(rve, name, prop)
    sym = phase_symmetrize(rve, name)
    if MFH_Core.is_homogeneous_inclusion(geom)
        (sym isa NoSymmetrize || P_i isa TensND.TensISO) && return P_i ⋅ A_dil
        P₀_proj = _project_matrix(P₀, sym)
        A_raw = MFH_Core.gradient_gradient_loc(geom, P_i, P₀_proj; kw...)
        return _apply_symmetrize(P_i ⋅ A_raw, sym)
    end
    P₀_proj = _project_matrix(P₀, sym)
    return _apply_symmetrize(
        MFH_Core.flux_gradient_loc(geom, P_i, P₀_proj; kw...), sym
    )
end

"""
    _phase_stiffness_contribution(rve, name, prop::Symbol, P₀; kw...)

Aggregate contribution of phase `name` to the dilute *stiffness*
correction at reference `P₀` for property `prop` (`:C` for elasticity,
`:K` for conductivity).
"""
function _phase_stiffness_contribution(
        rve::RVE, name::Symbol, prop::Symbol,
        P₀::TensND.AbstractTens{4, 3}; kw...
    )
    a = rve.amounts[name]
    geom = rve.phases[name].geometry
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P₀, sym)
    if a isa VolumeFraction
        P_i = phase_property(rve, name, prop)
        N = MFH_Core.stiffness_contribution(geom, P_i, P₀_proj; kw...)
        return scale_by_amount(a, _apply_symmetrize(N, sym))
    else  # CrackDensity
        K_int = _crack_interface_K4(rve, name)
        N = MFH_Core.stiffness_contribution(
            geom, P₀_proj;
            K_interface = K_int, kw...
        )
        return _apply_symmetrize(MFH_Core.delta_stiffness(geom, N, amount_value(a)), sym)
    end
end

function _phase_stiffness_contribution(
        rve::RVE, name::Symbol, prop::Symbol,
        P₀::TensND.AbstractTens{2, 3}; kw...
    )
    a = rve.amounts[name]
    geom = rve.phases[name].geometry
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P₀, sym)
    if a isa VolumeFraction
        P_i = phase_property(rve, name, prop)
        N = MFH_Core.conductivity_contribution(geom, P_i, P₀_proj; kw...)
        return scale_by_amount(a, _apply_symmetrize(N, sym))
    else
        α_int = _crack_interface_α(rve, name)
        N = MFH_Core.conductivity_contribution(
            geom, P₀_proj;
            α_interface = α_int, kw...
        )
        return _apply_symmetrize(MFH_Core.delta_conductivity(geom, N, amount_value(a)), sym)
    end
end

# ── Helpers : pull optional interface-stiffness properties from the RVE ────
#
# A crack phase can carry an optional spring-like interface stiffness via
# either of two property keys :
#   * `:K_interface`  for elasticity   (a `Tens{2,3}` 3×3 symmetric)
#   * `:α_interface`  for conductivity (a `Real` scalar conductance)
# Returns `nothing` when the property is absent — the existing
# traction-free / free-flux pipeline is then used unchanged.

function _crack_interface_K4(rve::RVE, name::Symbol)
    props = rve.phases[name].properties
    return haskey(props, :K_interface) ? props[:K_interface] : nothing
end

function _crack_interface_α(rve::RVE, name::Symbol)
    props = rve.phases[name].properties
    return haskey(props, :α_interface) ? props[:α_interface] : nothing
end

# ── Compliance / resistivity contribution ────────────────────────────────────
#
#  Heterogeneous inclusions (`LayeredSphere`, `LayeredSpheroid`) have NO
#  representative phase property: `phase_property(rve, name, prop)` is a
#  placeholder that every stiffness-side kernel deliberately ignores (see
#  `LayeredSpheres/scheme_integration.jl`).  Feeding it to the generic
#  `compliance_contribution` — which does use `inv(C₁)` — silently returns a
#  value that depends on that meaningless declaration.  The compliance
#  contribution is instead obtained from the stiffness one through the exact
#  identity
#
#      ℍ = − 𝕊₀ : 𝐍 : 𝕊₀ ,      (resp.  ℍ_R = − 𝐑₀ ⋅ 𝐍_K ⋅ 𝐑₀)
#
#  which holds for ANY inclusion (for a homogeneous one it reproduces
#  `(𝕊₁ − 𝕊₀) : 𝔸_σσ` exactly, since 𝔹 = ℂ₁ : 𝔸 there).  The same identity
#  is used on the ALV side by `stiffness_contribution_alv(crack, …)`.

"""
    _phase_compliance_contribution(rve, name, prop::Symbol, P₀; kw...)

Aggregate contribution of phase `name` to the dilute *compliance*
correction (resistivity for `:K`).
"""
function _phase_compliance_contribution(
        rve::RVE, name::Symbol, prop::Symbol,
        P₀::TensND.AbstractTens{4, 3}; kw...
    )
    a = rve.amounts[name]
    geom = rve.phases[name].geometry
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P₀, sym)
    if a isa VolumeFraction
        P_i = phase_property(rve, name, prop)
        H = if MFH_Core.is_homogeneous_inclusion(geom)
            compliance_contribution(geom, P_i, P₀_proj; kw...)
        else
            S₀ = inv(P₀_proj)
            -(S₀ ⊡ MFH_Core.stiffness_contribution(geom, P_i, P₀_proj; kw...) ⊡ S₀)
        end
        return scale_by_amount(a, _apply_symmetrize(H, sym))
    else
        K_int = _crack_interface_K4(rve, name)
        H = compliance_contribution(geom, P₀_proj; K_interface = K_int, kw...)
        return _apply_symmetrize(delta_compliance(geom, H, amount_value(a)), sym)
    end
end

function _phase_compliance_contribution(
        rve::RVE, name::Symbol, prop::Symbol,
        P₀::TensND.AbstractTens{2, 3}; kw...
    )
    a = rve.amounts[name]
    geom = rve.phases[name].geometry
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P₀, sym)
    if a isa VolumeFraction
        P_i = phase_property(rve, name, prop)
        R = if MFH_Core.is_homogeneous_inclusion(geom)
            MFH_Core.resistivity_contribution(geom, P_i, P₀_proj; kw...)
        else
            R₀ = inv(P₀_proj)
            -(R₀ ⋅ MFH_Core.conductivity_contribution(geom, P_i, P₀_proj; kw...) ⋅ R₀)
        end
        return scale_by_amount(a, _apply_symmetrize(R, sym))
    else
        α_int = _crack_interface_α(rve, name)
        R = compliance_contribution(geom, P₀_proj; α_interface = α_int, kw...)
        return _apply_symmetrize(delta_resistivity(geom, R, amount_value(a)), sym)
    end
end

# ── Strain-strain localization (Mori-Tanaka, SC, …) ──────────────────────────

"""
    _phase_dilute_concentration(rve, name, prop::Symbol, P₀; kw...) -> AbstractTens

Strain-strain (or gradient-gradient) dilute concentration tensor
``\\mathbb A_{\\varepsilon\\varepsilon}^{(i)}`` for phase `name` in the
reference medium `P₀`. Used by Mori-Tanaka and self-consistent kernels
when a per-phase tensor ``\\mathbb A_i`` is required (rather than just
the contribution sum).

For cracks, the strain concentration tensor is singular (the crack is
infinitely compliant in its normal direction); the helper returns the
zero tensor as a placeholder — schemes that need crack handling should
short-circuit on `geom isa AbstractCrack`.
"""
function _phase_dilute_concentration(
        rve::RVE, name::Symbol, prop::Symbol,
        P₀::TensND.AbstractTens{4, 3}; kw...
    )
    geom = rve.phases[name].geometry
    geom isa MFH_Core.AbstractCrack && return zero(P₀)   # caller must handle cracks separately
    P_i = phase_property(rve, name, prop)
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P₀, sym)
    A = MFH_Core.strain_strain_loc(geom, P_i, P₀_proj; kw...)
    return _apply_symmetrize(A, sym)
end

function _phase_dilute_concentration(
        rve::RVE, name::Symbol, prop::Symbol,
        P₀::TensND.AbstractTens{2, 3}; kw...
    )
    geom = rve.phases[name].geometry
    geom isa MFH_Core.AbstractCrack && return zero(P₀)
    P_i = phase_property(rve, name, prop)
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P₀, sym)
    A = MFH_Core.gradient_gradient_loc(geom, P_i, P₀_proj; kw...)
    return _apply_symmetrize(A, sym)
end

# ── Voigt / Reuss phase averages ────────────────────────────────────────────
#
# The bounds average the phase property directly.  Two corrections are needed
# for the general case:
#   * an internally heterogeneous inclusion has no single property — the Voigt
#     (resp. Reuss) average must run over its layers;
#   * a phase declaring an orientation distribution must have its property
#     averaged over that distribution.

#  Dispatch on the *geometry*, so that a heterogeneous inclusion which exposes
#  no layer-wise average falls through to the informative error below rather
#  than to a `MethodError` from three frames down.  (Keying the specificity on
#  the tensor order instead would make the catch-all unreachable.)
const _Layered = Union{LayeredSpheres.LayeredSphere, LayeredSpheroids.LayeredSpheroid}

_layer_voigt(sphere::_Layered, ::TensND.AbstractTens{4, 3}) =
    layer_stiffness_average(sphere)
_layer_voigt(sphere::_Layered, ::TensND.AbstractTens{2, 3}) =
    layer_conductivity_average(sphere)
_layer_reuss(sphere::_Layered, ::TensND.AbstractTens{4, 3}) =
    layer_compliance_average(sphere)
_layer_reuss(sphere::_Layered, ::TensND.AbstractTens{2, 3}) =
    layer_resistivity_average(sphere)

_layer_voigt(geom, _ref) = _no_layer_average(geom, "Voigt")
_layer_reuss(geom, _ref) = _no_layer_average(geom, "Reuss")

_no_layer_average(geom, bound) = throw(
    ArgumentError(
        "the $bound bound needs a single phase property, but $(typeof(geom)) " *
            "reports `is_homogeneous_inclusion == false` and exposes no " *
            "layer-wise average. A bound has to average the *constituent* " *
            "properties over the inclusion, which takes the internal volume " *
            "fractions — supply them by implementing `Schemes._layer_voigt` / " *
            "`Schemes._layer_reuss` (and `Schemes.has_layer_average`) for your " *
            "type, or use a scheme that consumes contribution tensors instead."
    )
)

"""
    has_layer_average(geom) -> Bool

Whether the bounds can be evaluated on `geom`: `true` for a homogeneous
inclusion, whose declared phase property *is* the average, and for a
heterogeneous one that implements [`_layer_voigt`](@ref) / [`_layer_reuss`](@ref).

The point of asking is that a bound needs the internal volume fractions of an
internally heterogeneous pattern, which the RVE does not carry. Schemes that
merely *consult* a bound — `AsymmetricSelfConsistent` picks its iteration form
from one — use this to fall back rather than to fail.
"""
has_layer_average(geom) = MFH_Core.is_homogeneous_inclusion(geom)
has_layer_average(::_Layered) = true

"Whether every inclusion phase of `rve` can enter a Voigt / Reuss bound."
_bounds_available(rve::RVE) =
    all(has_layer_average(rve.phases[n].geometry) for n in inclusion_phase_names(rve))

"""
    _phase_voigt_property(rve, name, prop, ref) -> AbstractTens

Phase property entering the Voigt bound: `⟨C_r⟩` over the layers when the
inclusion is heterogeneous, then over the orientation distribution.
"""
function _phase_voigt_property(rve::RVE, name::Symbol, prop::Symbol, ref)
    geom = rve.phases[name].geometry
    P = MFH_Core.is_homogeneous_inclusion(geom) ?
        phase_property(rve, name, prop) : _layer_voigt(geom, ref)
    return _apply_symmetrize(P, phase_symmetrize(rve, name))
end

"""
    _phase_reuss_property(rve, name, prop, ref) -> AbstractTens

Phase compliance entering the Reuss bound, layer- then orientation-averaged.
"""
function _phase_reuss_property(rve::RVE, name::Symbol, prop::Symbol, ref)
    geom = rve.phases[name].geometry
    S = MFH_Core.is_homogeneous_inclusion(geom) ?
        inv(phase_property(rve, name, prop)) : _layer_reuss(geom, ref)
    return _apply_symmetrize(S, phase_symmetrize(rve, name))
end

# =============================================================================
#  Bundled phase helpers — one localization solve instead of two
#
#  Mori-Tanaka and the self-consistent kernels ask, for the same phase and the
#  same reference medium, for BOTH a concentration tensor and a contribution
#  tensor.  Each of the four helpers above independently descends to
#  `hill_tensor` / `cod_tensor` — the dominant cost — so the pair currently
#  costs exactly twice what it needs to.
#
#  The four original helpers are deliberately left untouched: `Dilute`,
#  `DiluteDual`, `Maxwell`, `PonteCastanedaWillis`, `DifferentialScheme` and
#  the ASC compliance form each use only ONE of them, have no duplication to
#  remove, and therefore act as exact controls for this change.
#
#  Two invariants make the bundles bitwise identical rather than merely close:
#
#    1. `_apply_symmetrize` does NOT commute with the tensor product, so the
#       bundles thread the RAW (pre-symmetrization) localization tensor and
#       reproduce each original expression verbatim.
#
#    2. Every helper now evaluates its phase in the SAME reference medium,
#       `P₀_proj = _project_matrix(P₀, sym)`.  This was not always the case:
#       `_phase_stiffness_contribution` (order 2) and
#       `_phase_compliance_contribution` (both orders) used to pass the RAW
#       `P₀`, so `A_dil` and `N` of one and the same phase could be computed
#       in two different reference media as soon as
#       `symmetrize ≠ NoSymmetrize`.  See the CHANGELOG entry for v0.1.1.
# =============================================================================

"""
    _phase_dilute_and_contribution(rve, name, prop, P₀; kw...) -> (A_dil, N)

`(_phase_dilute_concentration, _phase_stiffness_contribution)` for a
`VolumeFraction` phase, sharing one localization solve.  Bitwise identical to
calling the two helpers separately.  `CrackDensity` phases must go through
[`_phase_compliance_and_contribution`](@ref).
"""
function _phase_dilute_and_contribution(
        rve::RVE, name::Symbol, prop::Symbol,
        P₀::TensND.AbstractTens{4, 3}; kw...
    )
    geom = rve.phases[name].geometry
    P_i = phase_property(rve, name, prop)
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P₀, sym)
    a = rve.amounts[name]
    A_raw, N_raw = MFH_Core.loc_and_stiffness(geom, P_i, P₀_proj; kw...)
    return (
        _apply_symmetrize(A_raw, sym),
        scale_by_amount(a, _apply_symmetrize(N_raw, sym)),
    )
end

function _phase_dilute_and_contribution(
        rve::RVE, name::Symbol, prop::Symbol,
        P₀::TensND.AbstractTens{2, 3}; kw...
    )
    geom = rve.phases[name].geometry
    P_i = phase_property(rve, name, prop)
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P₀, sym)
    a = rve.amounts[name]
    A_raw, N_raw = MFH_Core.loc_and_stiffness(geom, P_i, P₀_proj; kw...)
    return (
        _apply_symmetrize(A_raw, sym),
        scale_by_amount(a, _apply_symmetrize(N_raw, sym)),
    )
end

"""
    _phase_compliance_and_contribution(rve, name, prop, P₀; kw...) -> (H, N)

`(_phase_compliance_contribution, _phase_stiffness_contribution)` for a
`CrackDensity` phase, sharing one `cod_tensor` solve.
"""
function _phase_compliance_and_contribution(
        rve::RVE, name::Symbol, prop::Symbol,
        P₀::TensND.AbstractTens{4, 3}; kw...
    )
    geom = rve.phases[name].geometry
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P₀, sym)
    a = rve.amounts[name]
    K_int = _crack_interface_K4(rve, name)
    H_raw, N_raw = compliance_and_stiffness_contribution(
        geom, P₀_proj; K_interface = K_int, kw...
    )
    return (
        _apply_symmetrize(delta_compliance(geom, H_raw, amount_value(a)), sym),
        _apply_symmetrize(MFH_Core.delta_stiffness(geom, N_raw, amount_value(a)), sym),
    )
end

function _phase_compliance_and_contribution(
        rve::RVE, name::Symbol, prop::Symbol,
        P₀::TensND.AbstractTens{2, 3}; kw...
    )
    geom = rve.phases[name].geometry
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P₀, sym)
    a = rve.amounts[name]
    α_int = _crack_interface_α(rve, name)
    R_raw, N_raw = compliance_and_stiffness_contribution(
        geom, P₀_proj; α_interface = α_int, kw...
    )
    return (
        _apply_symmetrize(delta_resistivity(geom, R_raw, amount_value(a)), sym),
        _apply_symmetrize(MFH_Core.delta_conductivity(geom, N_raw, amount_value(a)), sym),
    )
end

"""
    _phase_dilute_and_stress_average(rve, name, prop, P₀; kw...) -> (A_dil, CA)

`(_phase_dilute_concentration, _phase_stress_strain_average)` sharing one
localization solve.  Reproduces the three branches of
`_phase_stress_strain_average` verbatim — including the
`sym isa NoSymmetrize || P_i isa TensISO` shortcut, which returns
`P_i ⊡ A_dil` (the **symmetrized** `A`) where the slow branch returns
`_apply_symmetrize(P_i ⊡ A_raw, sym)` (the **raw** `A`).  Those two forms
agree only to ~1e-16, so collapsing them would break bit-identity on the
existing suite.
"""
function _phase_dilute_and_stress_average(
        rve::RVE, name::Symbol, prop::Symbol,
        P₀::TensND.AbstractTens{4, 3}; kw...
    )
    geom = rve.phases[name].geometry
    geom isa MFH_Core.AbstractCrack && return (zero(P₀), zero(P₀))
    P_i = phase_property(rve, name, prop)
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P₀, sym)
    if MFH_Core.is_homogeneous_inclusion(geom)
        A_raw = MFH_Core.strain_strain_loc(geom, P_i, P₀_proj; kw...)
        A_dil = _apply_symmetrize(A_raw, sym)
        (sym isa NoSymmetrize || P_i isa TensND.TensISO) && return (A_dil, P_i ⊡ A_dil)
        return (A_dil, _apply_symmetrize(P_i ⊡ A_raw, sym))
    end
    A_raw, B_raw = MFH_Core.loc_and_stress_average(geom, P_i, P₀_proj; kw...)
    return (_apply_symmetrize(A_raw, sym), _apply_symmetrize(B_raw, sym))
end

function _phase_dilute_and_stress_average(
        rve::RVE, name::Symbol, prop::Symbol,
        P₀::TensND.AbstractTens{2, 3}; kw...
    )
    geom = rve.phases[name].geometry
    geom isa MFH_Core.AbstractCrack && return (zero(P₀), zero(P₀))
    P_i = phase_property(rve, name, prop)
    sym = phase_symmetrize(rve, name)
    P₀_proj = _project_matrix(P₀, sym)
    if MFH_Core.is_homogeneous_inclusion(geom)
        A_raw = MFH_Core.gradient_gradient_loc(geom, P_i, P₀_proj; kw...)
        A_dil = _apply_symmetrize(A_raw, sym)
        (sym isa NoSymmetrize || P_i isa TensND.TensISO) && return (A_dil, P_i ⋅ A_dil)
        return (A_dil, _apply_symmetrize(P_i ⋅ A_raw, sym))
    end
    A_raw, B_raw = MFH_Core.loc_and_stress_average(geom, P_i, P₀_proj; kw...)
    return (_apply_symmetrize(A_raw, sym), _apply_symmetrize(B_raw, sym))
end
