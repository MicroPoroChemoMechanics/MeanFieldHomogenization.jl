# =============================================================================
#  evaluate.jl — `homogenize` on a `Laminate`, and the per-layer localization.
#
#  The physics lives in `Core/laminate_algebra.jl`; this file is the bridge
#  between the cell (named layers, thicknesses, interfaces, a frame) and that
#  pure block algebra, plus the choice of the returned TensND type.
# =============================================================================

# ── Gathering the cell into the layer frame ─────────────────────────────────

# Kelvin-Mandel components of a 4th-order layer property **in the laminate
# frame**. `KM(C, basis)` is the general route and the correct one: a layer
# property may be stored in any basis of its own (a `TensTI` about its own
# axis, a `TensRotated`, …), so its canonical components are not what the
# kernel needs. When the laminate frame is canonical, this is a no-op.
@inline _layer_km4(C, basis) = SMatrix{6, 6}(TensND.KM(C, basis))

# Components of a 2nd-order layer property in the laminate frame. Both bases
# used by a laminate are orthonormal, so co- and contravariant components
# coincide and `components(t, ℬ)` is unambiguous.
@inline _layer_km2(K, basis) = SMatrix{3, 3}(TensND.components(K, basis))

"""
    _gather(lam, prop::Symbol, ::Val{order}) -> (C6s, fs, T)

Collect the per-layer property matrices in the layer frame and the volume
fractions `f_i = h_i / L`, promoted to one common element type so that a
`Dual` modulus, a `Dual` thickness or a symbolic entry propagates through the
whole kernel.
"""
function _gather(lam::Laminate, prop::Symbol, ::Val{4})
    props, fs, Tel = _gather_common(lam, prop)
    mats = SMatrix{6, 6, Tel}[SMatrix{6, 6, Tel}(_layer_km4(p, lam.basis)) for p in props]
    return mats, fs, Tel
end

function _gather(lam::Laminate, prop::Symbol, ::Val{2})
    props, fs, Tel = _gather_common(lam, prop)
    mats = SMatrix{3, 3, Tel}[SMatrix{3, 3, Tel}(_layer_km2(p, lam.basis)) for p in props]
    return mats, fs, Tel
end

function _gather_common(lam::Laminate, prop::Symbol)
    names = lam.layer_names
    isempty(names) && throw(ArgumentError("Laminate has no layer; call add_layer! first"))
    props = [layer_property(lam, nm, prop) for nm in names]
    L = laminate_period(lam)
    hs = [lam.thicknesses[nm] for nm in names]
    Tel = promote_type(
        mapreduce(eltype, promote_type, props),
        mapreduce(typeof, promote_type, hs),
        typeof(L),
    )
    fs = Tel[Tel(h / L) for h in hs]
    return props, fs, Tel
end

# ── The returned TensND type ────────────────────────────────────────────────
#
# Symmetry is decided STRUCTURALLY, from the declared classes of the inputs —
# never from a numerical test on the output, and never through `best_fit_ti`
# (which is a projection, documented as reporting-only, and not
# symbolic-safe). The rule is exact and holds for `Dual` and symbolic element
# types alike: the TI tensors about a common axis are closed under product
# and inverse, so a laminate of layers that are all TI about its own normal
# is exactly TI about that normal. Every interface type is in-plane
# isotropic, so none of them breaks it.

@inline function _parallel(a, b)
    T = promote_type(eltype(a), eltype(b))
    T <: Real || return false
    da = sqrt(sum(x -> x^2, a))
    db = sqrt(sum(x -> x^2, b))
    (da > 0 && db > 0) || return false
    return isapprox(abs(sum(a .* b)), da * db; rtol = 1.0e-12, atol = 1.0e-14)
end

_ti_about(::TensND.TensISO{4, 3}, n) = true
# `TensTI{4,T,5}` ONLY — the major-symmetric Walpole form. The 6- and
# 8-component variants are transversely isotropic too, but not
# major-symmetric: `TensTI{4,T,8}` is exactly what the exact rotation-group
# average (`transverse_isotropify`) produces, with ℓ₃ ≠ ℓ₄ and the
# antisymmetric azimuthal couplings ℓ₇, ℓ₈. Claiming the TI branch for those
# would send the result through the 5-parameter read-off and silently discard
# that content, so they fall through to the generic `Tens`, which keeps
# everything.
_ti_about(C::TensND.TensTI{4, T, 5}, n) where {T} = _parallel(TensND.axis(C), n)
_ti_about(::TensND.TensTI{4}, n) = false
_ti_about(::Any, n) = false

_ti_about2(::TensND.TensISO{2, 3}, n) = true
# Likewise at order 2: `TensTI{2,T,2}` is `a(δ−n⊗n) + b n⊗n` and is fully
# described by the two numbers `_wrap2` reads off; the 3-component variant
# carries an extra part that the read-off would drop.
_ti_about2(K::TensND.TensTI{2, T, 2}, n) where {T} = _parallel(TensND.axis(K), n)
_ti_about2(::TensND.TensTI{2}, n) = false
_ti_about2(::Any, n) = false

"""
    _ti_preserving(itf) -> Bool

Whether an interface leaves a transversely isotropic stack transversely
isotropic. The scalar-valued models are in-plane isotropic by construction and
do; the tensor-valued ones carry an arbitrary in-plane texture and, in
general, do not — so they are refused, structurally and conservatively, even
when the tensor they hold happens to be isotropic.
"""
_ti_preserving(::PerfectInterface) = true
_ti_preserving(::SpringInterface) = true
_ti_preserving(::MembraneInterface) = true
_ti_preserving(::KapitzaInterface) = true
_ti_preserving(::SurfaceConductiveInterface) = true
_ti_preserving(::AnisotropicSpringInterface) = false
_ti_preserving(::AnisotropicMembraneInterface) = false
_ti_preserving(::AnisotropicSurfaceConductiveInterface) = false

# All layers TI about the laminate normal — AND every interface in-plane
# isotropic. Both are needed: an anisotropic interface breaks the symmetry of
# an otherwise transversely isotropic stack just as surely as an anisotropic
# layer does, and claiming TI would send the result through the
# five-coefficient read-off and silently discard the in-plane texture.
function _all_ti(lam::Laminate, prop::Symbol, ::Val{4})
    all(_ti_preserving, lam.interfaces) || return false
    n = laminate_normal(lam)
    return all(nm -> _ti_about(layer_property(lam, nm, prop), n), lam.layer_names)
end

function _all_ti(lam::Laminate, prop::Symbol, ::Val{2})
    all(_ti_preserving, lam.interfaces) || return false
    n = laminate_normal(lam)
    return all(nm -> _ti_about2(layer_property(lam, nm, prop), n), lam.layer_names)
end

"""
    _wrap4(lam, prop, M6) -> AbstractTens{4,3}

Turn the effective Kelvin-Mandel matrix (expressed in the layer frame) back
into a TensND tensor, picking the tightest **exact** type:

- `TensTI{4,T,5}` about the layer normal when every layer is isotropic or TI
  about that normal — the Walpole coefficients are *read off* the layer-frame
  matrix in closed form, not fitted. Worth having beyond tidiness: a `TensTI`
  fed back into a multiscale chain hits the analytic TI-coaxial Hill branch of
  `Core/dispatch.jl` instead of a cubature;
- a generic `Tens{4,3}` in the laminate basis otherwise.
"""
function _wrap4(lam::Laminate, prop::Symbol, M6::AbstractMatrix)
    if _all_ti(lam, prop, Val(4))
        ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆ = TensND.ti_params_from_KM(M6)
        return TensND.TensTI{4}(ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆, laminate_normal(lam))
    end
    return TensND.inv_KM(Matrix(M6), lam.basis)
end

"""
    _wrap2(lam, prop, M3) -> AbstractTens{2,3}

Order-2 counterpart of [`_wrap4`](@ref): `TensTI{2}(a, b, n)` with `a` the
transverse and `b` the axial conductivity when every layer is isotropic or TI
about the normal, a generic `Tens{2,3}` otherwise.
"""
function _wrap2(lam::Laminate, prop::Symbol, M3::AbstractMatrix)
    if _all_ti(lam, prop, Val(2))
        a = (M3[1, 1] + M3[2, 2]) / 2
        b = M3[3, 3]
        return TensND.TensTI{2}(a, b, laminate_normal(lam))
    end
    return TensND.Tens(Matrix(M3), lam.basis)
end

# A single layer with only perfect interfaces IS the effective medium — return
# its property object untouched, so the degeneracy is bit-exact and keeps the
# input's own type.
_trivial_cell(lam::Laminate) =
    layer_count(lam) == 1 && all(itf -> itf isa PerfectInterface, lam.interfaces)

# ── `Laminated` — the exact periodic solution ───────────────────────────────

"""
    _evaluate(lam::Laminate, ::Laminated, ::Val{p}; kw...)

Exact effective property of the periodic multilayer cell. Serves elasticity
and transport from one implementation, dispatching on the **order** of the
stored property exactly as `_mt_dispatch` does for Mori-Tanaka.
"""
function _evaluate(lam::Laminate, ::Laminated, ::Val{p}; kw...) where {p}
    first_prop = layer_property(lam, first(lam.layer_names), p)
    return _laminated_dispatch(lam, first_prop, Val(p); kw...)
end

function _laminated_dispatch(
        lam::Laminate, ::TensND.AbstractTens{4, 3}, ::Val{p}; kw...
    ) where {p}
    _trivial_cell(lam) && return layer_property(lam, first(lam.layer_names), p)
    C6s, fs, T = _gather(lam, p, Val(4))
    P_int, C_surf = _interface_terms(lam, T, Val(4))
    Chom = MFH_Core.laminate_stiffness(C6s, fs, P_int, C_surf)
    return _wrap4(lam, p, Chom)
end

function _laminated_dispatch(
        lam::Laminate, ::TensND.AbstractTens{2, 3}, ::Val{p}; kw...
    ) where {p}
    _trivial_cell(lam) && return layer_property(lam, first(lam.layer_names), p)
    K3s, fs, T = _gather(lam, p, Val(2))
    P_int, K_surf = _interface_terms(lam, T, Val(2))
    Khom = MFH_Core.laminate_conductivity(K3s, fs, P_int, K_surf)
    return _wrap2(lam, p, Khom)
end

# ── Bounds on a laminate ────────────────────────────────────────────────────
#
# Voigt and Reuss need no matrix phase, so they apply to a laminate as they do
# to an RVE. They are genuinely useful here (the exact answer is bracketed by
# them) and they are free oracles for the test suite: the laminate is EXACTLY
# Voigt in the in-plane block and EXACTLY Reuss in the out-of-plane one.
# Imperfect interfaces are ignored by both, as bounds on the layers alone.

function _evaluate(lam::Laminate, scheme::Union{Voigt, Reuss}, ::Val{p}; kw...) where {p}
    first_prop = layer_property(lam, first(lam.layer_names), p)
    return _bounds_dispatch(lam, scheme, first_prop, Val(p))
end

function _bounds_dispatch(lam::Laminate, ::Voigt, ::TensND.AbstractTens{4, 3}, ::Val{p}) where {p}
    mats, fs, _ = _gather(lam, p, Val(4))
    return _wrap4(lam, p, sum(fs[i] * mats[i] for i in eachindex(mats)))
end

function _bounds_dispatch(lam::Laminate, ::Voigt, ::TensND.AbstractTens{2, 3}, ::Val{p}) where {p}
    mats, fs, _ = _gather(lam, p, Val(2))
    return _wrap2(lam, p, sum(fs[i] * mats[i] for i in eachindex(mats)))
end

function _bounds_dispatch(lam::Laminate, ::Reuss, ::TensND.AbstractTens{4, 3}, ::Val{p}) where {p}
    mats, fs, _ = _gather(lam, p, Val(4))
    S = sum(fs[i] * MFH_Core._inv_km6(mats[i]) for i in eachindex(mats))
    return _wrap4(lam, p, MFH_Core._inv_km6(S))
end

function _bounds_dispatch(lam::Laminate, ::Reuss, ::TensND.AbstractTens{2, 3}, ::Val{p}) where {p}
    mats, fs, _ = _gather(lam, p, Val(2))
    S = sum(fs[i] * MFH_Core._inv3(mats[i]) for i in eachindex(mats))
    return _wrap2(lam, p, MFH_Core._inv3(S))
end

# ── Per-layer fields ────────────────────────────────────────────────────────

"""
    laminate_hill(lam, name::Symbol; property = :C) -> (ℙ, ℚ)

The two Hill tensors of one layer, as TensND tensors in the laminate basis:
`ℙ = n ⊗ˢ 𝐊⁻¹ ⊗ˢ n` — the flat-inclusion limit of the Hill polarization
tensor of an ellipsoid embedded in that layer's own material — and
`ℚ = ℂ − ℂ:ℙ:ℂ`, the in-plane Schur complement.

Exposed for the theory page and for checking the identities `ℙ:ℂ:ℙ = ℙ` and
`ℚ:ℙ = 0`; the kernel itself never materializes them as tensors.
"""
function laminate_hill(lam::Laminate, name::Symbol; property::Symbol = :C)
    C = layer_property(lam, name, property)
    C isa TensND.AbstractTens{4, 3} ||
        throw(ArgumentError("laminate_hill needs a 4th-order property; got $(typeof(C))"))
    C6 = _layer_km4(C, lam.basis)
    P = MFH_Core.flat_hill(C6)
    Q = C6 - C6 * P * C6
    return (
        TensND.inv_KM(Matrix(P), lam.basis),
        TensND.inv_KM(Matrix(Q), lam.basis),
    )
end

"""
    layer_strain_localization(lam, name::Symbol; property = :C) -> Tens{4,3}

Strain localization tensor `𝔸_i` of one layer: `ε_i = 𝔸_i : E`, with
`𝔸_i = 𝕀 + ℙ_i : (ℂ^{hom} − ℂ_i)` and `Σ_i f_i 𝔸_i = 𝕀`.

Its in-plane block is the identity and its in-plane/out-of-plane coupling
block vanishes — the macroscopic in-plane strain reaches every layer
unchanged, which is the compatibility condition `ε_i = E + a_i ⊗ˢ n`.

Defined for perfect interfaces; with a primal (spring / Kapitza) interface
part of the macroscopic strain is carried by the jumps, so the layer strains
no longer average to `E` — see [`interface_jump`](@ref).
"""
function layer_strain_localization(lam::Laminate, name::Symbol; property::Symbol = :C)
    C6, Chom = _loc_setup4(lam, name, property)
    return TensND.inv_KM(
        Matrix(MFH_Core.laminate_strain_localization(C6, Chom)), lam.basis
    )
end

"""
    layer_stress_localization(lam, name::Symbol; property = :C) -> Tens{4,3}

Stress localization tensor `𝔹_i` of one layer: `σ_i = 𝔹_i : Σ`, with
`𝔹_i = ℂ_i : 𝔸_i : (ℂ^{hom})^{-1}` and `Σ_i f_i 𝔹_i = 𝕀`.
"""
function layer_stress_localization(lam::Laminate, name::Symbol; property::Symbol = :C)
    C6, Chom = _loc_setup4(lam, name, property)
    return TensND.inv_KM(
        Matrix(MFH_Core.laminate_stress_localization(C6, Chom)), lam.basis
    )
end

function _loc_setup4(lam::Laminate, name::Symbol, property::Symbol)
    C = layer_property(lam, name, property)
    C isa TensND.AbstractTens{4, 3} || throw(
        ArgumentError(
            "layer localization at order 4 needs a 4th-order property; got $(typeof(C))"
        )
    )
    C6s, fs, T = _gather(lam, property, Val(4))
    P_int, C_surf = _interface_terms(lam, T, Val(4))
    Chom = MFH_Core.laminate_stiffness(C6s, fs, P_int, C_surf)
    return (_layer_km4(C, lam.basis), Chom)
end

"""
    layer_gradient_localization(lam, name::Symbol; property = :K) -> Tens{2,3}

Transport counterpart of [`layer_strain_localization`](@ref):
`∇T_i = 𝐀_i · ∇T`, with `Σ_i f_i 𝐀_i = 𝟏`.
"""
function layer_gradient_localization(lam::Laminate, name::Symbol; property::Symbol = :K)
    K3, Khom = _loc_setup2(lam, name, property)
    return TensND.Tens(
        Matrix(MFH_Core.laminate_gradient_localization(K3, Khom)), lam.basis
    )
end

"""
    layer_flux_localization(lam, name::Symbol; property = :K) -> Tens{2,3}

Transport counterpart of [`layer_stress_localization`](@ref): `q_i = 𝐁_i · q`,
with `Σ_i f_i 𝐁_i = 𝟏`.
"""
function layer_flux_localization(lam::Laminate, name::Symbol; property::Symbol = :K)
    K3, Khom = _loc_setup2(lam, name, property)
    return TensND.Tens(
        Matrix(MFH_Core.laminate_flux_localization(K3, Khom)), lam.basis
    )
end

function _loc_setup2(lam::Laminate, name::Symbol, property::Symbol)
    K = layer_property(lam, name, property)
    K isa TensND.AbstractTens{2, 3} || throw(
        ArgumentError(
            "layer localization at order 2 needs a 2nd-order property; got $(typeof(K))"
        )
    )
    K3s, fs, T = _gather(lam, property, Val(2))
    P_int, K_surf = _interface_terms(lam, T, Val(2))
    Khom = MFH_Core.laminate_conductivity(K3s, fs, P_int, K_surf)
    return (_layer_km2(K, lam.basis), Khom)
end

"""
    interface_jump(lam, k::Integer, E; property = :C) -> NTuple{3}

Displacement jump `[u] = 𝕂_k · (Σ · n)` across the `k`-th interface, under the
macroscopic strain `E`, with `Σ = ℂ^{hom} : E`. Returned in canonical
components. Zero for a perfect or a dual (membrane) interface, which produce
no jump.

This is the observable that distinguishes a spring interface from a softer
layer: the compliance shows up as a discontinuity, not as a strain.
"""
function interface_jump(lam::Laminate, k::Integer, E; property::Symbol = :C)
    itf = layer_interface(lam, k)
    Chom = homogenize(lam, Laminated(), property)
    Σ = Chom ⊡ E
    Σarr = TensND.components(Σ, lam.basis)
    nloc = (0, 0, 1)                       # `n` is the third axis OF THAT FRAME
    t = SVector{3}(ntuple(i -> sum(Σarr[i, j] * nloc[j] for j in 1:3), 3))
    jump_local = _interface_jump_local(itf, t)
    # back to canonical components
    R = MFH_Core._basis_matrix(lam.basis)
    return ntuple(i -> sum(R[i, j] * jump_local[j] for j in 1:3), 3)
end

_interface_jump_local(::PerfectInterface, t) = zero(t)
_interface_jump_local(itf::SpringInterface, t) =
    SVector(itf.kt * t[1], itf.kt * t[2], itf.kn * t[3])
_interface_jump_local(::MembraneInterface, t) = zero(t)
_interface_jump_local(::KapitzaInterface, t) = zero(t)
_interface_jump_local(::SurfaceConductiveInterface, t) = zero(t)
