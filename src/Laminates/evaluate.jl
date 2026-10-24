# =============================================================================
#  evaluate.jl — `homogenize` on a `Laminate`, and the per-layer concentration.
#
#  The physics lives in `Core/laminate_algebra.jl`; this file is the bridge
#  between the cell (named layers, thicknesses, interfaces, a frame) and that
#  pure block algebra, plus the choice of the returned TensND type.
# =============================================================================

# ── Gathering the cell into the layer frame ─────────────────────────────────

# Kelvin-Mandel components of a 4th-order layer property **in the laminate
# frame**. `mandel(C, basis)` is the general route and the correct one: a layer
# property may be stored in any basis of its own (a `TensTI` about its own
# axis, a `TensRotated`, …), so its canonical components are not what the
# kernel needs. When the laminate frame is canonical, this is a no-op.
@inline _layer_km4(C, basis) = SMatrix{6, 6}(TensND.mandel(C, basis))

# Components of a 2nd-order layer property in the laminate frame. Both bases
# used by a laminate are orthonormal, so co- and contravariant components
# coincide and `components(t, ℬ)` is unambiguous.
@inline _layer_km2(K, basis) = SMatrix{3, 3}(TensND.components(K, basis))

"""
    _gather(lam, prop::Symbol, ::Val{order}) -> (C6s, fs, T)

Collect the per-layer property matrices in the layer frame and the volume
fractions ``f_i = h_i/L``, promoted to one common element type so that a
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
        _frame_eltype(lam.basis),
    )
    fs = Tel[Tel(h / L) for h in hs]
    return props, fs, Tel
end

# The frame enters the layer components through the change of basis, so its
# element type does too: a dual or symbolic normal. The canonical frame has
# exact zeros and ones and imposes no type of its own.
_frame_eltype(::TensND.CanonicalBasis) = Bool
_frame_eltype(basis::TensND.AbstractBasis) = eltype(basis)

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

# TensND's `isparallel`, at the tolerance this rule always had (1e-12 relative on
# |a·b|, about 2e-12 on its square): a tolerance on comparable numbers, an exact
# structural test on symbolic ones, never a false positive.
_parallel(a, b) = TensND.isparallel(a, b; rtol = 2.0e-12)

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
    _frame_axis(basis, ::Type{Tel}) -> NTuple{3,Tel}

The layer normal, in the element type of the **result** rather than of the
frame.

A `TensTI` axis is geometry, but `TensND` converts it to the element type of the
tensor's *data* (`map(T, n)`), so a `Float64` `1.0` read off a
`CanonicalBasis{3,Float64}` comes back as a symbolic `1.0` — and, the components
being rebuilt from the Walpole basis of axis `n`, that factor multiplies every
coefficient of the answer. Reading a canonical frame exactly, whatever its own
element type, is what keeps a symbolic laminate free of stray floats.

A genuinely oblique numeric frame still yields floating-point axis components,
which is correct: the geometry itself is floating point.
"""
@inline _frame_axis(::TensND.CanonicalBasis{3}, ::Type{Tel}) where {Tel} =
    (zero(Tel), zero(Tel), one(Tel))
@inline function _frame_axis(basis::TensND.AbstractBasis{3}, ::Type{Tel}) where {Tel}
    n = MFH_Core._basis_axis(basis, 3)
    return ntuple(i -> convert(Tel, n[i]), 3)
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
        ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆ = TensND.ti_params_from_mandel(M6)
        n = _frame_axis(lam.basis, eltype(M6))
        return TensND.TensTI{4}(ℓ₁, ℓ₂, ℓ₃, ℓ₅, ℓ₆, n)
    end
    return TensND.from_mandel(Matrix(M6), lam.basis)
end

"""
    _wrap4_general(lam, prop, M6) -> AbstractTens{4,3}

Same as [`_wrap4`](@ref) for a tensor that is transversely isotropic about the
layer normal but **not major-symmetric** — every concentration tensor, ``\\mathbb{A}_i``
first of all: ``\\mathbb{A}_i = \\mathbb{I} + \\mathbb{P}_i:(\\mathbb{C}^{\\mathrm{hom}} - \\mathbb{C}_i)`` is a product of major-symmetric
tensors and a product of two such is not one.

`_wrap4` cannot be reused because `TensND.ti_params_from_mandel` *projects* onto
the 5-coefficient major-symmetric Walpole span: fed a concentration tensor it
would replace ``\\ell_3`` and ``\\ell_4`` by their half-sum and report the result as exact.
The read-off here is `TensND.ti8_params_from_mandel`, which is the identity on the
whole axially-invariant subspace.

``\\ell_7`` and ``\\ell_8``, the antisymmetric azimuthal couplings, are **structurally** zero
here and are dropped rather than carried: `_all_ti` has already established
that every layer is `TensISO` or major-symmetric `TensTI{4,T,5}` about ``\\underline{n}`` —
both with ``\\ell_7 = \\ell_8 = 0`` — that every interface is in-plane isotropic, and the
six-coefficient Walpole span is closed under product and inverse. So the result
is exactly a `TensTI{4,T,6}`, the tightest type that loses nothing.

Why it is worth the trouble: a generic `Tens{4,3}` carries 81 components and no
structure, so the caller pays a full array contraction for every product, an
`inv` through a 6×6 factorization, and — symbolically — expressions an order of
magnitude larger than the six coefficients that actually describe the tensor.
"""
function _wrap4_general(lam::Laminate, prop::Symbol, M6::AbstractMatrix)
    if _all_ti(lam, prop, Val(4))
        ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆, _, _ = TensND.ti8_params_from_mandel(Matrix(M6))
        n = _frame_axis(lam.basis, eltype(M6))
        return TensND.TensTI{4}(ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆, n)
    end
    return TensND.from_mandel(Matrix(M6), lam.basis)
end

"""
    _wrap2(lam, prop, M3) -> AbstractTens{2,3}

Order-2 counterpart of [`_wrap4`](@ref): `TensTI{2}(a, b, n)` with `a` the
transverse and `b` the axial conductivity when every layer is isotropic or TI
about the normal, a generic `Tens{2,3}` otherwise.
"""
function _wrap2(lam::Laminate, prop::Symbol, M3::AbstractMatrix)
    if _all_ti(lam, prop, Val(2))
        a, b, _ = TensND.ti2_params_from_matrix(M3)
        return TensND.TensTI{2}(a, b, _frame_axis(lam.basis, eltype(M3)))
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
    S = sum(fs[i] * TensND.mandel_inv_schur(mats[i]) for i in eachindex(mats))
    return _wrap4(lam, p, TensND.mandel_inv_schur(S))
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
``\\mathbb{P} = \\underline{n}\\stackrel{s}{\\otimes}\\boldsymbol{K}^{-1}\\stackrel{s}{\\otimes}\\underline{n}`` — the flat-inclusion limit of the Hill polarization
tensor of an ellipsoid embedded in that layer's own material — and
``\\mathbb{Q} = \\mathbb{C} - \\mathbb{C}:\\mathbb{P}:\\mathbb{C}``, the in-plane Schur complement.

Exposed for the theory page and for checking the identities ``\\mathbb{P}:\\mathbb{C}:\\mathbb{P} = \\mathbb{P}`` and
``\\mathbb{Q}:\\mathbb{P} = 0``; the kernel itself never materializes them as tensors.
"""
function laminate_hill(lam::Laminate, name::Symbol; property::Symbol = :C)
    C = layer_property(lam, name, property)
    C isa TensND.AbstractTens{4, 3} ||
        throw(ArgumentError("laminate_hill needs a 4th-order property; got $(typeof(C))"))
    C6 = _layer_km4(C, lam.basis)
    P = MFH_Core.flat_hill(C6)
    Q = C6 - C6 * P * C6
    return (
        TensND.from_mandel(Matrix(P), lam.basis),
        TensND.from_mandel(Matrix(Q), lam.basis),
    )
end

"""
    layer_gradient_concentration(lam, name::Symbol; property = :C) -> AbstractTens

Gradient concentration tensor of one layer: with a fourth-order stiffness, the
strain concentration tensor ``\\mathbb{A}_i``, ``\\boldsymbol{\\varepsilon}_i = \\mathbb{A}_i:\\boldsymbol{E}``, with
``\\mathbb{A}_i = \\mathbb{I} + \\mathbb{P}_i:(\\mathbb{C}^{\\mathrm{hom}} - \\mathbb{C}_i)`` and ``\\sum_i f_i\\,\\mathbb{A}_i = \\mathbb{I}``; with a
second-order conductivity (`property = :K`), ``\\nabla T_i = \\boldsymbol{A}_i\\cdot\\nabla T``.

Its in-plane block is the identity and its in-plane/out-of-plane coupling
block vanishes — the macroscopic in-plane gradient reaches every layer
unchanged, which is the compatibility condition
``\\boldsymbol{\\varepsilon}_i = \\boldsymbol{E} + \\underline{a}_i\\stackrel{s}{\\otimes}\\underline{n}`` read backwards.

The same object as [`gradient_gradient_concentration`](@ref)`(lam, name)`, the
package-wide generic. Defined for perfect interfaces; with a primal (spring /
Kapitza) interface part of the macroscopic gradient is carried by the jumps,
so the layer gradients no longer average to it — see [`interface_jump`](@ref).
Explicit name in elasticity: [`layer_strain_concentration`](@ref).
"""
function layer_gradient_concentration(lam::Laminate, name::Symbol; property::Symbol = :C)
    P, Phom = _concentration_setup(lam, name, property)
    return _wrap_km(lam, property, MFH_Core.laminate_gradient_concentration(P, Phom))
end

"""
    layer_flux_concentration(lam, name::Symbol; property = :C) -> AbstractTens

Flux concentration tensor of one layer: the stress concentration tensor
``\\mathbb{B}_i``, ``\\boldsymbol{\\sigma}_i = \\mathbb{B}_i:\\boldsymbol{\\Sigma}``, with
``\\mathbb{B}_i = \\mathbb{C}_i:\\mathbb{A}_i:(\\mathbb{C}^{\\mathrm{hom}})^{-1}`` and ``\\sum_i f_i\\,\\mathbb{B}_i = \\mathbb{I}``; in
conduction, ``\\underline{q}_i = \\boldsymbol{B}_i\\cdot\\underline{q}``. The same object as
[`flux_flux_concentration`](@ref)`(lam, name)`. Explicit name in elasticity:
[`layer_stress_concentration`](@ref).
"""
function layer_flux_concentration(lam::Laminate, name::Symbol; property::Symbol = :C)
    P, Phom = _concentration_setup(lam, name, property)
    return _wrap_km(lam, property, MFH_Core.laminate_flux_concentration(P, Phom))
end

# The Kelvin-Mandel matrices of the layer and of the laminate, at the order of
# the property: the property tensor of the layer is the function barrier.
_concentration_setup(lam::Laminate, name::Symbol, property::Symbol) =
    _concentration_setup(lam, property, layer_property(lam, name, property))

function _concentration_setup(lam::Laminate, property::Symbol, C::TensND.AbstractTens{4, 3})
    C6s, fs, T = _gather(lam, property, Val(4))
    P_int, C_surf = _interface_terms(lam, T, Val(4))
    return (_layer_km4(C, lam.basis), MFH_Core.laminate_stiffness(C6s, fs, P_int, C_surf))
end

function _concentration_setup(lam::Laminate, property::Symbol, K::TensND.AbstractTens{2, 3})
    K3s, fs, T = _gather(lam, property, Val(2))
    P_int, K_surf = _interface_terms(lam, T, Val(2))
    return (_layer_km2(K, lam.basis), MFH_Core.laminate_conductivity(K3s, fs, P_int, K_surf))
end

_concentration_setup(::Laminate, property::Symbol, P) = throw(
    ArgumentError(
        "layer concentration needs a 4th- or 2nd-order property; :$(property) is a $(typeof(P))"
    )
)

# Back to a tensor, at the order of the Kelvin-Mandel matrix.
_wrap_km(lam::Laminate, property::Symbol, M::StaticMatrix{6, 6}) = _wrap4_general(lam, property, M)
_wrap_km(lam::Laminate, property::Symbol, M::StaticMatrix{3, 3}) = _wrap2(lam, property, M)

# ── The package-wide concentration generics, on a layer ──────────────────────
#
#  `Laminate` is a cell, not an `AbstractInclusion`: there is no matrix, no
#  reference medium and no Eshelby auxiliary problem, so the `(incl, P₁, P₀)`
#  signature of `src/concentration.jl` does not apply. What plays the role of
#  "which inclusion" is the LAYER NAME — the same choice as
#  `gradient_gradient_concentration(::LayeredSphere, C₀; layer::Int)`. The
#  property, `:C` by default as for `homogenize`, sets the order.
#
#  Each method is one call to `_concentration_setup` plus one kernel from
#  `Core/laminate_algebra.jl`, which inverts by cofactors and never by LU — so
#  every one of them is exact under `ForwardDiff.Dual` and evaluable on
#  `SymPy.Sym` / `Symbolics.Num`.
#
#  Sum rules: `Σ_i f_i 𝔸_i = 𝕀`, `Σ_i f_i 𝔹_i = 𝕀`, `Σ_i f_i ℂ_i:𝔸_i = ℂ^{hom}`
#  and `Σ_i f_i 𝔸_i:𝕊^{hom} = 𝕊^{hom}`. They hold for perfect and for dual
#  (membrane) interfaces; with a primal (spring / Kapitza) interface part of the
#  macroscopic gradient is carried by the jumps (`interface_jump`) and the
#  gradient-side rules no longer close.

"""
    gradient_gradient_concentration(lam::Laminate, name::Symbol; property = :C) -> AbstractTens

Gradient concentration tensor ``\\mathbb{A}_i`` of layer `name`:
``\\boldsymbol{\\varepsilon}_i = \\mathbb{A}_i:\\boldsymbol{E}`` in elasticity,
``\\nabla T_i = \\boldsymbol{A}_i\\cdot\\nabla T`` in conduction. Identical to
[`layer_gradient_concentration`](@ref).
"""
function gradient_gradient_concentration(lam::Laminate, name::Symbol; property::Symbol = :C)
    return layer_gradient_concentration(lam, name; property = property)
end

"""
    flux_gradient_concentration(lam::Laminate, name::Symbol; property = :C) -> AbstractTens

Flux of layer `name` per unit macroscopic gradient,
``\\mathbb{A}_{\\sigma\\varepsilon,i} = \\mathbb{C}_i:\\mathbb{A}_i``:
``\\boldsymbol{\\sigma}_i = \\mathbb{A}_{\\sigma\\varepsilon,i}:\\boldsymbol{E}``, with
``\\sum_i f_i\\,\\mathbb{A}_{\\sigma\\varepsilon,i} = \\mathbb{C}^{\\mathrm{hom}}``.

Unlike [`flux_flux_concentration`](@ref) it forms no inverse of the effective
property, so it is the cheaper and better-conditioned of the two flux-side
tensors.
"""
function flux_gradient_concentration(lam::Laminate, name::Symbol; property::Symbol = :C)
    P, Phom = _concentration_setup(lam, name, property)
    return _wrap_km(lam, property, MFH_Core.laminate_flux_gradient_concentration(P, Phom))
end

"""
    gradient_flux_concentration(lam::Laminate, name::Symbol; property = :C) -> AbstractTens

Gradient of layer `name` per unit macroscopic flux,
``\\mathbb{A}_{\\varepsilon\\sigma,i} = \\mathbb{A}_i:(\\mathbb{C}^{\\mathrm{hom}})^{-1}``:
``\\boldsymbol{\\varepsilon}_i = \\mathbb{A}_{\\varepsilon\\sigma,i}:\\boldsymbol{\\Sigma}``, with
``\\sum_i f_i\\,\\mathbb{A}_{\\varepsilon\\sigma,i} = (\\mathbb{C}^{\\mathrm{hom}})^{-1}``.
"""
function gradient_flux_concentration(lam::Laminate, name::Symbol; property::Symbol = :C)
    P, Phom = _concentration_setup(lam, name, property)
    return _wrap_km(lam, property, MFH_Core.laminate_gradient_flux_concentration(P, Phom))
end

"""
    flux_flux_concentration(lam::Laminate, name::Symbol; property = :C) -> AbstractTens

Flux concentration tensor ``\\mathbb{B}_i`` of layer `name`:
``\\boldsymbol{\\sigma}_i = \\mathbb{B}_i:\\boldsymbol{\\Sigma}``. Identical to
[`layer_flux_concentration`](@ref).
"""
function flux_flux_concentration(lam::Laminate, name::Symbol; property::Symbol = :C)
    return layer_flux_concentration(lam, name; property = property)
end

# The names of elasticity, for fourth-order stiffnesses only.

"""
    layer_strain_concentration(lam, name::Symbol; property = :C) -> AbstractTens{4,3}

Name in elasticity of [`layer_gradient_concentration`](@ref), for a fourth-order
property.
"""
layer_strain_concentration(lam::Laminate, name::Symbol; kw...) = MFH_Core._order_checked(
    Val(4), :layer_strain_concentration, :layer_gradient_concentration,
    layer_gradient_concentration(lam, name; kw...)
)

"""
    layer_stress_concentration(lam, name::Symbol; property = :C) -> AbstractTens{4,3}

Name in elasticity of [`layer_flux_concentration`](@ref), for a fourth-order
property.
"""
layer_stress_concentration(lam::Laminate, name::Symbol; kw...) = MFH_Core._order_checked(
    Val(4), :layer_stress_concentration, :layer_flux_concentration,
    layer_flux_concentration(lam, name; kw...)
)

"""
    interface_jump(lam, k::Integer, E; property = :C) -> NTuple{3}

Displacement jump ``[\\![\\underline{u}]\\!] = \\boldsymbol{\\mathcal{K}}_k\\cdot(\\boldsymbol{\\Sigma}\\cdot\\underline{n})`` across the `k`-th interface, under the
macroscopic strain `E`, with ``\\boldsymbol{\\Sigma} = \\mathbb{C}^{\\mathrm{hom}}:\\boldsymbol{E}``. Returned in canonical
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
    # back to canonical components — `_frame_matrix`, not `_basis_matrix`:
    # the latter pins the frame to `Float64` and `Float64(::Sym)` throws.
    R = Matrix(TensND.vecbasis(lam.basis, :cov))
    return ntuple(i -> sum(R[i, j] * jump_local[j] for j in 1:3), 3)
end

_interface_jump_local(::PerfectInterface, t) = zero(t)
function _interface_jump_local(itf::SpringInterface, t)
    sn, st = spring_compliances(itf)      # stored `kn`, `kt` are stiffnesses
    return SVector(st * t[1], st * t[2], sn * t[3])
end
_interface_jump_local(::MembraneInterface, t) = zero(t)
_interface_jump_local(::KapitzaInterface, t) = zero(t)
_interface_jump_local(::SurfaceConductiveInterface, t) = zero(t)

# ── Concentration tensors of every layer ────────────────────────────────────
#
# The exact solution gives each layer its strain and stress concentration
# tensors directly. With a primal (spring / Kapitza) interface part of the
# macroscopic strain is carried by the jumps, and the strain average over the
# layers no longer closes on 𝕀.

function Schemes._concentration_tensors(lam::Laminate, scheme::Laminated, ::Val{p}; kw...) where {p}
    P_hom = Schemes._evaluate(lam, scheme, Val(p); kw...)
    names = layer_names(lam)
    pairs = [_layer_concentration_pair(lam, n, p) for n in names]
    strain = Dict{Symbol, TensND.AbstractTens}(n => A for (n, (A, _)) in zip(names, pairs))
    stress = Dict{Symbol, TensND.AbstractTens}(n => B for (n, (_, B)) in zip(names, pairs))
    weights = Dict(n => layer_volume_fraction(lam, n) for n in names)
    properties = Dict{Symbol, TensND.AbstractTens}(n => layer_property(lam, n, p) for n in names)
    return Schemes._concentration_result(P_hom, names, strain, stress, weights, properties, nothing)
end

_layer_concentration_pair(lam::Laminate, name::Symbol, p::Symbol) = (
    gradient_gradient_concentration(lam, name; property = p),
    flux_gradient_concentration(lam, name; property = p),
)
