# =============================================================================
#  elastic.jl — elasticity and conduction of a `LayeredCylinder` with numbers:
#  reading the moduli, solving the radial problems, packing the results in
#  transversely isotropic tensors about the cylinder axis.
# =============================================================================

_not_coaxial(P, axis) = throw(
    ArgumentError(
        "a LayeredCylinder of axis $axis takes isotropic properties or transversely " *
            "isotropic ones about that axis; got a $(nameof(typeof(P))) of axis $(TensND.axis(P))"
    )
)

_not_ti(P, drel) = throw(
    ArgumentError(
        "a LayeredCylinder takes properties that are isotropic or transversely isotropic " *
            "about its axis; the $(nameof(typeof(P))) it received is at a relative distance " *
            "$drel from that symmetry"
    )
)

_symbolic_general(P) = throw(
    ArgumentError(
        "a LayeredCylinder takes a symbolic property as a TensISO or as a TensTI about its " *
            "axis; a $(nameof(typeof(P))) would have to be compared with its transversely " *
            "isotropic part, which symbols do not allow"
    )
)

# Relative distance to transverse isotropy under which a general tensor (the
# running estimate of a scheme, stored in a rotated basis) is read as such.
const _TI_RTOL = 1.0e-8

# A general tensor: its transversely isotropic part about the axis, provided
# it is the whole of it. Numbers only: the distance is compared.
function _ti_part(P::TensND.AbstractTens, axis)
    is_hard_numeric(eltype(P)) || _symbolic_general(P)
    Pti, _, drel = TensND.project(P, Val(:TI), axis)
    drel ≤ _TI_RTOL || _not_ti(P, drel)
    return Pti
end

"""
    _hill_moduli(C, axis) -> (; k, l, m, n, p, lt)

Hill's moduli of a stiffness that is isotropic, or transversely isotropic about
`axis`: plane-strain bulk modulus ``k = (C_{1111} + C_{1122})/2``,
``l = C_{1133}``, transverse shear modulus ``m = (C_{1111} - C_{1122})/2``,
``n = C_{3333}`` and axial shear modulus ``p = C_{2323}``. `lt` is ``C_{3311}``, equal to
``l`` for a stiffness and distinct from it for an operator without major symmetry
(a running estimate in aging viscoelasticity).
"""
function _hill_moduli(C::TensND.TensISO{4, 3}, axis)
    κ, μ = TensND.k_mu(C)
    l = κ - 2 * μ / 3
    return (; k = κ + μ / 3, l, m = μ, n = κ + 4 * μ / 3, p = μ, lt = l)
end

# Read on the Walpole coefficients, as the block matrices of aging
# viscoelasticity are: ℓ₁ = n, ℓ₂ = 2k, ℓ₃ = √2 C₃₃₁₁, ℓ₄ = √2 C₁₁₃₃, ℓ₅ = 2m,
# ℓ₆ = 2p. An operator without major symmetry (ℓ₃ ≠ ℓ₄) is read as it is; one
# that is not invariant under the mirrors through the axis (ℓ₇, ℓ₈ ≠ 0) is not
# transversely isotropic in the sense of the cylinder, and is refused at the
# tolerance of a general tensor.
function _hill_moduli(C::TensND.TensTI{4}, axis)
    TensND.isparallel(TensND.axis(C), axis) || _not_coaxial(C, axis)
    ℓ = TensND.walpole_coeffs(C, Val(8))
    _check_mirror_invariant(C, ℓ)
    sq2 = sqrt(oftype(ℓ[1], 2))
    return (; k = ℓ[2] / 2, l = ℓ[4] / sq2, m = ℓ[5] / 2, n = ℓ[1], p = ℓ[6] / 2, lt = ℓ[3] / sq2)
end

_check_mirror_invariant(C, ℓ) = _check_mirror_invariant(Val(is_hard_numeric(eltype(ℓ))), C, ℓ)
# The coefficients ℓ₇ and ℓ₈ of a symbolic tensor are those it is built with.
_check_mirror_invariant(::Val{false}, C, ℓ) = nothing
function _check_mirror_invariant(::Val{true}, C, ℓ)
    drel = max(abs(ℓ[7]), abs(ℓ[8])) / maximum(abs, ℓ)
    drel ≤ _TI_RTOL || _not_ti(C, drel)
    return nothing
end

_hill_moduli(C::TensND.AbstractTens{4, 3}, axis) = _hill_moduli(_ti_part(C, axis), axis)
_hill_moduli(C::TensND.AbstractTens{2, 3}, axis) = _wrong_order(C, "a stiffness")

"""
    _conductivities(K, axis) -> (; t, a)

Transverse and axial conductivities of a conductivity tensor that is isotropic,
or transversely isotropic about `axis`.
"""
function _conductivities(K::TensND.TensISO{2, 3}, axis)
    k = first(TensND.get_data(K))
    return (; t = k, a = k)
end

function _conductivities(K::TensND.TensTI{2, <:Any, 2}, axis)
    TensND.isparallel(TensND.axis(K), axis) || _not_coaxial(K, axis)
    t, a = TensND.get_data(K)
    return (; t, a)
end

_conductivities(K::TensND.AbstractTens{2, 3}, axis) = _conductivities(_ti_part(K, axis), axis)
_conductivities(K::TensND.AbstractTens{4, 3}, axis) = _wrong_order(K, "a conductivity")

_wrong_order(P, expected) = throw(
    ArgumentError(
        "a LayeredCylinder was given a property of order $(TensND.get_order(P)) where " *
            "$expected is expected: the layers, the reference and the remote loading must " *
            "belong to the same physics"
    )
)

_entries_eltype(nt::NamedTuple) = promote_type(map(typeof, values(nt))...)

# The moduli an elastic layer needs are finite, and its shear moduli and
# plane-strain bulk modulus positive: the recurrences divide by them. Checked
# where the numbers compare.
_check_stiffness(L, where_) = _check_stiffness(Val(is_hard_numeric(_entries_eltype(L))), L, where_)
_check_stiffness(::Val{false}, L, where_) = nothing
function _check_stiffness(::Val{true}, L, where_)
    all(isfinite, values(L)) && L.k > 0 && L.m > 0 && L.p > 0 || throw(
        ArgumentError(
            "$where_ of the LayeredCylinder has Hill moduli (k, l, m, n, p) = $((L.k, L.l, L.m, L.n, L.p)): " *
                "they must be finite, with k, m and p positive (an incompressible or a " *
                "void layer is not supported)"
        )
    )
    return nothing
end

# A layer may be impermeable (zero conductivity); the reference medium carries
# the remote gradient and may not.
_check_conductivity(K, where_; positive::Bool = false) =
    _check_conductivity(Val(is_hard_numeric(_entries_eltype(K))), K, where_, positive)
_check_conductivity(::Val{false}, K, where_, positive) = nothing
function _check_conductivity(::Val{true}, K, where_, positive)
    all(isfinite, values(K)) && (positive ? K.t > 0 && K.a > 0 : K.t ≥ 0 && K.a ≥ 0) || throw(
        ArgumentError(
            "$where_ of the LayeredCylinder has conductivities (transverse, axial) = " *
                "$(values(K)): they must be finite and $(positive ? "positive" : "nonnegative")"
        )
    )
    return nothing
end

"""
    _elastic_solution(cyl, C₀) -> NamedTuple

The four radial problems of the cylinder in the reference `C₀`, each under its
unit remote loading: `dilatation` and `axial` (axisymmetric), `shear`
(transverse), `antiplane`, with the layer moduli `Ls`, the reference moduli
`L₀`, the radii, the area fractions and the interfaces.
"""
function _elastic_solution(cyl::LayeredCylinder{T, N}, C₀::TensND.AbstractTens{4, 3}) where {T, N}
    MFH_Core._bump!(MFH_Core.LAYER_RECURRENCES)
    L₀ = _hill_moduli(C₀, cyl.axis)
    # `map` over the tuple of layers: an index would lose the type of each.
    Ls = map(C -> _hill_moduli(C, cyl.axis), cyl.moduli)
    _check_stiffness(L₀, "the reference medium")
    foreach(k -> _check_stiffness(Ls[k], "layer $k"), 1:N)
    S = promote_type(
        T, _entries_eltype(L₀), map(_entries_eltype, Ls)..., interfaces_eltype(cyl.interfaces)
    )
    return _elastic_solution(S, cyl, L₀, Ls)
end

# The entries of the moduli in the common type `S`: numbers, or the Volterra
# matrices of aging viscoelasticity.
_in_type(::Type{S}, x::Number) where {S} = S(x)
_in_type(::Type{S}, x::AbstractMatrix) where {S} = convert(Matrix{S}, x)
_in_type(::Type{S}, nt::NamedTuple) where {S} = map(x -> _in_type(S, x), nt)

# Behind a function barrier: `S` is a type computed from the inputs, which the
# conversions below must see as a static parameter to stay inferred. The same
# method solves the aging problem, the entries being Volterra matrices.
function _elastic_solution(::Type{S}, cyl::LayeredCylinder{T, N}, L₀, Ls) where {S, T, N}
    radii = map(x -> S(x), cyl.radii)
    L₀ = _in_type(S, L₀)
    Ls = map(L -> _in_type(S, L), Ls)
    itfs = cyl.interfaces
    dilatation, axial = _solve_axisymmetric(radii, Ls, itfs, L₀)
    shear = _solve_shear(radii, Ls, itfs, L₀)
    antiplane = _solve_potential(Antiplane(), radii, map(L -> L.p, Ls), itfs, L₀.p)
    fractions = ntuple(k -> S(layer_volume_fraction(cyl, k)), Val(N))
    return (; radii, Ls, L₀, fractions, interfaces = itfs, dilatation, axial, shear, antiplane)
end

"""
    _conduction_solution(cyl, K₀) -> NamedTuple

The transverse conduction problem of the cylinder in the reference `K₀`, under
a unit remote transverse gradient, with the layer conductivities `Ks` and the
reference ones `K₀`.
"""
function _conduction_solution(cyl::LayeredCylinder{T, N}, K₀::TensND.AbstractTens{2, 3}) where {T, N}
    MFH_Core._bump!(MFH_Core.LAYER_RECURRENCES)
    K₀e = _conductivities(K₀, cyl.axis)
    Ks = map(K -> _conductivities(K, cyl.axis), cyl.moduli)
    _check_conductivity(K₀e, "the reference medium"; positive = true)
    foreach(k -> _check_conductivity(Ks[k], "layer $k"), 1:N)
    S = promote_type(
        T, _entries_eltype(K₀e), map(_entries_eltype, Ks)..., interfaces_eltype(cyl.interfaces)
    )
    return _conduction_solution(S, cyl, K₀e, Ks)
end

function _conduction_solution(::Type{S}, cyl::LayeredCylinder{T, N}, K₀e, Ks) where {S, T, N}
    radii = map(x -> S(x), cyl.radii)
    K₀e = _in_type(S, K₀e)
    Ks = map(K -> _in_type(S, K), Ks)
    transverse = _solve_potential(
        TransverseConduction(), radii, map(K -> K.t, Ks), cyl.interfaces, K₀e.t
    )
    fractions = ntuple(k -> S(layer_volume_fraction(cyl, k)), Val(N))
    return (; radii, Ks, K₀ = K₀e, fractions, interfaces = cyl.interfaces, transverse)
end

# ── Packing ──────────────────────────────────────────────────────────────────

"""
    _ti_tensor(c, axis) -> TensTI{4}

Fourth-order transversely isotropic tensor of entries `(zz, ze, mz, me, T, L)`
about `axis`: Walpole coefficients ``\\ell_1 = zz``, ``\\ell_2 = me``,
``\\ell_3 = ze/\\sqrt 2``, ``\\ell_4 = \\sqrt 2\\,mz``, ``\\ell_5 = T``, ``\\ell_6 = L``.
"""
function _ti_tensor(c::NamedTuple{(:zz, :ze, :mz, :me, :T, :L)}, axis)
    sq2 = sqrt(oftype(c.me, 2))
    return TensND.TensTI{4}(c.zz, c.me, c.ze / sq2, sq2 * c.mz, c.T, c.L, axis)
end

_ti_tensor(c::NamedTuple{(:t, :a)}, axis) = TensND.TensTI{2}(c.t, c.a, axis)

_stiffness_tensor(L, axis) =
    TensND.ti_from_constants(L.k + L.m, L.k - L.m, L.l, L.n, L.p, axis)
