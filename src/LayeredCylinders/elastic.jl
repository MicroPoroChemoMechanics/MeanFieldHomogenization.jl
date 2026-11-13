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

_not_ti(P) = throw(
    ArgumentError(
        "a LayeredCylinder takes isotropic properties (TensISO) or transversely isotropic " *
            "ones about its axis (TensTI); got a $(nameof(typeof(P))). A general tensor that " *
            "has this symmetry can be converted with TensND's `best_fit_ti`"
    )
)

"""
    _hill_moduli(C, axis) -> (; k, l, m, n, p)

Hill's moduli of a stiffness that is isotropic, or transversely isotropic about
`axis`: plane-strain bulk modulus ``k = (C_{1111} + C_{1122})/2``,
``l = C_{1133}``, transverse shear modulus ``m = (C_{1111} - C_{1122})/2``,
``n = C_{3333}`` and axial shear modulus ``p = C_{2323}``.
"""
function _hill_moduli(C::TensND.TensISO{4, 3}, axis)
    κ, μ = TensND.k_mu(C)
    return (; k = κ + μ / 3, l = κ - 2 * μ / 3, m = μ, n = κ + 4 * μ / 3, p = μ)
end

function _hill_moduli(C::TensND.TensTI{4}, axis)
    TensND.isparallel(TensND.axis(C), axis) || _not_coaxial(C, axis)
    C₁₁₁₁, C₁₁₂₂, C₁₁₃₃, C₃₃₃₃, C₂₃₂₃ = TensND.ti_constants(C)
    return (; k = (C₁₁₁₁ + C₁₁₂₂) / 2, l = C₁₁₃₃, m = (C₁₁₁₁ - C₁₁₂₂) / 2, n = C₃₃₃₃, p = C₂₃₂₃)
end

_hill_moduli(C, axis) = _not_ti(C)

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

_conductivities(K, axis) = _not_ti(K)

_entries_eltype(nt::NamedTuple) = promote_type(map(typeof, values(nt))...)

# The moduli an elastic layer needs are finite, and its shear moduli and
# plane-strain bulk modulus positive: the recurrences divide by them. Checked
# where the numbers compare.
_check_stiffness(L, where_) = _check_stiffness(Val(is_hard_numeric(_entries_eltype(L))), L, where_)
_check_stiffness(::Val{false}, L, where_) = nothing
function _check_stiffness(::Val{true}, L, where_)
    all(isfinite, values(L)) && L.k > 0 && L.m > 0 && L.p > 0 || throw(
        ArgumentError(
            "$where_ of the LayeredCylinder has Hill moduli (k, l, m, n, p) = $(values(L)): " *
                "they must be finite, with k, m and p positive (an incompressible or a " *
                "void layer is not supported)"
        )
    )
    return nothing
end

_check_conductivity(K, where_) = _check_conductivity(Val(is_hard_numeric(_entries_eltype(K))), K, where_)
_check_conductivity(::Val{false}, K, where_) = nothing
function _check_conductivity(::Val{true}, K, where_)
    all(isfinite, values(K)) && K.t ≥ 0 && K.a ≥ 0 || throw(
        ArgumentError(
            "$where_ of the LayeredCylinder has conductivities (transverse, axial) = " *
                "$(values(K)): they must be finite and nonnegative"
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
    Ls = ntuple(k -> _hill_moduli(layer_modulus(cyl, k), cyl.axis), Val(N))
    _check_stiffness(L₀, "the reference medium")
    foreach(k -> _check_stiffness(Ls[k], "layer $k"), 1:N)
    S = promote_type(
        T, _entries_eltype(L₀), map(_entries_eltype, Ls)..., interfaces_eltype(cyl.interfaces)
    )
    radii = map(S, cyl.radii)
    L₀ = map(S, L₀)
    Ls = map(L -> map(S, L), Ls)
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
    Ks = ntuple(k -> _conductivities(layer_modulus(cyl, k), cyl.axis), Val(N))
    _check_conductivity(K₀e, "the reference medium")
    foreach(k -> _check_conductivity(Ks[k], "layer $k"), 1:N)
    S = promote_type(
        T, _entries_eltype(K₀e), map(_entries_eltype, Ks)..., interfaces_eltype(cyl.interfaces)
    )
    radii = map(S, cyl.radii)
    K₀e = map(S, K₀e)
    Ks = map(K -> map(S, K), Ks)
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
