# =============================================================================
#  fracture_sets.jl — a background medium crossed by sets of parallel
#  fractures declared by their compliances of linear slip.
#
#  Each set of normal n adds the change sym(n ⊗ Z ⊗ n) to the compliance of the
#  background (Schoenberg and Sayers 1995), Z being the compliance of the set
#  per unit length normal to it: a fracture of compliance 𝕂 every length L
#  gives Z = 𝕂/L. The changes add up,
#
#       𝕊 = 𝕊_b + Σ_s sym(n_s ⊗ Z_s ⊗ n_s) ,
#
#  which for one set is exactly the periodic laminate of one layer of the
#  background and its spring interface; the sets do not interact.
# =============================================================================

"""
    linear_slip_change(n, Z) → Δ𝕊

The change of compliance of a set of parallel fractures of normal `n` and
compliance `Z` (a symmetric second-order tensor, a 3×3 matrix of canonical
components whose symmetric part is taken,
or the pair `(Z_N, Z_T)` of a set invariant by rotation about its normal,
``\\boldsymbol{Z} = Z_N\\,\\underline{n}\\otimes\\underline{n} + Z_T\\,(\\boldsymbol{1} - \\underline{n}\\otimes\\underline{n})``):
``\\Delta\\mathbb{S} = \\operatorname{sym}(\\underline{n}\\otimes\\boldsymbol{Z}\\otimes\\underline{n})``,
minor-symmetrized. The converse of [`slip_compliance`](@ref).
"""
linear_slip_change(n::TensND.AbstractTens{1, 3}, Z::TensND.AbstractTens{2, 3}) =
    _slip_change(_vec(n), _sym2(canonical_components(Z)))
linear_slip_change(n::TensND.AbstractTens{1, 3}, Z::AbstractMatrix) = _slip_change(_vec(n), _sym2(Z))
function linear_slip_change(n::TensND.AbstractTens{1, 3}, (Z_N, Z_T)::Tuple{Number, Number})
    v = _vec(n)
    N = otimes(v)
    return _slip_change(v, Z_N * N + Z_T * (one(N) - N))
end
linear_slip_change(n::AbstractVector, Z) = linear_slip_change(Tens(_unit(n)), Z)

# n_i Z_jk n_l = (n ⊗ n ⊠̄ Z)_ijkl, `otimesl` of Tensors.jl, minor-symmetrized:
# the kernels of statically typed tensors keep the result inferable.
_slip_change(n::Vec{3}, Z::SymmetricTensor{2, 3}) = Tens(minorsymmetric(otimesl(otimes(n), Z)))
_vec(n::TensND.AbstractTens{1, 3}) = (c = canonical_components(n); Vec{3}((c[1], c[2], c[3])))
# The symmetric part entry by entry: the constructor from a matrix compares its
# entries, which symbols do not allow.
_sym2(Z) = SymmetricTensor{2, 3}((i, j) -> (Z[i, j] + Z[j, i]) / 2)
_unit(n::AbstractVector) = SVector{3}(n) / sqrt(transpose(n) * n)

"""
    fractured_stiffness(ℂ_b, sets...) → ℂ

The stiffness of the background `ℂ_b` crossed by sets of parallel fractures,
each a pair `n => Z` of a normal and a compliance of linear slip per unit
length ([`linear_slip_change`](@ref) for the forms of `Z`):

```math
\\mathbb{S} = \\mathbb{C}_b^{-1} + \\sum_s \\operatorname{sym}(\\underline{n}_s\\otimes\\boldsymbol{Z}_s\\otimes\\underline{n}_s)
```

[schoenberg1995](@cite). One set is exactly the periodic laminate of the
background with a spring interface of compliance ``L\\,\\boldsymbol{Z}`` every
length ``L``. Numbers, dual numbers and symbols.
"""
fractured_stiffness(C_b::TensND.AbstractTens{4, 3}, sets::Pair...) =
    inv(inv(C_b) + mapreduce(((n, Z),) -> linear_slip_change(n, Z), +, sets))

"""
    orthogonal_fracture_compliances(Δ𝕊, frame) → (normals, Z_N, Z_T)

The compliances of linear slip of three sets of fractures normal to the vectors
of the orthonormal `frame` (the columns of a 3×3 matrix, or a basis of TensND),
each invariant by rotation about its normal, read on a change of compliance
`Δ𝕊`. A set of normal ``\\underline{e}_i`` changes ``\\Delta S_{iiii}`` by
``Z_N^i`` and ``4\\,\\Delta S_{ijij}`` by ``Z_T^i`` for ``j \\neq i``
([`linear_slip_change`](@ref)), and no other component of the frame, so that

```math
Z_N^i = \\Delta S_{iiii},
\\qquad
Z_T^i = 2\\,(\\Delta S_{ijij} + \\Delta S_{ikik} - \\Delta S_{jkjk}),
```

``(i, j, k)`` a permutation of ``(1, 2, 3)``, the components being those of the
frame: the six moduli that best represent `Δ𝕊` by three such sets, the other
components (couplings between normal compliances, components outside the
orthotropic pattern of the frame) being left out. For three sets,
`fractured_stiffness(ℂ_b, (eᵢ => (Z_N[i], Z_T[i]) for i in 1:3)...)` is the
converse. The method of [rasolofosaonGJI2000](@citet), eq. (3), who find the frame
as the best orthotropic one of the cracked medium
([`project`](@extref TensND) with `:ORTHO` in TensND).
"""
function orthogonal_fracture_compliances(ΔS::TensND.AbstractTens{4, 3}, frame::AbstractMatrix)
    D = _sym4(canonical_components(ΔS))
    e = ntuple(i -> Vec{3}((frame[1, i], frame[2, i], frame[3, i])), Val(3))
    q(a, b) = (P = otimes(a, b); dcontract(P, dcontract(D, P)))     # ΔS_abab in the frame
    s = (q(e[2], e[3]), q(e[1], e[3]), q(e[1], e[2]))               # s[k] = ΔS_ijij, {i, j, k} = {1, 2, 3}
    Z_N = ntuple(i -> q(e[i], e[i]), Val(3))
    Z_T = ntuple(i -> 2 * (sum(s) - 2 * s[i]), Val(3))
    return (normals = map(Tens, e), Z_N = Z_N, Z_T = Z_T)
end
orthogonal_fracture_compliances(ΔS::TensND.AbstractTens{4, 3}, frame::TensND.AbstractBasis{3}) =
    orthogonal_fracture_compliances(ΔS, TensND.vecbasis(frame, :cov))

_sym4(A) = SymmetricTensor{4, 3}((i, j, k, l) -> A[i, j, k, l])
