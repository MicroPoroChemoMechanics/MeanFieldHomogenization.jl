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
