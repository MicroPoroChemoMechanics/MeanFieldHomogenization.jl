# =============================================================================
#  kinematics.jl — what a law receives: the small-strain tensor, in Kelvin–Mandel
#  components.
#
#  A law is evaluated millions of times in a structural computation, inside a
#  local Newton loop and under automatic differentiation. Its strain and stress
#  are therefore plain static vectors of six numbers, not TensND tensors: the
#  Kelvin–Mandel basis is orthonormal, so the double contraction σ:ε is the dot
#  product of the two vectors, the norm of a tensor is the norm of its vector,
#  and a fourth-order tensor with the minor symmetries is a 6×6 matrix whose
#  products are the tensor products. Nothing in a law has to know about
#  factors of 2 on the shears.
#
#  The conversion to and from TensND tensors happens once, at the boundary
#  (`material_response`), through the conversions of Tensors.jl.
# =============================================================================

"""
    SmallStrain()

The kinematics of a law written in infinitesimal strain: the law receives the
symmetric strain tensor ``\\boldsymbol{\\varepsilon}`` and returns the Cauchy
stress ``\\boldsymbol{\\sigma}``. It is the only kinematics so far; the trait
[`kinematics`](@ref) exists so that a finite-strain one can be added without
changing the laws written for this one.
"""
struct SmallStrain end

# The Kelvin–Mandel vectors the laws are written on, and their operations
# (`Mandel`, `identity_mandel`, `trace_mandel`, `deviator`, `sqnorm_deviator`,
# `von_mises`, `equivalent_strain`), are TensND's: imported in MaterialLaws.jl
# and exported from there with the rest of the module.

# ── The boundary with TensND ─────────────────────────────────────────────────
# Components in the canonical basis (`to_tensors`), whatever basis the tensor is
# written in; then Tensors.jl's Mandel ordering, which is that of `Mandel`.

_mandel(t::TensND.AbstractTens{2, 3}) = Tensors.tomandel(SArray, to_tensors(t))
_mandel(t::TensND.AbstractTens{4, 3}) = Tensors.tomandel(SArray, to_tensors(t))

_tensor(v::StaticVector{6}) = TensND.from_mandel(v)
_tensor(A::StaticMatrix{6, 6}) = TensND.from_mandel(A)

# Symmetric N×N matrices by their N(N+1)/2 upper components, row by row: the
# unknowns of the self-consistent fixed points on a compliance or a stiffness.
@inline _upper_index(i, j, N) = (a = min(i, j); b = max(i, j); (a - 1) * N - ((a - 1) * (a - 2)) ÷ 2 + (b - a + 1))
_sym_matrix(v::SVector{K, T}, ::Val{N}) where {K, T, N} =
    SMatrix{N, N, T}(ntuple(q -> v[_upper_index((q - 1) % N + 1, (q - 1) ÷ N + 1, N)], Val(N * N)))
@generated function _sym_components(M::StaticMatrix{N, N, T}) where {N, T}
    entries = [:((M[$i, $j] + M[$j, $i]) / 2) for i in 1:N for j in i:N]
    return :(SVector{$(length(entries))}($(entries...)))
end
