# =============================================================================
#  block_solve.jl — linear systems whose unknowns are tensors.
#
#  Both N-body schemes end on a system of the form
#
#       Σ_K  𝕄_{IK} : X_K = B_I           (I = 1 … N)
#
#  where the unknowns X_K are 4th-order tensors (elasticity) or 2nd-order
#  tensors (conduction).  Rather than reimplementing the tensorial Gauss
#  elimination that Molinari & El Mouden describe, the system is flattened
#  onto the Kelvin-Mandel basis — where the double contraction `:` becomes an
#  ordinary matrix product — assembled as one dense matrix, and handed to `\`.
#
#  Kelvin-Mandel is the right basis precisely because it is orthonormal: the
#  contraction of two minor-symmetric tensors equals the matrix product of
#  their Mandel matrices with no metric factor to carry around, so a formula
#  transcribed from a paper survives the flattening unchanged.
#
#  Sizes: 6 Mandel components in 3D elasticity, 3 in 2D elasticity, and d in
#  conduction — so the assembled matrix is (N·m)² with m ≤ 6.  For the
#  assemblies these schemes target (tens to a few hundred particles) that is a
#  dense solve of at most a few thousand unknowns: entirely unremarkable, and
#  far cheaper than the O(N²) interaction assembly that precedes it.
# =============================================================================

"""
    _mandel_size(P₀) -> Int

Number of Kelvin-Mandel components carried by a tensor of the same order and
dimension as the reference property `P₀`.
"""
_mandel_size(::TensND.AbstractTens{4, 3}) = 6
_mandel_size(::TensND.AbstractTens{4, 2}) = 3
_mandel_size(::TensND.AbstractTens{2, 3}) = 3
_mandel_size(::TensND.AbstractTens{2, 2}) = 2

# ─── Flattening ──────────────────────────────────────────────────────────────
#
# 4th order: TensND's `mandel(·; symmetrize = true)` / `from_mandel` pair, in 3D
# and in 2D alike.
# 2nd order: the identity map — a symmetric 2nd-order tensor is already its own
# Mandel vector in the sense used by `⋅`.
#
# Canonical components throughout: the blocks of one system come in different
# bases (a Hill tensor in its particle's, a reference in the canonical one) and
# `_from_mandel` rebuilds canonical tensors. `get_array` read each block in its
# own basis, which mixed frames in one matrix for every oriented particle.
# A block already canonical is read as it is.

_to_mandel(A, ::TensND.AbstractTens{4}) = TensND.mandel(TensND.canonical_components(A); symmetrize = true)
_to_mandel(A, ::TensND.AbstractTens{2}) = Matrix(TensND.canonical_components(A))

_from_mandel(M, ::TensND.AbstractTens{4}) = TensND.from_mandel(M)
_from_mandel(M, ::TensND.AbstractTens{2}) = TensND.Tens(M)

"""
    _solve_tensor_system(blocks, rhs, P₀) -> Vector

Solve ``\\sum_K \\mathbb{M}_{IK}:\\mathbb{X}_K = \\mathbb{B}_I`` for the tensor unknowns ``\\mathbb{X}_K``.

`blocks[i][k]` holds ``\\mathbb{M}_{IK}`` and `rhs[i]` holds ``\\mathbb{B}_I``, both as TensND tensors
of the same order as the reference `P₀`. The result is returned in the same
representation.

The whole system is flattened onto the Kelvin-Mandel basis, solved with a
single dense factorization, and unflattened. The element type is promoted
across every block, so a `ForwardDiff.Dual` anywhere in the assembly carries
through the solve — which is what makes a sensitivity to a particle position or
radius work.
"""
function _solve_tensor_system(blocks, rhs, P₀::TensND.AbstractTens)
    n = length(rhs)
    m = _mandel_size(P₀)
    Mb = [[_to_mandel(blocks[i][k], P₀) for k in 1:n] for i in 1:n]
    Rb = [_to_mandel(rhs[i], P₀) for i in 1:n]
    # `hcat` and `vcat` promote across the blocks they join: the blocks live in an
    # untyped vector (their TensND symmetry classes differ) and under AD mix
    # `Dual` blocks with `Float64` ones, so a `Dual` anywhere in the assembly
    # carries through the solve.
    A = reduce(vcat, [reduce(hcat, row) for row in Mb])
    X = A \ reduce(vcat, Rb)
    return [_from_mandel(X[rows, :], P₀) for rows in Iterators.partition(axes(X, 1), m)]
end
