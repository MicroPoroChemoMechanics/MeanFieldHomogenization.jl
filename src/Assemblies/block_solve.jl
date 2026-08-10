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
# 4th order: the package-wide `mandel66_minor` / `array_from_mandel66` pair in
# 3D, and its 2D counterpart written out here (three components, weights √2).
# 2nd order: the identity map — a symmetric 2nd-order tensor is already its own
# Mandel vector in the sense used by `⋅`.

const _MANDEL_IDX_2D = ((1, 1), (2, 2), (1, 2))

_to_mandel(A, ::TensND.AbstractTens{4, 3}) = MFH_Core.mandel66_minor(TensND.get_array(A))
_to_mandel(A, ::TensND.AbstractTens{2}) = Matrix(TensND.get_array(A))

function _to_mandel(A, ::TensND.AbstractTens{4, 2})
    arr = TensND.get_array(A)
    T = eltype(arr)
    sq2 = sqrt(T(2))
    Λ(I) = I ≤ 2 ? one(T) : sq2
    M = Matrix{T}(undef, 3, 3)
    for I in 1:3, J in 1:3
        (i, j) = _MANDEL_IDX_2D[I]
        (k, l) = _MANDEL_IDX_2D[J]
        v = (arr[i, j, k, l] + arr[j, i, k, l] + arr[i, j, l, k] + arr[j, i, l, k]) / 4
        M[I, J] = Λ(I) * Λ(J) * v
    end
    return M
end

_from_mandel(M, ::TensND.AbstractTens{4, 3}) = TensND.Tens(MFH_Core.array_from_mandel66(M))
_from_mandel(M, ::TensND.AbstractTens{2}) = TensND.Tens(M)

function _from_mandel(M, ::TensND.AbstractTens{4, 2})
    T = eltype(M)
    sq2 = sqrt(T(2))
    Λ(I) = I ≤ 2 ? one(T) : sq2
    arr = Array{T, 4}(undef, 2, 2, 2, 2)
    for I in 1:3, J in 1:3
        (i, j) = _MANDEL_IDX_2D[I]
        (k, l) = _MANDEL_IDX_2D[J]
        v = M[I, J] / (Λ(I) * Λ(J))
        arr[i, j, k, l] = v
        arr[j, i, k, l] = v
        arr[i, j, l, k] = v
        arr[j, i, l, k] = v
    end
    return TensND.Tens(arr)
end

"""
    _solve_tensor_system(blocks, rhs, P₀) -> Vector

Solve `Σ_K 𝕄_{IK} : X_K = B_I` for the tensor unknowns `X_K`.

`blocks[i][k]` holds `𝕄_{IK}` and `rhs[i]` holds `B_I`, both as TensND tensors
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
    # The element type is read off the assembled matrices themselves, not from
    # `eltype` of their container: the blocks live in an untyped vector (their
    # TensND symmetry classes differ), and under AD they mix `Dual` blocks with
    # `Float64` ones, so container inference lands on `Any` and `zeros(Any, …)`
    # then fails. Scanning the actual matrices is what keeps a `Dual` anywhere
    # in the assembly from being silently lost.
    T = promote_type(
        mapreduce(eltype, promote_type, Iterators.flatten(Mb)),
        mapreduce(eltype, promote_type, Rb)
    )
    A = zeros(T, n * m, n * m)
    B = zeros(T, n * m, m)
    for i in 1:n
        rows = ((i - 1) * m + 1):(i * m)
        B[rows, :] .= Rb[i]
        for k in 1:n
            A[rows, ((k - 1) * m + 1):(k * m)] .= Mb[i][k]
        end
    end
    X = A \ B
    return [_from_mandel(X[((i - 1) * m + 1):(i * m), :], P₀) for i in 1:n]
end
