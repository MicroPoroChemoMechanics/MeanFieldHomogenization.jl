# =============================================================================
#  ti_schemes_alv.jl — TI-symmetry fast paths for ALV schemes.
#
#  When **every** ALV block matrix in a homogenization pipeline is a
#  TI 4-tensor with the **same canonical axis** `n = e₃` (each `(6×6)`
#  Mandel block is parametrized by the six Walpole coefficients
#  `(ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆)`), the algebra collapses to:
#
#    * a `(2n)×(2n)` block-Volterra problem on the synthetic 2×2 matrix
#         `[[ℓ₁, ℓ₃]; [ℓ₄, ℓ₂]]`
#    * two `n×n` scalar Volterra problems on `ℓ₅` and `ℓ₆` (in-plane and
#      out-of-plane shears).
#
#  Storage:    6 · n²  entries vs  36 · n²  for the full `(6n×6n)` form
#  Inverse:   ~3× cheaper than generic `(6n×6n)` block forward-substitution
#             (block_size = 2 forward sub + 2 scalar inverses)
#  Product:   8 n×n GEMMs + 2 n×n GEMMs vs the full 6n×6n GEMM
#
#  Convention: the synthetic 2×2 Walpole matrix is `[[ℓ₁, ℓ₃]; [ℓ₄, ℓ₂]]`
#  (matching `TensND.TensTI`).  Major-symmetric input (ℓ₃ ≡ ℓ₄, e.g. an
#  elastic stiffness or relaxation tensor) is preserved as a 6-tuple
#  with `ℓ₃` and `ℓ₄` carrying equal data; the algebra closure does not
#  require restricting to the major-symmetric subspace.
# =============================================================================

# ── TI 6-tuple primitives ───────────────────────────────────────────────────

"""
    _ti_identity(n, T) -> NTuple{6, Matrix{T}}

TI form of the ``6n\\times 6n`` block-diagonal identity matrix:
``(\\ell_1, \\ell_2, \\ell_3, \\ell_4, \\ell_5, \\ell_6) = (\\mathbb 1, \\mathbb 1, 0, 0, \\mathbb 1, \\mathbb 1)`` (because
``\\boldsymbol{1}\\stackrel{s}{\\boxtimes}\\boldsymbol{1} = \\mathbb{W}_1 + \\mathbb{W}_2 + \\mathbb{W}_5 + \\mathbb{W}_6`` for the full 4-tensor identity).
"""
# Promote `T` with the eltype of every TI 6-tuple in `list` (Dual-safe :
# phases may carry a wider eltype than the matrix, e.g. geometry Duals).
@inline function _ti_list_eltype(T::Type, list::AbstractVector)
    for a in list
        T = promote_type(T, eltype(a[1]))
    end
    return T
end

@inline function _ti_identity(n::Int, T::Type)
    Iₙ = Matrix{T}(LinearAlgebra.I, n, n)
    Zₙ = zeros(T, n, n)
    return (Iₙ, copy(Iₙ), Zₙ, copy(Zₙ), copy(Iₙ), copy(Iₙ))
end

"""
    _ti_add!(acc, c, a) -> acc

In-place TI scalar AXPY: `acc[k] .+= c · a[k]` for ``k = 1, \\dots, 6``.
"""
@inline function _ti_add!(acc::NTuple{6, <:Matrix}, c::Real, a::NTuple{6, <:Matrix})
    @inbounds for k in 1:6
        @. acc[k] += c * a[k]
    end
    return acc
end

"""
    _ti_prod(a, b) -> NTuple{6, Matrix}

TI Volterra product ``M_a\\circ M_b``.  In Walpole synthetic notation:

```math
\\begin{bmatrix} c_1 & c_3 \\\\ c_4 & c_2 \\end{bmatrix}
= \\begin{bmatrix} a_1 & a_3 \\\\ a_4 & a_2 \\end{bmatrix}
  \\begin{bmatrix} b_1 & b_3 \\\\ b_4 & b_2 \\end{bmatrix},
\\qquad
c_5 = a_5\\circ b_5,
\\qquad
c_6 = a_6\\circ b_6,
```

with ``c_5`` the in-plane shear and ``c_6`` the out-of-plane shear (scalars).
Each entry of the ``2\\times 2`` product expands to two ``n\\times n`` Volterra products.
"""
@inline function _ti_prod(a::NTuple{6, <:Matrix}, b::NTuple{6, <:Matrix})
    c1 = a[1] * b[1] + a[3] * b[4]
    c2 = a[4] * b[3] + a[2] * b[2]
    c3 = a[1] * b[3] + a[3] * b[2]
    c4 = a[4] * b[1] + a[2] * b[4]
    c5 = a[5] * b[5]
    c6 = a[6] * b[6]
    return (c1, c2, c3, c4, c5, c6)
end

"""
    _ti_prod!(out, a, b) -> out

In-place TI Volterra product : 8 `mul!` calls for the ``2\\times 2`` normal
block + 2 `mul!` for the shears.  No allocation if `out`'s entries
have the right size.
"""
@inline function _ti_prod!(
        out::NTuple{6, <:Matrix},
        a::NTuple{6, <:Matrix},
        b::NTuple{6, <:Matrix}
    )
    T = eltype(out[1])
    one_T = one(T)
    # c1 = a[1] * b[1] + a[3] * b[4]
    mul!(out[1], a[1], b[1])
    mul!(out[1], a[3], b[4], one_T, one_T)
    # c2 = a[4] * b[3] + a[2] * b[2]
    mul!(out[2], a[4], b[3])
    mul!(out[2], a[2], b[2], one_T, one_T)
    # c3 = a[1] * b[3] + a[3] * b[2]
    mul!(out[3], a[1], b[3])
    mul!(out[3], a[3], b[2], one_T, one_T)
    # c4 = a[4] * b[1] + a[2] * b[4]
    mul!(out[4], a[4], b[1])
    mul!(out[4], a[2], b[4], one_T, one_T)
    # c5 = a[5] * b[5], c6 = a[6] * b[6]
    mul!(out[5], a[5], b[5])
    mul!(out[6], a[6], b[6])
    return out
end

# ── Pack / unpack the Walpole 2×2 part as a (2n)×(2n) block-Volterra ──────
#
# Time-interleaved layout: block (i, j) of size 2×2 in the packed matrix
# corresponds to `[[ℓ₁[i,j], ℓ₃[i,j]]; [ℓ₄[i,j], ℓ₂[i,j]]]`.  Because
# every ℓᵢ is lower-triangular in (i, j) (Volterra causality), the
# packed matrix is block-lower-triangular with `block_size = 2`, ready
# for `volterra_inverse(_; block_size = 2)` and
# `volterra_left_divide(_, _; block_size = 2)`.

@inline function _ti_pack_walpole(a::NTuple{6, <:Matrix})
    n = size(a[1], 1)
    T = promote_type(eltype(a[1]), eltype(a[2]), eltype(a[3]), eltype(a[4]))
    M = zeros(T, 2 * n, 2 * n)
    @inbounds for i in 1:n, j in 1:n
        M[2i - 1, 2j - 1] = a[1][i, j]   # ℓ₁
        M[2i - 1, 2j] = a[3][i, j]   # ℓ₃
        M[2i, 2j - 1] = a[4][i, j]   # ℓ₄
        M[2i, 2j] = a[2][i, j]   # ℓ₂
    end
    return M
end

@inline function _ti_unpack_walpole(M::AbstractMatrix, n::Int)
    T = eltype(M)
    ℓ₁ = zeros(T, n, n); ℓ₂ = zeros(T, n, n)
    ℓ₃ = zeros(T, n, n); ℓ₄ = zeros(T, n, n)
    @inbounds for i in 1:n, j in 1:n
        ℓ₁[i, j] = M[2i - 1, 2j - 1]
        ℓ₃[i, j] = M[2i - 1, 2j]
        ℓ₄[i, j] = M[2i, 2j - 1]
        ℓ₂[i, j] = M[2i, 2j]
    end
    return ℓ₁, ℓ₂, ℓ₃, ℓ₄
end

"""
    _ti_inv(a) -> NTuple{6, Matrix}

TI Volterra inverse.  Inverts the synthetic Walpole ``2\\times 2`` part via a
``2n\\times 2n`` block-Volterra inverse and the two scalar shears
independently.
"""
function _ti_inv(a::NTuple{6, <:Matrix})
    n = size(a[1], 1)
    M_pack = _ti_pack_walpole(a)
    M_inv = volterra_inverse(M_pack; block_size = 2)
    ℓ₁, ℓ₂, ℓ₃, ℓ₄ = _ti_unpack_walpole(M_inv, n)
    ℓ₅ = volterra_inverse(a[5]; block_size = 1)
    ℓ₆ = volterra_inverse(a[6]; block_size = 1)
    return (ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆)
end

"""
    _ti_left_divide(S, M) -> NTuple{6, Matrix}

TI form of the Volterra left-divide ``T = S^{-\\circ}\\circ M``.  Solves the
Walpole ``2\\times 2`` part as a ``2n\\times 2n`` block-Volterra system and the two
scalar shears independently.
"""
function _ti_left_divide(S::NTuple{6, <:Matrix}, M::NTuple{6, <:Matrix})
    n = size(S[1], 1)
    S_pack = _ti_pack_walpole(S)
    M_pack = _ti_pack_walpole(M)
    T_pack = volterra_left_divide(S_pack, M_pack; block_size = 2)
    ℓ₁, ℓ₂, ℓ₃, ℓ₄ = _ti_unpack_walpole(T_pack, n)
    ℓ₅ = volterra_left_divide(S[5], M[5]; block_size = 1)
    ℓ₆ = volterra_left_divide(S[6], M[6]; block_size = 1)
    return (ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆)
end

# ── TI form detection and conversion ────────────────────────────────────────

"""
    _is_ti_block(M; axis = (0, 0, 1), tol = 1e-12) -> Bool

Heuristic: return `true` if the ``6n\\times 6n`` block matrix `M` is in TI
form (each ``6\\times 6`` block is a TI 4-tensor with the canonical axis ``\\underline{e}_3``).

Allocation-free: extracts the 6 Walpole coefficients from the canonical
entries of each block on the fly and checks every entry against the TI
Mandel pattern, without building intermediate parameter matrices nor a
reconstructed ``6n\\times 6n`` matrix for comparison.
"""
function _is_ti_block(
        M::AbstractMatrix;
        axis::NTuple{3} = (0.0, 0.0, 1.0),
        tol::Real = 1.0e-12
    )
    sz = size(M, 1)
    sz == size(M, 2) || return false
    sz % 6 == 0 || return false
    n = sz ÷ 6
    iszero(n) && return true
    axis == (0.0, 0.0, 1.0) || return false
    scale = max(maximum(abs, M), one(real(eltype(M))))
    abstol = tol * scale
    s2 = sqrt(2)
    @inbounds for i in 1:n, j in 1:n
        r = 6 * (i - 1)
        c = 6 * (j - 1)
        # Extract the 6 Walpole coefficients from canonical block entries.
        ℓ₁ = M[r + 3, c + 3]
        ℓ₂ = M[r + 1, c + 1] + M[r + 1, c + 2]
        ℓ₅ = M[r + 1, c + 1] - M[r + 1, c + 2]
        ℓ₃ = s2 * M[r + 3, c + 1]
        ℓ₄ = s2 * M[r + 1, c + 3]
        ℓ₆ = M[r + 4, c + 4]
        block_diag_in = (ℓ₂ + ℓ₅) / 2
        block_off_in = (ℓ₂ - ℓ₅) / 2
        ℓ₃_o_s2 = ℓ₃ / s2
        ℓ₄_o_s2 = ℓ₄ / s2
        # Walk all 36 Mandel entries and compare.
        for k in 1:6, l in 1:6
            expected = if k ≤ 2 && l ≤ 2
                k == l ? block_diag_in : block_off_in
            elseif k == 3 && l == 3
                ℓ₁
            elseif k == 3 && l ≤ 2
                ℓ₃_o_s2
            elseif k ≤ 2 && l == 3
                ℓ₄_o_s2
            elseif (k == 4 && l == 4) || (k == 5 && l == 5)
                ℓ₆
            elseif k == 6 && l == 6
                ℓ₅
            else
                zero(eltype(M))
            end
            abs(M[r + k, c + l] - expected) ≤ abstol || return false
        end
    end
    return true
end

"""
    _ti_pair(M; axis) -> NTuple{6, Matrix}

Extract the six TI Walpole parameter matrices from a ``6n\\times 6n`` block
matrix.  Wrapper around [`ti_params_from_blocks`](@ref).
"""
@inline _ti_pair(M::AbstractMatrix; axis::NTuple{3} = (0.0, 0.0, 1.0)) =
    ti_params_from_blocks(M; axis = axis)

"""
    _ti_blocks(ℓ; axis) -> Matrix

Reassemble a ``6n\\times 6n`` TI block matrix from a 6-tuple of Walpole
Volterra matrices.  Wrapper around [`ti_blocks_from_params`](@ref).
"""
@inline _ti_blocks(
    ℓ::NTuple{6, <:AbstractMatrix};
    axis::NTuple{3} = (0.0, 0.0, 1.0)
) =
    ti_blocks_from_params(ℓ; axis = axis)

"""
    _iso_to_ti(αβ::Tuple) -> NTuple{6, Matrix}

Convert an iso ``(\\alpha, \\beta)`` pair into TI Walpole 6-tuple in the canonical
axis ``\\underline{e}_3``.  Iso ``= \\alpha\\,\\mathbb{J} + \\beta\\,\\mathbb{K}``; in Walpole basis with ``\\underline{n} = \\underline{e}_3``:

```math
\\mathbb{J} = \\tfrac{1}{3}\\bigl(\\mathbb{W}_1 + 2\\,\\mathbb{W}_2 + \\sqrt{2}\\,\\mathbb{W}_3 + \\sqrt{2}\\,\\mathbb{W}_4\\bigr),
\\qquad
\\mathbb{K} = \\boldsymbol{1}\\stackrel{s}{\\boxtimes}\\boldsymbol{1} - \\mathbb{J}.
```

Hence:

```math
\\ell_1 = \\tfrac{\\alpha}{3} + \\tfrac{2\\beta}{3}, \\quad
\\ell_2 = \\tfrac{2\\alpha}{3} + \\tfrac{\\beta}{3}, \\quad
\\ell_3 = \\ell_4 = \\tfrac{\\sqrt{2}}{3}\\,(\\alpha - \\beta), \\quad
\\ell_5 = \\beta, \\quad
\\ell_6 = \\beta.
```
"""
function _iso_to_ti(αβ::Tuple)
    α, β = αβ
    s2 = sqrt(2)
    ℓ₁ = α ./ 3 .+ (2 // 3) .* β
    ℓ₂ = (2 // 3) .* α .+ β ./ 3
    ℓ_off = ((α .- β) .* (s2 / 3))
    ℓ₃ = copy(ℓ_off)
    ℓ₄ = copy(ℓ_off)
    ℓ₅ = copy(β)
    ℓ₆ = copy(β)
    return (ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆)
end

# ── TI scheme implementations ───────────────────────────────────────────────

"""
    voigt_alv_ti(ℓ_phases, fractions) -> NTuple{6, Matrix}

TI-form Voigt bound: ``\\ell_k^{\\mathrm{hom}} = \\sum_i f_i\\,\\ell_{k,i}`` for each Walpole component ``k``.
"""
function voigt_alv_ti(ℓ_phases::AbstractVector, fractions::AbstractVector)
    length(ℓ_phases) == length(fractions) ||
        throw(ArgumentError("voigt_alv_ti: phase counts mismatch"))
    isempty(ℓ_phases) && throw(ArgumentError("voigt_alv_ti: at least one phase required"))
    n = size(ℓ_phases[1][1], 1)
    T = _ti_list_eltype(eltype(fractions), ℓ_phases)
    eff = ntuple(_ -> zeros(T, n, n), 6)
    @inbounds for r in eachindex(ℓ_phases)
        _ti_add!(eff, fractions[r], ℓ_phases[r])
    end
    return eff
end

"""
    reuss_alv_ti(ℓ_phases, fractions) -> NTuple{6, Matrix}

TI-form Reuss bound: invert per-phase compliances in TI Walpole form,
average, invert back.
"""
function reuss_alv_ti(ℓ_phases::AbstractVector, fractions::AbstractVector)
    length(ℓ_phases) == length(fractions) ||
        throw(ArgumentError("reuss_alv_ti: phase counts mismatch"))
    ℓ_inv_phases = [_ti_inv(ℓ) for ℓ in ℓ_phases]
    ℓ_inv_eff = voigt_alv_ti(ℓ_inv_phases, fractions)
    return _ti_inv(ℓ_inv_eff)
end

"""
    dilute_concentration_alv_ti(ℓ_E, ℓ_0, ℓ_P) -> NTuple{6, Matrix}

TI-form dilute concentration
``\\widetilde{\\mathbb{A}}^{\\mathrm{dil}} = (H\\,\\mathbb{I} + \\widetilde{\\mathbb{P}}\\circ\\Delta\\widetilde{\\mathbb{C}})^{-\\circ}``,
``\\Delta\\widetilde{\\mathbb{C}} = \\widetilde{\\mathbb{C}}^{\\mathcal{E}} - \\widetilde{\\mathbb{C}}_0``, reduced
to a ``2n\\times 2n`` block-Volterra inverse + two ``n\\times n`` scalar inverses.
"""
function dilute_concentration_alv_ti(
        ℓ_E::NTuple{6, <:Matrix},
        ℓ_0::NTuple{6, <:Matrix},
        ℓ_P::NTuple{6, <:Matrix}
    )
    n = size(ℓ_E[1], 1)
    T = promote_type(eltype(ℓ_E[1]), eltype(ℓ_0[1]), eltype(ℓ_P[1]))
    Δ = ntuple(k -> ℓ_E[k] .- ℓ_0[k], 6)
    PΔ = _ti_prod(ℓ_P, Δ)
    Id = _ti_identity(n, T)
    sum_ti = ntuple(k -> Id[k] .+ PΔ[k], 6)
    return _ti_inv(sum_ti)
end

"""
    dilute_contribution_alv_ti(ℓ_E, ℓ_0, ℓ_P) -> NTuple{6, Matrix}

TI-form dilute contribution ``\\widetilde{\\mathbb{N}} = \\Delta\\widetilde{\\mathbb{C}}\\circ\\widetilde{\\mathbb{A}}^{\\mathrm{dil}}``.
"""
function dilute_contribution_alv_ti(
        ℓ_E::NTuple{6, <:Matrix},
        ℓ_0::NTuple{6, <:Matrix},
        ℓ_P::NTuple{6, <:Matrix}
    )
    A_dil = dilute_concentration_alv_ti(ℓ_E, ℓ_0, ℓ_P)
    Δ = ntuple(k -> ℓ_E[k] .- ℓ_0[k], 6)
    return _ti_prod(Δ, A_dil)
end

"""
    dilute_alv_ti(ℓ_0, contribs_ti, fractions) -> NTuple{6, Matrix}

TI-form Dilute scheme: ``\\widetilde{\\mathbb{C}}^{\\mathrm{hom}} = \\widetilde{\\mathbb{C}}_0 + \\sum_i f_i\\,\\widetilde{\\mathbb{N}}_i`` (component-wise).
"""
function dilute_alv_ti(
        ℓ_0::NTuple{6, <:Matrix},
        contribs_ti::AbstractVector,
        fractions::AbstractVector
    )
    length(contribs_ti) == length(fractions) ||
        throw(ArgumentError("dilute_alv_ti: phase counts mismatch"))
    out = ntuple(k -> copy(ℓ_0[k]), 6)
    @inbounds for r in eachindex(contribs_ti)
        _ti_add!(out, fractions[r], contribs_ti[r])
    end
    return out
end

"""
    dilute_dual_alv_ti(ℓ_0, contribs_compliance_ti, fractions)
        -> NTuple{6, Matrix}

TI-form DiluteDual: invert to compliance Walpole form, average, invert
back.
"""
function dilute_dual_alv_ti(
        ℓ_0::NTuple{6, <:Matrix},
        contribs_compliance_ti::AbstractVector,
        fractions::AbstractVector
    )
    ℓ_J_0 = _ti_inv(ℓ_0)
    ℓ_J_eff = dilute_alv_ti(ℓ_J_0, contribs_compliance_ti, fractions)
    return _ti_inv(ℓ_J_eff)
end

"""
    mori_tanaka_alv_ti(ℓ_0, A_duts_ti, contribs_ti, fractions, f_M)
        -> NTuple{6, Matrix}

TI-form Mori-Tanaka:

```math
\\widetilde{\\mathbb{C}}^{\\mathrm{hom}} = \\widetilde{\\mathbb{C}}_0
  + \\Bigl(\\sum_i f_i\\,\\widetilde{\\mathbb{N}}_i\\Bigr)\\circ
    \\Bigl(f_0\\,H\\,\\mathbb{I} + \\sum_j f_j\\,\\widetilde{\\mathbb{A}}_j^{\\mathrm{dil}}\\Bigr)^{-\\circ},
```

all in TI Walpole form.
"""
function mori_tanaka_alv_ti(
        ℓ_0::NTuple{6, <:Matrix},
        A_duts_ti::AbstractVector,
        contribs_ti::AbstractVector,
        fractions::AbstractVector, f_M::Real
    )
    length(A_duts_ti) == length(contribs_ti) == length(fractions) ||
        throw(ArgumentError("mori_tanaka_alv_ti: phase counts mismatch"))
    n = size(ℓ_0[1], 1)
    T = promote_type(eltype(ℓ_0[1]), eltype(fractions), typeof(f_M))
    T = _ti_list_eltype(_ti_list_eltype(T, A_duts_ti), contribs_ti)
    Id = _ti_identity(n, T)
    num = ntuple(_ -> zeros(T, n, n), 6)
    den = ntuple(k -> T(f_M) .* Id[k], 6)
    @inbounds for r in eachindex(A_duts_ti)
        _ti_add!(num, fractions[r], contribs_ti[r])
        _ti_add!(den, fractions[r], A_duts_ti[r])
    end
    factor = _ti_left_divide(den, num)
    return ntuple(k -> ℓ_0[k] .+ factor[k], 6)
end

"""
    maxwell_alv_ti(ℓ_0, contribs_ti, fractions, ℓ_H_0) -> NTuple{6, Matrix}

TI-form Maxwell scheme.
"""
function maxwell_alv_ti(
        ℓ_0::NTuple{6, <:Matrix},
        contribs_ti::AbstractVector,
        fractions::AbstractVector,
        ℓ_H_0::NTuple{6, <:Matrix}
    )
    length(contribs_ti) == length(fractions) ||
        throw(ArgumentError("maxwell_alv_ti: phase counts mismatch"))
    n = size(ℓ_0[1], 1)
    T = promote_type(eltype(ℓ_0[1]), eltype(fractions), eltype(ℓ_H_0[1]))
    T = _ti_list_eltype(T, contribs_ti)
    Id = _ti_identity(n, T)
    Σ = ntuple(_ -> zeros(T, n, n), 6)
    @inbounds for r in eachindex(contribs_ti)
        _ti_add!(Σ, fractions[r], contribs_ti[r])
    end
    HΣ = _ti_prod(ℓ_H_0, Σ)
    inv_arg = ntuple(k -> Id[k] .- HΣ[k], 6)
    factor = _ti_prod(Σ, _ti_inv(inv_arg))
    return ntuple(k -> ℓ_0[k] .+ factor[k], 6)
end
