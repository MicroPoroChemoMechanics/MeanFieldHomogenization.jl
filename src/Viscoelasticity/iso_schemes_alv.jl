# =============================================================================
#  iso_schemes_alv.jl — iso-symmetry fast paths for ALV schemes.
#
#  When **every** ALV block matrix in a homogenization pipeline is in
#  **iso** form (each `(6 × 6)` Mandel block of the
#  `(6n × 6n)` matrix is `α[i,j] · 𝕁_M + β[i,j] · 𝕂_M`), the entire
#  algebra collapses to two `n × n` scalar Volterra matrices `α` and
#  `β` because `𝕁` and `𝕂` are orthogonal idempotents that commute with
#  every iso 4-tensor:
#
#      𝕁 + 𝕂 = 𝕀 ,   𝕁²= 𝕁 ,   𝕂² = 𝕂 ,   𝕁·𝕂 = 𝕂·𝕁 = 0.
#
#  Hence for iso matrices `M_a = α_a 𝕁 + β_a 𝕂` and
#  `M_b = α_b 𝕁 + β_b 𝕂` :
#
#      M_a + M_b   = (α_a + α_b) 𝕁 + (β_a + β_b) 𝕂
#      M_a ∘ M_b   = (α_a · α_b) 𝕁 + (β_a · β_b) 𝕂      (Volterra)
#      M_a^{-vol}  = α_a^{-vol} 𝕁 + β_a^{-vol} 𝕂.
#
#  All scheme algebra (Voigt, Reuss, Dilute, Mori-Tanaka, Maxwell) thus
#  reduces to two independent scalar Volterra problems on `α` and `β`.
#
#  Speedup vs the generic `(6n × 6n)` path :
#    - matrix-matrix product : ~108× cheaper (216 n³ → 2 n³)
#    - matrix inverse        : ~18× cheaper (block-LU on 6×6 blocks vs n×n)
#    - storage               : 18× smaller (2 × n² floats vs 36 n²).
#
#  These iso primitives are exported as **internal** helpers — the
#  public `homogenize_alv` API still returns a `(6n × 6n)` matrix.  The
#  iso fast path is selected automatically when all phases are iso.
# =============================================================================

# ── Iso scalar primitives ───────────────────────────────────────────────────

"""
    _iso_identity(n, T) -> (I_n, I_n)

Iso form of the ``6n\\times 6n`` block-diagonal identity matrix in the
``(\\alpha, \\beta)`` parameter space — both components reduce to the scalar
``n\\times n`` identity.
"""
@inline _iso_identity(n::Int, T::Type) = (
    Matrix{T}(LinearAlgebra.I, n, n),
    Matrix{T}(LinearAlgebra.I, n, n),
)

"""
    _iso_add!(αβ_acc, c, αβ_r) -> αβ_acc

In-place iso-form scalar AXPY: `αβ_acc .+= c · αβ_r` componentwise on
``(\\alpha, \\beta)``.  Returns `αβ_acc` for chaining.
"""
@inline function _iso_add!(αβ_acc::Tuple, c::Real, αβ_r::Tuple)
    @. αβ_acc[1] += c * αβ_r[1]
    @. αβ_acc[2] += c * αβ_r[2]
    return αβ_acc
end

"""
    _iso_prod(a, b) -> (α, β)

Iso-form Volterra product ``M_a\\circ M_b``.  Because ``\\mathbb{J}:\\mathbb{K} = \\mathbb{K}:\\mathbb{J} = 0`` the
cross terms vanish and the product reduces to two independent scalar
Volterra products ``(\\alpha_a\\circ\\alpha_b, \\beta_a\\circ\\beta_b)``.
"""
@inline _iso_prod(a::Tuple, b::Tuple) = (a[1] * b[1], a[2] * b[2])

"""
    _iso_prod!(out, a, b) -> out

In-place iso-form Volterra product : `out[1] .= a[1] * b[1]`,
`out[2] .= a[2] * b[2]`.  Allocation-free (uses `mul!` BLAS).
"""
@inline function _iso_prod!(out::Tuple, a::Tuple, b::Tuple)
    mul!(out[1], a[1], b[1])
    mul!(out[2], a[2], b[2])
    return out
end

"""
    _iso_inv(a) -> (α^{-vol}, β^{-vol})

Iso-form Volterra inverse.  The two components are inverted
independently as scalar ``n\\times n`` Volterra matrices.
"""
@inline _iso_inv(a::Tuple) = (
    volterra_inverse(a[1]; block_size = 1),
    volterra_inverse(a[2]; block_size = 1),
)

"""
    _iso_inv!(out, a) -> out

In-place iso-form Volterra inverse: `out[i]` receives ``a_i^{-\\circ}``
(``i = 1, 2``) via `volterra_inverse!`.
"""
@inline function _iso_inv!(out::Tuple, a::Tuple)
    volterra_inverse!(out[1], a[1]; block_size = 1)
    volterra_inverse!(out[2], a[2]; block_size = 1)
    return out
end

"""
    _iso_left_divide(S, M) -> (S_α^{-vol} · M_α, S_β^{-vol} · M_β)

Iso-form ``T = S^{-\\circ}\\circ M`` (left divide).  Both components solve the
scalar Volterra system ``S_x\\circ T_x = M_x`` by forward substitution
(see [`volterra_left_divide`](@ref)).
"""
@inline function _iso_left_divide(S::Tuple, M::Tuple)
    return (
        volterra_left_divide(S[1], M[1]; block_size = 1),
        volterra_left_divide(S[2], M[2]; block_size = 1),
    )
end

# ── Iso form detection and conversion ───────────────────────────────────────

"""
    _is_iso_block(M::AbstractMatrix; tol = 0) -> Bool

Heuristic: return `true` if the ``6n\\times 6n`` block matrix `M` is in iso
form (each ``6\\times 6`` block is an iso 4-tensor in Mandel form).  The check
verifies the canonical iso pattern ``(\\mathbf{M}_{ij})_{kk} = (\\alpha + 2\\beta)/3``
(diag), ``(\\alpha - \\beta)/3`` (off-diag in the 1..3 block), ``\\beta`` on the 4..6
diagonal, and zero elsewhere — to within absolute tolerance `tol`
times the maximum modulus of `M`.

Used by `homogenize_alv` to dispatch to the iso fast path automatically.

Allocation-free: extracts ``(\\alpha, \\beta)`` from the canonical block entries
on the fly without building the parameter matrices.
"""
function _is_iso_block(M::AbstractMatrix; tol::Real = 1.0e-12)
    sz = size(M, 1)
    sz == size(M, 2) || return false
    sz % 6 == 0 || return false
    n = sz ÷ 6
    iszero(n) && return true
    scale = max(maximum(abs, M), one(real(eltype(M))))
    abstol = tol * scale
    @inbounds for i in 1:n, j in 1:n
        r = 6 * (i - 1)
        c = 6 * (j - 1)
        # Extract (α, β) from the canonical entries of block (i, j).
        a = M[r + 1, c + 1] + 2 * M[r + 1, c + 2]
        b = M[r + 4, c + 4]
        diag_top = (a + 2b) / 3
        offdiag_top = (a - b) / 3
        # Upper-left 3×3 block: diag = (α+2β)/3, off-diag = (α-β)/3.
        for k in 1:3, l in 1:3
            expected = (k == l) ? diag_top : offdiag_top
            abs(M[r + k, c + l] - expected) ≤ abstol || return false
        end
        # Cross blocks (1..3 × 4..6 and 4..6 × 1..3) must be zero.
        for k in 1:3, l in 4:6
            abs(M[r + k, c + l]) ≤ abstol || return false
        end
        for k in 4:6, l in 1:3
            abs(M[r + k, c + l]) ≤ abstol || return false
        end
        # Lower-right 3×3 block: diagonal = β, off-diag = 0.
        for k in 4:6, l in 4:6
            expected = (k == l) ? b : zero(b)
            abs(M[r + k, c + l] - expected) ≤ abstol || return false
        end
    end
    return true
end

"""
    _iso_pair(M::AbstractMatrix) -> (α::Matrix, β::Matrix)

Extract the iso ``(\\alpha, \\beta)`` parameter matrices from a ``6n\\times 6n`` iso
block matrix.  Wrapper around [`iso_params_from_blocks`](@ref) returning
a tuple suitable for the iso scheme primitives above.
"""
@inline _iso_pair(M::AbstractMatrix) = iso_params_from_blocks(M)

"""
    _iso_blocks(αβ::Tuple) -> Matrix

Reassemble a ``6n\\times 6n`` iso block matrix from an ``(\\alpha, \\beta)`` pair.
Inverse of [`_iso_pair`](@ref).
"""
@inline _iso_blocks(αβ::Tuple) = iso_blocks_from_params(αβ[1], αβ[2])

# ── Iso scheme implementations ──────────────────────────────────────────────

"""
    voigt_alv_iso(αβ_phases, fractions) -> (α_eff, β_eff)

Iso-form Voigt bound: ``\\alpha^{\\mathrm{hom}} = \\sum_i f_i\\,\\alpha_i``, ``\\beta^{\\mathrm{hom}} = \\sum_i f_i\\,\\beta_i``.
"""
function voigt_alv_iso(αβ_phases::AbstractVector, fractions::AbstractVector)
    length(αβ_phases) == length(fractions) ||
        throw(ArgumentError("voigt_alv_iso: phase counts mismatch"))
    isempty(αβ_phases) && throw(ArgumentError("voigt_alv_iso: at least one phase required"))
    n = size(αβ_phases[1][1], 1)
    T = _promote_iso_eltype(αβ_phases, fractions)
    αβ_eff = (zeros(T, n, n), zeros(T, n, n))
    @inbounds for r in eachindex(αβ_phases)
        _iso_add!(αβ_eff, fractions[r], αβ_phases[r])
    end
    return αβ_eff
end

"""
    _promote_iso_eltype(αβ_list, fractions, αβ_0...) -> Type

Promote element types of an iso ALV pipeline : `αβ_list` is a vector of
``(\\alpha, \\beta)`` tuples, `fractions` is a numeric vector, optional positional
``(\\alpha, \\beta)`` tuples (e.g. the matrix reference) participate in the
promotion.  Used to lift `Tuple{Matrix{Float64}, Matrix{Float64}}` to
the right element type when sensitivities are run with `Dual` fractions.
"""
function _promote_iso_eltype(
        αβ_list::AbstractVector, fractions::AbstractVector,
        αβ_0::Tuple...
    )
    T = eltype(fractions)
    for ab in αβ_list          # loop over ALL phases — eltypes may differ
        T = promote_type(T, eltype(ab[1]), eltype(ab[2]))
    end
    for ab in αβ_0
        T = promote_type(T, eltype(ab[1]), eltype(ab[2]))
    end
    return T
end

"""
    reuss_alv_iso(αβ_phases, fractions) -> (α_eff, β_eff)

Iso-form Reuss bound: invert per-phase compliances, average, invert
back — done independently on the two scalar Volterra components.
"""
function reuss_alv_iso(αβ_phases::AbstractVector, fractions::AbstractVector)
    length(αβ_phases) == length(fractions) ||
        throw(ArgumentError("reuss_alv_iso: phase counts mismatch"))
    αβ_inv_phases = [_iso_inv(αβ) for αβ in αβ_phases]
    αβ_inv_eff = voigt_alv_iso(αβ_inv_phases, fractions)
    return _iso_inv(αβ_inv_eff)
end

"""
    dilute_alv_iso(αβ_0, contribs_iso, fractions) -> (α_eff, β_eff)

Iso-form Dilute scheme: ``\\widetilde{\\mathbb{C}}^{\\mathrm{hom}} = \\widetilde{\\mathbb{C}}_0 + \\sum_i f_i\\,\\widetilde{\\mathbb{N}}_i``,
evaluated on the ``(\\alpha, \\beta)`` iso components.
"""
function dilute_alv_iso(
        αβ_0::Tuple,
        contribs_iso::AbstractVector,
        fractions::AbstractVector
    )
    length(contribs_iso) == length(fractions) ||
        throw(ArgumentError("dilute_alv_iso: phase counts mismatch"))
    T = _promote_iso_eltype(contribs_iso, fractions, αβ_0)
    αβ = (Matrix{T}(αβ_0[1]), Matrix{T}(αβ_0[2]))
    @inbounds for r in eachindex(contribs_iso)
        _iso_add!(αβ, fractions[r], contribs_iso[r])
    end
    return αβ
end

"""
    dilute_concentration_alv_iso(αβ_E, αβ_0, αβ_P) -> (α_A, β_A)

Iso-form dilute concentration
``\\widetilde{\\mathbb{A}}^{\\mathrm{dil}} = (H\\,\\mathbb{I} + \\widetilde{\\mathbb{P}}\\circ\\Delta\\widetilde{\\mathbb{C}})^{-\\circ}``,
``\\Delta\\widetilde{\\mathbb{C}} = \\widetilde{\\mathbb{C}}^{\\mathcal{E}} - \\widetilde{\\mathbb{C}}_0``, reduced
to two scalar Volterra problems on ``(\\alpha^{\\mathcal{E}} - \\alpha_0, \\beta^{\\mathcal{E}} - \\beta_0)``.
"""
function dilute_concentration_alv_iso(αβ_E::Tuple, αβ_0::Tuple, αβ_P::Tuple)
    n = size(αβ_E[1], 1)
    T = promote_type(eltype(αβ_E[1]), eltype(αβ_0[1]), eltype(αβ_P[1]))
    Δαβ = (αβ_E[1] .- αβ_0[1], αβ_E[2] .- αβ_0[2])
    PΔ = _iso_prod(αβ_P, Δαβ)
    Id = _iso_identity(n, T)
    return _iso_inv((Id[1] .+ PΔ[1], Id[2] .+ PΔ[2]))
end

"""
    dilute_contribution_alv_iso(αβ_E, αβ_0, αβ_P) -> (α_N, β_N)

Iso-form dilute contribution ``\\widetilde{\\mathbb{N}} = \\Delta\\widetilde{\\mathbb{C}}\\circ\\widetilde{\\mathbb{A}}^{\\mathrm{dil}}``.
"""
function dilute_contribution_alv_iso(αβ_E::Tuple, αβ_0::Tuple, αβ_P::Tuple)
    A_dil = dilute_concentration_alv_iso(αβ_E, αβ_0, αβ_P)
    Δαβ = (αβ_E[1] .- αβ_0[1], αβ_E[2] .- αβ_0[2])
    return _iso_prod(Δαβ, A_dil)
end

"""
    mori_tanaka_alv_iso(αβ_0, A_duts_iso, contribs_iso, fractions, f_M)
        -> (α_eff, β_eff)

Iso-form Mori-Tanaka:

```math
\\widetilde{\\mathbb{C}}^{\\mathrm{hom}} = \\widetilde{\\mathbb{C}}_0
  + \\Bigl(\\sum_i f_i\\,\\widetilde{\\mathbb{N}}_i\\Bigr)\\circ
    \\Bigl(f_0\\,H\\,\\mathbb{I} + \\sum_j f_j\\,\\widetilde{\\mathbb{A}}_j^{\\mathrm{dil}}\\Bigr)^{-\\circ},
```

reduced to two scalar Volterra problems on ``(\\alpha, \\beta)``.
"""
function mori_tanaka_alv_iso(
        αβ_0::Tuple, A_duts_iso::AbstractVector,
        contribs_iso::AbstractVector,
        fractions::AbstractVector, f_M::Real
    )
    length(A_duts_iso) == length(contribs_iso) == length(fractions) ||
        throw(ArgumentError("mori_tanaka_alv_iso: phase counts mismatch"))
    n = size(αβ_0[1], 1)
    T = promote_type(
        _promote_iso_eltype(A_duts_iso, fractions, αβ_0),
        _promote_iso_eltype(contribs_iso, fractions),
        typeof(f_M)
    )
    Id = _iso_identity(n, T)
    num = (zeros(T, n, n), zeros(T, n, n))
    den = (T(f_M) .* Id[1], T(f_M) .* Id[2])
    @inbounds for r in eachindex(A_duts_iso)
        _iso_add!(num, fractions[r], contribs_iso[r])
        _iso_add!(den, fractions[r], A_duts_iso[r])
    end
    factor = _iso_prod(num, _iso_inv(den))
    return (T.(αβ_0[1]) .+ factor[1], T.(αβ_0[2]) .+ factor[2])
end

"""
    maxwell_alv_iso(αβ_0, contribs_iso, fractions, αβ_H_0) -> (α_eff, β_eff)

Iso-form Maxwell scheme:

```math
\\widetilde{\\mathbb{C}}^{\\mathrm{hom}} = \\widetilde{\\mathbb{C}}_0
  + \\widetilde{\\mathbb{N}}\\circ\\bigl(H\\,\\mathbb{I} - \\widetilde{\\mathbb{P}}_{\\Omega}\\circ\\widetilde{\\mathbb{N}}\\bigr)^{-\\circ},
\\qquad
\\widetilde{\\mathbb{N}} = \\sum_i f_i\\,\\widetilde{\\mathbb{N}}_i ,
```

with ``\\widetilde{\\mathbb{P}}_{\\Omega}`` the Hill kernel of the distribution shape,
reduced to two scalar Volterra problems on ``(\\alpha, \\beta)``.
"""
function maxwell_alv_iso(
        αβ_0::Tuple, contribs_iso::AbstractVector,
        fractions::AbstractVector, αβ_H_0::Tuple
    )
    length(contribs_iso) == length(fractions) ||
        throw(ArgumentError("maxwell_alv_iso: phase counts mismatch"))
    n = size(αβ_0[1], 1)
    T = _promote_iso_eltype(contribs_iso, fractions, αβ_0, αβ_H_0)
    Id = _iso_identity(n, T)
    Σ = (zeros(T, n, n), zeros(T, n, n))
    @inbounds for r in eachindex(contribs_iso)
        _iso_add!(Σ, fractions[r], contribs_iso[r])
    end
    HΣ = _iso_prod(αβ_H_0, Σ)
    factor = _iso_prod(Σ, _iso_inv((Id[1] .- HΣ[1], Id[2] .- HΣ[2])))
    return (T.(αβ_0[1]) .+ factor[1], T.(αβ_0[2]) .+ factor[2])
end

"""
    dilute_dual_alv_iso(αβ_0, contribs_compliance_iso, fractions) -> (α_eff, β_eff)

Iso-form DiluteDual: invert the matrix to compliance space, average,
invert back.
"""
function dilute_dual_alv_iso(
        αβ_0::Tuple, contribs_compliance_iso::AbstractVector,
        fractions::AbstractVector
    )
    αβ_J_0 = _iso_inv(αβ_0)
    αβ_J_eff = dilute_alv_iso(αβ_J_0, contribs_compliance_iso, fractions)
    return _iso_inv(αβ_J_eff)
end
