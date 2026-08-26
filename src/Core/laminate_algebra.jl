# =============================================================================
#  laminate_algebra.jl — in-plane / out-of-plane block algebra of a periodic
#  laminate, in the Kelvin-Mandel representation.
#
#  Everything here is pure linear algebra on `SMatrix`, with no dependency on
#  `Schemes` or `Laminates`: the laminate kernel, its ageing-viscoelastic
#  twin and user code all reach it from `Core`.
#
#  ── The partition ─────────────────────────────────────────────────────────
#  In an orthonormal frame whose THIRD axis is the layer normal `n`, the
#  Kelvin-Mandel order used by `TensND` / `Tensors.jl`
#
#       (11, 22, 33, √2·23, √2·13, √2·12)
#
#  reads, with `(e₁, e₂, e₃) = (ℓ, m, n)`,
#
#       (ℓℓ, mm, nn, √2·mn, √2·ℓn, √2·ℓm) ,
#
#  so the reordered basis of the multilayer literature
#
#       B* = (ℓ⊗ℓ, m⊗m, √2 ℓ⊗ˢm | n⊗n, √2 m⊗ˢn, √2 n⊗ˢℓ)
#
#  is nothing but the index partition
#
#       KM_IP = (1, 2, 6)      in-plane      (ε₁₁, ε₂₂, ε₁₂ — continuous)
#       KM_OP = (3, 4, 5)      out-of-plane  (σ₃₃, σ₂₃, σ₁₃ — continuous)
#
#  The √2 factors that B* calls for are ALREADY carried by the Mandel basis:
#  no permutation matrix is ever formed and no rescaling is ever applied.
#  The two index sets are interleaved rather than contiguous, which costs
#  nothing — the block algebra below is ordering-agnostic.
#
#  ── The index reversal (read this before touching anything) ───────────────
#  The out-of-plane Mandel slots (3, 4, 5) correspond to the index pairs
#  ((3,3), (2,3), (1,3)), hence to the acoustic-tensor indices
#
#       K_ab = n_i C_iajb n_j = C_3a3b
#
#  in the REVERSED order (3, 2, 1).  Writing `s = (1, √2, √2)` for the Mandel
#  weights of those three slots,
#
#       KM(ℂ)[OP_a, OP_b] = s_a · s_b · K[π(a), π(b)] ,   π = (3, 2, 1) .
#
#  This reversal is invisible for isotropic layers (`K` is then diagonal),
#  which is exactly why it must be guarded by a test on a triclinic
#  stiffness — see `test/Laminates/test_km_blocks.jl`.
#
#  ── Why no `pinv` ─────────────────────────────────────────────────────────
#  `⟨ℙ⟩` is supported on the 3-dimensional out-of-plane subspace, so its
#  Moore-Penrose pseudo-inverse is the ordinary inverse of its `OP` block,
#  embedded back.  Computing it as a 3×3 cofactor inversion (`_inv3`)
#  rather than through `LinearAlgebra.pinv` keeps the whole laminate kernel
#  differentiable by `ForwardDiff` and evaluable on `SymPy.Sym` /
#  `Symbolics.Num` — an SVD is neither.  For the same reason every matrix
#  here is an `SMatrix` and never an `MMatrix`: `MMatrix{6,6,T}(undef)` is
#  not constructible for a non-`isbits` `T` such as `SymPy.Sym`.
# =============================================================================

using TensND: _KM_rotation

"""
    KM_IP

Kelvin-Mandel slots of the **in-plane** subspace `(ℓ⊗ℓ, m⊗m, √2 ℓ⊗ˢm)` in a
frame whose third axis is the layer normal — the components of the strain
that stay continuous across a laminate's interfaces.
"""
const KM_IP = SVector(1, 2, 6)

"""
    KM_OP

Kelvin-Mandel slots of the **out-of-plane** subspace `(n⊗n, √2 m⊗ˢn, √2 n⊗ˢℓ)`
— the components of the stress that stay continuous across a laminate's
interfaces, and the support of the flat-inclusion Hill tensor `ℙ`.
"""
const KM_OP = SVector(3, 4, 5)

# Slot look-ups, resolved at compile time. Return 0 for an index that is not
# in the corresponding set, so the embedding helpers can branch on it.
@inline _op_slot(i::Int) = i == 3 ? 1 : (i == 4 ? 2 : (i == 5 ? 3 : 0))
@inline _ip_slot(i::Int) = i == 1 ? 1 : (i == 2 ? 2 : (i == 6 ? 3 : 0))

"""
    _op_block(M) -> SMatrix{3,3}

Out-of-plane block of a 6×6 Kelvin-Mandel matrix, in **slot** indexing
(rows/columns ordered as the Mandel slots `(3, 4, 5)`, i.e. `(nn, mn, ℓn)`).
No unscaling and no permutation — see [`acoustic_tensor`](@ref) for the
physical acoustic tensor.
"""
@inline _op_block(M::AbstractMatrix) = SMatrix{3, 3}(M[KM_OP, KM_OP])

"""
    _ip_block(M) -> SMatrix{3,3}

In-plane block of a 6×6 Kelvin-Mandel matrix, in slot indexing (Mandel slots
`(1, 2, 6)`, i.e. `(ℓℓ, mm, ℓm)`).
"""
@inline _ip_block(M::AbstractMatrix) = SMatrix{3, 3}(M[KM_IP, KM_IP])

"""
    _op_embed(B) -> SMatrix{6,6}

Scatter a slot-indexed 3×3 out-of-plane block back into a full 6×6
Kelvin-Mandel matrix, zero everywhere else. Left inverse of [`_op_block`](@ref)
on out-of-plane-supported matrices.
"""
@inline function _op_embed(B::AbstractMatrix{T}) where {T}
    z = zero(T)
    return SMatrix{6, 6, T}(
        ntuple(Val(36)) do k
            i = mod1(k, 6)
            j = (k - 1) ÷ 6 + 1
            a = _op_slot(i)
            b = _op_slot(j)
            (a == 0 || b == 0) ? z : @inbounds(B[a, b])
        end
    )
end

"""
    _ip_embed(B) -> SMatrix{6,6}

Scatter a slot-indexed 3×3 in-plane block back into a full 6×6 Kelvin-Mandel
matrix, zero everywhere else. Used for the membrane (surface-elastic)
interface term, which lives entirely in the in-plane block.
"""
@inline function _ip_embed(B::AbstractMatrix{T}) where {T}
    z = zero(T)
    return SMatrix{6, 6, T}(
        ntuple(Val(36)) do k
            i = mod1(k, 6)
            j = (k - 1) ÷ 6 + 1
            a = _ip_slot(i)
            b = _ip_slot(j)
            (a == 0 || b == 0) ? z : @inbounds(B[a, b])
        end
    )
end

"""
    plane_pinv(M) -> SMatrix{6,6}

Moore-Penrose pseudo-inverse of a 6×6 Kelvin-Mandel matrix whose range and
kernel are **known** to coincide with the out-of-plane subspace [`KM_OP`] of
the frame in which it is written — true for `⟨ℙ⟩` and for the out-of-plane
part of any stiffness.

Implemented as a 3×3 cofactor inversion of the `OP` block, embedded back:

```math
\\mathrm{Mat}(\\langle\\mathbb P\\rangle^{\\dagger}, \\mathcal B^{*}) =
\\begin{pmatrix} 0 & 0 \\\\ 0 & P_{\\mathcal O}^{-1}\\end{pmatrix},
\\qquad
\\langle\\mathbb P\\rangle^{\\dagger} : \\langle\\mathbb P\\rangle = \\Pi^{\\mathcal O}.
```

Never `LinearAlgebra.pinv`: its SVD is differentiable by neither `ForwardDiff`
nor a symbolic backend, and would in any case be wasted on an exactly-rank-3
input.
"""
@inline plane_pinv(M::AbstractMatrix) = _op_embed(_inv3(_op_block(M)))

"""
    flat_hill(C6) -> SMatrix{6,6}

Hill polarisation tensor `ℙ = n ⊗ˢ 𝐊⁻¹ ⊗ˢ n` of a flat (layer) inclusion of
Kelvin-Mandel stiffness `C6`, expressed in the layer frame — the limit of the
Hill tensor of a flat ellipsoid as its smallest aspect ratio tends to zero.

`ℙ` and [`plane_pinv`](@ref) of the stiffness are the **same object**: the
out-of-plane block of `ℙ` is the matrix inverse of the out-of-plane block of
`ℂ`, and every other block vanishes. Hence also `ℙ : ℂ : ℙ = ℙ` and
`ℚ : ℙ = 0` with `ℚ = ℂ − ℂ:ℙ:ℂ` the second Hill tensor.
"""
@inline flat_hill(C6::AbstractMatrix) = plane_pinv(C6)

"""
    acoustic_tensor(M) -> SMatrix{3,3}

Acoustic tensor `𝐊 = n · ℂ · n`, i.e. `K_ab = C_{3a3b}`, read off the
out-of-plane block of a 6×6 Kelvin-Mandel matrix written in a frame whose
third axis is `n`. Undoes both the Mandel `√2` weights and the `(3, 2, 1)`
index reversal documented at the top of this file.

Used for reporting and for the exact out-of-plane oracle of a laminate,
`(n · ℂ^{hom} · n)^{-1} = Σ_i f_i (n · ℂ_i · n)^{-1} + Σ_j 𝕂_j / L`; the
laminate kernel itself never materializes `𝐊`.
"""
@inline function acoustic_tensor(M::AbstractMatrix{T}) where {T}
    B = _op_block(M)
    r2 = sqrt(2 * one(T))
    # K[i,j] = B[π(i), π(j)] / (ŝ_i ŝ_j),  π = (3,2,1),  ŝ = (√2, √2, 1)
    K11 = @inbounds B[3, 3] / 2
    K12 = @inbounds B[3, 2] / 2
    K13 = @inbounds B[3, 1] / r2
    K22 = @inbounds B[2, 2] / 2
    K23 = @inbounds B[2, 1] / r2
    K33 = @inbounds B[1, 1]
    return SMatrix{3, 3, T}(K11, K12, K13, K12, K22, K23, K13, K23, K33)
end

"""
    compliance_op_block(X) -> SMatrix{3,3}

Slot-indexed out-of-plane block of an out-of-plane **compliance** given by its
physical 3×3 matrix `X` in the frame `(ℓ, m, n)` — the inverse map of
[`acoustic_tensor`](@ref) on the compliance side:

```
Ŝ[a, b] = X[π(a), π(b)] / (s_a s_b) ,   π = (3, 2, 1),  s = (1, √2, √2).
```

Its use is the imperfect interface of spring type, whose compliance `𝕂`
produces the added strain `(𝕂·(σ·n)) ⊗ˢ n`. `X` may be **any symmetric 3×3
compliance** — all six entries are used — so an anisotropic interface is
handled exactly like an isotropic one; the isotropic case
`𝕂 = k_n\\,n⊗n + k_t\\,(δ − n⊗n)` simply gives the block
`diag(k_n, k_t/2, k_t/2)`.

Note the factor 2 on the tangential entries: a displacement jump enters the
strain through a *symmetrized* product, so those compliances are halved in
Mandel slots — exactly as `ℙ` is `𝐊⁻¹` divided, not multiplied, by the Mandel
weights.
"""
@inline function compliance_op_block(X::AbstractMatrix{T}) where {T}
    r2 = sqrt(2 * one(T))
    S11 = @inbounds X[3, 3]
    S12 = @inbounds X[3, 2] / r2
    S13 = @inbounds X[3, 1] / r2
    S22 = @inbounds X[2, 2] / 2
    S23 = @inbounds X[2, 1] / 2
    S33 = @inbounds X[1, 1] / 2
    return SMatrix{3, 3, T}(S11, S12, S13, S12, S22, S23, S13, S23, S33)
end

"""
    _inv_km6(M) -> SMatrix{6,6}

Inverse of a 6×6 Kelvin-Mandel matrix by block elimination on the in-plane /
out-of-plane partition: two `_inv3` cofactor inversions and a Schur
complement, hence **no pivoting and no factorization**. Stays exact for
`ForwardDiff.Dual` and evaluable for symbolic element types, where the LU
fallback behind `inv(::SMatrix{6,6})` is not.

Requires the out-of-plane block and the in-plane Schur complement to be
invertible — both hold for any stiffness with a positive-definite acoustic
tensor.
"""
@inline function _inv_km6(M::AbstractMatrix)
    A = _ip_block(M)
    D = _op_block(M)
    B = SMatrix{3, 3}(M[KM_IP, KM_OP])
    C = SMatrix{3, 3}(M[KM_OP, KM_IP])
    Di = _inv3(D)
    BDi = B * Di
    DiC = Di * C
    S = _inv3(A - BDi * C)          # inverse of the in-plane Schur complement
    X11 = S
    X12 = -S * BDi
    X21 = -DiC * S
    X22 = Di + DiC * S * BDi
    return _km6_from_blocks(X11, X12, X21, X22)
end

"""
    _km6_from_blocks(X_II, X_IO, X_OI, X_OO) -> SMatrix{6,6}

Reassemble a 6×6 Kelvin-Mandel matrix from its four slot-indexed 3×3 blocks
on the in-plane / out-of-plane partition.
"""
@inline function _km6_from_blocks(
        X11::AbstractMatrix{T}, X12::AbstractMatrix,
        X21::AbstractMatrix, X22::AbstractMatrix
    ) where {T}
    return SMatrix{6, 6, T}(
        ntuple(Val(36)) do k
            i = mod1(k, 6)
            j = (k - 1) ÷ 6 + 1
            ai = _ip_slot(i)
            aj = _ip_slot(j)
            bi = _op_slot(i)
            bj = _op_slot(j)
            @inbounds if ai != 0 && aj != 0
                X11[ai, aj]
            elseif ai != 0
                X12[ai, bj]
            elseif aj != 0
                X21[bi, aj]
            else
                X22[bi, bj]
            end
        end
    )
end

# ── Frame handling ──────────────────────────────────────────────────────────

"""
    _bond6(R) -> SMatrix{6,6}

6×6 Kelvin-Mandel (Bond) matrix of the 3×3 rotation `R`, such that the Mandel
matrix of a 4th-order tensor in the frame whose basis vectors are the
**columns** of `R` is `Q' * KM(C) * Q`. `Q` is orthogonal.

A thin alias over `TensND._KM_rotation`, which builds the same object with
`SMatrix`/`ntuple` and is therefore usable for `ForwardDiff.Dual` and for
symbolic element types alike.
"""
@inline _bond6(R::AbstractMatrix) = TensND._KM_rotation(R)

# ── The laminate kernels ────────────────────────────────────────────────────

"""
    laminate_stiffness(C6s, f, P_int, C_surf; opinv = plane_pinv,
                       opinv_avg = plane_pinv) -> SMatrix{6,6}

Effective Kelvin-Mandel stiffness of a periodic laminate, in the layer frame:

```math
\\mathbb C^{hom} = \\langle\\mathbb Q\\rangle
 + \\langle\\mathbb C : \\mathbb P\\rangle : \\langle\\mathbb P\\rangle^{\\dagger}
   : \\langle\\mathbb P : \\mathbb C\\rangle
 + \\sum_j \\mathbb C^{s}_j / L ,
```

with `⟨·⟩ = Σ_i f_i (·)_i`, `ℙ_i = flat_hill(ℂ_i)` and
`ℚ_i = ℂ_i − ℂ_i:ℙ_i:ℂ_i` (the in-plane Schur complement of `ℂ_i`).

Arguments:

- `C6s` — per-layer Kelvin-Mandel stiffnesses in the layer frame;
- `f` — per-layer volume fractions;
- `P_int` — `Σ_j ℙ^{int}_j / L`, the primal (spring / Kapitza) interface term,
  which is out-of-plane-supported and adds to `⟨ℙ⟩` alone;
- `C_surf` — `Σ_j ℂ^{s}_j / L`, the dual (membrane) surface term, which is
  in-plane-supported and adds to `ℂ^{hom}` directly, the interfaces being
  planar (`divₛ σˢ = 0`, hence no traction jump).

`opinv` / `opinv_avg` are the out-of-plane inversion used for a layer
stiffness and for the average `⟨ℙ⟩` respectively. They are function arguments
so that the ageing-viscoelastic laminate reuses this very kernel with the
Volterra block inversion substituted; Julia specializes on the function type,
so there is no abstraction cost and only one place where the physics lives.
"""
function laminate_stiffness(
        C6s, f, P_int, C_surf;
        opinv = plane_pinv, opinv_avg = plane_pinv
    )
    Z = zero(first(C6s))
    Pavg = Z
    CPavg = Z
    PCavg = Z
    Qavg = Z
    @inbounds for i in eachindex(C6s)
        C = C6s[i]
        P = opinv(C)                 # ℙ_i
        CP = C * P                   # in Kelvin-Mandel, ⊡ is the matrix product
        PC = P * C
        fi = f[i]
        Pavg = Pavg + fi * P
        CPavg = CPavg + fi * CP
        PCavg = PCavg + fi * PC
        Qavg = Qavg + fi * (C - CP * C)
    end
    Pavg = Pavg + P_int
    return Qavg + CPavg * opinv_avg(Pavg) * PCavg + C_surf
end

"""
    laminate_strain_localization(C6, Chom6; opinv = plane_pinv) -> SMatrix{6,6}

Strain localization tensor of one layer, `𝔸_i = 𝕀 + ℙ_i : (ℂ^{hom} − ℂ_i)`,
in Kelvin-Mandel form. Its in-plane block is the identity and its
in-plane/out-of-plane coupling block vanishes — the macroscopic in-plane
strain is transmitted unchanged to every layer. The fractions weight to the
identity, `Σ_i f_i 𝔸_i = 𝕀`.
"""
@inline function laminate_strain_localization(C6, Chom6; opinv = plane_pinv)
    return one(SMatrix{6, 6, eltype(Chom6)}) + opinv(C6) * (Chom6 - C6)
end

"""
    laminate_stress_localization(C6, Chom6; opinv = plane_pinv) -> SMatrix{6,6}

Stress localization tensor of one layer,
`𝔹_i = ℂ_i : 𝔸_i : (ℂ^{hom})^{-1}`, in Kelvin-Mandel form, with
`Σ_i f_i 𝔹_i = 𝕀`. The inverse goes through [`_inv_km6`](@ref), so the result
stays differentiable and symbolically evaluable.
"""
@inline function laminate_stress_localization(C6, Chom6; opinv = plane_pinv)
    return C6 * laminate_strain_localization(C6, Chom6; opinv = opinv) * _inv_km6(Chom6)
end

"""
    laminate_stress_strain_localization(C6, Chom6; opinv = plane_pinv) -> SMatrix{6,6}

Mixed localization tensor of one layer, `𝔸^{σε}_i = ℂ_i : 𝔸_i`, mapping the
**macroscopic strain** to the layer stress: `σ_i = 𝔸^{σε}_i : E`. It weights to
the effective stiffness, `Σ_i f_i 𝔸^{σε}_i = ℂ^{hom}`, which is the mean-field
identity `ℂ^{hom} = ⟨ℂ : 𝔸⟩` written layer by layer.

No inverse of `ℂ^{hom}` is formed, unlike [`laminate_stress_localization`](@ref).
"""
@inline function laminate_stress_strain_localization(C6, Chom6; opinv = plane_pinv)
    return C6 * laminate_strain_localization(C6, Chom6; opinv = opinv)
end

"""
    laminate_strain_stress_localization(C6, Chom6; opinv = plane_pinv) -> SMatrix{6,6}

Mixed localization tensor of one layer, `𝔸^{εσ}_i = 𝔸_i : (ℂ^{hom})^{-1}`,
mapping the **macroscopic stress** to the layer strain: `ε_i = 𝔸^{εσ}_i : Σ`,
with `Σ_i f_i 𝔸^{εσ}_i = (ℂ^{hom})^{-1}`. The inverse goes through
[`_inv_km6`](@ref), never `inv(::SMatrix{6,6})`.
"""
@inline function laminate_strain_stress_localization(C6, Chom6; opinv = plane_pinv)
    return laminate_strain_localization(C6, Chom6; opinv = opinv) * _inv_km6(Chom6)
end

# ── Conduction (order 2) ────────────────────────────────────────────────────
#
# A second-order property needs no Mandel weights: in the frame (ℓ, m, n) the
# partition is simply `IP = (1, 2)` / `OP = (3)`, and the out-of-plane
# subspace is ONE-dimensional, so the pseudo-inverse degenerates to a scalar
# reciprocal.

"""
    plane_pinv2(K3) -> SMatrix{3,3}

Order-2 analog of [`plane_pinv`](@ref): the pseudo-inverse of a 3×3
conductivity-like matrix restricted to the out-of-plane direction `n`, i.e.
`(1/k_{nn}) n⊗n` in a frame whose third axis is `n`.
"""
@inline function plane_pinv2(K3::AbstractMatrix{T}) where {T}
    z = zero(T)
    p = inv(@inbounds K3[3, 3])
    return SMatrix{3, 3, T}(z, z, z, z, z, z, z, z, p)
end

"""
    laminate_conductivity(K3s, f, P_int, K_surf; opinv = plane_pinv2,
                          opinv_avg = plane_pinv2) -> SMatrix{3,3}

Effective conductivity (or diffusivity, permeability, …) of a periodic
laminate, in the layer frame — the order-2 transposition of
[`laminate_stiffness`](@ref), with the in-plane gradient continuous and the
normal flux continuous. `P_int` collects `Σ_j ρ_j / L · n⊗n` (Kapitza) and
`K_surf` collects `Σ_j (k^s_j / L)(δ − n⊗n)` (highly conductive surface
layer).

Because the out-of-plane subspace is one-dimensional, the normal component
obeys the exact series law `1/k^{hom}_{nn} = Σ_i f_i / k_{i,nn} + Σ_j ρ_j / L`.
"""
function laminate_conductivity(
        K3s, f, P_int, K_surf;
        opinv = plane_pinv2, opinv_avg = plane_pinv2
    )
    Z = zero(first(K3s))
    Pavg = Z
    KPavg = Z
    PKavg = Z
    Qavg = Z
    @inbounds for i in eachindex(K3s)
        K = K3s[i]
        P = opinv(K)
        KP = K * P
        PK = P * K
        fi = f[i]
        Pavg = Pavg + fi * P
        KPavg = KPavg + fi * KP
        PKavg = PKavg + fi * PK
        Qavg = Qavg + fi * (K - KP * K)
    end
    Pavg = Pavg + P_int
    return Qavg + KPavg * opinv_avg(Pavg) * PKavg + K_surf
end

"""
    laminate_gradient_localization(K3, Khom3; opinv = plane_pinv2) -> SMatrix{3,3}

Gradient localization tensor of one layer in a transport problem,
`𝐀_i = 𝟏 + 𝐏_i · (𝐊^{hom} − 𝐊_i)`, with `Σ_i f_i 𝐀_i = 𝟏`.
"""
@inline function laminate_gradient_localization(K3, Khom3; opinv = plane_pinv2)
    return one(SMatrix{3, 3, eltype(Khom3)}) + opinv(K3) * (Khom3 - K3)
end

"""
    laminate_flux_localization(K3, Khom3; opinv = plane_pinv2) -> SMatrix{3,3}

Flux localization tensor of one layer,
`𝐁_i = 𝐊_i · 𝐀_i · (𝐊^{hom})^{-1}`, with `Σ_i f_i 𝐁_i = 𝟏`.
"""
@inline function laminate_flux_localization(K3, Khom3; opinv = plane_pinv2)
    return K3 * laminate_gradient_localization(K3, Khom3; opinv = opinv) * _inv3(Khom3)
end

"""
    laminate_flux_gradient_localization(K3, Khom3; opinv = plane_pinv2) -> SMatrix{3,3}

Mixed transport localization of one layer, `𝐀^{qg}_i = 𝐊_i · 𝐀_i`, mapping the
**macroscopic gradient** to the layer flux, with `Σ_i f_i 𝐀^{qg}_i = 𝐊^{hom}`.
Order-2 twin of [`laminate_stress_strain_localization`](@ref).
"""
@inline function laminate_flux_gradient_localization(K3, Khom3; opinv = plane_pinv2)
    return K3 * laminate_gradient_localization(K3, Khom3; opinv = opinv)
end

"""
    laminate_gradient_flux_localization(K3, Khom3; opinv = plane_pinv2) -> SMatrix{3,3}

Mixed transport localization of one layer, `𝐀^{gq}_i = 𝐀_i · (𝐊^{hom})^{-1}`,
mapping the **macroscopic flux** to the layer gradient, with
`Σ_i f_i 𝐀^{gq}_i = (𝐊^{hom})^{-1}`. Order-2 twin of
[`laminate_strain_stress_localization`](@ref).
"""
@inline function laminate_gradient_flux_localization(K3, Khom3; opinv = plane_pinv2)
    return laminate_gradient_localization(K3, Khom3; opinv = opinv) * _inv3(Khom3)
end
