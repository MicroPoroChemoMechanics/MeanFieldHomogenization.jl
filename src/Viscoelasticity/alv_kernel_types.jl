# =============================================================================
#  alv_kernel_types.jl — structured ALV kernel types analogous to TensISO /
#  TensTI / TensOrtho but parametrized over discretized Volterra operators.
#
#  An ALV kernel `M̃(t,t')` stored as a `(6n × 6n)` block matrix of memory
#  has 36 n² Float64 entries.  In iso / TI / ortho form the same kernel
#  is exactly described by 2 / 6 / 12 `n × n` matrices respectively
#  (sanahuja2013 ; barthelemyIJES2019).  The structured types
#  defined here keep that compact storage **plus** the AbstractMatrix
#  interface, so iso / TI / ortho ALV operators can flow through Julia
#  generic matrix code while preserving:
#
#    * memory savings :  iso 18×, TI 6×, ortho 3× cheaper than (6n×6n)
#    * algebra closure :  +, *, volterra_inverse stay in the structured
#      type — the (6n×6n) materialization is needed only at API
#      boundaries.
#
#  Iso ⊂ TI ⊂ ortho ⊂ generic aniso.  Promotion / conversion functions
#  follow this ladder (e.g. `ALVKernelTI(K::ALVKernelISO)`) so mixed
#  arithmetic auto-promotes to the most general operand.
#
#  This is a **prototype** : the types are fully usable on their own
#  (algebra closure verified by tests) but `homogenize_alv` does not
#  yet accept them — pass `Matrix(K)` to materialize.  Full integration
#  with the dispatcher is left to a follow-up PR.
# =============================================================================

"""
    AbstractALVKernel{T} <: AbstractMatrix{T}

Abstract supertype for structured ALV kernel wrappers.  Concrete
subtypes (`ALVKernelISO`, `ALVKernelTI`, `ALVKernelOrtho`) store the
symmetry-class parameters compactly and present a `(6n × 6n)`
AbstractMatrix view via lazy `getindex`.
"""
abstract type AbstractALVKernel{T} <: AbstractMatrix{T} end

# ─── ALVKernelISO ──────────────────────────────────────────────────────────

"""
    ALVKernelISO{T}(α::Matrix{T}, β::Matrix{T})
    ALVKernelISO(M::AbstractMatrix)

Iso ALV kernel: stores the two `n × n` Volterra parameter matrices `α`
and `β` (with `α = 3K(t,t')` and `β = 2μ(t,t')` in Mandel form).
Materializes as a `(6n × 6n)` block matrix on demand via
`Matrix(K)` or `getindex`.
"""
struct ALVKernelISO{T} <: AbstractALVKernel{T}
    α::Matrix{T}
    β::Matrix{T}
    function ALVKernelISO{T}(α::AbstractMatrix, β::AbstractMatrix) where {T}
        size(α) == size(β) ||
            throw(ArgumentError("ALVKernelISO: α and β must have the same size"))
        size(α, 1) == size(α, 2) ||
            throw(ArgumentError("ALVKernelISO: α must be square"))
        return new{T}(Matrix{T}(α), Matrix{T}(β))
    end
end

ALVKernelISO(α::AbstractMatrix{T1}, β::AbstractMatrix{T2}) where {T1, T2} =
    ALVKernelISO{promote_type(T1, T2)}(α, β)

ALVKernelISO(M::AbstractMatrix) =
    (αβ = iso_params_from_blocks(M); ALVKernelISO(αβ[1], αβ[2]))

# ─── ALVKernelTI ───────────────────────────────────────────────────────────

"""
    ALVKernelTI{T}(ℓ::NTuple{6, Matrix{T}}; axis = (0, 0, 1))
    ALVKernelTI(M::AbstractMatrix; axis = (0, 0, 1))

TI ALV kernel: stores the six `n × n` Walpole parameter matrices
`(ℓ₁, ℓ₂, ℓ₃, ℓ₄, ℓ₅, ℓ₆)` with the canonical axis (currently only
`e₃` supported).  Materializes as a `(6n × 6n)` block matrix on demand.
"""
struct ALVKernelTI{T} <: AbstractALVKernel{T}
    ℓ::NTuple{6, Matrix{T}}
    axis::NTuple{3, Float64}
    function ALVKernelTI{T}(
            ℓ::NTuple{6, <:AbstractMatrix},
            axis::NTuple{3, <:Real}
        ) where {T}
        n = size(ℓ[1], 1)
        @inbounds for k in 1:6
            size(ℓ[k]) == (n, n) ||
                throw(ArgumentError("ALVKernelTI: all components must be n×n"))
        end
        return new{T}(
            ntuple(k -> Matrix{T}(ℓ[k]), 6),
            (Float64(axis[1]), Float64(axis[2]), Float64(axis[3]))
        )
    end
end

function ALVKernelTI(
        ℓ::NTuple{6, <:AbstractMatrix};
        axis::NTuple{3, <:Real} = (0.0, 0.0, 1.0)
    )
    T = promote_type(map(eltype, ℓ)...)
    return ALVKernelTI{T}(ℓ, axis)
end

ALVKernelTI(
    M::AbstractMatrix;
    axis::NTuple{3, <:Real} = (0.0, 0.0, 1.0)
) =
    ALVKernelTI(ti_params_from_blocks(M; axis = (Float64.(axis)...,)); axis = axis)

# ─── ALVKernelOrtho ────────────────────────────────────────────────────────

"""
    ALVKernelOrtho{T}(o::NTuple{12, Matrix{T}}; axes = canonical)
    ALVKernelOrtho(M::AbstractMatrix; axes = canonical)

Ortho ALV kernel: stores the twelve `n × n` parameter matrices of the
ortho closure (9 entries of the full unsymmetric 3×3 normal block in
Mandel form + 3 shears) with the canonical material frame
`(e₁, e₂, e₃)`.  Materializes as a `(6n × 6n)` block matrix on demand.
"""
struct ALVKernelOrtho{T} <: AbstractALVKernel{T}
    o::NTuple{12, Matrix{T}}
    axes::NTuple{3, NTuple{3, Float64}}
    function ALVKernelOrtho{T}(
            o::NTuple{12, <:AbstractMatrix},
            axes::NTuple{3, <:NTuple{3, <:Real}}
        ) where {T}
        n = size(o[1], 1)
        @inbounds for k in 1:12
            size(o[k]) == (n, n) ||
                throw(ArgumentError("ALVKernelOrtho: all components must be n×n"))
        end
        return new{T}(
            ntuple(k -> Matrix{T}(o[k]), 12),
            ntuple(
                i -> (
                    Float64(axes[i][1]), Float64(axes[i][2]),
                    Float64(axes[i][3]),
                ), 3
            )
        )
    end
end

const _CANON_AXES_F64 = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

function ALVKernelOrtho(
        o::NTuple{12, <:AbstractMatrix};
        axes::NTuple{3, <:NTuple{3, <:Real}} = _CANON_AXES_F64
    )
    T = promote_type(map(eltype, o)...)
    return ALVKernelOrtho{T}(o, axes)
end

ALVKernelOrtho(
    M::AbstractMatrix;
    axes::NTuple{3, <:NTuple{3, <:Real}} = _CANON_AXES_F64
) =
    ALVKernelOrtho(
    ortho_params_from_blocks(
        M;
        axes = ntuple(i -> (Float64.(axes[i])...,), 3)
    );
    axes = axes
)

# ─── AbstractMatrix interface ──────────────────────────────────────────────

# Number of `(t, t')` time steps stored.
@inline _ntimes(K::ALVKernelISO) = size(K.α, 1)
@inline _ntimes(K::ALVKernelTI) = size(K.ℓ[1], 1)
@inline _ntimes(K::ALVKernelOrtho) = size(K.o[1], 1)

@inline Base.size(K::AbstractALVKernel) = (n = _ntimes(K); (6 * n, 6 * n))
@inline Base.size(K::AbstractALVKernel, d::Integer) = size(K)[d]

@inline LinearAlgebra.istril(::AbstractALVKernel) = true   # Volterra causality

# `getindex` — compute the (i,j) entry of the equivalent (6n × 6n) Mandel
# block matrix lazily from the stored parameters.

@inline function Base.getindex(K::ALVKernelISO, i::Integer, j::Integer)
    @boundscheck checkbounds(K, i, j)
    bi, ki = divrem(i - 1, 6); ki += 1; bi += 1
    bj, kj = divrem(j - 1, 6); kj += 1; bj += 1
    α_ij = K.α[bi, bj]; β_ij = K.β[bi, bj]
    if ki ≤ 3 && kj ≤ 3
        return ki == kj ? (α_ij + 2β_ij) / 3 : (α_ij - β_ij) / 3
    elseif ki ≥ 4 && kj ≥ 4
        return ki == kj ? β_ij : zero(eltype(K))
    else
        return zero(eltype(K))
    end
end

@inline function Base.getindex(K::ALVKernelTI, i::Integer, j::Integer)
    @boundscheck checkbounds(K, i, j)
    bi, ki = divrem(i - 1, 6); ki += 1; bi += 1
    bj, kj = divrem(j - 1, 6); kj += 1; bj += 1
    ℓ₁ = K.ℓ[1][bi, bj]; ℓ₂ = K.ℓ[2][bi, bj]
    ℓ₃ = K.ℓ[3][bi, bj]; ℓ₄ = K.ℓ[4][bi, bj]
    ℓ₅ = K.ℓ[5][bi, bj]; ℓ₆ = K.ℓ[6][bi, bj]
    s2 = sqrt(2)
    if ki ≤ 2 && kj ≤ 2
        # In-plane normal block (1:2, 1:2)
        return ki == kj ? (ℓ₂ + ℓ₅) / 2 : (ℓ₂ - ℓ₅) / 2
    elseif ki == 3 && kj == 3
        return ℓ₁
    elseif ki == 3 && kj ≤ 2
        return ℓ₃ / s2
    elseif ki ≤ 2 && kj == 3
        return ℓ₄ / s2
    elseif (ki == 4 && kj == 4) || (ki == 5 && kj == 5)
        return ℓ₆
    elseif ki == 6 && kj == 6
        return ℓ₅
    else
        return zero(eltype(K))
    end
end

@inline function Base.getindex(K::ALVKernelOrtho, i::Integer, j::Integer)
    @boundscheck checkbounds(K, i, j)
    bi, ki = divrem(i - 1, 6); ki += 1; bi += 1
    bj, kj = divrem(j - 1, 6); kj += 1; bj += 1
    o = K.o
    if ki ≤ 3 && kj ≤ 3
        # Normal 3×3 block: (ki, kj) → (3*(ki-1) + kj)
        return o[3 * (ki - 1) + kj][bi, bj]
    elseif ki == 4 && kj == 4
        return o[10][bi, bj]
    elseif ki == 5 && kj == 5
        return o[11][bi, bj]
    elseif ki == 6 && kj == 6
        return o[12][bi, bj]
    else
        return zero(eltype(K))
    end
end

# ─── Materialization to dense (6n × 6n) ────────────────────────────────────

Base.Matrix(K::ALVKernelISO) = iso_blocks_from_params(K.α, K.β)
Base.Matrix(K::ALVKernelTI) = ti_blocks_from_params(K.ℓ; axis = K.axis)
Base.Matrix(K::ALVKernelOrtho) = ortho_blocks_from_params(K.o; axes = K.axes)

Base.convert(::Type{Matrix}, K::AbstractALVKernel) = Matrix(K)
Base.convert(::Type{Matrix{T}}, K::AbstractALVKernel) where {T} =
    convert(Matrix{T}, Matrix(K))

# ─── Cross-symmetry conversions (iso ⊂ TI ⊂ ortho) ────────────────────────

ALVKernelTI(K::ALVKernelISO) =
    ALVKernelTI(_iso_to_ti((K.α, K.β)); axis = (0.0, 0.0, 1.0))

ALVKernelOrtho(K::ALVKernelISO) =
    ALVKernelOrtho(_iso_to_ortho((K.α, K.β)); axes = _CANON_AXES_F64)

function ALVKernelOrtho(K::ALVKernelTI)
    K.axis == (0.0, 0.0, 1.0) ||
        throw(ArgumentError("ALVKernelOrtho(::ALVKernelTI): only axis = e₃ is supported"))
    return ALVKernelOrtho(_ti_to_ortho(K.ℓ); axes = _CANON_AXES_F64)
end

# ─── Algebra closure within each symmetry class ───────────────────────────

# Iso
Base.:+(A::ALVKernelISO, B::ALVKernelISO) =
    ALVKernelISO(A.α .+ B.α, A.β .+ B.β)
Base.:-(A::ALVKernelISO, B::ALVKernelISO) =
    ALVKernelISO(A.α .- B.α, A.β .- B.β)
Base.:-(A::ALVKernelISO) = ALVKernelISO(-A.α, -A.β)
Base.:*(c::Number, A::ALVKernelISO) = ALVKernelISO(c .* A.α, c .* A.β)
Base.:*(A::ALVKernelISO, c::Number) = c * A
Base.:/(A::ALVKernelISO, c::Number) = ALVKernelISO(A.α ./ c, A.β ./ c)

function Base.:*(A::ALVKernelISO, B::ALVKernelISO)
    αβ_C = _iso_prod((A.α, A.β), (B.α, B.β))
    return ALVKernelISO(αβ_C[1], αβ_C[2])
end

# TI
Base.:+(A::ALVKernelTI, B::ALVKernelTI) =
    (_check_axis_ti(A, B); ALVKernelTI(ntuple(k -> A.ℓ[k] .+ B.ℓ[k], 6); axis = A.axis))
Base.:-(A::ALVKernelTI, B::ALVKernelTI) =
    (_check_axis_ti(A, B); ALVKernelTI(ntuple(k -> A.ℓ[k] .- B.ℓ[k], 6); axis = A.axis))
Base.:-(A::ALVKernelTI) = ALVKernelTI(ntuple(k -> -A.ℓ[k], 6); axis = A.axis)
Base.:*(c::Number, A::ALVKernelTI) =
    ALVKernelTI(ntuple(k -> c .* A.ℓ[k], 6); axis = A.axis)
Base.:*(A::ALVKernelTI, c::Number) = c * A
Base.:/(A::ALVKernelTI, c::Number) =
    ALVKernelTI(ntuple(k -> A.ℓ[k] ./ c, 6); axis = A.axis)

function Base.:*(A::ALVKernelTI, B::ALVKernelTI)
    _check_axis_ti(A, B)
    ℓ_C = _ti_prod(A.ℓ, B.ℓ)
    return ALVKernelTI(ℓ_C; axis = A.axis)
end

@inline function _check_axis_ti(A::ALVKernelTI, B::ALVKernelTI)
    A.axis == B.axis ||
        throw(ArgumentError("ALVKernelTI: incompatible axes $(A.axis) ≠ $(B.axis)"))
    return nothing
end

# Ortho
Base.:+(A::ALVKernelOrtho, B::ALVKernelOrtho) =
    (_check_axes_ortho(A, B); ALVKernelOrtho(ntuple(k -> A.o[k] .+ B.o[k], 12); axes = A.axes))
Base.:-(A::ALVKernelOrtho, B::ALVKernelOrtho) =
    (_check_axes_ortho(A, B); ALVKernelOrtho(ntuple(k -> A.o[k] .- B.o[k], 12); axes = A.axes))
Base.:-(A::ALVKernelOrtho) = ALVKernelOrtho(ntuple(k -> -A.o[k], 12); axes = A.axes)
Base.:*(c::Number, A::ALVKernelOrtho) =
    ALVKernelOrtho(ntuple(k -> c .* A.o[k], 12); axes = A.axes)
Base.:*(A::ALVKernelOrtho, c::Number) = c * A
Base.:/(A::ALVKernelOrtho, c::Number) =
    ALVKernelOrtho(ntuple(k -> A.o[k] ./ c, 12); axes = A.axes)

function Base.:*(A::ALVKernelOrtho, B::ALVKernelOrtho)
    _check_axes_ortho(A, B)
    o_C = _ortho_prod(A.o, B.o)
    return ALVKernelOrtho(o_C; axes = A.axes)
end

@inline function _check_axes_ortho(A::ALVKernelOrtho, B::ALVKernelOrtho)
    A.axes == B.axes ||
        throw(ArgumentError("ALVKernelOrtho: incompatible material frames"))
    return nothing
end

# ─── Cross-symmetry arithmetic (auto-promote up the iso ⊂ TI ⊂ ortho ladder) ──

# Iso × TI → TI (and symmetric)
for OP in (:+, :-, :*)
    @eval Base.$OP(A::ALVKernelISO, B::ALVKernelTI) = $OP(ALVKernelTI(A), B)
    @eval Base.$OP(A::ALVKernelTI, B::ALVKernelISO) = $OP(A, ALVKernelTI(B))
    @eval Base.$OP(A::ALVKernelISO, B::ALVKernelOrtho) = $OP(ALVKernelOrtho(A), B)
    @eval Base.$OP(A::ALVKernelOrtho, B::ALVKernelISO) = $OP(A, ALVKernelOrtho(B))
    @eval Base.$OP(A::ALVKernelTI, B::ALVKernelOrtho) = $OP(ALVKernelOrtho(A), B)
    @eval Base.$OP(A::ALVKernelOrtho, B::ALVKernelTI) = $OP(A, ALVKernelOrtho(B))
end

# ─── Volterra-specific operations ──────────────────────────────────────────

"""
    volterra_inverse(K::AbstractALVKernel) -> AbstractALVKernel

Volterra inverse of a structured ALV kernel.  Stays in the same
symmetry class : iso ↦ iso, TI ↦ TI, ortho ↦ ortho.  Avoids
materializing the `(6n × 6n)` matrix.
"""
function volterra_inverse(K::ALVKernelISO)
    αβ = _iso_inv((K.α, K.β))
    return ALVKernelISO(αβ[1], αβ[2])
end

function volterra_inverse(K::ALVKernelTI)
    ℓ = _ti_inv(K.ℓ)
    return ALVKernelTI(ℓ; axis = K.axis)
end

function volterra_inverse(K::ALVKernelOrtho)
    o = _ortho_inv(K.o)
    return ALVKernelOrtho(o; axes = K.axes)
end

"""
    volterra_left_divide(S::AbstractALVKernel, M::AbstractALVKernel)
        -> AbstractALVKernel

Volterra left-divide `T = S^{-vol} ∘ M` within the structured class.
Auto-promotes mixed inputs (e.g. iso S + TI M) to the more general
class before solving.
"""
function volterra_left_divide(S::ALVKernelISO, M::ALVKernelISO)
    αβ = _iso_left_divide((S.α, S.β), (M.α, M.β))
    return ALVKernelISO(αβ[1], αβ[2])
end

function volterra_left_divide(S::ALVKernelTI, M::ALVKernelTI)
    _check_axis_ti(S, M)
    ℓ = _ti_left_divide(S.ℓ, M.ℓ)
    return ALVKernelTI(ℓ; axis = S.axis)
end

function volterra_left_divide(S::ALVKernelOrtho, M::ALVKernelOrtho)
    _check_axes_ortho(S, M)
    o = _ortho_left_divide(S.o, M.o)
    return ALVKernelOrtho(o; axes = S.axes)
end

# Mixed: promote up the ladder before solving
volterra_left_divide(S::ALVKernelISO, M::ALVKernelTI) = volterra_left_divide(ALVKernelTI(S), M)
volterra_left_divide(S::ALVKernelTI, M::ALVKernelISO) = volterra_left_divide(S, ALVKernelTI(M))
volterra_left_divide(S::ALVKernelISO, M::ALVKernelOrtho) = volterra_left_divide(ALVKernelOrtho(S), M)
volterra_left_divide(S::ALVKernelOrtho, M::ALVKernelISO) = volterra_left_divide(S, ALVKernelOrtho(M))
volterra_left_divide(S::ALVKernelTI, M::ALVKernelOrtho) = volterra_left_divide(ALVKernelOrtho(S), M)
volterra_left_divide(S::ALVKernelOrtho, M::ALVKernelTI) = volterra_left_divide(S, ALVKernelOrtho(M))

# ─── Pretty printing ───────────────────────────────────────────────────────

function Base.show(io::IO, ::MIME"text/plain", K::ALVKernelISO{T}) where {T}
    n = _ntimes(K)
    println(
        io, "ALVKernelISO{$T} — iso ALV kernel, n = $n time steps, ",
        "size $(6n)×$(6n), 2 stored components (α, β)"
    )
    println(io, "  α (= 3K(t,t')) :")
    show(io, MIME"text/plain"(), K.α)
    println(io, "\n  β (= 2μ(t,t')) :")
    return show(io, MIME"text/plain"(), K.β)
end

function Base.show(io::IO, ::MIME"text/plain", K::ALVKernelTI{T}) where {T}
    n = _ntimes(K)
    return println(
        io, "ALVKernelTI{$T} — TI ALV kernel (axis $(K.axis)), n = $n, ",
        "size $(6n)×$(6n), 6 stored Walpole components"
    )
end

function Base.show(io::IO, ::MIME"text/plain", K::ALVKernelOrtho{T}) where {T}
    n = _ntimes(K)
    return println(
        io, "ALVKernelOrtho{$T} — ortho ALV kernel, n = $n, ",
        "size $(6n)×$(6n), 12 stored components (9 normal + 3 shear)"
    )
end
