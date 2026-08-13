# =============================================================================
#  order2_alv.jl — order-2 (vector-tensor) ageing linear viscoelasticity.
#
#  Mirrors the order-4 ALV machinery for second-order properties such as
#  thermal / electric conductivity, diffusivity, electrical permittivity,
#  etc.  All operators are stored as `(3n × 3n)` lower-block-triangular
#  matrices with 3×3 blocks (no Mandel scaling — the order-2 algebra is
#  matrix-direct).
#
#  Reproduces the order-2 `homogenize_visco(prop="Y", unitsize=3, …)`
#  workflow.
# =============================================================================

# ── Trapezoidal fillers for order-2 (3×3 block) kernels ─────────────────────
# Defined in trapezoidal.jl entry points; bodies live here for cohesion.

@inline function _fill_trapezoidal_order2_tens!(
        M::AbstractMatrix, law::ViscoLaw,
        times::AbstractVector
    )
    n = length(times)
    n == 0 && return M
    T = eltype(M)
    cache = Vector{Matrix{T}}(undef, n)
    @inbounds begin
        cache[1] = _to_order2_mat(visco_eval(law, times[1], times[1]))
        _set_block_order2!(M, 1, 1, cache[1])
        for i in 2:n
            for k in 1:i
                cache[k] = _to_order2_mat(visco_eval(law, times[i], times[k]))
            end
            _fill_row_blocks_order2_from_cache!(M, cache, i, T)
        end
    end
    return M
end

@inline function _fill_trapezoidal_order2_mat!(
        M::AbstractMatrix, law::ViscoLaw,
        times::AbstractVector
    )
    n = length(times)
    n == 0 && return M
    T = eltype(M)
    cache = Vector{Matrix{T}}(undef, n)
    @inbounds begin
        cache[1] = copy(visco_eval(law, times[1], times[1]))
        _set_block_order2!(M, 1, 1, cache[1])
        for i in 2:n
            for k in 1:i
                cache[k] = visco_eval(law, times[i], times[k])
            end
            _fill_row_blocks_order2_from_cache!(M, cache, i, T)
        end
    end
    return M
end

# Place row `i` blocks (3×3 each) into `M` using cached evaluations.
@inline function _fill_row_blocks_order2_from_cache!(
        M::AbstractMatrix,
        cache::Vector{<:AbstractMatrix},
        i::Int, ::Type{T}
    ) where {T}
    half = inv(T(2))
    @inbounds begin
        c1, c2 = cache[1], cache[2]
        for kk in 1:3, ll in 1:3
            M[3 * (i - 1) + kk, ll] = (c1[kk, ll] - c2[kk, ll]) * half
        end
        for j in 2:(i - 1)
            cm, cp = cache[j - 1], cache[j + 1]
            for kk in 1:3, ll in 1:3
                M[3 * (i - 1) + kk, 3 * (j - 1) + ll] =
                    (cm[kk, ll] - cp[kk, ll]) * half
            end
        end
        cm, cp = cache[i - 1], cache[i]
        for kk in 1:3, ll in 1:3
            M[3 * (i - 1) + kk, 3 * (i - 1) + ll] =
                (cm[kk, ll] + cp[kk, ll]) * half
        end
    end
    return M
end

@inline _to_order2_mat(s::AbstractMatrix) = s
@inline function _to_order2_mat(s::TensND.AbstractTens{2, 3})
    return TensND.get_array(s)
end

@inline function _set_block_order2!(
        M::AbstractMatrix, i::Int, j::Int,
        block::AbstractMatrix
    )
    rows = (3 * (i - 1) + 1):(3 * i)
    cols = (3 * (j - 1) + 1):(3 * j)
    @inbounds M[rows, cols] = block
    return M
end

# ── Iso order-2 parameter extraction (single scalar α per (i,j)) ───────────

"""
    iso_order2_params_from_blocks(M) -> α::Matrix

Decompose a `(3n × 3n)` block matrix whose every 3×3 block is `α[i,j]·𝐈`
(iso 2-tensor) into the scalar `n × n` Volterra matrix `α`.
"""
function iso_order2_params_from_blocks(M::AbstractMatrix)
    sz = size(M, 1)
    sz == size(M, 2) ||
        throw(ArgumentError("iso_order2_params_from_blocks: M must be square"))
    sz % 3 == 0 ||
        throw(ArgumentError("iso_order2_params_from_blocks: size $(sz) not divisible by 3"))
    n = sz ÷ 3
    T = eltype(M)
    α = zeros(T, n, n)
    @inbounds for i in 1:n, j in 1:n
        r = 3 * (i - 1)
        c = 3 * (j - 1)
        # Average of diagonal entries of the 3×3 block.
        α[i, j] = (M[r + 1, c + 1] + M[r + 2, c + 2] + M[r + 3, c + 3]) / 3
    end
    return α
end

"""
    iso_order2_blocks_from_params(α::AbstractMatrix) -> Matrix

Inverse of [`iso_order2_params_from_blocks`](@ref): build a `(3n × 3n)`
block matrix `α[i,j]·𝐈` per block.
"""
function iso_order2_blocks_from_params(α::AbstractMatrix)
    n = size(α, 1)
    n == size(α, 2) ||
        throw(ArgumentError("iso_order2_blocks_from_params: α must be square"))
    T = eltype(α)
    M = zeros(T, 3 * n, 3 * n)
    @inbounds for i in 1:n, j in 1:n
        a = T(α[i, j])
        rows = (3 * (i - 1) + 1):(3 * i)
        cols = (3 * (j - 1) + 1):(3 * j)
        M[rows[1], cols[1]] = a
        M[rows[2], cols[2]] = a
        M[rows[3], cols[3]] = a
    end
    return M
end

"""
    _is_iso_order2_block(M; tol = 1e-12) -> Bool

Return `true` if every 3×3 block of `M` is `α·𝐈` (iso 2-tensor in 3D).
"""
function _is_iso_order2_block(M::AbstractMatrix; tol::Real = 1.0e-12)
    sz = size(M, 1)
    sz == size(M, 2) || return false
    sz % 3 == 0 || return false
    n = sz ÷ 3
    iszero(n) && return true
    scale = max(maximum(abs, M), one(real(eltype(M))))
    abstol = tol * scale
    @inbounds for i in 1:n, j in 1:n
        r = 3 * (i - 1); c = 3 * (j - 1)
        a = (M[r + 1, c + 1] + M[r + 2, c + 2] + M[r + 3, c + 3]) / 3
        for k in 1:3, l in 1:3
            expected = (k == l) ? a : zero(a)
            abs(M[r + k, c + l] - expected) ≤ abstol || return false
        end
    end
    return true
end

# ── Hill order-2 kernel for iso ALV matrix + ellipsoidal inclusion ─────────
#
# Time-space decoupling: for an iso ALV matrix with conductivity α₀(t,t'),
#     P̃[block(i, j)] = α₀^{-vol}[i, j] · 𝐈^A
# where 𝐈^A is the purely geometric depolarization tensor of the ellipsoid
# (`tens_IA(ell)` in the canonical principal-axis frame, sums to 1).

"""
    hill_kernel_order2(ell, K_0_law::ViscoLaw, times) -> Matrix

Build the discrete Hill kernel `P̃` (size `(3n × 3n)`) for an
ellipsoidal inclusion in an isotropic ALV matrix.  Uses the time-space
decoupling formula  `P̃[block(i,j)] = α₀^{-vol}[i,j] · 𝐈^A`.
"""
function hill_kernel_order2(
        ell, K_0_law::ViscoLaw,
        times::AbstractVector{<:Real}
    )
    return hill_kernel_order2_at(ell, _trapezoidal_relaxation(K_0_law, times, 3))
end

"""
    hill_kernel_order2_at(ell, K_0::AbstractMatrix) -> Matrix

Variant of [`hill_kernel_order2`](@ref) taking an already-discretized
`(3n × 3n)` reference matrix instead of the matrix law — what the
differential scheme needs, its reference being the *running* effective
medium.  Same `_at` convention as the order-4 crack and layered-sphere
kernels.
"""
function hill_kernel_order2_at(ell, K_0::AbstractMatrix)
    _is_iso_order2_block(K_0) ||
        throw(ArgumentError("hill_kernel_order2: only iso ALV matrix is currently supported"))
    α_0 = iso_order2_params_from_blocks(K_0)
    α_0_inv = volterra_inverse(α_0; block_size = 1)
    IA = TensND.get_array(Elasticity.tens_IA(ell))
    n = size(α_0, 1)
    T = promote_type(eltype(α_0_inv), eltype(IA))
    P = zeros(T, 3 * n, 3 * n)
    @inbounds for i in 1:n, j in 1:n
        rows = (3 * (i - 1) + 1):(3 * i)
        cols = (3 * (j - 1) + 1):(3 * j)
        v = T(α_0_inv[i, j])
        for k in 1:3, l in 1:3
            P[rows[k], cols[l]] = v * IA[k, l]
        end
    end
    return P
end

# ── Generic order-2 ALV algebra (3n × 3n with block_size = 3) ──────────────

"""
    dilute_concentration_alv_order2(K_E, K_0, P) -> Matrix

Order-2 dilute concentration `Ã^dil = (𝟙 + P̃ ∘ ΔK̃)^{-vol}`,
all matrices `(3n × 3n)`.
"""
function dilute_concentration_alv_order2(
        K_E::AbstractMatrix, K_0::AbstractMatrix,
        P::AbstractMatrix
    )
    sz = size(K_E, 1)
    sz % 3 == 0 ||
        throw(ArgumentError("dilute_concentration_alv_order2: size not divisible by 3"))
    n = sz ÷ 3
    T = promote_type(eltype(K_E), eltype(K_0), eltype(P))
    Id = zeros(T, sz, sz)
    @inbounds for i in 1:n
        rows = (3 * (i - 1) + 1):(3 * i)
        Id[rows, rows] = Matrix{T}(LinearAlgebra.I, 3, 3)
    end
    arg = Id .+ P * (K_E .- K_0)
    return volterra_inverse(arg; block_size = 3)
end

"""
    dilute_contribution_alv_order2(K_E, K_0, P) -> Matrix

Order-2 dilute contribution `Ñ = ΔK̃ ∘ Ã^dil`.
"""
function dilute_contribution_alv_order2(
        K_E::AbstractMatrix, K_0::AbstractMatrix,
        P::AbstractMatrix
    )
    A_dil = dilute_concentration_alv_order2(K_E, K_0, P)
    return (K_E .- K_0) * A_dil
end

# ── Schemes ─────────────────────────────────────────────────────────────────

# Element type spanning EVERY per-phase block a kernel accumulates, not just
# the first one or the matrix block.  The kernels below allocate their
# accumulators up front, so a block the promotion misses is rejected by the
# broadcast that adds it: differentiating with respect to a phase property or
# an inclusion geometry makes `contribs` / `A_duts` (and any phase's `K̃_r`)
# `Dual` while `K̃_0` and the fractions stay `Float64`.  `Bool` is the neutral
# seed — it promotes away against any numeric type.
_alv2_blocks_eltype(mats) = mapreduce(eltype, promote_type, mats; init = Bool)

"""
    voigt_alv_order2(matrices, fractions) -> Matrix

Order-2 Voigt bound: `K̃_eff = Σ_r f_r K̃_r`.
"""
function voigt_alv_order2(
        matrices::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector
    )
    length(matrices) == length(fractions) ||
        throw(ArgumentError("voigt_alv_order2: phase counts mismatch"))
    isempty(matrices) && throw(ArgumentError("voigt_alv_order2: at least one phase required"))
    T = promote_type(_alv2_blocks_eltype(matrices), eltype(fractions))
    out = zeros(T, size(matrices[1])...)
    @inbounds for r in eachindex(matrices)
        @. out += fractions[r] * matrices[r]
    end
    return out
end

"""
    reuss_alv_order2(matrices, fractions) -> Matrix

Order-2 Reuss bound: invert each compliance, average, invert back.
"""
function reuss_alv_order2(
        matrices::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector
    )
    length(matrices) == length(fractions) ||
        throw(ArgumentError("reuss_alv_order2: phase counts mismatch"))
    inv_phases = [volterra_inverse(M; block_size = 3) for M in matrices]
    inv_eff = voigt_alv_order2(inv_phases, fractions)
    return volterra_inverse(inv_eff; block_size = 3)
end

"""
    dilute_alv_order2(K_0, contribs, fractions) -> Matrix

Order-2 Dilute scheme: `K̃_eff = K̃_0 + Σ_r f_r Ñ_r`.
"""
function dilute_alv_order2(
        K_0::AbstractMatrix,
        contribs::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector
    )
    length(contribs) == length(fractions) ||
        throw(ArgumentError("dilute_alv_order2: phase counts mismatch"))
    T = promote_type(
        eltype(K_0), _alv2_blocks_eltype(contribs), eltype(fractions)
    )
    out = T.(K_0)
    @inbounds for r in eachindex(contribs)
        @. out += fractions[r] * contribs[r]
    end
    return out
end

"""
    dilute_dual_alv_order2(K_0, contribs_compliance, fractions) -> Matrix

Order-2 DiluteDual: invert to compliance space, average, invert back.
"""
function dilute_dual_alv_order2(
        K_0::AbstractMatrix,
        contribs_compliance::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector
    )
    R_0 = volterra_inverse(K_0; block_size = 3)
    R_eff = dilute_alv_order2(R_0, contribs_compliance, fractions)
    return volterra_inverse(R_eff; block_size = 3)
end

"""
    mori_tanaka_alv_order2(K_0, A_duts, contribs, fractions, f_M) -> Matrix

Order-2 Mori-Tanaka:
   `K̃_eff = K̃_0 + (Σ_r f_r Ñ_r) ∘ (f_0 𝟙 + Σ_s f_s Ã_s)^{-vol}`.
"""
function mori_tanaka_alv_order2(
        K_0::AbstractMatrix,
        A_duts::AbstractVector{<:AbstractMatrix},
        contribs::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector, f_M::Real
    )
    length(A_duts) == length(contribs) == length(fractions) ||
        throw(ArgumentError("mori_tanaka_alv_order2: phase counts mismatch"))
    sz = size(K_0, 1)
    sz % 3 == 0 || throw(ArgumentError("mori_tanaka_alv_order2: size not divisible by 3"))
    n = sz ÷ 3
    T = promote_type(
        eltype(K_0), eltype(fractions), typeof(f_M),
        _alv2_blocks_eltype(A_duts), _alv2_blocks_eltype(contribs)
    )
    Id = zeros(T, sz, sz)
    @inbounds for i in 1:n
        rows = (3 * (i - 1) + 1):(3 * i)
        Id[rows, rows] = Matrix{T}(LinearAlgebra.I, 3, 3)
    end
    num = zeros(T, sz, sz)
    den = T(f_M) .* Id
    @inbounds for r in eachindex(A_duts)
        @. num += fractions[r] * contribs[r]
        @. den += fractions[r] * A_duts[r]
    end
    factor = volterra_left_divide(den, num; block_size = 3)
    return K_0 .+ factor
end

"""
    maxwell_alv_order2(K_0, contribs, fractions; H_0) -> Matrix

Order-2 Maxwell scheme.  `H_0` is the Hill kernel of the (matrix-only)
distribution shape — defaults to a sphere when not specified by
[`homogenize_alv_order2`](@ref).
"""
function maxwell_alv_order2(
        K_0::AbstractMatrix,
        contribs::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector;
        H_0::AbstractMatrix
    )
    length(contribs) == length(fractions) ||
        throw(ArgumentError("maxwell_alv_order2: phase counts mismatch"))
    sz = size(K_0, 1)
    sz % 3 == 0 || throw(ArgumentError("maxwell_alv_order2: size not divisible by 3"))
    n = sz ÷ 3
    T = promote_type(
        eltype(K_0), eltype(fractions), eltype(H_0),
        _alv2_blocks_eltype(contribs)
    )
    Id = zeros(T, sz, sz)
    @inbounds for i in 1:n
        rows = (3 * (i - 1) + 1):(3 * i)
        Id[rows, rows] = Matrix{T}(LinearAlgebra.I, 3, 3)
    end
    Σ = zeros(T, sz, sz)
    @inbounds for r in eachindex(contribs)
        @. Σ += fractions[r] * contribs[r]
    end
    factor = Σ * volterra_inverse(Id .- H_0 * Σ; block_size = 3)
    return K_0 .+ factor
end

# ── Order-2 ALV pipeline (internal — dispatched from homogenize_alv) ───────

"""
    _homogenize_alv_order2(rve, scheme, prop::Symbol; times) -> Matrix

Internal order-2 ALV pipeline.  Reached from [`homogenize_alv`](@ref)
when the matrix property law samples to a 3×3 / `TensND.AbstractTens{2,3}`
value.  Returns the effective `K̃_eff` of size `(3n × 3n)`.

Supports iso ALV matrix + ellipsoidal inclusions of any aspect ratio.
The result is generally anisotropic (TI for spheroids, ortho for
triaxial ellipsoids).
"""
function _homogenize_alv_order2(
        rve::RVE, scheme::HomogenizationScheme,
        prop::Symbol; times::AbstractVector{<:Real}, kw...
    )
    K_M_law = matrix_property(rve, prop)
    K_M_law isa ViscoLaw ||
        throw(ArgumentError("homogenize_alv_order2: matrix property $prop is not a ViscoLaw"))
    K_0 = _trapezoidal_relaxation(K_M_law, times, 3)
    f_M = matrix_volume_fraction(rve)

    incl_names = inclusion_phase_names(rve)
    # `fractions` must carry whatever element type the RVE amounts store —
    # typically `Float64` but also `ForwardDiff.Dual` for autodiff
    # sensitivities via `set_param(rve, AmountParameter(...), Dual(...))`.
    # A hard-coded `Float64[]` here silently breaks AD through volume
    # fractions in this order-2 (conductivity/diffusion) path, unlike the
    # order-4 path which already promotes correctly (`homogenize_alv.jl`).
    T_amount = isempty(incl_names) ? Float64 :
        promote_type((typeof(_amount_value(rve, n)) for n in incl_names)...)

    # The per-phase blocks are computed BEFORE their containers are typed:
    # their element type is not implied by the matrix law either.
    # Differentiating with respect to a phase property (through `K_r`) or an
    # inclusion geometry (through the Hill kernel `P_r`, hence `A_dut` /
    # `N_dut`) makes them `Dual` while `K_0` stays `Float64`, and a container
    # typed from `eltype(K_0)` alone rejects them on `push!` — the same trap
    # as the hard-coded `Float64[]` documented above for `fractions`.
    per_phase = map(incl_names) do name
        ph = rve.phases[name]
        K_r_law = phase_property(rve, name, prop)
        K_r_law isa ViscoLaw ||
            throw(ArgumentError("homogenize_alv_order2: phase $name property is not a ViscoLaw"))
        K_r = _trapezoidal_relaxation(K_r_law, times, 3)
        P_r = hill_kernel_order2(ph.geometry, K_M_law, times)
        # `symmetrize` is honored on the dilute quantities, exactly as the
        # order-4 pipeline does (`homogenize_alv.jl`), and with the order-2
        # projector: the blocks here are 2-tensors, not 6×6 Mandel blocks.
        sym = phase_symmetrize(rve, name)
        return (
            K_r = K_r,
            A_dut = _maybe_symmetrize_alv2(
                dilute_concentration_alv_order2(K_r, K_0, P_r), sym
            ),
            N_dut = _maybe_symmetrize_alv2(
                dilute_contribution_alv_order2(K_r, K_0, P_r), sym
            ),
        )
    end
    T_block = isempty(per_phase) ? eltype(K_0) :
        promote_type(
            eltype(K_0),
            (
                promote_type(eltype(q.K_r), eltype(q.A_dut), eltype(q.N_dut))
                for q in per_phase
            )...
        )

    fractions = T_amount[_amount_value(rve, name) for name in incl_names]
    K_phases = Matrix{T_block}[K_0]
    A_duts = Matrix{T_block}[]
    contribs = Matrix{T_block}[]
    for q in per_phase
        push!(K_phases, q.K_r)
        push!(A_duts, q.A_dut)
        push!(contribs, q.N_dut)
    end

    return _homogenize_alv2_dispatch(
        rve, scheme, prop, times,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law; kw...
    )
end

# Dispatch table for order-2 schemes.

function _homogenize_alv2_dispatch(
        ::RVE, ::Voigt, ::Symbol, ::AbstractVector,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law; kw...
    )
    return voigt_alv_order2(K_phases, [f_M; fractions])
end

function _homogenize_alv2_dispatch(
        ::RVE, ::Reuss, ::Symbol, ::AbstractVector,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law; kw...
    )
    return reuss_alv_order2(K_phases, [f_M; fractions])
end

function _homogenize_alv2_dispatch(
        ::RVE, ::Dilute, ::Symbol, ::AbstractVector,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law; kw...
    )
    return dilute_alv_order2(K_0, contribs, fractions)
end

function _homogenize_alv2_dispatch(
        ::RVE, ::DiluteDual, ::Symbol, ::AbstractVector,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law; kw...
    )
    # The dual scheme accumulates in RESISTIVITY space, but `contribs`
    # carries the conductivity contributions Ñ.  Map them with the exact
    # identity (the order-2 twin of the order-4 rule in
    # `homogenize_alv.jl`) before averaging:
    #     R̃_r = −R̃_0 ∘ Ñ_r ∘ R̃_0 ,   R̃_0 = K̃_0^{-vol} ,
    # which is `(R_r − R_0) ∘ B̃^dil` written out.  Returning the
    # relaxation-side `dilute_alv_order2` here instead — as this dispatch
    # used to — made `DiluteDual` a silent alias of `Dilute`.
    R_0 = volterra_inverse(K_0; block_size = 3)
    contribs_R = [-(R_0 * N̄ * R_0) for N̄ in contribs]
    return dilute_dual_alv_order2(K_0, contribs_R, fractions)
end

function _homogenize_alv2_dispatch(
        ::RVE, ::MoriTanaka, ::Symbol, ::AbstractVector,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law; kw...
    )
    return mori_tanaka_alv_order2(K_0, A_duts, contribs, fractions, f_M)
end

function _homogenize_alv2_dispatch(
        rve::RVE, ::Maxwell, ::Symbol,
        times::AbstractVector,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law; kw...
    )
    # Default distribution shape: spherical
    H_0 = hill_kernel_order2(Spheroid(1.0), K_M_law, times)
    return maxwell_alv_order2(K_0, contribs, fractions; H_0 = H_0)
end

function _homogenize_alv2_dispatch(
        rve::RVE, sch::DifferentialScheme, prop::Symbol,
        times::AbstractVector,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law; kw...
    )
    return differential_alv_order2(
        rve, prop; times = times,
        _diff_alv_options(sch)...
    )
end

# =============================================================================
#  Differential ALV, order 2 (viscous conduction / diffusion) — SciML ODE on
#  the fictitious incorporation time τ, mirror of the order-4
#  `differential_alv`:
#
#      dK̃/dτ = Σ_α dφ_α/dτ · (K̃_α − K̃) ∘ Ã_α^dil(K̃)
#
#  with the same Sherman-Morrison volume balance and the same
#  trajectories.  State: `vec(K̃)` of size `(3n)²`.
# =============================================================================

"""
    differential_alv_order2(rve, prop::Symbol; times, nsteps = 100,
                            trajectory = nothing, abstol = 1e-8,
                            reltol = 1e-6, alg = nothing,
                            formulation = :stiffness) -> Matrix

Order-2 (conductivity / diffusion) counterpart of
[`differential_alv`](@ref): the same incorporation-sequence ODE on
`τ ∈ [0, 1]`, integrated on the `(3n × 3n)` ALV conductivity block
matrix.

`formulation = :compliance` integrates the resistivity `R̃ = K̃^{-vol}`
instead, through `H̃_α = −R̃ ∘ Ñ_α ∘ R̃`, and inverts the result.

As in the order-4 case the reference of the ODE is the running
effective medium, and the order-2 ALV Hill kernel exists for an
isotropic reference only — every inclusion must therefore be spherical
or carry an isotropic orientation average.
"""
function differential_alv_order2(
        rve::RVE, prop::Symbol;
        times::AbstractVector{<:Real},
        nsteps::Int = 100,
        trajectory = nothing,
        abstol::Real = 1.0e-8,
        reltol::Real = 1.0e-6,
        alg = nothing,
        formulation::Symbol = :stiffness,
        solver_kwargs::NamedTuple = NamedTuple()
    )
    formulation in (:stiffness, :compliance) ||
        throw(
        ArgumentError(
            "differential_alv_order2: formulation must be :stiffness or " *
                ":compliance; got :$(formulation)"
        )
    )
    K_M_law = matrix_property(rve, prop)
    K_M_law isa ViscoLaw ||
        throw(ArgumentError("differential_alv_order2: matrix property is not a ViscoLaw"))
    K_0 = _trapezoidal_relaxation(K_M_law, times, 3)
    _is_iso_order2_block(K_0) ||
        throw(
        ArgumentError(
            "differential_alv_order2: the ALV matrix must be isotropic (the " *
                "differential ODE evaluates the order-2 ALV Hill kernel against " *
                "its running effective medium, and that kernel exists for an " *
                "isotropic reference only)."
        )
    )
    n = length(times)

    solid_data = NamedTuple[]
    for name in inclusion_phase_names(rve)
        amt = rve.amounts[name]
        amt isa VolumeFraction ||
            throw(
            ArgumentError(
                "differential_alv_order2: phase :$(name) carries a crack " *
                    "density; order-2 ALV cracks are not supported by the " *
                    "differential scheme."
            )
        )
        ph = rve.phases[name]
        sym = phase_symmetrize(rve, name)
        K_r_law = phase_property(rve, name, prop)
        K_r_law isa ViscoLaw ||
            throw(ArgumentError("differential_alv_order2: phase $name property is not a ViscoLaw"))
        K_r = _trapezoidal_relaxation(K_r_law, times, 3)
        _alv_diff_keeps_iso(ph.geometry, sym, nothing) ||
            _alv_diff_iso_error(name, "the shape")
        push!(
            solid_data, (
                name = name, geom = ph.geometry, K_r = K_r,
                target = _amount_value(rve, name), sym = sym,
            )
        )
    end

    paths = trajectory === nothing ?
        _resolve_paths_alv(Schemes.Proportional(), rve, nsteps) :
        _resolve_paths_alv(trajectory, rve, nsteps)

    # State eltype: the matrix law, the phase laws and the targets are not
    # the only inputs the RHS touches — it also rebuilds the order-2 ALV Hill
    # kernel against the running medium at every step
    # (`hill_kernel_order2_at(r.geom, K_curr)`), so an inclusion GEOMETRY
    # reaches the state through that kernel alone.  Probing it once per phase
    # (against hundreds of evaluations during the integration) reports the
    # eltype it actually produces, exactly as `T_contrib` does for the
    # elastic `DifferentialScheme`.
    Tp = eltype(K_0)
    for sd in solid_data
        Tp = promote_type(
            Tp, eltype(sd.K_r), typeof(sd.target),
            eltype(hill_kernel_order2_at(sd.geom, K_0))
        )
    end
    sz = 3 * n
    dual = formulation === :compliance
    x0 = vec(Tp.(dual ? volterra_inverse(K_0; block_size = 3) : K_0))
    ode_p = (n = n, sz = sz, solid_data = solid_data, paths = paths, dual = dual)
    rhs! = (du, u, p, τ) -> _diff_alv2_ode_rhs!(du, u, p, τ)
    prob = ODEProblem(rhs!, x0, (0.0, 1.0), ode_p)
    sol = solve(
        prob,
        alg === nothing ? Tsit5() : alg;
        abstol, reltol,
        saveat = range(0.0, 1.0; length = max(nsteps, 1) + 1),
        dense = false,
        solver_kwargs...
    )
    P_end = reshape(sol.u[end], sz, sz)
    return dual ? volterra_inverse(P_end; block_size = 3) : P_end
end

function _diff_alv2_ode_rhs!(du, u, p, τ)
    sz = p.sz
    P_curr = reshape(u, sz, sz)
    K_curr = p.dual ? volterra_inverse(P_curr; block_size = 3) : P_curr
    Δ = zeros(eltype(u), sz, sz)

    n_solid = length(p.solid_data)
    n_solid == 0 && (du .= vec(Δ); return nothing)

    # Sherman-Morrison : dφ_α/dτ = df_α/dτ + (f_α / f_0) · sum(df).
    f = Vector{eltype(u)}(undef, n_solid)
    df = Vector{eltype(u)}(undef, n_solid)
    @inbounds for (i, r) in enumerate(p.solid_data)
        nt = p.paths[r.name]
        f[i] = nt.f(τ) * r.target
        df[i] = nt.df(τ) * r.target
    end
    f0 = one(eltype(u)) - sum(f; init = zero(eltype(u)))
    sum_df = sum(df; init = zero(eltype(u)))

    @inbounds for (i, r) in enumerate(p.solid_data)
        dφᵢ = df[i] + (f[i] / f0) * sum_df
        iszero(dφᵢ) && continue
        P_r = hill_kernel_order2_at(r.geom, K_curr)
        contrib = dilute_contribution_alv_order2(r.K_r, K_curr, P_r)
        # Order-2 projector: `K_curr` is a (3n × 3n) matrix of 2-tensor
        # blocks.  Using the order-4 (6×6 Mandel) one here silently mixed two
        # consecutive TIME blocks, so the "averaged" contribution was not
        # isotropic and the running medium drifted out of the class the
        # order-2 Hill kernel needs.
        contrib = _maybe_symmetrize_alv2(contrib, r.sym)
        term = p.dual ? -(P_curr * contrib * P_curr) : contrib
        @. Δ += dφᵢ * term
    end

    du .= vec(Δ)
    return nothing
end

"""
    homogenize_alv_order2(rve, scheme, prop; times)

Backwards-compatible alias for [`homogenize_alv`](@ref) when the matrix
property is order-2.  New code should call `homogenize_alv` directly —
the dispatch on order-2 vs order-4 is automatic from the law sample.
"""
homogenize_alv_order2(
    rve::RVE, scheme::HomogenizationScheme,
    prop::Symbol; kw...
) =
    homogenize_alv(rve, scheme, prop; kw...)
