# =============================================================================
#  order2_alv.jl — order-2 (vector-tensor) aging linear viscoelasticity.
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

_fill_trapezoidal_order2_tens!(M::AbstractMatrix, law::ViscoLaw, times::AbstractVector) =
    _fill_trapezoidal_blocks!(M, law, times, Val(3), _to_order2_mat)
_fill_trapezoidal_order2_mat!(M::AbstractMatrix, law::ViscoLaw, times::AbstractVector) =
    _fill_trapezoidal_blocks!(M, law, times, Val(3), identity)

@inline _to_order2_mat(s::AbstractMatrix) = s
@inline function _to_order2_mat(s::TensND.AbstractTens{2, 3})
    return TensND.canonical_components(s)   # the ALV matrices are canonical
end

# ── Iso order-2 parameter extraction (single scalar α per (i,j)) ───────────

"""
    iso_order2_params_from_blocks(M) -> α::Matrix

Decompose a ``3n\\times 3n`` block matrix whose every ``3\\times 3`` block is ``\\alpha_{ij}\\,\\boldsymbol{1}``
(iso 2-tensor) into the scalar ``n\\times n`` Volterra matrix ``\\alpha``.
"""
function iso_order2_params_from_blocks(M::AbstractMatrix)
    sz = size(M, 1)
    sz == size(M, 2) ||
        throw(ArgumentError("iso_order2_params_from_blocks: M must be square"))
    sz % 3 == 0 ||
        throw(ArgumentError("iso_order2_params_from_blocks: size $(sz) not divisible by 3"))
    n = sz ÷ 3
    # A third of the trace of each 3×3 block.
    blk(i) = (3 * (i - 1) + 1):(3 * i)
    return [tr(@view M[blk(i), blk(j)]) / 3 for i in 1:n, j in 1:n]
end

"""
    iso_order2_blocks_from_params(α::AbstractMatrix) -> Matrix

Inverse of [`iso_order2_params_from_blocks`](@ref): build a ``3n\\times 3n``
block matrix ``\\alpha_{ij}\\,\\boldsymbol{1}`` per block.
"""
function iso_order2_blocks_from_params(α::AbstractMatrix)
    n = size(α, 1)
    n == size(α, 2) ||
        throw(ArgumentError("iso_order2_blocks_from_params: α must be square"))
    return kron(α, Matrix{eltype(α)}(I, 3, 3))
end

"""
    _is_iso_order2_block(M; tol = 1e-12) -> Bool

Return `true` if every ``3\\times 3`` block of `M` is ``\\alpha\\,\\boldsymbol{1}`` (iso 2-tensor in 3D).
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

Build the discrete Hill kernel ``\\widetilde{\\boldsymbol{P}}`` (size ``3n\\times 3n``) for an
ellipsoidal inclusion in an isotropic ALV matrix.  Uses the time-space
decoupling formula ``[\\widetilde{\\boldsymbol{P}}]_{ij} = \\bigl(\\widetilde{k}_0^{-\\circ}\\bigr)_{ij}\\,\\boldsymbol{I}^{\\boldsymbol{A}}``,
with ``\\widetilde{\\boldsymbol{K}}_0 = \\widetilde{k}_0\\,\\boldsymbol{1}`` the discretized isotropic reference.
"""
function hill_kernel_order2(
        ell, K_0_law::ViscoLaw,
        times::AbstractVector{<:Real}
    )
    return hill_kernel_order2_at(ell, _trapezoidal_relaxation(K_0_law, times, 3))
end

"""
    second_hill_kernel_order2(ell, K_0_law::ViscoLaw, times) -> Matrix

Discrete ALV second Hill tensor kernel (size ``3n\\times 3n``) of an ellipsoidal
inclusion in an isotropic ALV matrix of conductivity law `K_0_law`:
``\\widetilde{\\boldsymbol{Q}} = \\widetilde{\\boldsymbol{K}}_0 - \\widetilde{\\boldsymbol{K}}_0\\circ\\widetilde{\\boldsymbol{P}}\\circ\\widetilde{\\boldsymbol{K}}_0``,
with ``\\widetilde{\\boldsymbol{P}}`` the [`hill_kernel_order2`](@ref); the order-2
counterpart of [`second_hill_kernel`](@ref).
"""
function second_hill_kernel_order2(
        ell, K_0_law::ViscoLaw,
        times::AbstractVector{<:Real}
    )
    K_0 = _trapezoidal_relaxation(K_0_law, times, 3)
    return K_0 - K_0 * hill_kernel_order2_at(ell, K_0) * K_0
end

"""
    hill_kernel_order2_at(ell, K_0::AbstractMatrix) -> Matrix

Variant of [`hill_kernel_order2`](@ref) taking an already-discretized
``3n\\times 3n`` reference matrix instead of the matrix law — what the
differential scheme needs, its reference being the *running* effective
medium.  Same `_at` convention as the order-4 crack and layered-sphere
kernels.
"""
function hill_kernel_order2_at(ell, K_0::AbstractMatrix)
    _is_iso_order2_block(K_0) ||
        throw(ArgumentError("hill_kernel_order2: only iso ALV matrix is currently supported"))
    α_0 = iso_order2_params_from_blocks(K_0)
    α_0_inv = volterra_inverse(α_0; block_size = 1)
    # Canonical components: the 3n×3n matrices of the ALV algebra are written in
    # the canonical frame, and `tens_IA` is in the ellipsoid's. `get_array` read
    # it there, so a rotated ellipsoid was silently aligned with the axes.
    IA = TensND.canonical_components(Elasticity.tens_IA(ell))
    T = promote_type(eltype(α_0_inv), eltype(IA))
    return kron(Matrix{T}(α_0_inv), Matrix{T}(IA))   # block (i, j) = α⁻¹[i, j] I^A
end

# A layered sphere has no Hill kernel: its contribution comes from the
# recurrence of `layered_alv_order2.jl`. Refused by name rather than through a
# `MethodError` of `tens_IA`, should a path ask for one.
hill_kernel_order2_at(::LayeredSphere, ::AbstractMatrix) = throw(
    ArgumentError(
        "a LayeredSphere has no Hill kernel; its order-2 ALV contribution is " *
            "`property_contribution_alv`."
    )
)

# ── Generic order-2 ALV algebra (3n × 3n with block_size = 3) ──────────────

"""
    dilute_concentration_alv_order2(K_E, K_0, P) -> Matrix

Order-2 dilute concentration
``\\widetilde{\\boldsymbol{A}}^{\\mathrm{dil}} = (H\\,\\boldsymbol{1} + \\widetilde{\\boldsymbol{P}}\\circ\\Delta\\widetilde{\\boldsymbol{K}})^{-\\circ}``,
``\\Delta\\widetilde{\\boldsymbol{K}} = \\widetilde{\\boldsymbol{K}}^{\\mathcal{E}} - \\widetilde{\\boldsymbol{K}}_0``,
all matrices ``3n\\times 3n``.
"""
function dilute_concentration_alv_order2(
        K_E::AbstractMatrix, K_0::AbstractMatrix,
        P::AbstractMatrix
    )
    _alg_check_dense("dilute_concentration_alv_order2", _DenseALV{3}(), K_E, K_0, P)
    return _alg_dilute_concentration(_DenseALV{3}(), K_E, K_0, P)
end

"""
    dilute_contribution_alv_order2(K_E, K_0, P) -> Matrix

Order-2 dilute contribution ``\\widetilde{\\boldsymbol{N}} = \\Delta\\widetilde{\\boldsymbol{K}}\\circ\\widetilde{\\boldsymbol{A}}^{\\mathrm{dil}}``.
"""
function dilute_contribution_alv_order2(
        K_E::AbstractMatrix, K_0::AbstractMatrix,
        P::AbstractMatrix
    )
    _alg_check_dense("dilute_contribution_alv_order2", _DenseALV{3}(), K_E, K_0, P)
    return _alg_dilute_contribution(_DenseALV{3}(), K_E, K_0, P)
end

# ── Schemes ─────────────────────────────────────────────────────────────────

"""
    voigt_alv_order2(matrices, fractions) -> Matrix

Order-2 Voigt bound: ``\\widetilde{\\boldsymbol{K}}^{\\mathrm{hom}} = \\sum_i f_i\\,\\widetilde{\\boldsymbol{K}}_i``.
"""
function voigt_alv_order2(
        matrices::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector
    )
    _alg_check("voigt_alv_order2", matrices, fractions)
    isempty(matrices) && throw(ArgumentError("voigt_alv_order2: at least one phase required"))
    return _alg_voigt(_DenseALV{3}(), matrices, fractions)
end

"""
    reuss_alv_order2(matrices, fractions) -> Matrix

Order-2 Reuss bound: invert each compliance, average, invert back.
"""
function reuss_alv_order2(
        matrices::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector
    )
    _alg_check("reuss_alv_order2", matrices, fractions)
    isempty(matrices) && throw(ArgumentError("reuss_alv_order2: at least one phase required"))
    return _alg_reuss(_DenseALV{3}(), matrices, fractions)
end

"""
    dilute_alv_order2(K_0, contribs, fractions) -> Matrix

Order-2 Dilute scheme: ``\\widetilde{\\boldsymbol{K}}^{\\mathrm{hom}} = \\widetilde{\\boldsymbol{K}}_0 + \\sum_i f_i\\,\\widetilde{\\boldsymbol{N}}_i``.
"""
function dilute_alv_order2(
        K_0::AbstractMatrix,
        contribs::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector
    )
    _alg_check("dilute_alv_order2", contribs, fractions)
    return _alg_dilute(_DenseALV{3}(), K_0, contribs, fractions)
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
    _alg_check("dilute_dual_alv_order2", contribs_compliance, fractions)
    return _alg_dilute_dual(_DenseALV{3}(), K_0, contribs_compliance, fractions)
end

"""
    mori_tanaka_alv_order2(K_0, A_duts, contribs, fractions, f_M) -> Matrix

Order-2 Mori-Tanaka:

```math
\\widetilde{\\boldsymbol{K}}^{\\mathrm{hom}} = \\widetilde{\\boldsymbol{K}}_0
  + \\Bigl(\\sum_i f_i\\,\\widetilde{\\boldsymbol{N}}_i\\Bigr)\\circ
    \\Bigl(f_0\\,H\\,\\boldsymbol{1} + \\sum_j f_j\\,\\widetilde{\\boldsymbol{A}}_j^{\\mathrm{dil}}\\Bigr)^{-\\circ} .
```
"""
function mori_tanaka_alv_order2(
        K_0::AbstractMatrix,
        A_duts::AbstractVector{<:AbstractMatrix},
        contribs::AbstractVector{<:AbstractMatrix},
        fractions::AbstractVector, f_M::Real
    )
    _alg_check("mori_tanaka_alv_order2", A_duts, contribs, fractions)
    return _alg_mori_tanaka(_DenseALV{3}(), K_0, A_duts, contribs, fractions, f_M)
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
    _alg_check("maxwell_alv_order2", contribs, fractions)
    return _alg_maxwell(_DenseALV{3}(), K_0, contribs, fractions, H_0)
end

# ── Order-2 ALV pipeline (internal — dispatched from homogenize_alv) ───────

# The conductivity, dilute concentration and dilute contribution of one
# inclusion, as `_inclusion_alv_quantities` at order 4.
function _inclusion_alv2_quantities(geom, scheme, K_r_law, K_M_law, K_0, times)
    K_r = _trapezoidal_relaxation(K_r_law, times, 3)
    P_r = hill_kernel_order2(geom, K_M_law, times)
    return (
        K_r,
        dilute_concentration_alv_order2(K_r, K_0, P_r),
        dilute_contribution_alv_order2(K_r, K_0, P_r),
    )
end

# No Hill kernel: the recurrence gives the concentration and the contribution
# of the whole sphere, and its layers the conductivity each bound averages (the
# phase law is a placeholder).
_inclusion_alv2_quantities(s::LayeredSphere, scheme, K_r_law, K_M_law, K_0, times) = (
    _layer_bound_alv2(scheme, s, times),
    gradient_gradient_concentration_alv(s, K_M_law, times),
    property_contribution_alv(s, K_M_law, times),
)

# The relaxation kernel a phase enters the order-2 schemes with. A layered
# sphere's law is a placeholder, its layers carrying the kernels: the reference
# kernel `K_ref` stands in for it.
_alv2_phase_kernel(geom, law, K_ref, times) = _trapezoidal_relaxation(law, times, 3)
_alv2_phase_kernel(::LayeredSphere, law, K_ref, times) = K_ref

"""
    _homogenize_alv_order2(rve, scheme, prop::Symbol; times) -> Matrix

Internal order-2 ALV pipeline.  Reached from [`homogenize_alv`](@ref)
when the matrix property law samples to a ``3\\times 3`` / `TensND.AbstractTens{2,3}`
value.  Returns the effective ``\\widetilde{\\boldsymbol{K}}^{\\mathrm{hom}}`` of size ``3n\\times 3n``.

Supports iso ALV matrix + ellipsoidal inclusions of any aspect ratio.
The result is generally anisotropic (TI for spheroids, ortho for
triaxial ellipsoids).
"""
function _homogenize_alv_order2(
        rve::RVE, scheme::HomogenizationScheme,
        prop::Symbol; times::AbstractVector{<:Real}, kw...
    )
    m = matrix_name(scheme, rve)
    K_M_law = phase_property(rve, m, prop)
    K_M_law isa ViscoLaw ||
        throw(ArgumentError("homogenize_alv_order2: matrix property $prop is not a ViscoLaw"))
    K_0 = _trapezoidal_relaxation(K_M_law, times, 3)
    f_M = volume_fraction(rve, m)

    incl_names = inclusion_phase_names(rve, m)
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
        # `symmetrize` is honored on the dilute quantities, exactly as the
        # order-4 pipeline does (`homogenize_alv.jl`), and with the order-2
        # projector: the blocks here are 2-tensors, not 6×6 Mandel blocks.
        sym = phase_symmetrize(rve, name)
        K_r, A_dut, N_dut = _inclusion_alv2_quantities(
            ph.geometry, scheme, K_r_law, K_M_law, K_0, times
        )
        return (
            K_r = K_r, sym = sym,
            A_dut = _maybe_symmetrize_alv2(A_dut, sym),
            N_dut = _maybe_symmetrize_alv2(N_dut, sym),
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
    # The orientation average of each entry of `K_phases`, for the bounds.
    phase_syms = AbstractSymmetrize[phase_symmetrize(rve, m)]
    A_duts = Matrix{T_block}[]
    contribs = Matrix{T_block}[]
    for q in per_phase
        push!(K_phases, q.K_r)
        push!(phase_syms, q.sym)
        push!(A_duts, q.A_dut)
        push!(contribs, q.N_dut)
    end

    return _homogenize_alv2_dispatch(
        rve, scheme, prop, times,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law, m; phase_syms, kw...
    )
end

# Dispatch table for order-2 schemes.

# A scheme without an order-2 method is refused by name, not with a
# `MethodError` on this internal function.
function _homogenize_alv2_dispatch(
        ::RVE, scheme::HomogenizationScheme, ::Symbol, ::AbstractVector,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law, matrix::Symbol; kw...
    )
    throw(
        ArgumentError(
            "$(nameof(typeof(scheme))) has no order-2 (conduction, diffusion) ALV " *
                "implementation; Voigt, Reuss, Dilute, DiluteDual, MoriTanaka, Maxwell, " *
                "PonteCastanedaWillis, SelfConsistent and DifferentialScheme have one."
        )
    )
end

function _homogenize_alv2_dispatch(
        ::RVE, ::Voigt, ::Symbol, ::AbstractVector,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law, matrix::Symbol; phase_syms, kw...
    )
    # Each phase property averaged over its orientation distribution.
    return voigt_alv_order2(map(_maybe_symmetrize_alv2, K_phases, phase_syms), [f_M; fractions])
end

function _homogenize_alv2_dispatch(
        ::RVE, ::Reuss, ::Symbol, ::AbstractVector,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law, matrix::Symbol; phase_syms, kw...
    )
    fs = [f_M; fractions]
    all(s -> s isa NoSymmetrize, phase_syms) && return reuss_alv_order2(K_phases, fs)
    # The resistivity of each phase is averaged over its orientation
    # distribution, not its conductivity: the bound averages resistivities.
    Rs = map((K, sym) -> _maybe_symmetrize_alv2(volterra_inverse(K; block_size = 3), sym), K_phases, phase_syms)
    return volterra_inverse(voigt_alv_order2(Rs, fs); block_size = 3)
end

function _homogenize_alv2_dispatch(
        ::RVE, ::Dilute, ::Symbol, ::AbstractVector,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law, matrix::Symbol; kw...
    )
    return dilute_alv_order2(K_0, contribs, fractions)
end

function _homogenize_alv2_dispatch(
        ::RVE, ::DiluteDual, ::Symbol, ::AbstractVector,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law, matrix::Symbol; kw...
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
        fractions, f_M, K_M_law, matrix::Symbol; kw...
    )
    return mori_tanaka_alv_order2(K_0, A_duts, contribs, fractions, f_M)
end

function _homogenize_alv2_dispatch(
        rve::RVE, scheme::Maxwell, ::Symbol,
        times::AbstractVector,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law, matrix::Symbol; kw...
    )
    # The Hill kernel of the distribution shape the RVE declares, as at order
    # 4 and in the elastic scheme; it used to be a sphere whatever the RVE said.
    H_0 = hill_kernel_order2(Schemes.distribution_shape(rve, scheme).shape, K_M_law, times)
    return maxwell_alv_order2(K_0, contribs, fractions; H_0 = H_0)
end

# Ponte Castañeda–Willis: the Maxwell formula on the shape of a uniform
# distribution, as at order 4.
function _homogenize_alv2_dispatch(
        rve::RVE, scheme::PonteCastanedaWillis, ::Symbol,
        times::AbstractVector,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law, matrix::Symbol; kw...
    )
    dist = Schemes.distribution_shape(rve, scheme)
    dist isa UniformDistribution ||
        throw(ArgumentError("PCW-ALV: only UniformDistribution is currently supported"))
    H_d = hill_kernel_order2(dist.shape, K_M_law, times)
    return maxwell_alv_order2(K_0, contribs, fractions; H_0 = H_d)
end

function _homogenize_alv2_dispatch(
        rve::RVE, sc::SelfConsistent, prop::Symbol,
        times::AbstractVector,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law, matrix::Symbol; kw...
    )
    return self_consistent_alv_order2(rve, prop; times, matrix, sc.options...)
end

function _homogenize_alv2_dispatch(
        rve::RVE, sch::DifferentialScheme, prop::Symbol,
        times::AbstractVector,
        K_0, K_phases, A_duts, contribs,
        fractions, f_M, K_M_law, matrix::Symbol; kw...
    )
    return differential_alv_order2(
        rve, prop; times = times, matrix = matrix,
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
``\\tau \\in [0, 1]``, integrated on the ``3n\\times 3n`` ALV conductivity block
matrix.

`formulation = :compliance` integrates the resistivity ``\\widetilde{\\boldsymbol{R}} = \\widetilde{\\boldsymbol{K}}^{-\\circ}``
instead, through ``\\widetilde{\\boldsymbol{H}}_i = -\\widetilde{\\boldsymbol{R}}\\circ\\widetilde{\\boldsymbol{N}}_i\\circ\\widetilde{\\boldsymbol{R}}``, and inverts the result.

As in the order-4 case the reference of the ODE is the running
effective medium, and the order-2 ALV Hill kernel exists for an
isotropic reference only — every inclusion must therefore be spherical
or carry an isotropic orientation average.
"""
function differential_alv_order2(
        rve::RVE, prop::Symbol;
        times::AbstractVector{<:Real},
        matrix::Union{Nothing, Symbol} = nothing,
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
    m = host_phase_name(rve, matrix, "differential_alv_order2")
    K_M_law = phase_property(rve, m, prop)
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
    for name in inclusion_phase_names(rve, m)
        amt = rve.amounts[name]
        _is_crack_family(amt) &&
            throw(
            ArgumentError(
                "differential_alv_order2: phase :$(name) carries a crack " *
                    "density; order-2 ALV cracks are not supported by the " *
                    "differential scheme."
            )
        )
        amt isa Schemes.Remainder &&
            throw(
            ArgumentError(
                "differential_alv_order2: phase :$(name) carries the volume " *
                    "complement and so has no target to integrate up to; name it " *
                    "as the matrix, or give it an explicit fraction."
            )
        )
        ph = rve.phases[name]
        sym = phase_symmetrize(rve, name)
        K_r_law = phase_property(rve, name, prop)
        K_r_law isa ViscoLaw ||
            throw(ArgumentError("differential_alv_order2: phase $name property is not a ViscoLaw"))
        K_r = _alv2_phase_kernel(ph.geometry, K_r_law, K_0, times)
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
        _resolve_paths_alv(Schemes.Proportional(), rve, nsteps, m) :
        _resolve_paths_alv(trajectory, rve, nsteps, m)

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
            eltype(_diff_alv2_contribution(sd.geom, sd.K_r, K_0, times))
        )
    end
    sz = 3 * n
    dual = formulation === :compliance
    x0 = vec(Tp.(dual ? volterra_inverse(K_0; block_size = 3) : K_0))
    # A tuple of concrete records, each carrying its own trajectory: the
    # right-hand side runs over it without a dynamic dispatch per phase.
    ode_p = (
        sz = sz, dual = dual, times = times,
        solid_data = Tuple(merge(r, (path = paths[r.name],)) for r in solid_data),
    )
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

# The dilute contribution of one phase against the running medium `K_curr`:
# the Hill kernel of an ellipsoid, the recurrence of a layered sphere.
_diff_alv2_contribution(geom, K_r, K_curr, times) =
    dilute_contribution_alv_order2(K_r, K_curr, hill_kernel_order2_at(geom, K_curr))
_diff_alv2_contribution(s::LayeredSphere, K_r, K_curr, times) =
    property_contribution_alv(s, K_curr, times)

function _diff_alv2_ode_rhs!(du, u, p, τ)
    T = eltype(u)
    P_curr = reshape(u, p.sz, p.sz)
    K_curr = p.dual ? volterra_inverse(P_curr; block_size = 3) : P_curr
    Δ = reshape(du, p.sz, p.sz)
    fill!(Δ, zero(T))

    # Sherman-Morrison : dφ_α/dτ = df_α/dτ + (f_α / f_0) · sum(df).
    f = map(r -> r.path.f(τ) * r.target, p.solid_data)
    df = map(r -> r.path.df(τ) * r.target, p.solid_data)
    f0 = one(T) - sum(f; init = zero(T))
    sum_df = sum(df; init = zero(T))

    map(p.solid_data, f, df) do r, fᵢ, dfᵢ
        dφᵢ = dfᵢ + (fᵢ / f0) * sum_df
        iszero(dφᵢ) && return nothing
        # Order-2 projector: `K_curr` is a (3n × 3n) matrix of 2-tensor
        # blocks.  Using the order-4 (6×6 Mandel) one here silently mixed two
        # consecutive TIME blocks, so the "averaged" contribution was not
        # isotropic and the running medium drifted out of the class the
        # order-2 Hill kernel needs.
        contrib = _maybe_symmetrize_alv2(_diff_alv2_contribution(r.geom, r.K_r, K_curr, p.times), r.sym)
        term = p.dual ? -(P_curr * contrib * P_curr) : contrib
        @. Δ += dφᵢ * term
        return nothing
    end
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

# =============================================================================
#  Self-consistent ALV, order 2 (conduction, diffusion) — Picard on the running
#  estimate, the order-2 twin of `self_consistent_alv`:
#
#      K̃ ← (Σ_α f_α B̃_α(K̃)) ∘ (Σ_α f_α Ã_α(K̃))^{-∘},
#
#  over every phase, the matrix included, with Ã_α the dilute concentration of
#  phase α in the running medium and B̃_α = K̃_α ∘ Ã_α its flux. A layered
#  sphere gives Ã_α and B̃_α = Ñ_α + K̃ ∘ Ã_α through its recurrence.
# =============================================================================

# Whether a phase keeps the running estimate isotropic, as the order-2 Hill
# kernel against it requires: a sphere of isotropic conductivity, a layered
# sphere, or an isotropic orientation average.
_alv2_keeps_iso(geom, sym::AbstractSymmetrize, K_r) =
    sym isa IsoSymmetrize || geom isa LayeredSphere ||
    (_alv_geom_is_spherical(geom) && _is_iso_order2_block(K_r))

# The concentration and the flux of one phase in the running medium `K`.
function _sc2_phase_AB(geom, K_r, K, sym, times)
    A = dilute_concentration_alv_order2(K_r, K, hill_kernel_order2_at(geom, K))
    return _maybe_symmetrize_alv2(A, sym), _maybe_symmetrize_alv2(K_r * A, sym)
end
function _sc2_phase_AB(s::LayeredSphere, K_r, K, sym, times)
    A = gradient_gradient_concentration_alv(s, K, times)
    B = property_contribution_alv(s, K, times) .+ K * A
    return _maybe_symmetrize_alv2(A, sym), _maybe_symmetrize_alv2(B, sym)
end

"""
    self_consistent_alv_order2(rve, prop; times, matrix = nothing, abstol = 1e-10,
                               reltol = 1e-8, maxiters = 200, damping = 0.0,
                               verbose = false) -> Matrix

Order-2 (conduction, diffusion) self-consistent estimate in the aging linear
Volterra setting: the fixed point of
``\\widetilde{\\boldsymbol K} = \\bigl(\\sum_\\alpha f_\\alpha\\,\\widetilde{\\boldsymbol K}_\\alpha\\circ\\widetilde{\\boldsymbol A}_\\alpha\\bigr)\\circ\\bigl(\\sum_\\alpha f_\\alpha\\,\\widetilde{\\boldsymbol A}_\\alpha\\bigr)^{-\\circ}``,
every concentration ``\\widetilde{\\boldsymbol A}_\\alpha`` being taken in the running estimate
``\\widetilde{\\boldsymbol K}``, by Picard iteration from the matrix (the phase `matrix`, or
the one with `fraction = :rest`). A [`LayeredSphere`](@ref) phase enters
through its recurrence. Every phase must keep the estimate isotropic, as the
order-2 Hill kernel against it requires: a sphere of isotropic conductivity, a
layered sphere, or `symmetrize = :iso`. Reached through
`homogenize_alv(rve, SelfConsistent(), :K; times)`.
"""
function self_consistent_alv_order2(
        rve::RVE, prop::Symbol;
        times::AbstractVector{<:Real},
        matrix::Union{Nothing, Symbol} = nothing,
        abstol::Real = 1.0e-10,
        reltol::Real = 1.0e-8,
        maxiters::Int = 200,
        damping::Real = 0.0,
        verbose::Bool = false
    )
    m = host_phase_name(rve, matrix, "self_consistent_alv_order2")
    K_M_law = phase_property(rve, m, prop)
    K_M_law isa ViscoLaw ||
        throw(ArgumentError("self_consistent_alv_order2: matrix property is not a ViscoLaw"))
    K_M = _trapezoidal_relaxation(K_M_law, times, 3)
    data = map([m; inclusion_phase_names(rve, m)]) do name
        _is_crack_family(rve.amounts[name]) && throw(
            ArgumentError("self_consistent_alv_order2: phase :$(name) is a crack family, which the order-2 ALV schemes do not support.")
        )
        ph = rve.phases[name]
        sym = phase_symmetrize(rve, name)
        law = phase_property(rve, name, prop)
        law isa ViscoLaw ||
            throw(ArgumentError("self_consistent_alv_order2: phase $name property is not a ViscoLaw"))
        K_r = _alv2_phase_kernel(ph.geometry, law, K_M, times)
        _alv2_keeps_iso(ph.geometry, sym, K_r) ||
            _alv_diff_iso_error(name, "the shape or the anisotropy")
        f = name === m ? volume_fraction(rve, m) : _amount_value(rve, name)
        return (geom = ph.geometry, K_r = K_r, f = f, sym = sym)
    end
    K = K_M
    for iter in 1:maxiters
        # The comprehension captures this iteration's estimate, not `K`
        # itself, which is reassigned below and would be boxed.
        Kᵢ = K
        AB = [_sc2_phase_AB(d.geom, d.K_r, Kᵢ, d.sym, times) for d in data]
        A = sum(d.f .* ab[1] for (d, ab) in zip(data, AB))
        B = sum(d.f .* ab[2] for (d, ab) in zip(data, AB))
        K_new = B * volterra_inverse(A; block_size = 3)
        Δ = norm(K_new - K)
        verbose && @info "SC-ALV order 2, iteration $iter: ‖Δ‖ = $Δ"
        Δ ≤ abstol + reltol * norm(K) && return K_new
        K = (1 - damping) .* K_new .+ damping .* K
    end
    @debug "self_consistent_alv_order2: maxiters = $(maxiters) reached without convergence" abstol reltol
    return K
end
