using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import MeanFieldHomogenization.Viscoelasticity: _is_ti_block, _is_iso_block, _ti_pair,
    _ti_blocks, _ti_inv, _ti_prod, _ti_left_divide, _ti_identity, _iso_to_ti

# =============================================================================
#  test_ti_alv.jl — TI Walpole-basis ALV fast path: round-trips, primitives,
#  scheme equivalence vs the generic 6n×6n algebra.
# =============================================================================

# Helper: build a random lower-triangular n×n Volterra-like matrix with
# strictly positive diagonal so its Volterra inverse exists.
function _rand_lt(n)
    M = randn(n, n)
    @inbounds for i in 1:n, j in (i + 1):n
        M[i, j] = 0.0
    end
    @inbounds for i in 1:n
        M[i, i] = 1.0 + abs(randn())
    end
    return M
end

# Random TI ALV operator stored as a 6-tuple of n×n lower-triangular Volterra
# matrices (axis = e_3).
function _rand_ti_lt(n)
    return ntuple(_ -> _rand_lt(n), 6)
end

@testset "ti_alv — round-trip extract / reassemble" begin
    for n in (1, 3, 7)
        ℓ = ntuple(_ -> randn(n, n), 6)
        M = ti_blocks_from_params(ℓ)
        @test size(M) == (6n, 6n)
        @test _is_ti_block(M)
        ℓ_back = ti_params_from_blocks(M)
        for k in 1:6
            @test isapprox(ℓ[k], ℓ_back[k]; atol = 1.0e-14)
        end
    end
end

@testset "ti_alv — iso block is detected as TI" begin
    n = 4
    α = randn(n, n); β = randn(n, n)
    M_iso = iso_blocks_from_params(α, β)
    @test _is_iso_block(M_iso)
    @test _is_ti_block(M_iso)
    # ℓ extracted from iso M should match _iso_to_ti((α, β))
    ℓ_extracted = _ti_pair(M_iso)
    ℓ_via_helper = _iso_to_ti((α, β))
    for k in 1:6
        @test isapprox(ℓ_extracted[k], ℓ_via_helper[k]; atol = 1.0e-14)
    end
end

@testset "ti_alv — Volterra primitives match 6n×6n algebra" begin
    n = 5
    a = _rand_ti_lt(n)
    b = _rand_ti_lt(n)
    M_a = ti_blocks_from_params(a)
    M_b = ti_blocks_from_params(b)

    # Product
    c_ti = _ti_prod(a, b)
    M_c_via_ti = ti_blocks_from_params(c_ti)
    M_c_full = M_a * M_b
    @test isapprox(M_c_via_ti, M_c_full; atol = 1.0e-12)
    @test _is_ti_block(M_c_full)   # algebra closure

    # Inverse — random TI matrices can be ill-conditioned
    # (`‖M_a^{-1}‖ ∼ 10⁶`), so the two computational paths
    # (TI-inverse-then-assemble vs assemble-then-Volterra-inverse) only
    # agree up to ~1e-13 RELATIVE error.  Use an `rtol` scaled to
    # `‖M_a_inv_full‖` rather than a strict `atol`.
    a_inv = _ti_inv(a)
    M_a_inv_via_ti = ti_blocks_from_params(a_inv)
    M_a_inv_full = volterra_inverse(M_a; block_size = 6)
    @test isapprox(
        M_a_inv_via_ti, M_a_inv_full;
        rtol = 1.0e-10, atol = 1.0e-12
    )

    # Sanity: a · a⁻¹ = block-diag identity.  The residual of a numerically
    # formed inverse is bounded by ‖M_a‖·‖M_a⁻¹‖·eps, NOT by an absolute
    # constant: random TI draws reach ‖M_a⁻¹‖ ∼ 10⁶, and a fixed `atol = 1e-10`
    # made this assertion fail on unlucky draws.  Scale the tolerance the same
    # way as the inverse comparison above.
    H_id = zeros(6n, 6n)
    @inbounds for i in 1:n
        rows = (6 * (i - 1) + 1):(6 * i)
        H_id[rows, rows] = Matrix{Float64}(I, 6, 6)
    end
    atol_id = max(1.0e-12, 1.0e-15 * opnorm(M_a, 1) * opnorm(M_a_inv_via_ti, 1))
    @test isapprox(M_a * M_a_inv_via_ti, H_id; atol = atol_id)

    # Left divide — same conditioning concern as the inverse path.
    ainvb_ti = _ti_left_divide(a, b)
    M_via_ti = ti_blocks_from_params(ainvb_ti)
    M_full = volterra_left_divide(M_a, M_b; block_size = 6)
    @test isapprox(M_via_ti, M_full; rtol = 1.0e-10, atol = 1.0e-12)
end

@testset "ti_alv — schemes match 6n×6n in elastic limit" begin
    # Build a simple TI matrix + iso inclusion combo and verify scheme
    # results coincide with the generic 6n×6n path.
    n = 4
    times = collect(range(0.0, 1.0; length = n))

    # TI matrix: build via TensND, then heaviside to get a 6n×6n elastic-like
    # block matrix (constant in time).  Axis = e_3.
    n_axis = (0.0, 0.0, 1.0)
    # Use simple TI: ℓ₁ = 4, ℓ₂ = 6, ℓ₃ = ℓ₄ = 1, ℓ₅ = 1, ℓ₆ = 2  (positive defin.)
    ℓ_M = TensTI{4}(4.0, 6.0, 1.0, 1.0, 2.0, n_axis)   # major-symmetric
    C_M_law = heaviside_law(ℓ_M)
    C_M = trapezoidal_matrix(C_M_law, times)
    @test _is_ti_block(C_M)

    # Iso inclusion (will be embedded as TI automatically)
    C_I_t = TensISO{3}(3 * 10.0, 2 * 4.0)   # 3K=30, 2μ=8
    C_I_law = heaviside_law(C_I_t)
    C_I = trapezoidal_matrix(C_I_law, times)
    @test _is_iso_block(C_I)
    @test _is_ti_block(C_I)

    # Voigt: TI fast path vs generic
    f_I = 0.3; f_M = 1 - f_I
    voigt_full = voigt_alv([C_M, C_I], [f_M, f_I])
    ti_M = _ti_pair(C_M); ti_I = _ti_pair(C_I)
    voigt_ti = _ti_blocks(voigt_alv_ti([ti_M, ti_I], [f_M, f_I]))
    @test isapprox(voigt_ti, voigt_full; atol = 1.0e-12)
    @test _is_ti_block(voigt_full)

    # Reuss
    reuss_full = reuss_alv([C_M, C_I], [f_M, f_I])
    reuss_ti = _ti_blocks(reuss_alv_ti([ti_M, ti_I], [f_M, f_I]))
    @test isapprox(reuss_ti, reuss_full; atol = 1.0e-10)
end

@testset "ti_alv — homogenize_alv routes through TI for TI matrix" begin
    # End-to-end: build an RVE with TI matrix + iso spherical inclusion and
    # check the dispatcher returns the same answer as the generic path
    # (which we force by disabling iso/TI detection).
    n = 6
    times = collect(range(0.0, 2.0; length = n))
    n_axis = (0.0, 0.0, 1.0)

    # TI matrix (major-sym)
    ℓ_M_t = TensTI{4}(4.0, 6.0, 1.0, 1.5, 2.0, n_axis)
    C_M_law = heaviside_law(ℓ_M_t)
    # Iso inclusion
    C_I_t = TensISO{3}(3 * 10.0, 2 * 4.0)
    C_I_law = heaviside_law(C_I_t)

    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_M_law); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_I_law);
        fraction = 0.2
    )

    # Voigt — has a closed-form TI fast path
    C_eff = homogenize_alv(rve, Voigt(), :C; times = times)
    @test size(C_eff) == (6n, 6n)
    @test _is_ti_block(C_eff)

    # Generic reference (force fallback by passing the matrices manually)
    C_M = trapezoidal_matrix(C_M_law, times)
    C_I = trapezoidal_matrix(C_I_law, times)
    C_voigt_ref = voigt_alv([C_M, C_I], [0.8, 0.2])
    @test isapprox(C_eff, C_voigt_ref; atol = 1.0e-12)
end
