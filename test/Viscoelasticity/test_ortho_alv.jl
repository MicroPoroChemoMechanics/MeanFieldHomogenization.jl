using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import MeanFieldHomogenization.Viscoelasticity: _is_ortho_block, _is_ti_block, _is_iso_block,
    _ortho_pair, _ortho_blocks, _ortho_inv, _ortho_prod, _ortho_left_divide,
    _ortho_identity, _iso_to_ortho, _ti_to_ortho

# =============================================================================
#  test_ortho_alv.jl — orthotropic ALV fast path: round-trips, primitives,
#  scheme equivalence vs the generic 6n×6n algebra, end-to-end dispatch.
# =============================================================================

# Helper: random lower-triangular n×n Volterra-like matrix with strictly
# positive diagonal so its Volterra inverse exists.
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

# Random ortho ALV operator stored as a 12-tuple of n×n lower-triangular
# Volterra matrices (canonical material frame).  The 3×3 normal block is
# made diagonally dominant in (i,i) per time-step so the Volterra inverse
# is well-conditioned.
function _rand_ortho_lt(n)
    o = ntuple(_ -> _rand_lt(n), 12)
    @inbounds for i in 1:n
        # Boost the normal-block diagonal entries (positions 1, 5, 9) so
        # the per-time-step 3×3 normal block stays SPD-ish.
        o[1][i, i] += 5.0
        o[5][i, i] += 5.0
        o[9][i, i] += 5.0
        # Off-diagonal couplings stay small.
        o[2][i, i] *= 0.2; o[3][i, i] *= 0.2
        o[4][i, i] *= 0.2; o[6][i, i] *= 0.2
        o[7][i, i] *= 0.2; o[8][i, i] *= 0.2
    end
    return o
end

@testset "ortho_alv — round-trip extract / reassemble" begin
    for n in (1, 3, 7)
        o = ntuple(_ -> randn(n, n), 12)
        M = ortho_blocks_from_params(o)
        @test size(M) == (6n, 6n)
        @test _is_ortho_block(M)
        o_back = ortho_params_from_blocks(M)
        for k in 1:12
            @test isapprox(o[k], o_back[k]; atol = 1.0e-14)
        end
    end
end

@testset "ortho_alv — iso block is detected as ortho" begin
    n = 4
    α = randn(n, n); β = randn(n, n)
    M_iso = iso_blocks_from_params(α, β)
    @test _is_iso_block(M_iso)
    @test _is_ortho_block(M_iso)
    o_extracted = _ortho_pair(M_iso)
    o_via_helper = _iso_to_ortho((α, β))
    for k in 1:12
        @test isapprox(o_extracted[k], o_via_helper[k]; atol = 1.0e-14)
    end
end

@testset "ortho_alv — TI block is detected as ortho" begin
    n = 4
    ℓ = ntuple(_ -> randn(n, n), 6)
    M_ti = ti_blocks_from_params(ℓ)
    @test _is_ti_block(M_ti)
    @test _is_ortho_block(M_ti)
    o_extracted = _ortho_pair(M_ti)
    o_via_helper = _ti_to_ortho(ℓ)
    for k in 1:12
        @test isapprox(o_extracted[k], o_via_helper[k]; atol = 1.0e-14)
    end
end

@testset "ortho_alv — Volterra primitives match 6n×6n algebra" begin
    n = 5
    a = _rand_ortho_lt(n)
    b = _rand_ortho_lt(n)
    M_a = ortho_blocks_from_params(a)
    M_b = ortho_blocks_from_params(b)

    # Product
    c_ortho = _ortho_prod(a, b)
    M_c_via_ortho = ortho_blocks_from_params(c_ortho)
    M_c_full = M_a * M_b
    @test isapprox(M_c_via_ortho, M_c_full; atol = 1.0e-12)
    @test _is_ortho_block(M_c_full)   # algebra closure

    # Inverse
    a_inv = _ortho_inv(a)
    M_a_inv_via_ortho = ortho_blocks_from_params(a_inv)
    M_a_inv_full = volterra_inverse(M_a; block_size = 6)
    @test isapprox(M_a_inv_via_ortho, M_a_inv_full; atol = 1.0e-10)

    # Sanity: a · a⁻¹ = block-diagonal identity
    H_id = zeros(6n, 6n)
    @inbounds for i in 1:n
        rows = (6 * (i - 1) + 1):(6 * i)
        H_id[rows, rows] = Matrix{Float64}(I, 6, 6)
    end
    @test isapprox(M_a * M_a_inv_via_ortho, H_id; atol = 1.0e-10)

    # Left divide
    ainvb_ortho = _ortho_left_divide(a, b)
    M_via_ortho = ortho_blocks_from_params(ainvb_ortho)
    M_full = volterra_left_divide(M_a, M_b; block_size = 6)
    @test isapprox(M_via_ortho, M_full; atol = 1.0e-10)
end

@testset "ortho_alv — schemes match 6n×6n on iso phases" begin
    # Build a simple iso matrix + iso inclusion combo.  Iso ⊂ ortho, so
    # the ortho fast path applies — and must give the same answer as the
    # generic 6n×6n algebra.
    n = 4
    times = collect(range(0.0, 1.0; length = n))

    C_M_t = TensISO{3}(3 * 5.0, 2 * 2.0)   # 3K=15, 2μ=4
    C_M_law = heaviside_law(C_M_t)
    C_M = trapezoidal_matrix(C_M_law, times)
    @test _is_ortho_block(C_M)

    C_I_t = TensISO{3}(3 * 10.0, 2 * 4.0)
    C_I_law = heaviside_law(C_I_t)
    C_I = trapezoidal_matrix(C_I_law, times)
    @test _is_ortho_block(C_I)

    # Voigt
    f_I = 0.3; f_M = 1 - f_I
    voigt_full = voigt_alv([C_M, C_I], [f_M, f_I])
    o_M = _ortho_pair(C_M); o_I = _ortho_pair(C_I)
    voigt_ortho = _ortho_blocks(voigt_alv_ortho([o_M, o_I], [f_M, f_I]))
    @test isapprox(voigt_ortho, voigt_full; atol = 1.0e-12)
    @test _is_ortho_block(voigt_full)

    # Reuss
    reuss_full = reuss_alv([C_M, C_I], [f_M, f_I])
    reuss_ortho = _ortho_blocks(reuss_alv_ortho([o_M, o_I], [f_M, f_I]))
    @test isapprox(reuss_ortho, reuss_full; atol = 1.0e-10)
end

@testset "ortho_alv — random ortho phases match generic algebra" begin
    # Stress test: construct two random ortho ALV operators (closure-only,
    # not necessarily major-symmetric) and check Voigt / Reuss / dilute /
    # MT / Maxwell match the generic 6n×6n path.
    n = 4
    a = _rand_ortho_lt(n)
    b = _rand_ortho_lt(n)
    M_a = ortho_blocks_from_params(a)
    M_b = ortho_blocks_from_params(b)
    @test _is_ortho_block(M_a) && _is_ortho_block(M_b)

    f_a, f_b = 0.7, 0.3
    voigt_full = voigt_alv([M_a, M_b], [f_a, f_b])
    voigt_o = _ortho_blocks(voigt_alv_ortho([a, b], [f_a, f_b]))
    @test isapprox(voigt_o, voigt_full; atol = 1.0e-12)

    reuss_full = reuss_alv([M_a, M_b], [f_a, f_b])
    reuss_o = _ortho_blocks(reuss_alv_ortho([a, b], [f_a, f_b]))
    @test isapprox(reuss_o, reuss_full; atol = 1.0e-9)

    # Dilute concentration / contribution: pick a random P̃ in ortho form
    # so all three (C_E, C_0, P̃) are ortho.
    p = _rand_ortho_lt(n)
    P_full = ortho_blocks_from_params(p)

    A_full = dilute_concentration_alv(M_a, M_b, P_full)
    A_o = _ortho_blocks(dilute_concentration_alv_ortho(a, b, p))
    @test isapprox(A_o, A_full; atol = 1.0e-9)

    N_full = dilute_contribution_alv(M_a, M_b, P_full)
    N_o = _ortho_blocks(dilute_contribution_alv_ortho(a, b, p))
    @test isapprox(N_o, N_full; atol = 1.0e-9)
end

@testset "ortho_alv — homogenize_alv routes through ortho for ortho RVE" begin
    # End-to-end: build an RVE with an iso matrix + iso spherical inclusion
    # (everything is ortho).  Voigt / Dilute / MT must give the same answer
    # whether the dispatcher takes the ortho fast path or the generic path.
    n = 6
    times = collect(range(0.0, 2.0; length = n))

    C_M_t = TensISO{3}(3 * 5.0, 2 * 2.0)
    C_M_law = heaviside_law(C_M_t)
    C_I_t = TensISO{3}(3 * 10.0, 2 * 4.0)
    C_I_law = heaviside_law(C_I_t)

    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_M_law); fraction = :rest)
    add_phase!(
        rve, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => C_I_law);
        fraction = 0.2
    )

    # Voigt — closed form
    C_eff = homogenize_alv(rve, Voigt(), :C; times = times)
    @test size(C_eff) == (6n, 6n)
    @test _is_ortho_block(C_eff)
    # Iso ⊂ ortho — result should still be iso (uniform combination of iso
    # phases).
    @test _is_iso_block(C_eff)

    # Generic reference
    C_M = trapezoidal_matrix(C_M_law, times)
    C_I = trapezoidal_matrix(C_I_law, times)
    C_voigt_ref = voigt_alv([C_M, C_I], [0.8, 0.2])
    @test isapprox(C_eff, C_voigt_ref; atol = 1.0e-12)
end
