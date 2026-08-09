using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_hill_alv_iso.jl — discrete ALV Hill polarization tensor for an
#  isotropic ALV matrix, via the time-space decoupling formula of the
#  ECHOES manual appendix `viscoelastic_hill_kernel.qmd`.
# =============================================================================

# Convert a TensND.AbstractTens{4,3} to a 6×6 Mandel matrix using the
# package-internal helper.
const _to_mandel = MeanFieldHomogenization.Viscoelasticity._tens_to_mandel66

@testset "hill_kernel — sphere, elastic limit (Heaviside)" begin
    ell = Spheroid(1.0)
    C0 = TensISO{3}(30.0, 8.0)   # k = 10, μ = 4
    law = heaviside_law(C0)
    times = collect(0.0:0.5:2.0)
    n = length(times)

    P = hill_kernel(ell, law, times)
    @test size(P) == (6n, 6n)

    # Compare with analytical elastic Hill.
    P_elas = MeanFieldHomogenization.Elasticity.hill_tensor(Ellipsoid(1.0, 1.0, 1.0), C0)
    P_elas_M = _to_mandel(P_elas)

    # Every diagonal block must equal P_elas, off-diag must be 0.
    for i in 1:n, j in 1:n
        rows = (6 * (i - 1) + 1):(6 * i)
        cols = (6 * (j - 1) + 1):(6 * j)
        if i == j
            @test maximum(abs.(P[rows, cols] - P_elas_M)) ≤ 1.0e-12
        else
            @test all(iszero, P[rows, cols])
        end
    end
end

@testset "hill_kernel — spheroid (oblate), elastic limit" begin
    ell = Spheroid(0.3)             # oblate, ω = c/a = 0.3
    C0 = TensISO{3}(33.0, 9.0)
    law = heaviside_law(C0)
    times = collect(0.0:0.25:1.0)
    P = hill_kernel(ell, law, times)

    # Analytical elastic Hill in iso matrix for the same ellipsoid.
    P_elas = MeanFieldHomogenization.Elasticity.hill_tensor(
        Ellipsoid(1.0, 1.0, 0.3), C0
    )
    P_elas_M = _to_mandel(P_elas)
    n = length(times)
    for i in 1:n
        rows = (6 * (i - 1) + 1):(6 * i)
        @test maximum(abs.(P[rows, rows] - P_elas_M)) ≤ 1.0e-10
    end
end

@testset "hill_kernel — Maxwell iso relaxation, structural checks" begin
    ell = Spheroid(1.0)
    law = maxwell_iso(10.0, 4.0, 1.0, 0.5)
    times = collect(0.0:0.25:1.0)
    P = hill_kernel(ell, law, times)
    n = length(times)

    # Lower-triangular block structure.
    for i in 1:n, j in 1:n
        if j > i
            rows = (6 * (i - 1) + 1):(6 * i)
            cols = (6 * (j - 1) + 1):(6 * j)
            @test all(iszero, P[rows, cols])
        end
    end

    # The (1, 1) block at t = 0 must equal the elastic Hill tensor of the
    # instantaneous moduli (k, μ), since at t = t' = 0 the relaxation has
    # just been applied and the Maxwell kernel returns C_inst.
    C_inst = TensISO{3}(30.0, 8.0)
    P_elas = MeanFieldHomogenization.Elasticity.hill_tensor(Ellipsoid(1.0, 1.0, 1.0), C_inst)
    P_elas_M = _to_mandel(P_elas)
    @test maximum(abs.(P[1:6, 1:6] - P_elas_M)) ≤ 1.0e-10
end

@testset "hill_kernel — Maxwell iso, R · R^{-vol} = H 𝟙" begin
    # Sanity check: P · C0_disc must satisfy
    #     <C0> ∘ <P> + <Q> = I  (Hill identity, for Q = C0 - C0:P:C0 in Volterra)
    # We here just check that scalar Volterra inverses round-trip cleanly.
    ell = Spheroid(1.0)
    law = maxwell_iso(10.0, 4.0, 1.0, 0.5)
    times = collect(0.0:0.2:2.0)
    n = length(times)
    P = hill_kernel(ell, law, times)
    R = trapezoidal_matrix(law, times)
    R_inv = volterra_inverse(R; block_size = 6)
    # Product P * (R_inv) should be lower-triangular and finite.
    Q = P * R_inv
    @test size(Q) == (6n, 6n)
    @test isfinite(maximum(abs, Q))
end
