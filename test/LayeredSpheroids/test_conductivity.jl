using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_conductivity.jl — `LayeredSpheroid` geometry + transfer-matrix
#  recurrence, cross-checked against the classical Eshelby conduction
#  result and against physical interface limits.
#
#  `Spheroid(ω)`'s revolution axis is `ê₁` for prolate (`ω > 1`) and
#  `ê₃` for oblate (`ω < 1`) — see `Ellipsoid`'s docstring — so every
#  comparison below builds the matching `LayeredSpheroid` with
#  `axis = (1,0,0)` (prolate) or `(0,0,1)` (oblate).
# =============================================================================

@testset "LayeredSpheroid — trait is_homogeneous_inclusion" begin
    K1 = TensISO{3}(5.0)
    @test !is_homogeneous_inclusion(LayeredSpheroid((3.0,), (1.0,), (K1,)))
    @test !is_homogeneous_inclusion(
        LayeredSpheroid((2.9, 3.0), (sqrt(2.9^2 - 8.0), 1.0), (K1, K1))
    )
end

@testset "LayeredSpheroid — single layer reduces to the Eshelby spheroid" begin
    K0 = TensISO{3}(2.0)
    for (a, b, k1) in ((3.0, 1.0, 5.0), (0.5, 2.0, 5.0), (0.3, 1.5, 0.5), (10.0, 0.3, 8.0), (0.05, 3.0, 8.0))
        K1 = TensISO{3}(k1)
        # `Spheroid(ω) == Ellipsoid(1, 1, ω)`, whose symmetry axis is `e₃` for
        # *both* families — prolate included, since the constructor sorts the
        # semi-axes and permutes the basis to match (see its docstring). So the
        # comparison spheroid must be built about `e₃` in both branches.
        #
        # This used to read `ax = a > b ? (1, 0, 0) : (0, 0, 1)` and passed only
        # because the isotropic order-2 Hill kernel dropped the inclusion's
        # orientation, which for a sorted prolate put the distinct value back on
        # slot 1. Two compensating errors; fixing the kernel exposed this one.
        s = LayeredSpheroid((a,), (b,), (K1,); Nseries = 6, axis = (0.0, 0.0, 1.0))
        ell = Spheroid(a / b)
        P = hill_tensor(ell, K0)
        A_classic = inv(TensISO{3}(1.0) + P ⋅ (K1 - K0))
        A_mine = gradient_gradient_concentration(s, K1, K0)
        @test get_array(A_mine) ≈ get_array(A_classic) rtol = 1.0e-10

        B_mine = flux_gradient_concentration(s, K1, K0)
        N_mine = property_contribution(s, K1, K0)
        @test get_array(N_mine) ≈ get_array(B_mine - K0 ⋅ A_mine) rtol = 1.0e-10
        N_classic = (K1 - K0) ⋅ A_classic
        @test get_array(N_mine) ≈ get_array(N_classic) rtol = 1.0e-10
    end
end

@testset "LayeredSpheroid — two identical confocal layers ≡ one layer" begin
    K1 = TensISO{3}(5.0); K0 = TensISO{3}(2.0)
    s1 = LayeredSpheroid((3.0,), (1.0,), (K1,); Nseries = 6)
    focal2 = 3.0^2 - 1.0^2
    a_in = 2.9
    b_in = sqrt(a_in^2 - focal2)
    s2 = LayeredSpheroid((a_in, 3.0), (b_in, 1.0), (K1, K1); Nseries = 6)
    A1 = gradient_gradient_concentration(s1, K1, K0)
    A2 = gradient_gradient_concentration(s2, K1, K0)
    @test get_array(A1) ≈ get_array(A2) atol = 1.0e-12
end

@testset "LayeredSpheroid — Kapitza / surface-conductive interface limits" begin
    K1 = TensISO{3}(5.0); K0 = TensISO{3}(2.0)
    s_perfect = LayeredSpheroid((3.0,), (1.0,), (K1,); Nseries = 6)
    A_perfect = gradient_gradient_concentration(s_perfect, K1, K0)

    s_lc_tiny = LayeredSpheroid(
        (3.0,), (1.0,), (K1,);
        interfaces = (MeanFieldHomogenization.KapitzaInterface(1.0e-9),), Nseries = 6
    )
    @test get_array(gradient_gradient_concentration(s_lc_tiny, K1, K0)) ≈ get_array(A_perfect) atol = 1.0e-6

    s_hc_tiny = LayeredSpheroid(
        (3.0,), (1.0,), (K1,);
        interfaces = (MeanFieldHomogenization.SurfaceConductiveInterface(1.0e-9),), Nseries = 6
    )
    @test get_array(gradient_gradient_concentration(s_hc_tiny, K1, K0)) ≈ get_array(A_perfect) atol = 1.0e-6

    # Kapitza resistance → ∞ ⟺ fully insulated (impermeable) core.
    s_lc_big = LayeredSpheroid(
        (3.0,), (1.0,), (K1,);
        interfaces = (MeanFieldHomogenization.KapitzaInterface(1.0e10),), Nseries = 6
    )
    s_insulated = LayeredSpheroid((3.0,), (1.0,), (TensISO{3}(1.0e-14),); Nseries = 6)
    A_lc_big = gradient_gradient_concentration(s_lc_big, K1, K0)
    A_insulated = gradient_gradient_concentration(s_insulated, TensISO{3}(1.0e-14), K0)
    @test get_array(A_lc_big) ≈ get_array(A_insulated) rtol = 1.0e-5
end

@testset "layered_spheroid_from_fractions — recovers requested ω and fractions" begin
    K1 = TensISO{3}(5.0); K2 = TensISO{3}(20.0)
    s = layered_spheroid_from_fractions(3.0, 3.0, (0.3, 0.7), (K1, K2); Nseries = 6)
    a, b = outer_semiaxes(s)
    @test a ≈ 3.0
    @test a / b ≈ 3.0 rtol = 1.0e-8
    @test layer_volume_fraction(s, 1) ≈ 0.3 atol = 1.0e-8
    @test layer_volume_fraction(s, 2) ≈ 0.7 atol = 1.0e-8

    s_ob = layered_spheroid_from_fractions(0.3, 3.0, (0.4, 0.6), (K1, K2); Nseries = 6)
    a_ob, b_ob = outer_semiaxes(s_ob)
    @test a_ob ≈ 3.0
    @test a_ob / b_ob ≈ 0.3 rtol = 1.0e-8
    @test layer_volume_fraction(s_ob, 1) ≈ 0.4 atol = 1.0e-8
    @test layer_volume_fraction(s_ob, 2) ≈ 0.6 atol = 1.0e-8

    # Single layer still reduces exactly to the Eshelby result.
    K0 = TensISO{3}(2.0)
    # About `e₃`, as `Spheroid(3.0)` is — see the comment above.
    s1 = layered_spheroid_from_fractions(
        3.0, 3.0, (1.0,), (K1,); Nseries = 6, axis = (0.0, 0.0, 1.0)
    )
    A = gradient_gradient_concentration(s1, K1, K0)
    ell = Spheroid(3.0)
    P = hill_tensor(ell, K0)
    A_classic = inv(TensISO{3}(1.0) + P ⋅ (K1 - K0))
    @test get_array(A) ≈ get_array(A_classic) rtol = 1.0e-10
end

@testset "LayeredSpheroid — geometry validation errors" begin
    K1 = TensISO{3}(5.0)
    @test_throws ArgumentError LayeredSpheroid((1.0,), (1.0,), (K1,))       # ω = 1
    @test_throws ArgumentError LayeredSpheroid((3.0, 2.0), (1.0, 0.5), (K1, K1))  # not ascending
    @test_throws ArgumentError LayeredSpheroid((2.0, 3.0), (0.9, 1.0), (K1, K1))  # not confocal
end
