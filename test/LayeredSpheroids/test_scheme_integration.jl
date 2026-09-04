using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_scheme_integration.jl — `LayeredSpheroid` as a first-class RVE
#  phase, mirroring `LayeredSpheres/test_scheme_integration.jl`'s three
#  invariants:
#
#    1. degeneracy — a single-layer spheroid behaves exactly like the
#       equivalent `Ellipsoid`, through the full RVE/homogenize path;
#    2. independence — the result does NOT depend on the (meaningless)
#       phase property declared for the heterogeneous inclusion;
#    3. consistency — `N = ⟨K∇T⟩ - K₀·⟨∇T⟩`.
# =============================================================================

const K_M = TensISO{3}(2.0)
const K_I = TensISO{3}(20.0)

@testset "LayeredSpheroid — degeneracy vs Ellipsoid (Dilute, MoriTanaka)" begin
    ω = 3.0
    sph = LayeredSpheroid((3.0,), (1.0,), (K_I,); Nseries = 6, axis = (1.0, 0.0, 0.0))
    ell = Ellipsoid(ω, 1.0, 1.0)   # semi-axes sorted, longest ends up on ê₁ — matches `sph`'s axis

    for scheme in (Dilute(), MoriTanaka())
        r_sph = RVE()
        add_phase!(r_sph, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => K_M); fraction = :rest)
        add_phase!(r_sph, :I, sph, Dict(:K => K_I); fraction = 0.15)

        r_ell = RVE()
        add_phase!(r_ell, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => K_M); fraction = :rest)
        add_phase!(r_ell, :I, ell, Dict(:K => K_I); fraction = 0.15)

        K_sph = homogenize(r_sph, scheme, :K)
        K_ell = homogenize(r_ell, scheme, :K)
        @test get_array(K_sph) ≈ get_array(K_ell) rtol = 1.0e-10
    end
end

@testset "LayeredSpheroid — independent of the declared phase property" begin
    sph = LayeredSpheroid((3.0,), (1.0,), (K_I,); Nseries = 6)

    r_a = RVE()
    add_phase!(r_a, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => K_M); fraction = :rest)
    add_phase!(r_a, :I, sph, Dict(:K => K_I); fraction = 0.15)

    r_b = RVE()
    add_phase!(r_b, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:K => K_M); fraction = :rest)
    add_phase!(r_b, :I, sph, Dict(:K => TensISO{3}(1.0e-4)); fraction = 0.15)

    K_a = homogenize(r_a, MoriTanaka(), :K)
    K_b = homogenize(r_b, MoriTanaka(), :K)
    @test get_array(K_a) ≈ get_array(K_b) atol = 1.0e-14
end

@testset "LayeredSpheroid — N = ⟨K∇T⟩ - K₀·⟨∇T⟩" begin
    for (radii_a, radii_b) in (
            ((3.0,), (1.0,)),
            ((2.9, 3.0), (sqrt(2.9^2 - 8.0), 1.0)),
        )
        moduli = ntuple(k -> TensISO{3}(5.0 + 10k), length(radii_a))
        s = LayeredSpheroid(radii_a, radii_b, moduli; Nseries = 6)
        A = gradient_gradient_loc(s, K_I, K_M)
        B = flux_gradient_loc(s, K_I, K_M)
        N = conductivity_contribution(s, K_I, K_M)
        @test get_array(N) ≈ get_array(B - K_M ⋅ A) rtol = 1.0e-12
    end
end

@testset "LayeredSpheroid — Voigt / Reuss layer averages" begin
    K1 = TensISO{3}(5.0); K2 = TensISO{3}(20.0)
    s = LayeredSpheroid((2.9, 3.0), (sqrt(2.9^2 - 8.0), 1.0), (K1, K2); Nseries = 6)
    f1 = layer_volume_fraction(s, 1)
    f2 = layer_volume_fraction(s, 2)
    voigt = MeanFieldHomogenization.LayeredSpheroids.layer_conductivity_average(s)
    reuss = MeanFieldHomogenization.LayeredSpheroids.layer_resistivity_average(s)
    @test voigt.data[1] ≈ f1 * 5.0 + f2 * 20.0
    @test reuss.data[1] ≈ f1 / 5.0 + f2 / 20.0
end
