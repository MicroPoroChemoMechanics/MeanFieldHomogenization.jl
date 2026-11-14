# =============================================================================
#  test_scheme_integration.jl — the layered cylinder in the mean-field schemes.
#  A one-layer cylinder must give what the homogeneous `Cylinder` gives, in
#  every scheme; the declared phase property must not matter; and aligned
#  layered cylinders alone make a self-consistent problem whose reference is
#  transversely isotropic.
# =============================================================================
isdefined(@__MODULE__, :lc_ti) || include(joinpath(@__DIR__, "helpers.jl"))

const LC_HOM = MeanFieldHomogenization.Elasticity.Cylinder(1.0, 1.0, [0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 0.0 0.0])
lc_rel(a, b) = norm(lc_cc(a) .- lc_cc(b)) / norm(lc_cc(b))

function lc_eff(geom, scheme, prop; P_I, P_M, matrix_geom = LC_HOM, declared = P_I, kw...)
    r = RVE(; distribution_shape = LC_HOM)
    add_phase!(r, :M, matrix_geom, Dict(prop => P_M); fraction = :rest)
    add_phase!(r, :I, geom, Dict(prop => declared); fraction = 0.3, kw...)
    return homogenize(r, scheme, prop)
end

const LC_SCHEMES = (
    Voigt(), Reuss(), Dilute(), DiluteDual(), MoriTanaka(), Maxwell(), PonteCastanedaWillis(),
    SelfConsistent(), AsymmetricSelfConsistent(:M), DifferentialScheme(),
)

@testset "LayeredCylinder in the schemes — elasticity" begin
    Cf = lc_tensor(LC_F)
    CM = TensISO{3}(3 * 6.0, 2 * 2.5)
    lay = LayeredCylinder((1.0,), (Cf,))
    for scheme in LC_SCHEMES
        ref = lc_eff(LC_HOM, scheme, :C; P_I = Cf, P_M = CM)
        @test lc_rel(lc_eff(lay, scheme, :C; P_I = Cf, P_M = CM), ref) < 1.0e-8
        # the declared phase property is a placeholder
        @test lc_rel(lc_eff(lay, scheme, :C; P_I = Cf, P_M = CM, declared = CM), ref) < 1.0e-8
    end
    # every phase a layered cylinder: no Hill tensor is used, the reference is TI
    one_layer_matrix = LayeredCylinder((1.0,), (CM,))
    for scheme in (MoriTanaka(), SelfConsistent(), DifferentialScheme())
        @test lc_rel(
            lc_eff(lay, scheme, :C; P_I = Cf, P_M = CM, matrix_geom = one_layer_matrix),
            lc_eff(LC_HOM, scheme, :C; P_I = Cf, P_M = CM)
        ) < 1.0e-8
    end
    # an isotropic distribution of orientations
    @test lc_rel(
        lc_eff(lay, MoriTanaka(), :C; P_I = Cf, P_M = CM, symmetrize = :iso),
        lc_eff(LC_HOM, MoriTanaka(), :C; P_I = Cf, P_M = CM, symmetrize = :iso)
    ) < 1.0e-10
    # a coated fiber lies between the fiber made of its coating and of its core
    coated = LayeredCylinder((0.8, 1.0), (Cf, lc_tensor(LC_M)))
    p(C) = LCyl._hill_moduli(TensND.best_fit_ti(C, (0.0, 0.0, 1.0)), (0.0, 0.0, 1.0)).p
    lo = p(lc_eff(LC_HOM, MoriTanaka(), :C; P_I = lc_tensor(LC_M), P_M = CM))
    hi = p(lc_eff(LC_HOM, MoriTanaka(), :C; P_I = Cf, P_M = CM))
    @test lo < p(lc_eff(coated, MoriTanaka(), :C; P_I = Cf, P_M = CM)) < hi
end

@testset "LayeredCylinder in the schemes — conduction" begin
    Kf, KM = lc_K(5.0, 7.0), TensISO{3}(2.0)
    lay = LayeredCylinder((1.0,), (Kf,))
    for scheme in LC_SCHEMES
        ref = lc_eff(LC_HOM, scheme, :K; P_I = Kf, P_M = KM)
        @test lc_rel(lc_eff(lay, scheme, :K; P_I = Kf, P_M = KM), ref) < 1.0e-8
        @test lc_rel(lc_eff(lay, scheme, :K; P_I = Kf, P_M = KM, declared = KM), ref) < 1.0e-8
    end
    @test lc_rel(
        lc_eff(lay, SelfConsistent(), :K; P_I = Kf, P_M = KM, matrix_geom = LayeredCylinder((1.0,), (KM,))),
        lc_eff(LC_HOM, SelfConsistent(), :K; P_I = Kf, P_M = KM)
    ) < 1.0e-8
end

@testset "LayeredCylinder — layer averages for the bounds" begin
    Cf, Cm = lc_tensor(LC_F), TensISO{3}(30.0, 8.0)
    cyl = LayeredCylinder((0.6, 1.0), (Cf, Cm))
    L = MeanFieldHomogenization.LayeredSpheres
    @test L.layer_stiffness_average(cyl) ≈ 0.36 * Cf + 0.64 * Cm
    @test L.layer_compliance_average(cyl) ≈ 0.36 * inv(Cf) + 0.64 * inv(Cm)
    @test L.layer_property_average(cyl, Cf) ≈ L.layer_stiffness_average(cyl)
    @test L.layer_inverse_property_average(cyl, Cf) ≈ L.layer_compliance_average(cyl)
    ck = LayeredCylinder((0.6, 1.0), (lc_K(5.0, 7.0), TensISO{3}(2.0)))
    @test lc_cc(L.layer_conductivity_average(ck)) ≈ Diagonal([0.36 * 5 + 0.64 * 2, 0.36 * 5 + 0.64 * 2, 0.36 * 7 + 0.64 * 2])
    @test lc_cc(L.layer_resistivity_average(ck)) ≈ Diagonal([0.36 / 5 + 0.64 / 2, 0.36 / 5 + 0.64 / 2, 0.36 / 7 + 0.64 / 2])
end

@testset "LayeredCylinder — Newton self-consistent in a mixed cell" begin
    # Spheres in a cell with aligned cylinders: the estimate widens from isotropic
    # to transversely isotropic, then to a general tensor (an isotropic sphere in a
    # transversely isotropic reference). The Newton unknowns follow every widening,
    # for the homogeneous cylinder as for the layered one, and the root is Picard's.
    K = lc_K(5.0, 7.0)
    C = lc_tensor(LC_F)
    for (prop, P_I, P_M) in ((:K, K, TensISO{3}(2.0)), (:C, C, TensISO{3}(3 * 6.0, 2 * 2.5)))
        for geom in (LC_HOM, LayeredCylinder((1.0,), (P_I,)))
            r = RVE()
            add_phase!(r, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(prop => P_M); fraction = :rest)
            add_phase!(r, :I, geom, Dict(prop => P_I); fraction = 0.3)
            newton = homogenize(r, SelfConsistent(algorithm = NewtonDefault(), reltol = 1.0e-12), prop)
            @test lc_rel(newton, homogenize(r, SelfConsistent(reltol = 1.0e-12), prop)) < 1.0e-9
        end
    end
end
