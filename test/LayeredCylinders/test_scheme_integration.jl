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

@testset "LayeredCylinder in the schemes — aligned coated fibers" begin
    CM = TensISO{3}(3 * 6.0, 2 * 2.5)
    coated = LayeredCylinder((0.7, 1.0), (lc_tensor(LC_F), lc_tensor(LC_M)); interfaces = (SpringInterface(20.0, 10.0), PerfectInterface()))
    host = LayeredCylinder((1.0,), (CM,))
    # the self-consistent equation at its root: Σ fᵢ Nᵢ(C) = 0 against the estimate
    C = lc_eff(coated, SelfConsistent(; reltol = 1.0e-12), :C; P_I = CM, P_M = CM, matrix_geom = host)
    Cti = TensND.best_fit_ti(C, (0.0, 0.0, 1.0))
    residual = 0.7 * MeanFieldHomogenization.property_contribution(host, Cti) +
        0.3 * MeanFieldHomogenization.property_contribution(coated, Cti)
    @test norm(lc_data(residual)) < 1.0e-9 * norm(lc_data(Cti))
    # an orientation average about the fiber axis changes nothing
    @test lc_rel(
        lc_eff(coated, MoriTanaka(), :C; P_I = CM, P_M = CM, symmetrize = TISymmetrize((0.0, 0.0, 1.0))),
        lc_eff(coated, MoriTanaka(), :C; P_I = CM, P_M = CM)
    ) < 1.0e-12
    # a host whose layer is not its declared property is refused
    @test_throws "declared property" lc_eff(coated, MoriTanaka(), :C; P_I = CM, P_M = CM, matrix_geom = LayeredCylinder((1.0,), (lc_tensor(LC_M),)))
end

@testset "Self-consistent Newton — a class that narrows, a class that alternates" begin
    # a transversely isotropic matrix, every phase averaged over isotropic
    # orientations: the estimate narrows to the isotropic class
    r = RVE()
    CM = lc_ti(12.0, 4.0, 3.0, 20.0, 5.0)
    add_phase!(r, :M, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => CM); fraction = :rest, symmetrize = :iso)
    add_phase!(r, :I, Ellipsoid(1.0, 1.0, 1.0), Dict(:C => TensISO{3}(90.0, 40.0)); fraction = 0.3, symmetrize = :iso)
    newton = homogenize(r, SelfConsistent(algorithm = NewtonDefault(), reltol = 1.0e-12), :C)
    @test lc_rel(newton, homogenize(r, SelfConsistent(reltol = 1.0e-12), :C)) < 1.0e-9
    # a step that alternates between two classes has no class for the unknowns
    iso = TensISO{3}(3.0, 2.0)
    ti = lc_ti(1.5, 0.5, 1.0, 3.0, 1.0)
    alternating(x) = x isa TensISO ? ti : iso
    @test_throws "alternates" MeanFieldHomogenization.Schemes._sc_newton_seed(iso, alternating)
end

@testset "Aligned coated fibers in a transversely isotropic matrix, any axis" begin
    # Every phase transversely isotropic about the fiber axis n, the matrix a
    # one-layer cylinder: the self-consistent, asymmetric and differential
    # estimates are transversely isotropic about n. Turning the cell from e₃ to
    # n turns the estimate and leaves its Hill moduli unchanged.
    n = Tuple(normalize([1.0, -2.0, 2.0]))
    coat = (k = 8.0, l = 3.0, m = 2.5, n = 20.0, p = 3.0)
    function cell(axis)
        CM = lc_tensor(LC_M; axis)
        fiber = LayeredCylinder(
            (0.7, 1.0), (lc_tensor(LC_F; axis), lc_tensor(coat; axis));
            interfaces = (SpringInterface(20.0, 10.0), PerfectInterface()), axis
        )
        host = LayeredCylinder((1.0,), (CM,); axis)
        r = RVE(; distribution_shape = host)
        add_phase!(r, :M, host, Dict(:C => CM); fraction = :rest)
        add_phase!(r, :I, fiber, Dict(:C => CM); fraction = 0.3)
        return r, host, fiber
    end
    hill(C, axis) = LCyl._hill_moduli(TensND.best_fit_ti(C, axis), axis)
    r₃, _, _ = cell((0.0, 0.0, 1.0))
    rₙ, host, fiber = cell(n)
    dscheme(formulation) = DifferentialScheme(; formulation, abstol = 1.0e-11, reltol = 1.0e-11)
    for scheme in (SelfConsistent(; reltol = 1.0e-12), AsymmetricSelfConsistent(:M; reltol = 1.0e-12), dscheme(:stiffness), dscheme(:compliance))
        Cₙ = homogenize(rₙ, scheme, :C)
        # transversely isotropic about n
        @test norm(lc_cc(Cₙ) - lc_cc(TensND.best_fit_ti(Cₙ, n))) < 1.0e-12 * norm(lc_cc(Cₙ))
        h₃, hₙ = hill(homogenize(r₃, scheme, :C), (0.0, 0.0, 1.0)), hill(Cₙ, n)
        @test all(isapprox(getfield(hₙ, s), getfield(h₃, s); rtol = 1.0e-12) for s in (:k, :l, :m, :n, :p))
    end
    # the self-consistent equation at its root, against the estimate
    C = TensND.best_fit_ti(homogenize(rₙ, SelfConsistent(; reltol = 1.0e-12), :C), n)
    residual = 0.7 * MeanFieldHomogenization.property_contribution(host, C) + 0.3 * MeanFieldHomogenization.property_contribution(fiber, C)
    @test norm(lc_data(residual)) < 1.0e-9 * norm(lc_data(C))
    # the two formulations of the differential scheme
    @test lc_rel(homogenize(rₙ, dscheme(:stiffness), :C), homogenize(rₙ, dscheme(:compliance), :C)) < 1.0e-9
end
