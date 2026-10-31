using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

# =============================================================================
#  test_crack_order2_alv.jl — insulating cracks in an isotropic ALV conductor:
#  the resistivity R̃ = k b̃ n ⊗ n, b̃ = (4/3ℰ) λ̃^{-∘} (ellipse) or (π/2) λ̃^{-∘}
#  (ribbon), and the crack families in every order-2 ALV scheme. References:
#  the elastic thermal crack under constant kernels, and the flat limit of
#  insulating ellipsoids and cylinders through their own order-2 Hill kernel.
# =============================================================================

const _O2_ANGLES = (0.3, 1.1, -0.4)
_o2_components(K) = Array(canonical_components(K))

@testset "ALV crack resistivity — elastic limit, any shape and orientation, interfaces" begin
    times = [0.0, 0.5, 1.5]
    n = length(times)
    k = 2.5
    for crack in (PennyCrack(1.0; euler_angles = _O2_ANGLES), EllipticCrack(1.0, 0.3; euler_angles = _O2_ANGLES), RibbonCrack(1.0; euler_angles = _O2_ANGLES))
        R = inverse_property_contribution_alv(crack, heaviside_law(TensISO{3}(k)), times)
        @test R.class isa TensTI
        R_el = _o2_components(inverse_property_contribution(crack, TensISO{3}(k)))
        M = Matrix(R)
        @test all(M[(3i - 2):(3i), (3i - 2):(3i)] ≈ R_el for i in 1:n)
        @test all(maximum(abs, M[(3i - 2):(3i), (3j - 2):(3j)]) ≤ 1.0e-12 * norm(R_el) for i in 1:n for j in 1:(i - 1))
        # A Kapitza-like interface: the elastic b / (1 + b α b).
        Ri = Matrix(inverse_property_contribution_alv(crack, heaviside_law(TensISO{3}(k)), times; α_interface = heaviside_law(0.7)))
        @test Ri[1:3, 1:3] ≈ _o2_components(inverse_property_contribution(crack, TensISO{3}(k); α_interface = 0.7))
    end
    # Each interface belongs to its order; a crack of unknown conduction is refused.
    K = heaviside_law(TensISO{3}(k))
    @test_throws "Rn and Rt are the interfaces of a crack in elasticity" inverse_property_contribution_alv(PennyCrack(1.0), K, times; Rn = heaviside_law(1.0))
    @test_throws "α_interface is the interface of a crack in conduction" inverse_property_contribution_alv(PennyCrack(1.0), heaviside_law(TensISO{3}(3.0, 2.0)), times; α_interface = heaviside_law(1.0))
    @test_throws "implemented for insulating elliptic cracks" inverse_property_contribution_alv(ConductiveCrack(1.0; conductivity = 2.0), K, times)
    @test_throws "isotropic ALV conductor only" inverse_property_contribution_alv(PennyCrack(1.0), heaviside_law(TensTI{2}(1.0, 2.0, (0.0, 0.0, 1.0))), times)
end

@testset "ALV crack resistivity — the flat limit of insulating ellipsoids and cylinders, aging conductor" begin
    # A crack of density ε is the limit of insulating ellipsoids (a, b, ω b) at the
    # volume fraction (4π/3) ω ε, and a ribbon that of elliptic cylinders (b, ω b)
    # at the area fraction π ω ε, whose order-2 Hill kernels are computed
    # independently of the crack closed forms (error of order ω).
    times = [0.0, 0.3, 0.7, 1.2, 2.0]
    K0 = ViscoLaw((t, tp) -> t >= tp ? TensISO{3}(2.0 * (1 + 0.5tp) * (0.3 + exp(-(t - tp) / 0.9))) : TensISO{3}(0.0))
    insulating = ViscoLaw((t, tp) -> TensISO{3}(0.0))
    ε, ω = 0.1, 1.0e-6
    for (crack, flat, fraction) in (
            (EllipticCrack(1.0, 0.4; euler_angles = _O2_ANGLES), Ellipsoid(1.0, 0.4, 0.4ω; euler_angles = _O2_ANGLES), 4π / 3 * ω * ε),
            (RibbonCrack(1.0; euler_angles = _O2_ANGLES), Ellipsoid(Inf, 1.0, ω; euler_angles = _O2_ANGLES), π * ω * ε),
        )
        cracked, porous = RVE(), RVE()
        add_phase!(cracked, :M, Ellipsoid(1.0), Dict(:K => K0); fraction = :rest)
        add_phase!(cracked, :F, crack, Dict(:K => K0); density = ε)
        add_phase!(porous, :M, Ellipsoid(1.0), Dict(:K => K0); fraction = :rest)
        add_phase!(porous, :V, flat, Dict(:K => insulating); fraction)
        K_crack = homogenize_alv(cracked, Dilute(:M), :K; times)
        @test K_crack.class isa TensTI
        @test Matrix(K_crack) ≈ Matrix(homogenize_alv(porous, Dilute(:M), :K; times)) rtol = 1.0e-5
    end
end

@testset "ALV cracks in the order-2 schemes — elastic limit" begin
    # Under constant kernels every scheme gives the elastic estimate with thermal
    # cracks at every instant: oblique cracks for the schemes on the matrix (Voigt
    # and Reuss pass them over), averaged in orientation for those on the running
    # medium.
    times = [0.0, 1.0]
    k = 2.5
    for crack in (EllipticCrack(1.0, 0.3; euler_angles = _O2_ANGLES), RibbonCrack(1.0; euler_angles = _O2_ANGLES))
        for (scheme, kw, rtol) in (
                (Voigt(), (;), 1.0e-14), (Reuss(), (;), 1.0e-14),
                (Dilute(), (;), 1.0e-14), (DiluteDual(), (;), 1.0e-14), (MoriTanaka(), (;), 1.0e-14),
                (Maxwell(), (;), 1.0e-14), (PonteCastanedaWillis(), (;), 1.0e-14),
                (SelfConsistent(reltol = 1.0e-13), (; symmetrize = :iso), 1.0e-11),
                (DifferentialScheme(abstol = 1.0e-12, reltol = 1.0e-11), (; symmetrize = :iso), 1.0e-10),
                (DifferentialScheme(abstol = 1.0e-12, reltol = 1.0e-11, formulation = :compliance), (; symmetrize = :iso), 1.0e-10),
            )
            el, al = RVE(; distribution_shape = Spheroid(0.5)), RVE(; distribution_shape = Spheroid(0.5))
            add_phase!(el, :M, Ellipsoid(1.0), Dict(:K => TensISO{3}(k)); fraction = :rest)
            add_phase!(al, :M, Ellipsoid(1.0), Dict(:K => heaviside_law(TensISO{3}(k))); fraction = :rest)
            add_phase!(el, :I, Ellipsoid(1.0), Dict(:K => TensISO{3}(10k)); fraction = 0.2)
            add_phase!(al, :I, Ellipsoid(1.0), Dict(:K => heaviside_law(TensISO{3}(10k))); fraction = 0.2)
            add_phase!(el, :F, crack, Dict(:K => TensISO{3}(k)); density = 0.15, kw...)
            add_phase!(al, :F, crack, Dict(:K => heaviside_law(TensISO{3}(k))); density = 0.15, kw...)
            E = _o2_components(homogenize(el, scheme, :K))
            A = Matrix(homogenize_alv(al, scheme, :K; times))
            @test A[1:3, 1:3] ≈ E rtol = rtol
            @test A[4:6, 4:6] ≈ E rtol = rtol
        end
    end
end

@testset "ALV cracks, order-2 self-consistent — Newton and Picard, aging, interfaces" begin
    # The two solvers of the self-consistent equation reach the same estimate, with
    # a crack family held by an aging interface, in the classes of the operators and
    # on full matrices (a layered sphere).
    times = [0.0, 0.4, 1.5, 3.0]
    K0 = ViscoLaw((t, tp) -> t >= tp ? TensISO{3}(2.0 * (1 + 0.5tp) * (0.3 + exp(-(t - tp) / 0.9))) : TensISO{3}(0.0))
    K1 = ViscoLaw((t, tp) -> t >= tp ? TensISO{3}(5.0 * exp(-(t - tp) / 1.4)) : TensISO{3}(0.0))
    for inclusion in (Ellipsoid(1.0), LayeredSphere((0.7, 1.0), (TensISO{3}(20.0), TensISO{3}(1.0))))
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0), Dict(:K => K0); fraction = :rest)
        add_phase!(r, :I, inclusion, Dict(:K => K1); fraction = 0.3)
        add_phase!(r, :F, EllipticCrack(1.0, 0.4), Dict(:K => K0, :α_interface => ViscoLaw((t, tp) -> 0.5 * exp(-(t - tp)))); density = 0.2, symmetrize = :iso)
        picard = Matrix(homogenize_alv(r, SelfConsistent(reltol = 1.0e-13, maxiters = 3000), :K; times))
        @test Matrix(homogenize_alv(r, SelfConsistent(algorithm = NewtonDefault(), reltol = 1.0e-12), :K; times)) ≈ picard rtol = 1.0e-10
    end
    # Without an isotropic orientation average the estimate would leave its class.
    r = RVE()
    add_phase!(r, :M, Ellipsoid(1.0), Dict(:K => K0); fraction = :rest)
    add_phase!(r, :F, PennyCrack(1.0), Dict(:K => K0); density = 0.2)
    @test_throws "self_consistent_alv_order2: the transversely isotropic crack contribution of phase :F" homogenize_alv(r, SelfConsistent(), :K; times)
    @test_throws "differential_alv_order2: the transversely isotropic crack contribution of phase :F" homogenize_alv(r, DifferentialScheme(), :K; times)
    s = RVE()
    add_phase!(s, :M, Ellipsoid(1.0), Dict(:K => K0); fraction = :rest)
    add_phase!(s, :F, Ellipsoid(1.0), Dict(:K => K0); density = 0.2)
    for scheme in (Dilute(), SelfConsistent(), DifferentialScheme())
        @test_throws "has CrackDensity but geometry" homogenize_alv(s, scheme, :K; times)
    end
end

@testset "ALV cracks, order 2 — derivatives in the density" begin
    # The dilute estimate is affine in the density, of slope the crack contribution
    # −(4π/3) K₀ R K₀ of the elastic thermal crack under a constant kernel; the
    # self-consistent one has the slope of the elastic self-consistent estimate,
    # differentiated through its own solver.
    times = [0.0, 1.0]
    k = 2.5
    crack = EllipticCrack(1.0, 0.3; euler_angles = _O2_ANGLES)
    function cell(ε, law; kw...)
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1.0), Dict(:K => law(TensISO{3}(k))); fraction = :rest)
        add_phase!(r, :F, crack, Dict(:K => law(TensISO{3}(k))); density = ε, kw...)
        return r
    end
    slope = ForwardDiff.derivative(ε -> Matrix(homogenize_alv(cell(ε, heaviside_law), Dilute(), :K; times)), 0.1)
    K₀ = k * I(3)
    @test slope[1:3, 1:3] ≈ -(4π / 3) * K₀ * _o2_components(inverse_property_contribution(crack, TensISO{3}(k))) * K₀
    sc_alv = ForwardDiff.derivative(ε -> Matrix(homogenize_alv(cell(ε, heaviside_law; symmetrize = :iso), SelfConsistent(reltol = 1.0e-13), :K; times)), 0.1)
    sc_el = ForwardDiff.derivative(ε -> _o2_components(homogenize(cell(ε, identity; symmetrize = :iso), SelfConsistent(reltol = 1.0e-13), :K)), 0.1)
    @test sc_alv[4:6, 4:6] ≈ sc_el rtol = 1.0e-9
end
