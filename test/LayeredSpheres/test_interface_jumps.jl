using Test
using MeanFieldHomogenization
using TensND
using ForwardDiff

const MFH_IJ = MeanFieldHomogenization
const LS_IJ = MeanFieldHomogenization.LayeredSpheres

# =============================================================================
#  The opening of a primal interface in the average strain (gradient) of a
#  composite sphere.
#
#  The sum of the layer averages is the strain of the MATERIAL. Read from the
#  matrix, the sphere also strains by the opening of its spring interfaces,
#  and the strain average rule E = Σ fᵢ⟨ε⟩ᵢ needs that opening in some phase.
#  The concentration tensors used to leave it out, so every scheme was wrong
#  as soon as a spring or a Kapitza interface was present; the Kapitza jump
#  also had the wrong sign (a negative resistance).
#
#  The references come from three independent sources: closed forms, Echoes
#  (`sphere_nlayers`, C++), and the confocal spheroid of this package in the
#  spherical limit.
# =============================================================================

_iso(A) = TensND.get_data(A)

@testset "Interface jumps — a spring-bonded grain is an equivalent grain" begin
    # Under p𝟏 the grain strains by p/3kₛ and the spring opens by p/kₙ, so the
    # sphere responds as a homogeneous grain of modulus kₛ/(1 + 3kₛ/(kₙR)).
    k₀, μ₀ = 2.0, 1.5
    C₀ = iso_stiffness(k₀, μ₀)
    for (kₛ, μₛ, kₙ, R) in ((10.0, 1.0, 10.0, 1.0), (4.0, 3.0, 0.7, 2.5), (50.0, 20.0, 300.0, 0.3))
        s = LayeredSphere((R,), (iso_stiffness(kₛ, μₛ),); interfaces = (SpringInterface(kₙ, Inf),))
        k_eq = kₛ / (1 + 3kₛ / (kₙ * R))
        α_eq = (3k₀ + 4μ₀) / (3k_eq + 4μ₀)
        @test _iso(strain_strain_loc(s, C₀, C₀))[1] ≈ α_eq rtol = 1.0e-13
        # Without the outer opening, the strain of the grain alone.
        @test _iso(strain_strain_loc(s, C₀, C₀; external = false))[1] ≈
            α_eq * k_eq / kₛ rtol = 1.0e-13
        # The average stress does not see the opening: the traction is continuous.
        @test _iso(LS_IJ.stress_strain_loc(s, C₀, C₀))[1] ≈ 3k_eq * α_eq rtol = 1.0e-13
    end
end

@testset "Interface jumps — Kapitza: Hasselman–Johnson equivalent grain" begin
    # Across a resistance the temperature drops in the direction of the flux,
    # so the grain conducts as k₁/(1 + ρk₁/R): LOWER than k₁.
    k₀ = 2.0
    K₀ = TensISO{3}(k₀)
    for (k₁, ρ, R) in ((5.0, 0.1, 1.0), (0.3, 2.0, 0.5), (40.0, 0.01, 3.0))
        s = LayeredSphere((R,), (TensISO{3}(k₁),); interfaces = (KapitzaInterface(ρ),))
        k_eq = k₁ / (1 + ρ * k₁ / R)
        α_eq = 3k₀ / (2k₀ + k_eq)
        @test _iso(gradient_gradient_loc(s, K₀, K₀))[1] ≈ α_eq rtol = 1.0e-13
        @test _iso(gradient_gradient_loc(s, K₀, K₀; external = false))[1] ≈
            α_eq * k_eq / k₁ rtol = 1.0e-13
        @test _iso(LS_IJ.flux_gradient_loc(s, K₀, K₀))[1] ≈ k_eq * α_eq rtol = 1.0e-13
    end
end

@testset "Interface jumps — Echoes reference values" begin
    # Echoes `sphere_nlayers` with `PRIMALDISC` interfaces (stiffnesses for
    # elasticity, a conductance h = 1/ρ for transport), reference
    # `stiff_kmu(2, 1.5)` or `2 tId2`; `eE`, `sE`, `sphere_eE`, `layer_eE` read
    # as (α, β) = (d + 2o, d − o) from the 6×6 Mandel array.
    C₀ = iso_stiffness(2.0, 1.5)
    s1 = LayeredSphere((1.0,), (iso_stiffness(10.0, 1.0),); interfaces = (SpringInterface(10.0, 1.0),))
    @test collect(_iso(strain_strain_loc(s1, C₀, C₀))) ≈
        [0.8888888888888888, 1.438070030514018] rtol = 1.0e-12                  # eE
    @test collect(_iso(strain_strain_loc(s1, C₀, C₀; external = false))) ≈
        [0.22222222222222238, 0.8428949542289732] rtol = 1.0e-12                # sphere_eE(0, external=False)

    s2 = LayeredSphere(
        (0.6, 1.0), (iso_stiffness(10.0, 4.0), iso_stiffness(3.0, 2.0));
        interfaces = (SpringInterface(5.0, 2.0), SpringInterface(8.0, 3.0))
    )
    @test collect(_iso(strain_strain_loc(s2, C₀, C₀))) ≈
        [1.2295343215373222, 1.2460210239870269] rtol = 1.0e-12                 # eE
    @test collect(_iso(LS_IJ.stress_strain_loc(s2, C₀, C₀))) ≈
        [4.622794070776065, 2.26193692803892] rtol = 1.0e-12                    # sE
    @test collect(_iso(strain_strain_loc(s2, C₀, C₀; external = false))) ≈
        [0.6516850626903141, 0.7129372907003615] rtol = 1.0e-12                 # sphere_eE(1, external=False)
    layer_eE = (
        (ext = [0.9169408499600372, 1.1056931300520458], mat = [0.0833582590872761, 0.21152004027919052]),
        (
            ext = [1.3156570126861662, 1.284682790683399], mat = [0.5786043866057988, 0.6047290492463261],
            int = [0.8082648963360491, 0.8510828596939494],
        ),
    )
    for k in 1:2
        @test collect(_iso(strain_strain_loc(s2, C₀; layer = k, external = true))) ≈ layer_eE[k].ext rtol = 1.0e-12
        @test collect(_iso(strain_strain_loc(s2, C₀; layer = k))) ≈ layer_eE[k].mat rtol = 1.0e-12
    end
    @test collect(_iso(strain_strain_loc(s2, C₀; layer = 2, internal = true))) ≈ layer_eE[2].int rtol = 1.0e-12

    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = 0.6)
    add_phase!(rve, :S, s2, Dict(:C => C₀); fraction = 0.4)
    @test collect(k_mu(homogenize(rve, MoriTanaka(:M), :C))) ≈
        [1.6636285981441048, 1.2312245370614585] rtol = 1.0e-12
    @test collect(k_mu(homogenize(rve, SelfConsistent(), :C))) ≈
        [1.659332077419758, 1.2273096410408015] rtol = 1.0e-8

    # A grain bonded by springs, and a pore, in the self-consistent scheme.
    grain = LayeredSphere((1.0,), (iso_stiffness(10.0, 1.0),); interfaces = (SpringInterface(10.0, 1.0),))
    rock = RVE()
    add_phase!(rock, :S, grain, Dict(:C => iso_stiffness(1.0, 1.0)); fraction = 0.75)
    add_phase!(rock, :P, Ellipsoid(1.0), Dict(:C => iso_stiffness(0.0, 0.0)); fraction = 0.25)
    @test collect(k_mu(homogenize(rock, SelfConsistent(), :C))) ≈
        [0.696776997860494, 0.277209167665217] rtol = 1.0e-7

    K₀ = TensISO{3}(2.0)
    k1 = LayeredSphere((1.0,), (TensISO{3}(5.0),); interfaces = (KapitzaInterface(0.1),))
    @test _iso(gradient_gradient_loc(k1, K₀, K₀))[1] ≈ 0.8181818181818181 rtol = 1.0e-12
    @test _iso(gradient_gradient_loc(k1, K₀, K₀; external = false))[1] ≈ 0.5454545454545455 rtol = 1.0e-12
    k2 = LayeredSphere(
        (0.6, 1.0), (TensISO{3}(5.0), TensISO{3}(1.0));
        interfaces = (KapitzaInterface(0.1), KapitzaInterface(0.25))
    )
    @test _iso(gradient_gradient_loc(k2, K₀, K₀))[1] ≈ 1.2105384615384616 rtol = 1.0e-12
    @test _iso(LS_IJ.flux_gradient_loc(k2, K₀, K₀))[1] ≈ 1.1578461538461537 rtol = 1.0e-12
    rk = RVE()
    add_phase!(rk, :M, Ellipsoid(1.0), Dict(:K => K₀); fraction = 0.6)
    add_phase!(rk, :S, k2, Dict(:K => K₀); fraction = 0.4)
    @test _iso(homogenize(rk, MoriTanaka(:M), :K))[1] ≈ 1.533955785112 rtol = 1.0e-11
end

@testset "Interface jumps — the shells partition the sphere" begin
    C₀ = iso_stiffness(2.0, 1.5)
    s = LayeredSphere(
        (0.4, 0.7, 1.3), (iso_stiffness(10.0, 4.0), iso_stiffness(3.0, 2.0), iso_stiffness(6.0, 1.0));
        interfaces = (SpringInterface(5.0, 2.0), MembraneInterface(0.3, 0.2), SpringInterface(8.0, 3.0))
    )
    f = ntuple(k -> layer_volume_fraction(s, k), 3)
    # Each layer owns its outer interface…
    A_parts = sum(f[k] .* collect(_iso(strain_strain_loc(s, C₀; layer = k, external = true))) for k in 1:3)
    @test A_parts ≈ collect(_iso(strain_strain_loc(s, C₀, C₀))) rtol = 1.0e-12
    # …or its inner one, the outermost opening then going to the matrix.
    A_inner = sum(f[k] .* collect(_iso(strain_strain_loc(s, C₀; layer = k, internal = true))) for k in 1:3)
    @test A_inner ≈ collect(_iso(strain_strain_loc(s, C₀, C₀; external = false))) rtol = 1.0e-12
    # The material average is what the averages of `averages.jl` return.
    ε∞ = TensISO{3}(1.0)
    A_mat = sum(f[k] .* collect(_iso(strain_strain_loc(s, C₀; layer = k))) for k in 1:3)
    @test get_array(sphere_strain_average(s, C₀, ε∞)) ≈ get_array(TensISO{3}(A_mat...) ⊡ ε∞) rtol = 1.0e-12
end

@testset "Interface jumps — N = B − C₀:A for either owner of the outer interface" begin
    C₀ = iso_stiffness(2.0, 1.5)
    K₀ = TensISO{3}(2.0)
    for outer in (SpringInterface(8.0, 3.0), MembraneInterface(0.3, 0.2))
        s = LayeredSphere(
            (0.6, 1.0), (iso_stiffness(10.0, 4.0), iso_stiffness(3.0, 2.0));
            interfaces = (SpringInterface(5.0, 2.0), outer)
        )
        for external in (true, false)
            A = strain_strain_loc(s, C₀, C₀; external)
            B = LS_IJ.stress_strain_loc(s, C₀, C₀; external)
            Nc = stiffness_contribution(s, C₀, C₀; external)
            @test get_array(Nc) ≈ get_array(B - C₀ ⊡ A) rtol = 1.0e-12
            @test get_array(stiffness_contribution(s, C₀; external)) ≈ get_array(Nc) rtol = 1.0e-12
            # The bundles return the separate results bit for bit.
            AN = MFH_IJ.Core.loc_and_stiffness(s, C₀, C₀; external)
            AB = MFH_IJ.Core.loc_and_stress_average(s, C₀, C₀; external)
            @test AN[1] == A && AN[2] == Nc
            @test AB[1] == A && AB[2] == B
        end
    end
    for outer in (KapitzaInterface(0.25), SurfaceConductiveInterface(0.4))
        s = LayeredSphere(
            (0.6, 1.0), (TensISO{3}(5.0), TensISO{3}(1.0));
            interfaces = (KapitzaInterface(0.1), outer)
        )
        for external in (true, false)
            A = gradient_gradient_loc(s, K₀, K₀; external)
            B = LS_IJ.flux_gradient_loc(s, K₀, K₀; external)
            Nc = conductivity_contribution(s, K₀, K₀; external)
            @test _iso(Nc)[1] ≈ _iso(B)[1] - 2.0 * _iso(A)[1] rtol = 1.0e-12
            AN = MFH_IJ.Core.loc_and_stiffness(s, K₀, K₀; external)
            AB = MFH_IJ.Core.loc_and_stress_average(s, K₀, K₀; external)
            @test AN[1] == A && AN[2] == Nc
            @test AB[1] == A && AB[2] == B
        end
    end
end

@testset "Interface jumps — perfect and membrane interfaces are untouched" begin
    # No opening: the jump term is an exact zero, not a rounding residue.
    C₀ = iso_stiffness(2.0, 1.5)
    for itf in (PerfectInterface{Float64}(), MembraneInterface(0.3, 0.2))
        s = LayeredSphere((0.6, 1.0), (iso_stiffness(10.0, 4.0), iso_stiffness(3.0, 2.0)); interfaces = (itf, itf))
        α, β, f = LS_IJ._layer_localizations(s, C₀)
        @test collect(_iso(strain_strain_loc(s, C₀, C₀))) == [sum(f .* α), sum(f .* β)]
    end
end

@testset "Interface jumps — Kapitza agrees with the confocal spheroid" begin
    # Two independent solvers: the spheroid's coupled spheroidal harmonics,
    # which always counted the jump with the right sign, near the sphere.
    K₀ = TensISO{3}(2.0)
    K₁ = TensISO{3}(5.0)
    sphere = LayeredSphere((1.0,), (K₁,); interfaces = (KapitzaInterface(0.1),))
    α_sphere = _iso(gradient_gradient_loc(sphere, K₁, K₀))[1]
    spheroid = LayeredSpheroid((1.0,), (0.9999,), (K₁,); interfaces = (KapitzaInterface(0.1),), Nseries = 8)
    A = get_array(gradient_gradient_loc(spheroid, K₁, K₀))
    @test A[1, 1] ≈ α_sphere rtol = 1.0e-4
    @test A[3, 3] ≈ α_sphere rtol = 1.0e-4
end

@testset "Interface jumps — the ALV kernels in the elastic limit" begin
    C_M = TensISO{3}(30.0, 8.0)
    sphere = LayeredSphere(
        (0.5, 1.0), (TensISO{3}(60.0, 16.0), TensISO{3}(90.0, 24.0));
        interfaces = (SpringInterface(; sn = 0.05, st = 0.07), SpringInterface(; sn = 0.02, st = 0.03))
    )
    times = collect(0.0:0.5:1.0)
    n = length(times)
    to66 = MFH_IJ.Viscoelasticity._tens_to_mandel66
    for external in (true, false)
        A_alv = strain_strain_loc_alv(sphere, heaviside_law(C_M), times; external)
        N_alv = stiffness_contribution_alv(sphere, heaviside_law(C_M), times; external)
        A_el = to66(strain_strain_loc(sphere, C_M, C_M; external))
        N_el = to66(stiffness_contribution(sphere, C_M; external))
        for i in 1:n
            rows = (6 * (i - 1) + 1):(6 * i)
            @test isapprox(A_alv[rows, rows], A_el; rtol = 1.0e-10, atol = 1.0e-10)
            @test isapprox(N_alv[rows, rows], N_el; rtol = 1.0e-10, atol = 1.0e-10)
        end
    end
end

@testset "Interface jumps — the derivative with respect to kₙ is exact" begin
    k₀, μ₀, kₛ, μₛ = 2.0, 1.5, 10.0, 1.0
    C₀ = iso_stiffness(k₀, μ₀)
    α(kₙ) = _iso(
        strain_strain_loc(
            LayeredSphere((1.0,), (iso_stiffness(kₛ, μₛ),); interfaces = (SpringInterface(kₙ, Inf),)),
            C₀, C₀
        )
    )[1]
    kₙ = 3.0
    k_eq = kₛ / (1 + 3kₛ / kₙ)
    dk_eq = 3kₛ^2 / (kₙ + 3kₛ)^2
    exact = -3 * (3k₀ + 4μ₀) * dk_eq / (3k_eq + 4μ₀)^2
    @test ForwardDiff.derivative(α, kₙ) ≈ exact rtol = 1.0e-12
end

@testset "Interface jumps — Dormieux et al. (2010), self-consistent closed forms" begin
    # Dormieux, Jeannin, Bemer, Le & Sanahuja, IJNAMG 34 (2010) 249–271, grains bonded by
    # springs, and pores, in the self-consistent scheme. The average strain of a
    # grain is read on the outer lip of its interface (their eq. 43), which is
    # `external = true`. Eq. (49) holds for any kₛ, kₙ; eqs. (51)-(52) are
    # the limit kₛ = kₙ = ∞, κ = kₜR/μₛ.
    function sc(kₛ, μₛ, kₙ, kₜ, φ; R = 1.0)
        grain = LayeredSphere((R,), (iso_stiffness(kₛ, μₛ),); interfaces = (SpringInterface(kₙ, kₜ),))
        rve = RVE()
        add_phase!(rve, :S, grain, Dict(:C => iso_stiffness(1.0, 1.0)); fraction = 1 - φ)
        add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => iso_stiffness(0.0, 0.0)); fraction = φ)
        scheme = SelfConsistent(init = iso_stiffness(1.0, 1.0), abstol = 1.0e-15, reltol = 1.0e-13, maxiters = 500)
        return k_mu(homogenize(rve, scheme, :C))
    end
    M52(κ, φ) = (a = 16 * (5 + κ) * (3 - φ); b = (9 + 77φ) * κ + 114 * (3φ - 1); c = 57κ * (2φ - 1); (-b + sqrt(b^2 - 4a * c)) / (2a))
    for (κ, φ) in ((1.0, 0.25), (0.1, 0.35), (10.0, 0.3))
        k, μ = sc(Inf, 1.0, Inf, κ, φ)
        @test μ ≈ M52(κ, φ) rtol = 1.0e-9
        @test k ≈ 4 * (1 - φ) * μ / (3φ) rtol = 1.0e-9                       # (51)
    end
    for (kₛ, μₛ, kₙ, kₜ, φ, R) in ((10.0, 1.0, 10.0, 1.0, 0.25, 1.0), (5.0, 2.0, 3.0, 0.7, 0.3, 2.0))
        k, μ = sc(kₛ, μₛ, kₙ, kₜ, φ; R)
        @test k ≈ 4 * (1 - φ) * μ * R * kₙ * kₛ / (12μ * kₛ + 4μ * kₙ * R + 3φ * R * kₙ * kₛ) rtol = 1.0e-9   # (49)
    end
    # A Voigt seed is infinite with an incompressible grain: say so.
    grain = LayeredSphere((1.0,), (iso_stiffness(Inf, 1.0),); interfaces = (SpringInterface(Inf, 1.0),))
    rve = RVE()
    add_phase!(rve, :S, grain, Dict(:C => iso_stiffness(1.0, 1.0)); fraction = 0.75)
    add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => iso_stiffness(0.0, 0.0)); fraction = 0.25)
    @test_throws ArgumentError homogenize(rve, SelfConsistent(), :C)
end

@testset "Interface jumps — incompressible layers, exactly" begin
    k₀, μ₀ = 2.0, 1.5
    C₀ = iso_stiffness(k₀, μ₀)
    # kₛ → ∞ in the equivalent grain kₛ/(1 + 3kₛ/(kₙR)) leaves kₙR/3.
    kₙ, R = 10.0, 1.0
    g = LayeredSphere((R,), (iso_stiffness(Inf, 1.0),); interfaces = (SpringInterface(kₙ, 1.0),))
    α_eq = (3k₀ + 4μ₀) / (kₙ * R + 4μ₀)
    @test _iso(strain_strain_loc(g, C₀, C₀))[1] ≈ α_eq rtol = 1.0e-14
    @test _iso(strain_strain_loc(g, C₀, C₀; external = false))[1] == 0
    @test _iso(LS_IJ.stress_strain_loc(g, C₀, C₀))[1] ≈ kₙ * R * α_eq rtol = 1.0e-14
    # k = ∞ is the end of the k → ∞ road, not a separate branch.
    layered(k) = LayeredSphere(
        (0.6, 1.0), (iso_stiffness(k, 1.0), iso_stiffness(3.0, 2.0));
        interfaces = (SpringInterface(5.0, 2.0), PerfectInterface{Float64}())
    )
    for f in (
            s -> _iso(strain_strain_loc(s, C₀, C₀)),
            s -> _iso(stiffness_contribution(s, C₀, C₀)),
            s -> _iso(stiffness_contribution(s, C₀)),
        )
        @test collect(f(layered(Inf))) ≈ collect(f(layered(1.0e13))) rtol = 1.0e-10
    end
    # The layer averages of the stress add up to the stress of the sphere.
    s = layered(Inf)
    ε∞ = TensISO{3}(1.0) + TensISO{3}(0.0, 1.0) ⊡ Tens(Float64[1 0 0; 0 -1 0; 0 0 0])
    fr = ntuple(k -> layer_volume_fraction(s, k), 2)
    @test get_array(sum(fr[k] * layer_stress_average(s, C₀, ε∞, k) for k in 1:2)) ≈
        get_array(LS_IJ.stress_strain_loc(s, C₀, C₀) ⊡ ε∞) rtol = 1.0e-12
    # Pointwise: the strain is defined and traceless in the incompressible core;
    # the stress is not a function of it, and says so.
    fields = LayeredSphereFields(s, C₀)
    ε = get_array(local_strain(fields, [0.1, 0.2, 0.3], ε∞))
    @test all(isfinite, ε)
    @test abs(ε[1, 1] + ε[2, 2] + ε[3, 3]) ≤ 1.0e-14
    @test_throws ArgumentError local_stress(fields, [0.1, 0.2, 0.3], ε∞)
end
