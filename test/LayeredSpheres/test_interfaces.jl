using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  Imperfect interfaces: spring ( 2014) and membrane
#  (Gurtin-Murdoch 1975) — bulk mode.
# =============================================================================

@testset "SpringInterface — limits recover (bulk)" begin
    κ₀, μ₀ = 100.0, 70.0
    C₀ = TensISO{3}(3κ₀, 2μ₀)
    C₁ = TensISO{3}(3κ₀ * 2, 2μ₀ * 2)

    s_perfect = LayeredSphere((0.5, 1.0), (C₁, C₀))
    α_perfect = MeanFieldHomogenization.LayeredSpheres._bulk_localization(s_perfect, κ₀, μ₀)

    # k → 0 ≡ perfect interface (normal-only)
    intf_tight = (SpringInterface(; sn = 1.0e-14), PerfectInterface{Float64}())
    s_tight = LayeredSphere((0.5, 1.0), (C₁, C₀); interfaces = intf_tight)
    α_tight = MeanFieldHomogenization.LayeredSpheres._bulk_localization(s_tight, κ₀, μ₀)
    for k in 1:2
        @test α_tight[k] ≈ α_perfect[k] rtol = 1.0e-10
    end

    # k → ∞ : core decoupled → core bulk localization → 0
    intf_loose = (SpringInterface(; sn = 1.0e9), PerfectInterface{Float64}())
    s_loose = LayeredSphere((0.5, 1.0), (C₁, C₀); interfaces = intf_loose)
    α_loose = MeanFieldHomogenization.LayeredSpheres._bulk_localization(s_loose, κ₀, μ₀)
    @test abs(α_loose[1]) < 1.0e-6
end

@testset "SpringInterface — monotone in compliance" begin
    κ₀, μ₀ = 100.0, 70.0
    C₀ = TensISO{3}(3κ₀, 2μ₀)
    C₁ = TensISO{3}(3κ₀ * 2, 2μ₀ * 2)

    ks = [1.0e-5, 1.0e-3, 1.0e-1, 1.0]
    αs = Float64[]
    for k in ks
        s = LayeredSphere(
            (0.5, 1.0), (C₁, C₀);
            interfaces = (SpringInterface(; sn = k), PerfectInterface{Float64}())
        )
        push!(αs, MeanFieldHomogenization.LayeredSpheres._bulk_localization(s, κ₀, μ₀)[1])
    end
    for i in 1:(length(ks) - 1)
        @test αs[i] ≥ αs[i + 1]
    end
end

@testset "SpringInterface — stiffness and compliance spellings agree" begin
    # `kn`, `kt` are STIFFNESSES; `sn`, `st` the stored compliances.  The two
    # spellings must name the same interface, in both directions.
    s_norm = SpringInterface(; sn = 0.01)              # tangentially bonded
    s_full = SpringInterface(; sn = 0.01, st = 0.02)
    @test s_norm.sn ≈ 0.01
    @test s_norm.st == 0.0                             # bonded ⇒ exact zero
    @test s_norm.kn ≈ 100.0
    @test isinf(s_norm.kt)                             # bonded ⇒ infinite stiffness
    @test s_full.sn ≈ 0.01
    @test s_full.st ≈ 0.02
    @test s_full.kn ≈ 100.0
    @test s_full.kt ≈ 50.0

    @test SpringInterface(100.0, 50.0) == s_full       # positional = stiffnesses
    @test SpringInterface(; kn = 100.0, kt = 50.0) == s_full
    @test SpringInterface(100.0) == s_norm             # one argument ⇒ bonded
    @test spring_compliances(s_full) == (s_full.sn, s_full.st)
    @test all(spring_stiffnesses(s_full) .≈ (100.0, 50.0))
    @test Set(propertynames(s_full)) == Set((:kn, :kt, :sn, :st))

    # A perfect interface is an exact zero in compliance, not an Inf in
    # stiffness — which is what keeps the near-perfect regime differentiable.
    @test SpringInterface(; sn = 0.0, st = 0.0) == SpringInterface(Inf, Inf)

    @test_throws ArgumentError SpringInterface(; kn = 1.0, sn = 1.0)
    @test_throws ArgumentError SpringInterface(; kt = 1.0)
end

@testset "MembraneInterface — Gurtin-Murdoch bulk limit" begin
    κ₀, μ₀ = 100.0, 70.0
    C₀ = TensISO{3}(3κ₀, 2μ₀)
    C₁ = TensISO{3}(3κ₀ * 2, 2μ₀ * 2)

    # κs = μs = 0 recovers perfect interface.
    s_perfect = LayeredSphere((0.5, 1.0), (C₁, C₀))
    α_perfect = MeanFieldHomogenization.LayeredSpheres._bulk_localization(s_perfect, κ₀, μ₀)
    intf_zero = (MembraneInterface(0.0, 0.0), PerfectInterface{Float64}())
    s_zero = LayeredSphere((0.5, 1.0), (C₁, C₀); interfaces = intf_zero)
    α_zero = MeanFieldHomogenization.LayeredSpheres._bulk_localization(s_zero, κ₀, μ₀)
    for k in 1:2
        @test α_zero[k] ≈ α_perfect[k] rtol = 1.0e-12
    end

    # Positive κs reinforces the interface: bulk localization in layer 1
    # should differ from perfect case.
    intf_m = (MembraneInterface(1.0, 0.5), PerfectInterface{Float64}())
    s_m = LayeredSphere((0.5, 1.0), (C₁, C₀); interfaces = intf_m)
    α_m = MeanFieldHomogenization.LayeredSpheres._bulk_localization(s_m, κ₀, μ₀)
    @test α_m[1] != α_perfect[1]
end

@testset "MembraneInterface (DUALDISC) — Echoes concentration & moduli" begin
    # Whole-sphere strain concentration A_Ω = (Σ f_k α_k) 𝕁 + (Σ f_k β_k) 𝕂 of a
    # 2-layer sphere (core E=100 ν=0.49, shell E=10 ν=0.2) in matrix E=30 ν=0.3,
    # R = (1, 1.5), with a Gurtin–Murdoch membrane at the outer interface.
    # Reference values from the compiled Echoes `DUALDISC` (κs = λs + μs).
    C₀ = iso_stiffness_E_nu(30.0, 0.3)
    C₁ = iso_stiffness_E_nu(100.0, 0.49)
    C₂ = iso_stiffness_E_nu(10.0, 0.2)
    for ((κs, μs), (bulk, shear)) in (
            ((5.0, 3.0), (1.345305088, 1.0706066045)),
            ((50.0, 30.0), (0.5767671264, 0.6087140669)),
        )
        s = LayeredSphere(
            (1.0, 1.5), (C₁, C₂);
            interfaces = (PerfectInterface(), MembraneInterface(κs, μs)),
        )
        A = strain_strain_loc(s, C₁, C₀)
        a, b = TensND.get_data(A)                  # (Σf α, Σf β)
        @test a ≈ bulk rtol = 1.0e-8
        @test b ≈ shear rtol = 1.0e-8
    end

    # Effective Young's modulus (Mori–Tanaka), single-layer aggregate
    # (E=70 ν=0.2) + membrane (κs=5, μs=3) in matrix E=30 ν=0.3 — the N=1
    # composite-sphere path that must NOT ignore the interface.
    Cagg = iso_stiffness_E_nu(70.0, 0.2)
    agg = LayeredSphere((1.0,), (Cagg,); interfaces = (MembraneInterface(5.0, 3.0),))
    for (f, E_echoes) in ((0.1, 32.92649594), (0.5, 47.94244729))
        r = RVE()
        add_phase!(r, :CEMENT, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest)
        add_phase!(r, :AGG, agg, Dict(:C => Cagg); fraction = f)
        @test E_nu(homogenize(r, MoriTanaka(), :C))[1] ≈ E_echoes rtol = 1.0e-7
    end
end

@testset "SpringInterface eltype inference" begin
    s = LayeredSphere(
        (0.5, 1.0), (TensISO{3}(1.0, 1.0), TensISO{3}(1.0, 1.0));
        interfaces = (SpringInterface(; sn = 0.01), PerfectInterface{Float64}())
    )
    @test layer_interface(s, 1) isa SpringInterface
    @test layer_interface(s, 2) isa PerfectInterface
    @test eltype(layer_interface(s, 1)) === Float64
end
