using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using SymPy   # top level on purpose: `@syms` below is expanded before any
# `using` nested in a @testset body would have run.

# =============================================================================
#  Type-genericity tests for LayeredSphere: ForwardDiff.Dual + Sym.
# =============================================================================

@testset "LayeredSphere — ForwardDiff through bulk localization" begin
    # Derivative of α_1 w.r.t. κ_1 for single-layer composite.
    κ₀, μ₀ = 100.0, 70.0
    function α1_of_κ1(κ₁)
        C₁ = TensISO{3}(3 * κ₁, 2 * 140.0)
        s = LayeredSphere((1.0,), (C₁,))
        return MeanFieldHomogenization.LayeredSpheres._bulk_localization(s, κ₀, μ₀)[1]
    end
    κ₁ = 200.0
    d_auto = ForwardDiff.derivative(α1_of_κ1, κ₁)
    # Analytical: α₁ = (3κ₀ + 4μ₀)/(3κ₁ + 4μ₀) → dα/dκ₁ = -3(3κ₀+4μ₀)/(3κ₁+4μ₀)²
    d_analytic = -3 * (3κ₀ + 4μ₀) / (3 * κ₁ + 4μ₀)^2
    @test d_auto ≈ d_analytic rtol = 1.0e-10
end

@testset "LayeredSphere — ForwardDiff through conductivity localization" begin
    k₀ = 2.0
    function α1_of_k1(k₁)
        s = LayeredSphere((1.0,), (TensISO{3}(k₁),))
        return MeanFieldHomogenization.LayeredSpheres._cond_localization(s, k₀)[1]
    end
    k₁ = 5.0
    d_auto = ForwardDiff.derivative(α1_of_k1, k₁)
    # α₁ = 3k₀/(2k₀ + k₁) → dα/dk₁ = -3k₀/(2k₀+k₁)²
    d_analytic = -3 * k₀ / (2 * k₀ + k₁)^2
    @test d_auto ≈ d_analytic rtol = 1.0e-12
end

@testset "LayeredSphere — Symbolic (SymPy) bulk single-layer" begin
    @syms kk0 mm0 kk1 mm1
    C₀ = TensISO{3}(3 * kk0, 2 * mm0)
    C₁ = TensISO{3}(3 * kk1, 2 * mm1)
    s = LayeredSphere((Sym(1),), (C₁,))
    α_sym = MeanFieldHomogenization.LayeredSpheres._bulk_localization(s, kk0, mm0)[1]
    # Expected: (3k0 + 4m0)/(3k1 + 4m0)
    expected = (3 * kk0 + 4 * mm0) / (3 * kk1 + 4 * mm0)
    @test simplify(α_sym - expected) == 0
end

@testset "LayeredSphere — Symbolic conductivity single-layer" begin
    @syms ka kb
    s = LayeredSphere((Sym(1),), (TensISO{3}(kb),))
    α_sym = MeanFieldHomogenization.LayeredSpheres._cond_localization(s, ka)[1]
    expected = 3 * ka / (2 * ka + kb)
    @test simplify(α_sym - expected) == 0
end

@testset "LayeredSphere — pointwise field with a symbolic radius" begin
    # A symbolic radius cannot be located by comparison (`SymPy.Sym` is not
    # even `<: Real`), so the region has to be named.  Refusing with a message
    # that says so is the contract; the reconstruction itself stays symbolic.
    C₀ = TensISO{3}(3 * 100.0, 2 * 70.0)
    C₁ = TensISO{3}(3 * 250.0, 2 * 180.0)
    C₂ = TensISO{3}(3 * 60.0, 2 * 30.0)
    sphere = LayeredSphere((1.0, 2.0), (C₁, C₂))
    sol = LayeredSphereFields(sphere, C₀)

    SymPy.@syms r_sym::positive
    @test_throws ArgumentError local_strain_strain_loc(sol, r_sym, 0.4, 0.9)
    @test_throws ArgumentError get_layer(sphere, r_sym)

    # With the layer named, the whole reconstruction goes through symbolically.
    A = local_strain_strain_loc(sol, r_sym, 0.4, 0.9; layer = 2)
    @test eltype(A) <: Sym
    ℓ = TensND.get_ℓ(A)
    @test length(ℓ) == 6
    # Substituting a numeric radius must reproduce the numeric evaluation.
    A_num = local_strain_strain_loc(sol, 1.5, 0.4, 0.9; layer = 2)
    ℓ_num = TensND.get_ℓ(A_num)
    for i in 1:6
        @test Float64(subs(ℓ[i], Dict(r_sym => Sym(3) // 2))) ≈ ℓ_num[i] rtol = 1.0e-12
    end
end

@testset "LayeredSphere — symbolic grain with a spring and a Kapitza interface" begin
    # The deviatoric recurrence inverts its mode matrix in closed form: on SymPy
    # moduli the generic pivoted solve swelled for more than ten minutes, and
    # `Matrix{Sym}(I, 4, 4)` held the logical `True`, which refuses to multiply.
    @syms k::positive μ::positive ks::positive μs::positive kn::positive kt::positive R::positive
    C = iso_stiffness(k, μ)
    grain = LayeredSphere((R,), (iso_stiffness(ks, μs),); interfaces = (SpringInterface(kn, kt),))
    α, β = TensND.get_data(strain_strain_loc(grain, C, C))
    # The bonded grain answers a pressure as a grain of modulus kₛ/(1 + 3kₛ/(kₙR)).
    k_eq = ks / (1 + 3ks / (kn * R))
    @test simplify(α - (3k + 4μ) / (3k_eq + 4μ)) == 0
    # The deviatoric part agrees with the numeric recurrence.
    vals = Dict(k => 2.0, μ => 1.5, ks => 10.0, μs => 1.0, kn => 10.0, kt => 1.0, R => 1.0)
    β_num = TensND.get_data(
        strain_strain_loc(
            LayeredSphere((1.0,), (iso_stiffness(10.0, 1.0),); interfaces = (SpringInterface(10.0, 1.0),)),
            iso_stiffness(2.0, 1.5), iso_stiffness(2.0, 1.5)
        )
    )[2]
    @test Float64(N(subs(β, vals...))) ≈ β_num rtol = 1.0e-12
    # Kapitza: the Hasselman–Johnson grain k₁/(1 + ρk₁/R).
    @syms k0::positive k1::positive ρ::positive
    kgrain = LayeredSphere((R,), (TensISO{3}(k1),); interfaces = (KapitzaInterface(ρ),))
    αT = TensND.get_data(gradient_gradient_loc(kgrain, TensISO{3}(k0), TensISO{3}(k0)))[1]
    @test simplify(αT - 3k0 / (2k0 + k1 / (1 + ρ * k1 / R))) == 0
end
