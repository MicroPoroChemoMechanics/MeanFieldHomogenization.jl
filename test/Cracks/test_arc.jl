# =============================================================================
#  test_arc.jl — the circular arc crack and the Saturn ring.
#
#   1. The closed forms of ℍ and 𝐑 against Rice's formula integrated
#      numerically from the intensity factors (two independent transcriptions
#      of the literature, Kachanov & Sevostianov 2018, eqs. 1.9.27–1.10.12).
#   2. Limits: α → 0 is the ribbon crack; α = π is the cylindrical hole (the
#      core cut free carries no in-plane or anti-plane stress).
#   3. The derivative in α is the integrand of Rice's formula (AD on the closed
#      form against the intensity factors at the tips); symbolic closed forms.
#   4. The Saturn ring: Fett's factor, the pore, the penny limit, ∂/∂c.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using StaticArrays
using ForwardDiff
import Symbolics

const AR = MeanFieldHomogenization.Cracks
const AR_E, AR_ν, AR_k = 10.0, 0.3, 2.0
const AR_C = MeanFieldHomogenization.iso_stiffness_E_nu(AR_E, AR_ν)
const AR_K = TensISO{3}(AR_k)
const AR_Es = AR_E / (1 - AR_ν^2)
const AR_μ = AR_E / (2 * (1 + AR_ν))

_ar(t, ℬ) = Array(TensND.components(t, ℬ))

@testset "circular arc — closed forms against Rice's formula" begin
    for α in (0.3, 1.0, 2.5, π)
        arc = CircularArcCrack(1.3, α; euler_angles = (0.2, 0.9, -0.4))
        ℬ = crack_basis(arc)
        H = inverse_property_contribution(arc, AR_C)
        Hr = inverse_property_contribution(arc, AR_C; method = :rice)
        @test _ar(H, ℬ) ≈ _ar(Hr, ℬ) rtol = 1.0e-8
        R = inverse_property_contribution(arc, AR_K)
        Rr = inverse_property_contribution(arc, AR_K; method = :rice)
        @test _ar(R, ℬ) ≈ _ar(Rr, ℬ) rtol = 1.0e-8
        # the bundle is the pair
        Hb, Nb = MeanFieldHomogenization.inverse_and_property_contribution(arc, AR_C)
        @test TensND.canonical_components(Hb) ≈ TensND.canonical_components(H)
        @test TensND.canonical_components(Nb) ≈ TensND.canonical_components(-(AR_C ⊡ H ⊡ AR_C))
        Rb, NKb = MeanFieldHomogenization.inverse_and_property_contribution(arc, AR_K)
        @test TensND.canonical_components(Rb) ≈ TensND.canonical_components(R)
        @test TensND.canonical_components(NKb) ≈ TensND.canonical_components(-(AR_K ⋅ R ⋅ AR_K))
    end
end

@testset "circular arc — limits" begin
    R0 = 1.0
    @testset "α → 0: the ribbon" begin
        α = 1.0e-3
        arc = CircularArcCrack(R0, α)
        rib = RibbonCrack(R0 * α)
        ℬ = crack_basis(arc)
        @test _ar(inverse_property_contribution(arc, AR_C), ℬ) ≈ _ar(inverse_property_contribution(rib, AR_C), ℬ) rtol = 1.0e-5
        @test _ar(inverse_property_contribution(arc, AR_K), ℬ) ≈ _ar(inverse_property_contribution(rib, AR_K), ℬ) rtol = 1.0e-5
        @test crack_density_factor(arc) ≈ crack_density_factor(rib)
    end
    @testset "α = π: the cylindrical hole" begin
        arc = CircularArcCrack(R0, π)
        H = _ar(inverse_property_contribution(arc, AR_C), crack_basis(arc)) .* π^2   # per unit hole fraction
        # plane-strain hole: f/E*·(3, −1, 2) in plane, f/(2μ) anti-plane
        @test H[2, 2, 2, 2] ≈ 3 / AR_Es rtol = 1.0e-12
        @test H[3, 3, 3, 3] ≈ 3 / AR_Es rtol = 1.0e-12
        @test H[2, 2, 3, 3] ≈ -1 / AR_Es rtol = 1.0e-12
        @test H[2, 3, 2, 3] ≈ 2 / AR_Es rtol = 1.0e-12
        @test H[1, 2, 1, 2] ≈ 1 / (2AR_μ) rtol = 1.0e-12
        @test H[1, 3, 1, 3] ≈ 1 / (2AR_μ) rtol = 1.0e-12
        @test H[1, 1, 1, 1] == H[1, 1, 2, 2] == 0     # the core still carries the axial stress
        Rk = _ar(inverse_property_contribution(arc, AR_K), crack_basis(arc)) .* π^2
        @test Rk[2, 2] ≈ 2 / AR_k rtol = 1.0e-12      # insulating hole: 2f/k₀
        @test Rk[3, 3] ≈ 2 / AR_k rtol = 1.0e-12
    end
end

@testset "circular arc — derivative, symbols, intensity factors" begin
    R0 = 1.3
    @testset "d/dα is the integrand of Rice's formula" begin
        for α in (0.4, 1.7)
            VS(α) = π * (R0 * α)^2 * _ar(inverse_property_contribution(CircularArcCrack(R0, α), AR_C), TensND.CanonicalBasis{3, Float64}())
            dVS = ForwardDiff.derivative(VS, α)
            # Σ over the two tips of 2R (g⊗g/E* for modes I, II, g⊗g/2μ for mode III)
            ref = zeros(3, 3, 3, 3)
            for s in (1, -1)
                g = AR._arc_influence(R0, α, s)
                for q in 1:3
                    Bq = q == 1 ? 1 / (2AR_μ) : 1 / AR_Es    # (τ, ν, n): the frame of the tip
                    e = (g.τ, g.ν, g.n)[q]
                    G = [sum(e[p] * g.𝒦[p, i, j] for p in 1:3) for i in 1:3, j in 1:3]
                    ref .+= 2R0 * Bq .* [G[i, j] * G[k, l] for i in 1:3, j in 1:3, k in 1:3, l in 1:3]
                end
            end
            @test dVS ≈ ref rtol = 1.0e-10 atol = 1.0e-14
        end
    end

    @testset "symbolic closed form" begin
        Symbolics.@variables Rs αs Es νs ks
        arc = CircularArcCrack(Rs, αs)
        H = inverse_property_contribution(arc, MeanFieldHomogenization.iso_stiffness_E_nu(Es, νs))
        vals = Dict(Rs => R0, αs => 1.1, Es => AR_E, νs => AR_ν)
        Hn = TensND.canonical_components(inverse_property_contribution(CircularArcCrack(R0, 1.1), AR_C))
        Hs = TensND.canonical_components(H)
        for (i, j, k, l) in ((3, 3, 3, 3), (2, 2, 3, 3), (2, 3, 2, 3), (1, 3, 1, 3))
            @test Symbolics.value(Symbolics.substitute(Hs[i, j, k, l], vals; fold = Val(true))) ≈ Hn[i, j, k, l] rtol = 1.0e-12
        end
        R = inverse_property_contribution(arc, TensISO{3}(ks))
        @test Symbolics.value(Symbolics.substitute(TensND.canonical_components(R)[3, 3], Dict(Rs => R0, αs => 1.1, ks => AR_k); fold = Val(true))) ≈
            TensND.canonical_components(inverse_property_contribution(CircularArcCrack(R0, 1.1), AR_K))[3, 3] rtol = 1.0e-12
        # the numerical routes refuse symbols with a clear message
        @test_throws ArgumentError AR._annulus_solve(αs, 0.3)
        @test_throws ArgumentError AR._growth_integral(identity, αs, 1.0, 1.0e-8)
    end

    @testset "intensity factors" begin
        α = 1.0e-4
        arc = CircularArcCrack(R0, α)
        ℬ = crack_basis(arc)
        Σ = TensND.Tens(SMatrix{3, 3}(0.2, 0.5, 1.0, 0.5, -0.3, 0.7, 1.0, 0.7, 2.0), ℬ)
        # the curvature changes the factors at first order in α, with opposite
        # signs at the two tips: their mean is the straight crack's to O(α²)
        m = map(tip -> abs.(collect(sif(arc, AR_C, Σ; tip)[2])), (1, -1))
        mean = (m[1] .+ m[2]) ./ 2
        @test mean ≈ sqrt(π * R0 * α) .* [2.0, 0.7, 1.0] rtol = 1.0e-7
        @test all(abs.(m[1] .- mean) .< 10α .* mean)
        kT = map(tip -> sif(arc, AR_K, TensND.tens_basis(ℬ, 3); tip), (1, -1))
        @test sum(kT) / 2 ≈ sqrt(π * R0 * α) rtol = 1.0e-7
        @test_throws ArgumentError sif(arc, AR_C, Σ; tip = 2)
    end

    @testset "schemes, construction and guard rails" begin
        arc = CircularArcCrack(1.0, 1.2)
        ε = 0.05
        rve = RVE(; distribution_shape = Ellipsoid(1.0))
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => AR_C, :K => AR_K); fraction = :rest)
        add_phase!(rve, :F, arc, Dict(:C => AR_C, :K => AR_K); density = ε)
        H = inverse_property_contribution(arc, AR_C)
        @test TensND.canonical_components(inv(homogenize(rve, DiluteDual(), :C))) ≈ TensND.canonical_components(inv(AR_C) + π * ε * H) rtol = 1.0e-10
        Rk = inverse_property_contribution(arc, AR_K)
        @test TensND.canonical_components(inv(homogenize(rve, DiluteDual(), :K))) ≈ TensND.canonical_components(inv(AR_K) + π * ε * Rk) rtol = 1.0e-10
        @test homogenize(rve, MoriTanaka(), :C) isa TensND.AbstractTens
        @inferred inverse_property_contribution(arc, AR_C)
        @inferred inverse_property_contribution(arc, AR_C; method = :rice)
        @inferred inverse_property_contribution(arc, AR_K)
        @test MeanFieldHomogenization.shape_trait(arc) === ArcShape
        @test arc == CircularArcCrack(1.0, 1.2) && hash(arc) == hash(CircularArcCrack(1.0, 1.2))
        @test occursin("CircularArcCrack", sprint(show, arc))
        @test CircularArcCrack(1, 0.5, Matrix(1.0I, 3, 3)) isa CircularArcCrack{Float64}
        for (R, α) in ((1.0, 0.0), (1.0, 3.5), (-1.0, 1.0))
            @test_throws ArgumentError CircularArcCrack(R, α)
        end
        @test_throws ArgumentError inverse_property_contribution(arc, AR_C; method = :residues)
        @test_throws ArgumentError inverse_property_contribution(arc, AR_K; method = :residues)
        aniso = TensND.TensTI{4}(30.0, 8.0, 5.0, 12.0, 4.0, [0.0, 0.0, 1.0])
        @test_throws ArgumentError inverse_property_contribution(arc, aniso)
    end
end

@testset "Saturn ring" begin
    ν = AR_ν
    @test AR._fett_F(1.0e8, ν) ≈ 2 / π rtol = 1.0e-7
    @test AR._fett_F(0.0, ν) ≈ 1.122 * (39 - 5ν) / (27 - 5ν) rtol = 1.0e-12

    a = 1.0
    S = saturn_ring_compliance(a, 0.5a, AR_C)
    # a spherical void: V H_nnnn = (4π/3) a³ 3(1 − ν)(9 + 5ν)/(2E(7 − 5ν))
    @test S.pore ≈ 4π / 3 * a^3 * 3 * (1 - ν) * (9 + 5ν) / (2AR_E * (7 - 5ν)) rtol = 1.0e-10
    @test S.total ≈ S.pore + S.ring
    # a long ring is the penny of radius a + c
    penny(c) = 16 * (a + c)^3 / (3AR_Es)
    r = map(c -> saturn_ring_compliance(a, c, AR_C).total / penny(c), (2.0, 5.0, 20.0))
    @test issorted(abs.(r .- 1); rev = true)
    @test r[end] ≈ 1 rtol = 0.02
    # Rice's theorem: ∂(ring)/∂c = (4π/E*)(a + c) K_I(a + c)², K_I = F(c/a) √(πc)
    for c in (0.3, 1.0)
        dring = ForwardDiff.derivative(c -> saturn_ring_compliance(a, c, AR_C).ring, c)
        @test dring ≈ 4π / AR_Es * (a + c) * (AR._fett_F(c / a, ν) * sqrt(π * c))^2 rtol = 1.0e-7
    end
    @test_throws ArgumentError saturn_ring_compliance(-1.0, 0.5, AR_C)
    @test_throws ArgumentError saturn_ring_compliance(1.0, 0.5, TensND.TensTI{4}(30.0, 8.0, 5.0, 12.0, 4.0, [0.0, 0.0, 1.0]))
end
