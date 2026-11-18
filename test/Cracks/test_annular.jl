# =============================================================================
#  test_annular.jl — the annular crack (a penny bonded on a central disc).
#
#   1. The ring kernel s·Q₁ against direct quadrature, near the diagonal too.
#   2. The Galerkin solution: Chebyshev order, the narrow-ring limit (the
#      straight crack of half-width h, K = √(πh) in every mode).
#   3. Rice's formula is path-independent: the inner front grown inward, the
#      outer front grown outward and the mean jump of the final crack agree.
#   4. Rice's theorem differentiated: ∂B_nn/∂c = −(4c/(a³E*)) k_I(c)², through
#      ForwardDiff on the mean jump — an independent check of the theorem and of
#      the derivative.
#   5. Limits and literature: c → 0 tends to the penny; the fit of Kachanov &
#      Sevostianov (2018, Fig. 4.35b) of Smetanin's numerical factors.
#   6. Conduction, intensity factors, schemes, guard rails.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using StaticArrays
using QuadGK
using ForwardDiff

const AN = MeanFieldHomogenization.Cracks
const AN_E, AN_ν, AN_k = 10.0, 0.3, 2.0
const AN_C = MeanFieldHomogenization.iso_stiffness_E_nu(AN_E, AN_ν)
const AN_K = TensISO{3}(AN_k)
const AN_Es = AN_E / (1 - AN_ν^2)
const AN_μ = AN_E / (2 * (1 + AN_ν))

_an_B(B, ℬ) = Array(TensND.components(B, ℬ))

@testset "ring kernel s·Q₁" begin
    for (r, s) in ((0.7, 0.4), (1.0, 0.999), (1.0, 1 - 1.0e-7), (0.05, 1.0), (1.0, 0.01))
        # the integrand peaks over a width |r − s| at φ = 0: break the interval there
        d = abs(r - s)
        Q = 2 * quadgk(φ -> cos(φ) / sqrt((r - s)^2 + 4r * s * sin(φ / 2)^2), 0, d, 10d, 100d, 0.1, π; rtol = 1.0e-13)[1]
        @test AN._ring_kernel(r, s, r - s) ≈ s * Q rtol = 1.0e-9
    end
    @test AN._ring_kernel(0.5, 0.5, 0.0) == 0.0
end

@testset "annulus — Galerkin solution" begin
    @testset "Chebyshev order" begin
        for λ in (0.5, 0.1, 0.03)
            S = AN._annulus_solve(λ, AN_ν)
            R = AN._annulus_solve(λ, AN_ν; N = AN._annulus_order(λ) + 32)
            for f in (:kI, :kII, :kIII), side in (:in, :out)
                @test getfield(getfield(S, f), side) ≈ getfield(getfield(R, f), side) rtol = 1.0e-8
            end
            @test S.opening ≈ R.opening rtol = 1.0e-11
            @test S.slide ≈ R.slide rtol = 1.0e-11
        end
        @test_throws ArgumentError AN._annulus_solve(0.5, AN_ν; N = 1)
    end

    @testset "narrow ring: the straight crack" begin
        λ = 0.9999
        h = (1 - λ) / 2
        S = AN._annulus_solve(λ, AN_ν)
        for f in (:kI, :kII, :kIII), side in (:in, :out)
            @test getfield(getfield(S, f), side) / sqrt(π * h) ≈ 1 rtol = 1.0e-4
        end
        # mean jumps of a straight crack of length 2πa per unit traction:
        # opening 4h/E*·(πh/2) and sliding (4h/E* + 2h/μ)(πh/2)/2 per unit length
        @test S.opening ≈ 2π * 2π * h^2 rtol = 1.0e-3
        @test S.slide ≈ π * (2π * h^2 * (1 - AN_ν) / 2 + π * h^2) rtol = 1.0e-3
    end

end

@testset "AnnularCrack — Rice's formula" begin
    crack = AnnularCrack(1.0, 0.5)
    ℬ = crack_basis(crack)

    @testset "path independence" begin
        Brice = cod_tensor(crack, AN_C)
        Bopen = cod_tensor(crack, AN_C; method = :opening)
        @test _an_B(Brice, ℬ) ≈ _an_B(Bopen, ℬ) rtol = 1.0e-8
        # the outer front grown outward from a ring of zero width at r = c
        c, a = crack.c, crack.a
        front(ρ) = begin
            S = AN._annulus_solve(c / ρ, AN_ν)
            sρ = sqrt(ρ)
            AN._circle_front(2π * ρ / 4, true, sρ * S.kI.out, sρ * S.kII.out, sρ * S.kIII.out)
        end
        Bout = rice_cod_tensor(front, c, a, AN_C, CircularFronts; size = a, basis = ℬ)
        @test _an_B(Bout, ℬ) ≈ _an_B(Bopen, ℬ) rtol = 1.0e-8
        # conduction
        @test cod_tensor(crack, AN_K) ≈ cod_tensor(crack, AN_K; method = :opening) rtol = 1.0e-8
        # the jump of temperature is (E*/2k₀) times the opening
        @test cod_tensor(crack, AN_K) ≈ AN_Es * _an_B(Bopen, ℬ)[3, 3] / (2AN_k) rtol = 1.0e-8
    end

    @testset "Rice's theorem differentiated" begin
        a = 1.0
        Bnn(c) = TensND.canonical_components(cod_tensor(AnnularCrack(a, c), AN_C; method = :opening))[3, 3]
        Btt(c) = TensND.canonical_components(cod_tensor(AnnularCrack(a, c), AN_C; method = :opening))[1, 1]
        for c in (0.3, 0.6)
            S = AN._annulus_solve(c / a, AN_ν)
            # B = (2/(π²a³)) ∫_c^a 2πρ [kI² Bᴿnn or (kII² Bᴿνν + kIII² Bᴿττ)/2] dρ
            @test ForwardDiff.derivative(Bnn, c) ≈ -4c / (a^3 * AN_Es) * S.kI.in^2 rtol = 1.0e-7
            @test ForwardDiff.derivative(Btt, c) ≈
                -2c / (a^3) * (S.kII.in^2 / AN_Es + S.kIII.in^2 / (2AN_μ)) rtol = 1.0e-7
        end
        # the Rice route carries the derivative too
        Bnn_rice(c) = TensND.canonical_components(cod_tensor(AnnularCrack(a, c), AN_C))[3, 3]
        @test ForwardDiff.derivative(Bnn_rice, 0.6) ≈ ForwardDiff.derivative(Bnn, 0.6) rtol = 1.0e-6
        # and in the Poisson ratio
        f(ν) = TensND.canonical_components(cod_tensor(AnnularCrack(a, 0.6), MeanFieldHomogenization.iso_stiffness_E_nu(AN_E, ν); method = :opening))[1, 1]
        g(ν) = TensND.canonical_components(cod_tensor(AnnularCrack(a, 0.6), MeanFieldHomogenization.iso_stiffness_E_nu(AN_E, ν)))[1, 1]
        @test ForwardDiff.derivative(g, AN_ν) ≈ ForwardDiff.derivative(f, AN_ν) rtol = 1.0e-6
    end

    @testset "limits and literature" begin
        Bp = _an_B(cod_tensor(PennyCrack(1.0), AN_C), ℬ)
        errs = map((0.04, 0.02, 0.01)) do c
            B = _an_B(cod_tensor(AnnularCrack(1.0, c), AN_C; method = :opening), ℬ)
            maximum(abs.(diag(B) ./ diag(Bp) .- 1))
        end
        @test issorted(errs; rev = true)
        @test errs[end] < 0.03
        # Kachanov & Sevostianov (2018), Fig. 4.35b: √(1 − λ) F(λ) ≈ 1 + Aλ at the inner
        # front, K = √(πw/2) F(λ), w = a − c the width, λ = w/a, from Smetanin's tables
        A = 4sqrt(2) / π^2 - 1
        for c in (0.2, 0.5, 0.8)
            λ = 1 - c
            S = AN._annulus_solve(c, AN_ν; shear = false)
            @test S.kI.in / sqrt(π * λ / 2) * sqrt(1 - λ) ≈ 1 + A * λ rtol = 0.03
        end
        # integrated, the fit gives V H_nnnn = (π²a³/E*) λ² (1 + 4Aλ/3 + A²λ²/2)
        λ = 0.5
        VH = π * _an_B(cod_tensor(AnnularCrack(1.0, 1 - λ), AN_C; method = :opening), ℬ)[3, 3]
        @test VH ≈ π^2 / AN_Es * λ^2 * (1 + 4A * λ / 3 + A^2 * λ^2 / 2) rtol = 0.04
    end

    @testset "intensity factors" begin
        rot = AnnularCrack(1.0, 0.5; euler_angles = (0.4, 1.1, -0.3))
        Σl = TensND.Tens(SMatrix{3, 3}(0.2, 0.5, 1.0, 0.5, -0.3, 0.7, 1.0, 0.7, 2.0), crack.basis)   # in the crack frame
        Σ = TensND.Tens(TensND.components(Σl, crack.basis), rot.basis)                            # same stress, rotated crack
        for front in (:outer, :inner)
            K0, m0 = sif(crack, AN_C, Σl; front)
            K1, m1 = sif(rot, AN_C, Σ; front, y₀ = TensND.tens_basis(rot.basis, 1))
            @test collect(m0) ≈ collect(m1) rtol = 1.0e-10
            S = AN._annulus_solve(0.5, AN_ν)
            side = front === :outer ? :out : :in
            @test m0[1] ≈ getfield(S.kI, side) * 2.0 rtol = 1.0e-10        # t·n = Σnn = 2
            @test abs(m0[2]) ≈ getfield(S.kII, side) * 1.0 rtol = 1.0e-10   # t·e_r = Σℓn = 1 at y₀ = ℓ
            @test abs(m0[3]) ≈ getfield(S.kIII, side) * 0.7 rtol = 1.0e-10  # t·e_θ = Σmn
        end
        @test sif(crack, AN_K, TensND.tens_basis(crack.basis, 3); front = :inner) ≈
            AN._annulus_solve(0.5, AN_ν; shear = false).kI.in rtol = 1.0e-12
        @test_throws ArgumentError sif(crack, AN_C, Σl; front = :middle)
    end

    @testset "schemes and interface" begin
        ε = 0.1
        H = inverse_property_contribution(crack, AN_C)
        flat = CustomInclusion(
            (1.0, 1.0, 0.0); basis = ℬ, density_factor = 4π / 3,
            inverse_property_contribution = (P₀; kw...) -> inverse_property_contribution(crack, P₀; kw...),
            property_contribution = (P₀; kw...) -> property_contribution(crack, P₀; kw...),
        )
        for scheme in (Dilute(), MoriTanaka())
            rves = map((crack, flat)) do geom
                rve = RVE(; distribution_shape = Ellipsoid(1.0))
                add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => AN_C); fraction = :rest)
                add_phase!(rve, :F, geom, Dict(:C => AN_C); density = ε)
                rve
            end
            @test TensND.canonical_components(homogenize(rves[1], scheme, :C)) ≈ TensND.canonical_components(homogenize(rves[2], scheme, :C)) rtol = 1.0e-10
        end
        # dilute compliance: S₀ + (4π/3) ε ℍ
        rve = RVE(; distribution_shape = Ellipsoid(1.0))
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => AN_C); fraction = :rest)
        add_phase!(rve, :F, crack, Dict(:C => AN_C); density = ε)
        @test TensND.canonical_components(inv(homogenize(rve, DiluteDual(), :C))) ≈ TensND.canonical_components(inv(AN_C) + 4π / 3 * ε * H) rtol = 1.0e-10
        # parallel cracks make a self-consistent reference anisotropic: refused, then accepted
        @test_throws ArgumentError homogenize(rve, SelfConsistent(), :C)
        rve_iso = RVE(; distribution_shape = Ellipsoid(1.0))
        add_phase!(rve_iso, :M, Ellipsoid(1.0), Dict(:C => AN_C); fraction = :rest)
        add_phase!(rve_iso, :F, crack, Dict(:C => AN_C); density = ε, symmetrize = IsoSymmetrize())
        @test homogenize(rve_iso, SelfConsistent(), :C) isa TensND.AbstractTens
        # interface corrections
        Ki = TensISO{3}(3.0)
        B = cod_tensor(crack, AN_C; method = :opening)
        @test _an_B(cod_tensor(crack, AN_C; method = :opening, K_interface = Ki), ℬ) ≈
            _an_B(AN._apply_interface_stiffness(B, Ki, 1.0), ℬ) rtol = 1.0e-12
        b = cod_tensor(crack, AN_K; method = :opening)
        @test cod_tensor(crack, AN_K; method = :opening, α_interface = 0.5) ≈ b / (1 + 0.5b) rtol = 1.0e-12
    end

    @testset "inference" begin
        @inferred cod_tensor(crack, AN_C)
        @inferred cod_tensor(crack, AN_C; method = :opening)
        @inferred cod_tensor(crack, AN_K)
        @inferred AN._ring_kernel(1.0, 0.5, 0.5)
    end

    @testset "construction and guard rails" begin
        @test crack == AnnularCrack(1.0, 0.5)
        @test hash(crack) == hash(AnnularCrack(1.0, 0.5))
        @test occursin("AnnularCrack", sprint(show, crack))
        @test MeanFieldHomogenization.shape_trait(crack) === CircularFronts
        @test semi_major(crack) == semi_minor(crack) == 1.0
        @test TensND.canonical_components(MeanFieldHomogenization.shape_tensor(crack)) ≈ Diagonal([1.0, 1.0, 0.0])
        @test AnnularCrack(1, 0.5, Matrix(1.0I, 3, 3)) isa AnnularCrack{Float64}
        for (a, c) in ((1.0, 1.0), (1.0, 0.0), (1.0, 1.5))
            @test_throws ArgumentError AnnularCrack(a, c)
        end
        aniso = TensND.TensTI{4}(30.0, 8.0, 5.0, 12.0, 4.0, [0.0, 0.0, 1.0])
        @test_throws ArgumentError cod_tensor(crack, aniso)
        @test_throws ArgumentError cod_tensor(crack, AN_C; method = :residues)
        @test_throws ArgumentError cod_tensor(crack, AN_K; method = :residues)
    end
end
