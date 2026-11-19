# =============================================================================
#  test_planar_crack.jl — a flat crack of any contour by the Gaussian
#  approximation of its integral equation.
#
#   1. The kernel: radial profiles against their angular integrals; the symbol
#      of an isotropic medium against the reduced kernel of `Cracks`; that of
#      conduction against the line integral it is the closed form of.
#   2. The edge rule: the discrete edge follows the edge of a line crack.
#   3. Exact solutions: disk and ellipse (isotropic, transversely isotropic and
#      anisotropic matrices, tilted cracks), the annular crack, conduction.
#   4. Contacts between faces, interface stiffness, opening field.
#   5. Intensity factors by the tangent ribbon, against the exact ellipse.
#   6. Derivatives with respect to the moduli; schemes; guard rails.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using StaticArrays
using QuadGK
using ForwardDiff

const GP = MeanFieldHomogenization.GaussianApproximation
const GP_C = MeanFieldHomogenization.iso_stiffness_E_nu(10.0, 0.3)
const GP_TI = TensND.TensTI{4}(30.0, 8.0, 5.0, 12.0, 4.0, (0.0, 0.0, 1.0))
const GP_K = TensISO{3}(2.5)
const GP_Ka = TensND.Tens([3.0 0.4 0.1; 0.4 2.0 0.3; 0.1 0.3 1.0])
const GP_ℬ = CanonicalBasis{3, Float64}()
const GP_ANGLES = (0.3, 0.9, 0.5)

_gp_err(B, Bref) = norm(TensND.canonical_components(B) - TensND.canonical_components(Bref)) /
    norm(TensND.canonical_components(Bref))

@testset "kernel" begin
    for k in 0:5, ρ in (0.0, 1.3, 5.0), H in (1.0, 2.0)
        z = ρ^2 / H
        ref = quadgk(φ -> cos(2k * φ) * (1 - 2z * cos(φ)^2) * exp(-z * cos(φ)^2), 0, 2π; rtol = 1.0e-13)[1] /
            (2π^1.5 * H^1.5)
        @test GP.harmonic_profile(k, ρ^2, H) ≈ ref atol = 1.0e-13
    end
    # far field of a point loop of unit moment
    @test GP.harmonic_profile(0, 40.0^2, 2.0) ≈ -1 / (2π * 40.0^3) rtol = 5.0e-3
    @inferred GP.harmonic_profile(2, 1.7, 2.0)

    # isotropic symbol: the closed form is the reduced kernel Q̂*ₙₙ of `Cracks`
    iso = GP._crack_symbol(GP_C, GP_ℬ)
    gen = GP._crack_symbol(TensND.Tens(TensND.get_array(GP_C)), GP_ℬ)
    @test all(θ -> iso(θ) ≈ gen(θ), (0.0, 0.7, 2.1))
    hs = GP.symbol_harmonics(iso, Val(3))
    @test length(hs.cosines) == 2 && iszero(hs.sines[1])
    # a symbol that 8 angles cannot resolve
    @test_throws ArgumentError GP.symbol_harmonics(θ -> SMatrix{1, 1}(1 / (1.01 + cos(2θ))), Val(1); angles = 8)

    # conduction symbol: -(1/2π) ∫ n·Ŝ(ξ + ζ n)·n dζ with Ŝ(k) = K - K·k⊗k·K / (k·K·k)
    K = SMatrix{3, 3}(TensND.get_array(GP_Ka))
    n = SVector(0.0, 0.0, 1.0)
    for θ in (0.0, 0.9, 2.3)
        e = SVector(cos(θ), sin(θ), 0.0)
        S(ζ) = (k = e + ζ * n; n' * K * n - (n' * K * k)^2 / (k' * K * k))
        @test only(GP._crack_symbol(GP_Ka, GP_ℬ)(θ)) ≈ quadgk(S, -Inf, Inf; rtol = 1.0e-11)[1] / (2π) rtol = 1.0e-9
    end
    @test only(GP._crack_symbol(GP_K, GP_ℬ)(0.4)) ≈ 2.5 / 2
end

@testset "edge rule" begin
    γ, c₀ = edge_rule(2.0)
    @test 0.7 < γ < 0.8 && 0.85 < c₀ < 0.9
    @test edge_rule(2) === (γ, c₀)                       # memoized
    # with the rule, the edge of the discrete line crack follows the edge of the
    # crack swept across a cell: without it (γ → ∞ drops a cell as soon as it is
    # cut) it moves by a whole step
    sweep(γ) = (o = GP._edge_offsets(2.0, γ); maximum(o) - minimum(o))
    @test sweep(γ) < 5.0e-3
    @test sweep(1.0e8) > 0.8
    # kernel of the line crack against its integral
    @test GP._line_kernel(1.3, 2.0) ≈ quadgk(k -> k * exp(-k^2 / 2) * cos(1.3k), 0, Inf)[1] / π rtol = 1.0e-10
end

@testset "exact solutions" begin
    penny = PlanarCrack(Disk(1.0); n = 48)
    @test penny.size == 1.0
    @test crack_area(penny) ≈ π rtol = 1.0e-3
    @test MeanFieldHomogenization.Core.shape_trait(penny) === ContourShape
    @test semi_minor(penny) == semi_major(penny) == 1.0
    @test TensND.get_array(MeanFieldHomogenization.Core.shape_tensor(penny)) == [1.0 0 0; 0 1.0 0; 0 0 0]
    @test occursin("PlanarCrack", sprint(show, penny)) && occursin("Ellipse", sprint(show, penny))
    @test _gp_err(cod_tensor(penny, GP_C), cod_tensor(PennyCrack(1.0), GP_C)) < 3.0e-3
    @test cod_tensor(penny, GP_K) ≈ cod_tensor(PennyCrack(1.0), GP_K) rtol = 3.0e-3
    # the solve is memoized per reference medium
    @test MeanFieldHomogenization.fe_assembly_count(penny) == 2
    cod_tensor(penny, GP_C)
    @test MeanFieldHomogenization.fe_assembly_count(penny) == 2

    # one grid only: an error of first order in the step, which the second grid removes
    coarse = PlanarCrack(Disk(1.0); n = 48, extrapolate = false)
    @test 3.0e-3 < _gp_err(cod_tensor(coarse, GP_C), cod_tensor(PennyCrack(1.0), GP_C)) < 3.0e-2

    # ℓ³ 𝐁 = a b² 𝐁ᴱ: the ellipse, in isotropic, transversely isotropic and
    # anisotropic matrices, aligned and tilted
    for C₀ in (GP_C, GP_TI), ang in ((), GP_ANGLES)
        ell = PlanarCrack(Ellipse(1.0, 0.5); n = 64, size = 1.0, euler_angles = ang)
        @test _gp_err(cod_tensor(ell, C₀), 0.25 * cod_tensor(EllipticCrack(1.0, 0.5; euler_angles = ang), C₀)) < 3.0e-3
    end
    for ang in ((), GP_ANGLES)
        ell = PlanarCrack(Ellipse(1.0, 0.5); n = 64, size = 1.0, euler_angles = ang)
        @test cod_tensor(ell, GP_Ka) ≈ 0.25 * cod_tensor(EllipticCrack(1.0, 0.5; euler_angles = ang), GP_Ka) rtol = 3.0e-3
    end
    # an ellipse turned by 30° in its own plane: the same principal values of 𝐁,
    # its in-plane principal directions turned by 30°
    turned = TensND.get_array(cod_tensor(PlanarCrack(Ellipse(1.0, 0.5; angle = π / 6); n = 64, size = 1.0), GP_C))
    aligned = 0.25 * TensND.get_array(cod_tensor(EllipticCrack(1.0, 0.5), GP_C))
    R = [cos(π / 6) -sin(π / 6) 0; sin(π / 6) cos(π / 6) 0; 0 0 1]
    @test turned ≈ R * aligned * R' rtol = 1.0e-2
    # the annular crack, against the Galerkin solution of `Cracks`
    ring = PlanarCrack(Annulus(1.0, 0.5); n = 64, size = 1.0)
    @test _gp_err(cod_tensor(ring, GP_C), cod_tensor(AnnularCrack(1.0, 0.5), GP_C; method = :opening)) < 1.0e-2
    # default size: the radius of the disk of the same area, counted on the grid
    @test PlanarCrack(Annulus(1.0, 0.5); n = 32).size ≈ sqrt(0.75) rtol = 5.0e-3
    @test PlanarCrack(Rectangle(1.0, 0.5); n = 16).size ≈ sqrt(2 / π)

    @testset "inference" begin
        @inferred cod_tensor(penny, GP_C)
        @inferred cod_tensor(penny, GP_K)
    end
end

@testset "contacts, interface, opening" begin
    free = PlanarCrack(Disk(1.0); n = 48)
    island = PlanarCrack(setdiff(Disk(1.0), Disk(0.25; center = (0.3, 0.1))); n = 48, size = 1.0)
    B, Bi = cod_tensor(free, GP_C), cod_tensor(island, GP_C)
    # a contact stiffens the crack in every mode, and nearly as much in each
    ratios = diag(TensND.get_array(Bi)) ./ diag(TensND.get_array(B))
    @test all(0.5 .< ratios .< 0.95) && maximum(ratios) - minimum(ratios) < 0.05
    # compliance and resistance of a flat crack in an isotropic body are tied:
    # π ℓ³ Bₙₙ / (π ℓ³ b) = k₀ (1 - ν) / μ, contacts or not
    μ = 10.0 / (2 * 1.3)
    for crack in (free, island)
        @test TensND.get_array(cod_tensor(crack, GP_C))[3, 3] / cod_tensor(crack, GP_K) ≈ 2.5 * (1 - 0.3) / μ rtol = 1.0e-8
    end

    # interface stiffness: none in the limit K → 0, the springs alone for K → ∞
    # (uniform opening K⁻¹·t: π ℓ³ 𝐁 = S K⁻¹)
    @test _gp_err(cod_tensor(free, GP_C; K_interface = TensISO{3}(1.0e-9)), B) < 1.0e-6
    Kbig = TensND.Tens([3.0 0.5 0.0; 0.5 2.0 0.2; 0.0 0.2 4.0] .* 1.0e6)
    Bs = cod_tensor(free, GP_C; K_interface = Kbig)
    @test TensND.get_array(Bs) ≈ inv(TensND.get_array(Kbig)) rtol = 2.0e-2
    @test cod_tensor(free, GP_K; α_interface = 1.0e7) ≈ 1.0e-7 rtol = 2.0e-2
    # in between, the opening is not uniform: the closed form of an ellipse
    # with uniform springs is an approximation, a few percent away
    Kmid = TensISO{3}(3.0)
    approx = MeanFieldHomogenization.Cracks._apply_interface_stiffness(cod_tensor(PennyCrack(1.0), GP_C), Kmid, 1.0)
    @test 1.0e-3 < _gp_err(cod_tensor(free, GP_C; K_interface = Kmid), approx) < 0.1

    # opening field: its integral is π ℓ³ 𝐁·𝚺·n on the grid it is computed on,
    # up to the part of the edge Gaussians that lies outside the crack
    one_grid = PlanarCrack(Disk(1.0); n = 32, extrapolate = false, euler_angles = GP_ANGLES)
    n̂ = TensND.tens_basis(one_grid.basis, 3)
    l̂ = TensND.tens_basis(one_grid.basis, 1)
    Σ = n̂ ⊗ n̂ + 0.4 * (l̂ ⊗ n̂ + n̂ ⊗ l̂)
    x, u = crack_opening(one_grid, GP_C, Σ)
    @test length(x) == length(u) == length(first(one_grid.levels).inside)
    h = first(one_grid.levels).grid.h
    @test h^2 * sum(u) ≈ π * TensND.components(cod_tensor(one_grid, GP_C) ⋅ Σ ⋅ n̂, one_grid.basis) rtol = 3.0e-2
    xT, uT = crack_opening(one_grid, GP_K, 2.0 * n̂)
    # the same values laid on the grid, for a surface plot
    gx, gy, Z = opening_surface(one_grid, uT)
    @test size(Z) == (length(gx), length(gy)) && sum(Z) ≈ sum(uT) && count(!iszero, Z) == length(uT)
    @test gx[2] - gx[1] ≈ h && first(gx) ≈ -1 + h / 2
    @test_throws DimensionMismatch opening_surface(one_grid, uT[1:3])
    @test h^2 * sum(uT) ≈ 2π * cod_tensor(one_grid, GP_K) rtol = 3.0e-2
    # elliptic profile of the penny-shaped crack, away from the edge
    flat = PlanarCrack(Disk(1.0); n = 64)
    e₃ = TensND.tens_basis(GP_ℬ, 3)
    xs, us = crack_opening(flat, GP_C, e₃ ⊗ e₃)
    j = argmin(norm.(xs))
    @test us[j][3] ≈ 4 * (1 - 0.3) / (π * μ) rtol = 1.0e-2
    # and over the whole crack, edge included: the sum of the Gaussians, not
    # their coefficients, which overshoot on the nodes next to the edge
    profile(x) = 4 * (1 - 0.3) / (π * μ) * sqrt(max(1 - sum(abs2, x), 0.0))
    @test maximum(abs(u[3] - profile(x)) for (x, u) in zip(xs, us)) < 0.03 * profile((0.0, 0.0))
    @test @inferred(GP._nodal_field(first(flat.levels), last.(us))) isa Vector{Float64}
end

@testset "intensity factors" begin
    for (C₀, ang) in ((GP_C, ()), (GP_TI, GP_ANGLES))
        exact = EllipticCrack(1.0, 0.5; euler_angles = ang)
        l̂, m̂, n̂ = (TensND.tens_basis(crack_basis(exact), i) for i in 1:3)
        Σ = n̂ ⊗ n̂ + 0.3 * (l̂ ⊗ n̂ + n̂ ⊗ l̂) + 0.2 * (m̂ ⊗ n̂ + n̂ ⊗ m̂)
        crack = PlanarCrack(Ellipse(1.0, 0.5); n = 64, size = 1.0, euler_angles = ang)
        for φ in (0.0, 0.8, π / 2)
            Kex, _ = sif(exact, C₀, Σ; y₀ = cos(φ) * l̂ + sin(φ) * m̂)
            K, modes = sif(crack, C₀, Σ; point = (cos(φ), 0.5sin(φ)), normal = (cos(φ), 2sin(φ)))
            @test _gp_err(K, Kex) < 2.5e-2
            @test length(modes) == 3
        end
    end
    exact = EllipticCrack(1.0, 0.5)
    l̂, m̂, n̂ = (TensND.tens_basis(crack_basis(exact), i) for i in 1:3)
    crack = PlanarCrack(Ellipse(1.0, 0.5); n = 64, size = 1.0)
    @test sif(crack, GP_Ka, n̂; point = (cos(0.7), 0.5sin(0.7)), normal = (cos(0.7), 2sin(0.7))) ≈
        sif(exact, GP_Ka, n̂; y₀ = cos(0.7) * l̂ + sin(0.7) * m̂) rtol = 2.5e-2

    # the local relation itself, on the crack it is exact for: a ribbon of
    # half-width b has the uniform N = √(b/π) 𝐁ᴿ·𝚺·n and K = √(π b) 𝚺·n
    ribbon = RibbonCrack(0.7; euler_angles = GP_ANGLES)
    l̂, m̂, n̂ = (TensND.tens_basis(crack_basis(ribbon), i) for i in 1:3)
    Σ = n̂ ⊗ n̂ + 0.3 * (l̂ ⊗ n̂ + n̂ ⊗ l̂) + 0.2 * (m̂ ⊗ n̂ + n̂ ⊗ m̂)
    N = sqrt(0.7 / π) * (cod_tensor(ribbon, GP_TI) ⋅ Σ ⋅ n̂)
    K, (KI, KII, KIII) = sif_from_dif(N, m̂, n̂, GP_TI)
    @test TensND.canonical_components(K) ≈ TensND.canonical_components(first(sif(ribbon, GP_TI, Σ))) rtol = 1.0e-8
    @test KI ≈ sqrt(0.7π) && KII ≈ 0.2 * sqrt(0.7π) && abs(KIII) ≈ 0.3 * sqrt(0.7π)
    @test sif_from_dif(sqrt(0.7 / π) * cod_tensor(ribbon, GP_Ka) * (n̂ ⋅ (2.0 * n̂)), m̂, n̂, GP_Ka) ≈
        sif(ribbon, GP_Ka, 2.0 * n̂) rtol = 1.0e-8
end

@testset "derivatives" begin
    # 𝐁 scales as 1/E at fixed ν: ∂𝐁/∂E = -𝐁/E, ∂²𝐁/∂E² = 2𝐁/E², whatever the contour
    region = setdiff(Ellipse(1.0, 0.6), Disk(0.15))
    Bnn(E, ν) = TensND.get_array(
        cod_tensor(PlanarCrack(region; n = 24, size = 1.0), MeanFieldHomogenization.iso_stiffness_E_nu(E, ν))
    )[3, 3]
    B₀ = Bnn(10.0, 0.3)
    @test ForwardDiff.derivative(E -> Bnn(E, 0.3), 10.0) ≈ -B₀ / 10 rtol = 1.0e-7
    @test ForwardDiff.derivative(E -> ForwardDiff.derivative(e -> Bnn(e, 0.3), E), 10.0) ≈ 2B₀ / 100 rtol = 1.0e-6
    # with respect to the Poisson ratio, the sliding mode of the disk against the
    # closed form of the penny-shaped crack, within the error of the grid
    B11(crack, ν) = TensND.get_array(cod_tensor(crack(), MeanFieldHomogenization.iso_stiffness_E_nu(10.0, ν)))[1, 1]
    @test ForwardDiff.derivative(ν -> B11(() -> PlanarCrack(Disk(1.0); n = 32), ν), 0.3) ≈
        ForwardDiff.derivative(ν -> B11(() -> PennyCrack(1.0), ν), 0.3) rtol = 1.0e-2
    # conduction: b = g / k₀
    b(k) = cod_tensor(PlanarCrack(region; n = 24, size = 1.0), TensISO{3}(k))
    @test ForwardDiff.derivative(b, 2.5) ≈ -b(2.5) / 2.5 rtol = 1.0e-7
end

@testset "schemes" begin
    ε = 0.1
    gauss, exact = PlanarCrack(Disk(1.0); n = 48), PennyCrack(1.0)
    for scheme in (Dilute(), MoriTanaka(), SelfConsistent())
        C = map((gauss, exact)) do crack
            rve = RVE(; distribution_shape = Ellipsoid(1.0))
            add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => GP_C); fraction = :rest)
            add_phase!(rve, :F, crack, Dict(:C => GP_C); density = ε, symmetrize = IsoSymmetrize())
            homogenize(rve, scheme, :C)
        end
        @test _gp_err(C[1], C[2]) < 1.0e-3
    end
    K = map((gauss, exact)) do crack
        rve = RVE(; distribution_shape = Ellipsoid(1.0))
        add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => GP_K); fraction = :rest)
        add_phase!(rve, :F, crack, Dict(:K => GP_K); density = ε)
        homogenize(rve, MoriTanaka(), :K)
    end
    @test _gp_err(K[1], K[2]) < 1.0e-3
    @test check_inclusion_interface(gauss; amount = :density, verbose = false)
end

@testset "guard rails" begin
    @test_throws ArgumentError PlanarCrack(Disk(1.0); n = 2)                       # extrapolation
    @test_throws ArgumentError PlanarCrack(Ellipse(1.0, 0.02); n = 16)             # no node kept
    @test_throws ArgumentError PlanarCrack(Disk(1.0); size = -1.0)
    @test_throws ArgumentError PlanarCrack(Disk(1.0); n = 512, memory_limit = 10^6)
    @test PlanarCrack(Disk(1.0); n = 3, extrapolate = false) isa PlanarCrack
    # the contour is not differentiable: refused at construction and by the schemes
    @test_throws ArgumentError ForwardDiff.derivative(a -> PlanarCrack(Disk(a)).size, 1.0)
    @test_throws ArgumentError MeanFieldHomogenization.Schemes._replace_geom_field(
        PlanarCrack(Disk(1.0); n = 8), Val(:size), nothing, 2.0
    )
    @test_throws ArgumentError MeanFieldHomogenization.Schemes._replace_geom_field(
        PlanarCrack(Disk(1.0); n = 8), Val(:size), 1, 2.0
    )
end
