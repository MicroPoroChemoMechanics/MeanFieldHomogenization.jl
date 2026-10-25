using Test
using MeanFieldHomogenization
using TensND

@testset "Cracks — :residues accuracy (post polish + vectorised QuadGK)" begin

    basis = TensND.CanonicalBasis{3, Float64}()
    δ(a, b) = a == b ? 1.0 : 0.0

    # Cubic stiffness (Fe-like) as a 6×6 KM matrix → Tens via from_mandel
    c11, c12, c44 = 237.0, 141.0, 116.0
    KM_cubic = [
        c11 c12 c12 0.0 0.0 0.0;
        c12 c11 c12 0.0 0.0 0.0;
        c12 c12 c11 0.0 0.0 0.0;
        0.0 0.0 0.0 2c44 0.0 0.0;
        0.0 0.0 0.0 0.0 2c44 0.0;
        0.0 0.0 0.0 0.0 0.0 2c44
    ]
    C_cubic = TensND.from_mandel(KM_cubic, basis)

    # Triclinic stiffness — hand-picked symmetric-positive KM (6×6)
    KM_tri = [
        210.0 80.0 75.0 5.0 4.0 3.0;
        80.0 195.0 90.0 -2.0 3.0 -1.0;
        75.0 90.0 220.0 1.0 -2.0 2.0;
        5.0 -2.0 1.0 60.0 2.5 1.5;
        4.0 3.0 -2.0 2.5 65.0 -1.0;
        3.0 -1.0 2.0 1.5 -1.0 55.0
    ]
    C_tric = TensND.from_mandel(KM_tri, basis)

    @testset "Penny / cubic — :residues vs :nestedquadgk" begin
        crack = PennyCrack(1.0)
        B_res = cod_tensor(crack, C_cubic; method = :residues)
        B_nqg = cod_tensor(crack, C_cubic; method = :nestedquadgk, reltol = 1.0e-12)

        err = maximum(abs, components_canon(B_res - B_nqg))
        scale = maximum(abs, components_canon(B_nqg))
        @test err / scale < 1.0e-8
    end

    @testset "Penny / triclinic — :residues vs :nestedquadgk" begin
        crack = PennyCrack(1.0)
        B_res = cod_tensor(crack, C_tric; method = :residues)
        B_nqg = cod_tensor(crack, C_tric; method = :nestedquadgk, reltol = 1.0e-12)

        err = maximum(abs, components_canon(B_res - B_nqg))
        scale = maximum(abs, components_canon(B_nqg))
        @test err / scale < 1.0e-6
    end

    @testset "Elliptic / cubic η=0.3 — :residues vs :nestedquadgk" begin
        crack = EllipticCrack(1.0, 0.3)
        B_res = cod_tensor(crack, C_cubic; method = :residues)
        B_nqg = cod_tensor(crack, C_cubic; method = :nestedquadgk, reltol = 1.0e-12)

        err = maximum(abs, components_canon(B_res - B_nqg))
        scale = maximum(abs, components_canon(B_nqg))
        @test err / scale < 1.0e-6
    end

    @testset "Roots — |Q(zᵣ)| at the rounding level" begin
        # Build a cubic C_arr that yields a well-conditioned degree-6 Q.
        Cr = zeros(3, 3, 3, 3)
        for i in 1:3, j in 1:3, k in 1:3, l in 1:3
            Cr[i, j, k, l] = (
                c12 * δ(i, j) * δ(k, l) +
                    c44 * (δ(i, k) * δ(j, l) + δ(i, l) * δ(j, k)) +
                    (c11 - c12 - 2c44) * (i == j == k == l ? 1.0 : 0.0)
            )
        end
        α0 = [0.6, 0.3, 0.2]
        α1 = [0.1, 0.2, 0.9]
        sys = MeanFieldHomogenization.Core._build_poly_system(Cr, α0, α1)
        @test !isempty(sys.roots_uhp)
        max_resid = maximum(abs(sys.Q(zr)) for zr in sys.roots_uhp)
        Qscale = maximum(abs, MeanFieldHomogenization.Core.Polynomials.coeffs(sys.Q))
        @test max_resid < 1.0e-10 * Qscale
    end

    @testset "Multiple roots — summed, not skipped" begin
        # An isotropic matrix entered as a general tensor: the acoustic polynomial
        # has a triple root, whose contribution used to be skipped (relative error
        # 1 on a penny-shaped crack). Against the closed forms.
        λ, μ = 18.0, 12.0
        KM_iso = [
            λ + 2μ λ λ 0.0 0.0 0.0;
            λ λ + 2μ λ 0.0 0.0 0.0;
            λ λ λ + 2μ 0.0 0.0 0.0;
            0.0 0.0 0.0 2μ 0.0 0.0;
            0.0 0.0 0.0 0.0 2μ 0.0;
            0.0 0.0 0.0 0.0 0.0 2μ
        ]
        C_iso = TensND.from_mandel(KM_iso, basis)
        C_iso_t = TensISO{3}(3λ + 2μ, 2μ)
        for crack in (
                PennyCrack(1.0), EllipticCrack(1.0, 0.5), RibbonCrack(1.0),
                EllipticCrack(1.0, 0.5; euler_angles = (0.3, 0.7, 0.1)),
            )
            @test components_canon(cod_tensor(crack, C_iso; method = :residues)) ≈
                components_canon(cod_tensor(crack, C_iso_t)) rtol = 1.0e-10
        end
        # Transversely isotropic, with (C13 + C44)² = (C11 - C44)(C33 - C44): a
        # double root. Against the cubature.
        C11, C33, C44 = 60.0, 45.0, 20.0
        C13 = sqrt((C11 - C44) * (C33 - C44)) - C44
        KM_ti = [
            C11 C11 - 2C44 C13 0.0 0.0 0.0;
            C11 - 2C44 C11 C13 0.0 0.0 0.0;
            C13 C13 C33 0.0 0.0 0.0;
            0.0 0.0 0.0 2C44 0.0 0.0;
            0.0 0.0 0.0 0.0 2C44 0.0;
            0.0 0.0 0.0 0.0 0.0 2C44
        ]
        C_ti = TensND.from_mandel(KM_ti, basis)
        crack = EllipticCrack(1.0, 0.5; euler_angles = (0.3, 0.7, 0.1))
        @test components_canon(cod_tensor(crack, C_ti; method = :residues)) ≈
            components_canon(cod_tensor(crack, C_ti; method = :nestedquadgk, abstol = 1.0e-12, reltol = 1.0e-11)) rtol = 1.0e-10
    end
end

@testset "crack kernel: the residue sums and their quadrature fallback agree" begin
    # Below the rounding of any sum, the tolerance sends the kernel to the
    # quadrature that replaces residue sums whose error bound is too large: both
    # give the same kernel.
    CRK = MeanFieldHomogenization.Cracks
    KM = [
        210.0 80.0 75.0 5.0 4.0 3.0;
        80.0 195.0 90.0 -2.0 3.0 -1.0;
        75.0 90.0 220.0 1.0 -2.0 2.0;
        5.0 -2.0 1.0 60.0 2.5 1.5;
        4.0 3.0 -2.0 2.5 65.0 -1.0;
        3.0 -1.0 2.0 1.5 -1.0 55.0
    ]
    C = Float64.(Array(components_canon(TensND.from_mandel(KM, TensND.CanonicalBasis{3, Float64}()))))
    n̂ = [0.0, 0.0, 1.0]
    for α in (0.2, 1.0, 2.5)
        ξs = [cos(α), sin(α), 0.0]
        R = CRK._Qnn_star_residue(C, ξs, n̂)
        F = CRK._Qnn_star_residue(C, ξs, n̂; rtol = 5.0e-16)
        @test F ≈ R rtol = 1.0e-13
    end
end
