using Test
using MeanFieldHomogenization
using TensND

@testset "Cracks — :residues accuracy (post polish + vectorised QuadGK)" begin

    basis = TensND.CanonicalBasis{3, Float64}()
    δ(a, b) = a == b ? 1.0 : 0.0

    # Cubic stiffness (Fe-like) as a 6×6 KM matrix → Tens via inv_KM
    c11, c12, c44 = 237.0, 141.0, 116.0
    KM_cubic = [
        c11 c12 c12 0.0 0.0 0.0;
        c12 c11 c12 0.0 0.0 0.0;
        c12 c12 c11 0.0 0.0 0.0;
        0.0 0.0 0.0 2c44 0.0 0.0;
        0.0 0.0 0.0 0.0 2c44 0.0;
        0.0 0.0 0.0 0.0 0.0 2c44
    ]
    C_cubic = TensND.inv_KM(KM_cubic, basis)

    # Triclinic stiffness — hand-picked symmetric-positive KM (6×6)
    KM_tri = [
        210.0 80.0 75.0 5.0 4.0 3.0;
        80.0 195.0 90.0 -2.0 3.0 -1.0;
        75.0 90.0 220.0 1.0 -2.0 2.0;
        5.0 -2.0 1.0 60.0 2.5 1.5;
        4.0 3.0 -2.0 2.5 65.0 -1.0;
        3.0 -1.0 2.0 1.5 -1.0 55.0
    ]
    C_tric = TensND.inv_KM(KM_tri, basis)

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

    @testset "Polished roots — |Q(zᵣ)| near machine epsilon" begin
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
end
