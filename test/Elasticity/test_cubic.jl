# The cubic symmetry class.
#
# Three constants where the isotropic class has two, and no TensND storage type
# to carry them -- so what is tested is the algebra: that the projection is an
# orthogonal projection onto a class that *contains* the isotropic one, that its
# residual is a genuine distance, and that the two group-theoretic facts the
# implementation leans on actually hold.

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import ForwardDiff

@testset "cubic symmetry class" begin

    @testset "it contains the isotropic class" begin
        for (k, μ) in ((2.0, 0.75), (1.0, 1.0), (30.0, 12.0))
            C = iso_stiffness(k, μ)
            p = cubic_parameters(C)
            @test p.C11 ≈ k + 4μ / 3
            @test p.C12 ≈ k - 2μ / 3
            @test p.C44 ≈ μ
            @test p.alpha ≈ 3k
            @test p.beta ≈ 2μ
            @test p.gamma ≈ 2μ
            @test cubic_residual(C) < 1.0e-14
            @test abs(cubic_anisotropy(C)) < 1.0e-14
            # and the projection returns it unchanged (relative: these moduli
            # span more than an order of magnitude)
            @test norm(KM(best_fit_cubic(C)) - KM(C)) < 1.0e-14 * norm(KM(C))
        end

        # Conversely, a cubic tensor built with C11 = C12 + 2 C44 *is* isotropic.
        k, μ = 2.0, 0.75
        @test norm(KM(cubic_stiffness(k + 4μ / 3, k - 2μ / 3, μ)) - KM(iso_stiffness(k, μ))) <
            1.0e-14 * norm(KM(iso_stiffness(k, μ)))
        @test abs(cubic_anisotropy(cubic_stiffness(3.0, 1.0, 1.0))) < 1.0e-14
    end

    @testset "constructor and parameters invert one another" begin
        for (C11, C12, C44) in ((3.0, 1.0, 1.5), (10.0, 4.0, 2.0), (1.0, -0.2, 0.6))
            p = cubic_parameters(cubic_stiffness(C11, C12, C44))
            @test p.C11 ≈ C11
            @test p.C12 ≈ C12
            @test p.C44 ≈ C44
            @test cubic_residual(cubic_stiffness(C11, C12, C44)) < 1.0e-14
            @test cubic_anisotropy(cubic_stiffness(C11, C12, C44)) ≈
                (C11 - C12 - 2C44) / C11
        end
        @test eltype(KM(cubic_stiffness(3, 1, 1))) === Float64   # Integers promoted
    end

    @testset "it is an orthogonal projection" begin
        # Something genuinely off-class, built by perturbing component pairs of
        # an isotropic tensor; the minor symmetries are already encoded by
        # Kelvin-Mandel, so they survive by construction.
        M = Matrix(KM(iso_stiffness(2.0, 0.8)))
        M[3, 3] += 0.7
        M[1, 2] -= 0.3
        M[2, 1] -= 0.3
        t = Tens(inv_KM(M))
        P = best_fit_cubic(t)
        Pm = Matrix(KM(P))
        @test norm(Pm - Matrix(KM(best_fit_cubic(P)))) < 1.0e-13     # idempotent
        @test abs(dot(M - Pm, Pm)) < 1.0e-12 * norm(M)^2             # orthogonal
        @test cubic_residual(t) ≈ norm(M - Pm) / norm(M)
        @test cubic_residual(t) > 1.0e-2                             # and it noticed
        @test cubic_residual(P) < 1.0e-14
    end

    @testset "the class is major-symmetric by group theory" begin
        # Under the octahedral group the Kelvin-Mandel space splits as
        # A1g + Eg + T2g, three inequivalent irreducible representations each of
        # multiplicity one, so the commutant is spanned by three *symmetric*
        # projectors. A tensor with no major symmetry at all therefore comes out
        # major-symmetric once projected -- which is exactly what makes the
        # projection usable on a localization tensor.
        M = Matrix(KM(iso_stiffness(2.0, 0.8)))
        M[1, 4] += 0.4          # destroys the major symmetry
        M[3, 5] -= 0.25
        M[2, 2] += 0.6          # and the cubic symmetry
        t = Tens(inv_KM(M))
        @test norm(M - M') > 0.5
        Pm = Matrix(KM(best_fit_cubic(t)))
        @test norm(Pm - Pm') < 1.0e-13
    end

    @testset "the cube axes are the canonical basis" begin
        # A cubic tensor read in a rotated frame is not cubic in the canonical
        # one, and the residual must say so. This also pins that the components
        # used are the CANONICAL ones: `get_array` on a rotated tensor returns
        # the components in its own basis, and reading those instead would make
        # the residual come out at zero for every rotation.
        Cc = cubic_stiffness(3.0, 1.0, 1.5)
        @test cubic_residual(Cc) < 1.0e-14
        b = RotatedBasis(0.3, 0.4, 0.5)
        rotated = Tens(get_array(Cc), b)      # same components, other basis
        @test cubic_residual(rotated) > 1.0e-2
        # A rotation belonging to the cube's own symmetry group leaves it alone.
        # The basis is checked to be a genuine rotation first, or the test would
        # pass on the identity and say nothing.
        qb = RotatedBasis(π / 2, 0.0, 0.0)
        @test norm(Matrix(TensND.vecbasis(qb, :cov)) - Matrix(1.0I, 3, 3)) > 0.5
        quarter = Tens(get_array(Cc), qb)
        @test cubic_residual(quarter) < 1.0e-12
    end

    @testset "at order two the cubic class is the isotropic class" begin
        K = TensND.TensISO{3}(1.7)                    # spherical, order 2
        @test norm(
            get_array(change_tens_canon(best_fit_cubic(K))) -
                get_array(change_tens_canon(K))
        ) < 1.0e-14
        A = Tens([2.0 0.3 0.0; 0.3 1.0 0.0; 0.0 0.0 0.5])
        @test norm(
            get_array(change_tens_canon(best_fit_cubic(A))) -
                get_array(change_tens_canon(best_fit_iso(A)))
        ) < 1.0e-14
    end

    @testset "differentiation" begin
        f = μ -> cubic_parameters(iso_stiffness(2.0, μ)).C44
        @test ForwardDiff.derivative(f, 0.75) ≈ 1
        g = C44 -> cubic_anisotropy(cubic_stiffness(3.0, 1.0, C44))
        @test ForwardDiff.derivative(g, 1.5) ≈ -2 / 3
        h = μ -> cubic_residual(iso_stiffness(2.0, μ))
        @test isfinite(ForwardDiff.derivative(h, 0.75))
    end
end
