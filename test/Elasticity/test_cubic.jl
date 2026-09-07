# The cubic symmetry class, as this package uses it.
#
# The algebra is TensND's and is tested there: the storage, the componentwise
# products and inverse, the promotions, the projection and the exact rotational
# average all live in `TensND/test/test_tens_cubic.jl`. What is tested here is
# the two things this package adds -- a name for the projection residual, and
# the symmetry trait -- and, more importantly, the *reading* those two support,
# which is the reason the class is here at all.

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

@testset "cubic symmetry, as MeanFieldHomogenization uses it" begin

    can = TensND.CanonicalBasis{3, Float64}()
    rot = TensND.RotatedBasis(0.3, 0.4, 0.5)

    @testset "cubic_residual is the third return of proj_tens" begin
        C = TensND.tens_cubic(10.0, 4.0, 2.0, can)
        @test cubic_residual(C) == TensND.proj_tens(Val(:CUBIC), C, can)[3]
        @test cubic_residual(C) < 1.0e-14
        # The default frame is the canonical one, which is exactly why a rotated
        # cube must be given its own: it is not cubic about the canonical axes.
        Cr = TensND.tens_cubic(10.0, 4.0, 2.0, rot)
        @test cubic_residual(Cr, rot) < 1.0e-14
        @test cubic_residual(Cr) > 1.0e-2
        # An isotropic tensor is cubic about every cube.
        @test cubic_residual(iso_stiffness(2.0, 0.75)) < 1.0e-14
        @test cubic_residual(iso_stiffness(2.0, 0.75), rot) < 1.0e-14
        # A genuinely transversely isotropic one is not. (Taking the TI
        # projection of an *isotropic* tensor would not do: it gives that same
        # isotropic tensor back, which is cubic.)
        @test cubic_residual(TensND.tens_TI(10.0, 3.0, 2.5, 12.0, 2.0, [0.0, 0.0, 1.0])) >
            1.0e-2
    end

    @testset "residual and anisotropy answer different questions" begin
        # This is the pairing the whole class is here for. `cubic_residual` is
        # the distance to a class the answer belongs to *by group theory*, so it
        # is discretization error and nothing else. `cubic_anisotropy` lives
        # inside the class. An artifact breaks the symmetry; a real
        # morphological anisotropy does not.
        iso = iso_stiffness(2.0, 0.75)
        @test cubic_residual(iso) < 1.0e-14
        @test abs(TensND.cubic_anisotropy(best_fit_cubic(iso, can))) < 1.0e-14

        cub = TensND.tens_cubic(10.0, 4.0, 2.0, can)      # genuinely anisotropic
        @test cubic_residual(cub) < 1.0e-14               # ...and exactly in class
        @test TensND.cubic_anisotropy(cub) ≈ (10 - 4 - 4) / 10

        # Perturb *out* of the class and only the residual moves.
        M = Matrix(KM(cub))
        M[1, 1] += 0.5
        M[3, 3] -= 0.5
        off = TensND.Tens(TensND.inv_KM(M))
        @test cubic_residual(off) > 1.0e-2
        @test TensND.cubic_anisotropy(best_fit_cubic(off, can)) ≈
            TensND.cubic_anisotropy(cub) atol = 1.0e-2
    end

    @testset "the symmetry trait" begin
        # It can dispatch now, and only now: `material_symmetry` answers from
        # the container, and before TensND had `TensCubic` a `CubicSym` would
        # have claimed a structure the object did not carry.
        @test material_symmetry(TensND.tens_cubic(10.0, 4.0, 2.0, can)) isa CubicSym
        @test CubicSym() isa MaterialSymmetry
        @test material_symmetry(iso_stiffness(2.0, 0.75)) isa IsotropicSym
        # A cubic tensor written as a plain array is still, correctly, reported
        # as unstructured: the trait is type-level, not a detection.
        plain = TensND.Tens(get_array(TensND.tens_cubic(10.0, 4.0, 2.0, can)))
        @test material_symmetry(plain) isa GeneralAnisotropicSym
        @test cubic_residual(plain) < 1.0e-14      # ...though it *is* cubic
    end
end
