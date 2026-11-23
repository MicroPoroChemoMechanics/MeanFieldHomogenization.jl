# =============================================================================
#  The coefficient a `property` lens designates, in the class of the tensor or
#  in a class that contains it.
#
#  References, independent of the lens under test:
#   1. the coefficients of an isotropic stiffness, read from its Kelvin-Mandel
#      matrix (k from the normal block, μ from a shear term);
#   2. the chain rule: a derivative with respect to the coefficient p of any
#      class is Σ_{i≤j} ∂f/∂M_ij · ∂M_ij/∂p, where ∂M/∂p is the derivative of
#      the Kelvin-Mandel matrix of the tensor alone (no homogenization), and
#      ∂f/∂M_ij the derivatives through the Kelvin-Mandel lenses; the same
#      derivative is thus obtained by two routes;
#   3. refusals: a class that does not contain the tensor, a name foreign to
#      the class, a component out of range.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import ForwardDiff
import Symbolics

const LENS = MeanFieldHomogenization.Schemes

lens_n = TensND.normalize([0.0, 1.0, 1.0])
lens_frame = TensND.CanonicalBasis{3, Float64}()
lens_C = iso_stiffness(10.0, 4.0)
lens_p(sel; kw...) = property(:I, :C, sel; kw...)
# Kelvin-Mandel matrix in the canonical basis, whatever the basis of the tensor.
km(X::TensND.AbstractTens{order, dim}) where {order, dim} = mandel(X, TensND.CanonicalBasis{dim, eltype(X)}())

function lens_rve(C = lens_C)
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => iso_stiffness(5.0, 3.0)); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0, 1.0, 0.3), Dict(:C => C); fraction = 0.2)
    return rve
end
lens_indexer(X) = canonical_components(X)[1, 1, 3, 3]
lens_derivative(p, C = lens_C) = derivative(lens_rve(C), MoriTanaka(), p; indexer = lens_indexer)

# Route 2: the chain rule through the Kelvin-Mandel components, the derivative
# of the Kelvin-Mandel matrix of the tensor `tensor(x)` taken alone.
function lens_by_mandel(tensor, x₀, C = lens_C)
    dM = ForwardDiff.derivative(x -> Matrix(km(tensor(x))), x₀)
    return sum(lens_derivative(lens_p(Symbol("M$i$j")), C) * dM[i, j] for i in 1:6 for j in i:6 if dM[i, j] != 0; init = 0.0)
end

@testset "Coefficient lenses" begin

    @testset "the coefficients of an isotropic stiffness" begin
        M = km(lens_C)
        @test LENS.property_coefficient(lens_C, lens_p(:k)) ≈ (M[1, 1] + 2M[1, 2]) / 3 rtol = 1.0e-14
        @test LENS.property_coefficient(lens_C, lens_p(:μ)) ≈ M[4, 4] / 2 rtol = 1.0e-14
        @test LENS.property_coefficient(lens_C, lens_p(:bulk)) == LENS.property_coefficient(lens_C, lens_p(:k))
        @test LENS.property_coefficient(lens_C, lens_p(:shear)) == LENS.property_coefficient(lens_C, lens_p(:μ))
        @test LENS.property_coefficient(lens_C, lens_p(:α)) ≈ 30.0
        @test LENS.property_coefficient(lens_C, lens_p(:β)) ≈ 8.0
        # In the plane, α = 2k with k = λ + μ, the bulk modulus of the plane.
        C₂ = TensISO{2}(2 * 7.0, 2 * 3.0)
        @test LENS.property_coefficient(C₂, lens_p(:k)) ≈ 7.0
        @test LENS.property_coefficient(C₂, lens_p(:μ)) ≈ 3.0
        @test LENS.property_coefficient(C₂, lens_p(:M12)) ≈ km(C₂)[1, 2]
        # Setting k, then μ, gives the stiffness of those moduli.
        C = LENS.with_property_coefficient(lens_C, lens_p(:k), 12.0)
        C = LENS.with_property_coefficient(C, lens_p(:μ), 5.0)
        @test km(C) ≈ km(iso_stiffness(12.0, 5.0)) rtol = 1.0e-14
    end

    @testset "the tensor written in an enclosing class is the tensor" begin
        for p in (
                lens_p(:ℓ₁; symmetry = (:TI, lens_n)), lens_p(:C1212; symmetry = (:ORTHO, lens_frame)),
                lens_p(:γ; symmetry = (:CUBIC, lens_frame)), lens_p(:M23), lens_p(:M23; symmetry = :ANISO),
            )
            x = LENS.property_coefficient(lens_C, p)
            @test km(LENS.with_property_coefficient(lens_C, p, x)) ≈ km(lens_C) rtol = 1.0e-12
        end
        # Orthotropic components: C1212 is the tensor component, M66 = 2 C1212.
        @test LENS.property_coefficient(lens_C, lens_p(:C1212; symmetry = (:ORTHO, lens_frame))) ≈ 4.0
        @test LENS.property_coefficient(lens_C, lens_p(:M66)) ≈ 8.0
        # A Kelvin-Mandel component in a rotated basis is that of the tensor in that basis.
        b = TensND.RotatedBasis(0.3, 0.7, -0.2)
        T = LENS.with_property_coefficient(lens_C, lens_p(:M12), 9.0)
        @test LENS.property_coefficient(T, lens_p(:M13; symmetry = (:ANISO, b))) ≈ mandel(T, b)[1, 3] rtol = 1.0e-12
        # Off the diagonal the matrix stays symmetric.
        @test km(T)[1, 2] == km(T)[2, 1] == 9.0
    end

    @testset "one derivative, two routes" begin
        # ∂/∂k and ∂/∂μ of an isotropic phase: by the name, by the Kelvin-Mandel components.
        @test lens_derivative(lens_p(:k)) ≈ lens_by_mandel(k -> iso_stiffness(k, 4.0), 10.0) rtol = 1.0e-10
        @test lens_derivative(lens_p(:μ)) ≈ lens_by_mandel(μ -> iso_stiffness(10.0, μ), 4.0) rtol = 1.0e-10
        # ∂/∂α = ∂/∂k / 3 in space.
        @test lens_derivative(lens_p(:α)) ≈ lens_derivative(lens_p(:k)) / 3 rtol = 1.0e-12
        # A Walpole coefficient of the isotropic phase seen as transversely isotropic about n.
        ℓ = TensND.get_data(first(TensND.project(lens_C, :TI, lens_n)))
        for (r, sel) in enumerate((:ℓ₁, :ℓ₂, :ℓ₃, :ℓ₅, :ℓ₆))
            tensor(x) = TensND.rebuild(first(TensND.project(lens_C, :TI, lens_n)), Base.setindex(ℓ, x, r))
            @test lens_derivative(lens_p(sel; symmetry = (:TI, lens_n))) ≈ lens_by_mandel(tensor, ℓ[r]) rtol = 1.0e-9
        end
        # An orthotropic component of the isotropic phase.
        O = first(TensND.project(lens_C, :ORTHO, lens_frame))
        tensor_o(x) = TensND.rebuild(O, Base.setindex(TensND.get_data(O), x, 9))
        @test lens_derivative(lens_p(:C1212; symmetry = (:ORTHO, lens_frame))) ≈ lens_by_mandel(tensor_o, 4.0) rtol = 1.0e-9
        # ∂/∂k by the name equals the chain rule through the Walpole coefficients.
        dℓ_dk = ForwardDiff.derivative(k -> collect(TensND.get_data(first(TensND.project(iso_stiffness(k, 4.0), :TI, lens_n)))), 10.0)
        dℓ = [lens_derivative(lens_p(sel; symmetry = (:TI, lens_n))) for sel in (:ℓ₁, :ℓ₂, :ℓ₃, :ℓ₅, :ℓ₆)]
        @test lens_derivative(lens_p(:k)) ≈ sum(dℓ .* dℓ_dk) rtol = 1.0e-9
    end

    @testset "a second derivative, nested duals" begin
        d²(x) = ForwardDiff.derivative(
            k -> lens_derivative(lens_p(:μ), LENS.with_property_coefficient(lens_C, lens_p(:k), k)), x
        )
        reference(x) = ForwardDiff.derivative(
            k -> lens_by_mandel(μ -> iso_stiffness(k, μ), 4.0, iso_stiffness(k, 4.0)), x
        )
        @test d²(10.0) ≈ reference(10.0) rtol = 1.0e-8
    end

    @testset "symbolic coefficients" begin
        Symbolics.@variables k μ x
        Cs = iso_stiffness(k, μ)
        @test isequal(Symbolics.simplify(LENS.property_coefficient(Cs, lens_p(:k)) - k), 0)
        T = LENS.with_property_coefficient(Cs, lens_p(:μ), x)
        @test isequal(Symbolics.simplify(LENS.property_coefficient(T, lens_p(:μ)) - x), 0)
        # An isotropic tensor is contained in every class, decided by its type.
        @test LENS.property_coefficient(Cs, lens_p(:ℓ₁; symmetry = (:TI, lens_n))) isa Symbolics.Num
        # Another symbolic tensor cannot be compared with its projection.
        Ts = TensTI{4}(k, μ, x, k, μ, [0.0, 0.0, 1.0])
        @test_throws ArgumentError LENS.property_coefficient(Ts, lens_p(:C1111; symmetry = (:ORTHO, TensND.RotatedBasis(0.3, 0.2, 0.1))))
    end

    @testset "refusals" begin
        # The name of another class without `symmetry`.
        @test_throws ArgumentError LENS.property_coefficient(lens_C, lens_p(:ℓ₁))
        @test_throws ArgumentError LENS.property_coefficient(lens_C, lens_p(:K))
        # A class that does not contain the tensor.
        T = TensTI{4}(12.0, 9.0, 3.0, 4.0, 5.0, [0.0, 0.0, 1.0])
        @test_throws ArgumentError LENS.property_coefficient(T, lens_p(:ℓ₁; symmetry = (:TI, lens_n)))
        # A component out of range, or of the wrong order.
        @test_throws ArgumentError LENS.property_coefficient(lens_C, lens_p(:M77))
        @test_throws ArgumentError LENS.property_coefficient(lens_C, lens_p(:M3))
        # In the plane, no class of space.
        @test_throws ArgumentError LENS.property_coefficient(TensISO{2}(4.0, 2.0), lens_p(:ℓ₁; symmetry = (:TI, lens_n)))
    end
end
