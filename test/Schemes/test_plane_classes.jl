using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

# =============================================================================
#  test_plane_classes.jl — the orthotropic and the square classes of the plane
#  as properties of a two-dimensional cell.
#
#  A structured tensor of the plane and the general tensor of the same
#  components are the same material: every estimate must agree between the
#  two, the general one going through the paths tested elsewhere. The named
#  coefficients are checked against the components they stand for.
# =============================================================================

const PLC = MeanFieldHomogenization.Schemes

plc_matrix() = TensOrtho(40.0, 6.0, 2.5, 3.0, Basis(0.37))
plc_square() = cubic_from_constants(20.0, 12.0, 9.0, Basis(0.37))
plc_general(t) = Tens(get_array(t))
plc_ellipse() = Ellipsoid(1.0, 0.4)

function plc_cell(Cm, Ci; f = 0.2, shape = plc_ellipse())
    rve = RVE()
    add_phase!(rve, :matrix, Ellipsoid(1.0; dim = 2), Dict(:C => Cm); fraction = :rest)
    add_phase!(rve, :inclusions, shape, Dict(:C => Ci); fraction = f)
    return rve
end

@testset "Plane classes as properties of a cell" begin
    Cᵢ = iso_stiffness(Val(2), 50.0, 20.0)

    @testset "Hill tensor of an ellipse" begin
        for C₀ in (plc_matrix(), plc_square())
            P = hill_tensor(plc_ellipse(), C₀)
            @test mandel(P) ≈ mandel(hill_tensor(plc_ellipse(), plc_general(C₀))) rtol = 1.0e-13
            @test mandel(P) ≈ mandel(P)' rtol = 1.0e-12
        end
        # The isotropic tensor written in the square class: the closed form of the isotropic plane.
        L = iso_stiffness(Val(2), 5.0, 2.0)
        @test mandel(hill_tensor(plc_ellipse(), iso_to_cubic(L, Basis(0.9)))) ≈ mandel(hill_tensor(plc_ellipse(), L)) rtol = 1.0e-9
    end

    @testset "schemes: structured matrix against the general tensor" begin
        for C₀ in (plc_matrix(), plc_square()), scheme in (MoriTanaka(), SelfConsistent(), DifferentialScheme())
            structured = homogenize(plc_cell(C₀, Cᵢ), scheme, :C)
            general = homogenize(plc_cell(plc_general(C₀), Cᵢ), scheme, :C)
            @test mandel(structured) ≈ mandel(general) rtol = 1.0e-10
        end
        # The structured tensor as the property of the inclusions.
        L = iso_stiffness(Val(2), 5.0, 2.0)
        @test mandel(homogenize(plc_cell(L, plc_matrix()), MoriTanaka(), :C)) ≈
            mandel(homogenize(plc_cell(L, plc_general(plc_matrix())), MoriTanaka(), :C)) rtol = 1.0e-12
        # No inclusion: the matrix, in its class or not.
        @test mandel(homogenize(plc_cell(plc_matrix(), Cᵢ; f = 0.0), MoriTanaka(), :C)) ≈ mandel(plc_matrix()) rtol = 1.0e-13
    end

    @testset "conduction with an orthotropic tensor of the plane" begin
        K₀ = TensOrtho{2}(3.0, 1.0, Basis(0.3))
        cell(K) = (
            r = RVE(); add_phase!(r, :matrix, Ellipsoid(1.0; dim = 2), Dict(:K => K); fraction = :rest);
            add_phase!(r, :inclusions, plc_ellipse(), Dict(:K => TensISO{2}(10.0)); fraction = 0.2); r
        )
        @test get_array(homogenize(cell(K₀), MoriTanaka(), :K)) ≈ get_array(homogenize(cell(Tens(get_array(K₀))), MoriTanaka(), :K)) rtol = 1.0e-12
    end

    @testset "named coefficients" begin
        p(sel; kw...) = property(:matrix, :C, sel; kw...)
        O, O₅, Q = plc_matrix(), TensOrtho(40.0, 6.0, 2.5, 3.0, 1.5, Basis(0.37)), plc_square()
        for (name, value) in zip((:C1111, :C2222, :C1122, :C1212), (40.0, 6.0, 2.5, 3.0))
            @test PLC.property_coefficient(O, p(name)) == value
            @test PLC.property_coefficient(O₅, p(name)) == value
        end
        @test PLC.property_coefficient(O₅, p(:C2211)) == 1.5
        @test_throws ArgumentError PLC.property_coefficient(O, p(:C2211))
        @test_throws ArgumentError PLC.property_coefficient(O, p(:C3333))
        @test collect(PLC.property_coefficient.(Ref(Q), p.((:α, :β, :γ)))) == collect(get_data(Q))
        a = TensOrtho{2}(3.0, 1.0, Basis(0.3))
        pk(sel) = property(:matrix, :K, sel)
        @test (PLC.property_coefficient(a, pk(:a1)), PLC.property_coefficient(a, pk(:a2))) == (3.0, 1.0)
        # Setting a coefficient gives the tensor of that coefficient.
        O′ = PLC.with_property_coefficient(O, p(:C1212), 7.0)
        @test O′ isa TensOrtho{4, 2} && get_data(O′) == (40.0, 6.0, 2.5, 7.0)
        # A general tensor read in a class of the plane that contains it, the frame given.
        G = plc_general(O)
        ortho = p(:C1111; symmetry = (:ORTHO, Basis(0.37)))
        @test PLC.property_coefficient(G, ortho) ≈ 40.0
        @test mandel(PLC.with_property_coefficient(G, ortho, 44.0)) ≈ mandel(TensOrtho(44.0, 6.0, 2.5, 3.0, Basis(0.37)))
        @test PLC.property_coefficient(plc_general(Q), p(:γ; symmetry = (:CUBIC, Basis(0.37)))) ≈ get_data(Q)[3]
        @test PLC.property_coefficient(Q, p(:C1212; symmetry = :ORTHO)) ≈ 9.0          # a square tensor is orthotropic on its frame
        # A class that does not contain the tensor, and the class the plane does not have.
        @test_throws ArgumentError PLC.property_coefficient(G, p(:γ; symmetry = (:CUBIC, Basis(0.37))))
        @test_throws ArgumentError PLC.property_coefficient(G, p(:C1111; symmetry = (:ORTHO, Basis(0.1))))
        @test_throws ArgumentError PLC.property_coefficient(G, p(:ℓ₁; symmetry = (:TI, (1.0, 0.0))))
    end

    @testset "derivative of an estimate with respect to a named coefficient" begin
        # The shear modulus of the matrix, as a coefficient of its class and as a parameter of its constructor.
        index(X) = canonical_components(X)[1, 1, 2, 2]
        estimate(c66) = index(homogenize(plc_cell(TensOrtho(40.0, 6.0, 2.5, c66, Basis(0.37)), Cᵢ), MoriTanaka(), :C))
        d = derivative(plc_cell(plc_matrix(), Cᵢ), MoriTanaka(), property(:matrix, :C, :C1212); indexer = index)
        @test d ≈ ForwardDiff.derivative(estimate, 3.0) rtol = 1.0e-9
    end

    @testset "a cubic matrix in the differential scheme, in the plane and in space" begin
        Q₃ = cubic_from_constants(20.0, 12.0, 9.0, Basis(0.3, 0.5, 0.2))
        cell₃(C) = (
            r = RVE(); add_phase!(r, :matrix, Ellipsoid(1.0), Dict(:C => C); fraction = :rest);
            add_phase!(r, :inclusions, Ellipsoid(1.0, 1.0, 0.3), Dict(:C => iso_stiffness(50.0, 20.0)); fraction = 0.2); r
        )
        @test mandel(homogenize(cell₃(Q₃), DifferentialScheme(), :C)) ≈
            mandel(homogenize(cell₃(Tens(get_array(Q₃))), DifferentialScheme(), :C)) rtol = 1.0e-9
    end
end
