using Test
using MeanFieldHomogenization
using TensND
using ForwardDiff

const MFH_IS = MeanFieldHomogenization

# =============================================================================
#  Sensitivities with respect to INTERFACE parameters, for every interface
#  model and both physics.
#
#  These used to be unreachable. The transfer matrices widen locally to
#  `promote_type(eltype(intf), …)`, but the state buffers were sized from a
#  promotion that left the interfaces out, so a `ForwardDiff.Dual` interface
#  parameter produced a widened state vector that the narrower buffer refused
#  to store — a `MethodError`, not a wrong number. Nothing in the suite asked
#  for such a derivative, so nothing noticed.
#
#  Each test compares AD against a central difference; agreement to `1e-6`
#  relative is far beyond what a lost or truncated derivative could fake.
# =============================================================================

"Central difference of `f` at `x`, with a step scaled to `x`."
function _central(f, x; h = 1.0e-6)
    δ = h * max(abs(x), one(x))
    return (f(x + δ) - f(x - δ)) / (2δ)
end

@testset "LayeredSphere — sensitivity to interface parameters" begin
    C(κ, μ) = TensISO{3}(3κ, 2μ)
    K(k) = TensISO{3}(k)
    C₀, K₀ = C(2.0, 1.0), K(2.0)
    Cs = (C(1.0, 0.5), C(3.0, 1.5))
    Ks = (K(1.0), K(5.0))
    perfect = PerfectInterface{Float64}()

    @testset "SpringInterface — normal stiffness kn" begin
        f = function (kn)
            s = LayeredSphere((0.5, 1.0), Cs; interfaces = (SpringInterface(kn, 2.0), perfect))
            return strain_strain_loc(s, C₀; layer = 1)[1, 1, 1, 1]
        end
        @test ForwardDiff.derivative(f, 3.0) ≈ _central(f, 3.0) rtol = 1.0e-6
    end

    @testset "SpringInterface — tangential stiffness kt" begin
        f = function (kt)
            s = LayeredSphere((0.5, 1.0), Cs; interfaces = (SpringInterface(3.0, kt), perfect))
            return strain_strain_loc(s, C₀; layer = 1)[1, 2, 1, 2]
        end
        @test ForwardDiff.derivative(f, 2.0) ≈ _central(f, 2.0) rtol = 1.0e-6
    end

    @testset "SpringInterface — compliance spelling sn" begin
        # `sn` is the STORED field; differentiating it must agree with the chain
        # rule through `kn = 1/sn`.
        fs = function (sn)
            s = LayeredSphere((0.5, 1.0), Cs; interfaces = (SpringInterface(; sn, st = 0.5), perfect))
            return strain_strain_loc(s, C₀; layer = 1)[1, 1, 1, 1]
        end
        fk = function (kn)
            s = LayeredSphere(
                (0.5, 1.0), Cs;
                interfaces = (SpringInterface(; kn, st = 0.5), perfect)
            )
            return strain_strain_loc(s, C₀; layer = 1)[1, 1, 1, 1]
        end
        sn₀ = 0.25
        @test ForwardDiff.derivative(fs, sn₀) ≈ _central(fs, sn₀) rtol = 1.0e-6
        # dF/dsn = dF/dkn · dkn/dsn = dF/dkn · (-1/sn²)
        @test ForwardDiff.derivative(fs, sn₀) ≈
            ForwardDiff.derivative(fk, inv(sn₀)) * (-inv(sn₀^2)) rtol = 1.0e-8
    end

    @testset "MembraneInterface — surface moduli κs, μs" begin
        # The promoting outer constructor is what lets a single modulus be a
        # `Dual`; without it this is a `MethodError` at construction time.
        fκ = function (κs)
            s = LayeredSphere((0.5, 1.0), Cs; interfaces = (MembraneInterface(κs, 0.5), perfect))
            return strain_strain_loc(s, C₀; layer = 1)[1, 1, 1, 1]
        end
        fμ = function (μs)
            s = LayeredSphere((0.5, 1.0), Cs; interfaces = (MembraneInterface(0.3, μs), perfect))
            return strain_strain_loc(s, C₀; layer = 1)[1, 2, 1, 2]
        end
        @test ForwardDiff.derivative(fκ, 0.3) ≈ _central(fκ, 0.3) rtol = 1.0e-6
        @test ForwardDiff.derivative(fμ, 0.5) ≈ _central(fμ, 0.5) rtol = 1.0e-6
        @test MembraneInterface(1, 0.5) isa MembraneInterface{Float64}
    end

    @testset "KapitzaInterface — resistance" begin
        f = function (α)
            s = LayeredSphere(
                (0.5, 1.0), Ks;
                interfaces = (KapitzaInterface(α), perfect)
            )
            return gradient_gradient_loc(s, K₀; layer = 1)[1, 1]
        end
        @test ForwardDiff.derivative(f, 0.1) ≈ _central(f, 0.1) rtol = 1.0e-6
    end

    @testset "SurfaceConductiveInterface — conductance" begin
        f = function (β)
            s = LayeredSphere(
                (0.5, 1.0), Ks;
                interfaces = (SurfaceConductiveInterface(β), perfect)
            )
            return gradient_gradient_loc(s, K₀; layer = 1)[1, 1]
        end
        @test ForwardDiff.derivative(f, 0.2) ≈ _central(f, 0.2) rtol = 1.0e-6
    end

    @testset "the derivative reaches the EFFECTIVE moduli, not just the localization" begin
        # The whole point of these sensitivities: an optimizer sitting on the
        # homogenized property.
        f = function (kn)
            s = LayeredSphere((0.5, 1.0), Cs; interfaces = (SpringInterface(kn, 2.0), perfect))
            A = strain_strain_loc(s, C₀; layer = 1)
            return A[1, 1, 1, 1] + A[1, 2, 1, 2]
        end
        d = ForwardDiff.derivative(f, 3.0)
        @test isfinite(d)
        @test !iszero(d)
        @test d ≈ _central(f, 3.0) rtol = 1.0e-6
    end

    @testset "interfaces_eltype" begin
        ie = MeanFieldHomogenization.LayeredSpheres.interfaces_eltype
        @test ie((perfect, perfect)) === Float64
        @test ie((PerfectInterface{Float32}(), perfect)) === Float64
        @test ie(()) === Union{}
        D = ForwardDiff.Dual{Nothing, Float64, 1}
        @test ie((KapitzaInterface(one(D)), perfect)) === D
    end
end
