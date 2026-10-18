using Test
using MeanFieldHomogenization
using TensND
import Symbolics

# =============================================================================
#  Hill tensors of symbolic inclusions (`Symbolics.Num` semi-axes and moduli),
#  held to the numerical Hill tensor of the same inclusion.
#
#  `Symbolics.Num <: Real`, and the routes keyed on `T <: Real` sent a symbolic
#  spheroid, triaxial ellipsoid or cylinder into a numerical comparison
#  (`TypeError: non-boolean (Num) used in boolean context`); only the sphere
#  and the 2-D shapes went through. The spheroids now take their closed forms
#  in `acos`/`acosh`, independent of the numerical spheroid integrals, and the
#  triaxial ellipsoid keeps its incomplete elliptic integrals as calls.
# =============================================================================

Symbolics.@variables hs_a hs_b hs_c hs_k hs_μ
const HS_VARS = [hs_a, hs_b, hs_c, hs_k, hs_μ]
const HS_POINT = [1.0, 0.6, 0.3, 5.0, 2.0]

# A symbolic tensor, or vector, evaluated at `HS_POINT` through a compiled function.
_hs_eval_vec(v) = Symbolics.build_function(v, HS_VARS; expression = Val(false))[1](HS_POINT)
_hs_eval(P) = _hs_eval_vec(get_array(P))

@testset "Hill tensors of symbolic inclusions" begin
    shapes = (
        (Ellipsoid(hs_a, hs_a, hs_c), Ellipsoid(1.0, 1.0, 0.3)),
        (Ellipsoid(hs_a, hs_c, hs_c), Ellipsoid(1.0, 0.3, 0.3)),
        (Ellipsoid(hs_a, hs_b, hs_c), Ellipsoid(1.0, 0.6, 0.3)),
        (Cylinder(hs_a, hs_a), Cylinder(1.0, 1.0)),
        (Cylinder(hs_a, hs_c), Cylinder(1.0, 0.3)),
    )
    references = (
        (TensISO{3}(3hs_k, 2hs_μ), TensISO{3}(15.0, 4.0)),
        (TensISO{3}(hs_k), TensISO{3}(5.0)),
    )
    for (sym, num) in shapes, (C_sym, C_num) in references
        P = hill_tensor(sym, C_sym)
        @test eltype(get_array(P)) <: Symbolics.Num
        @test _hs_eval(P) ≈ get_array(hill_tensor(num, C_num)) rtol = 1.0e-13 atol = 1.0e-15
    end

    # The Newton potentials themselves, the sphere's included (its Hill tensor
    # takes Eshelby's closed form and never calls them).
    np3 = MeanFieldHomogenization.Core.newton_potential_3d
    for (sym, num) in (
            ((hs_a, hs_a, hs_a), (1.0, 1.0, 1.0)), ((hs_a, hs_a, hs_c), (1.0, 1.0, 0.3)),
            ((hs_a, hs_c, hs_c), (1.0, 0.3, 0.3)), ((hs_a, hs_b, hs_c), (1.0, 0.6, 0.3)),
        )
        Iv, IIv = np3(sym...)
        @test _hs_eval_vec([Iv..., IIv...]) ≈ [Iterators.flatten(np3(num...))...] rtol = 1.0e-13
    end

    # The shape class follows the structure of the expressions.
    @test Cylinder(hs_a, hs_a) isa Cylinder{CircularCylindrical}
    @test Cylinder(hs_a, hs_c) isa Cylinder{EllipticCylindrical}

    # The incomplete elliptic integrals stay calls, rather than 50 Carlson
    # duplications unrolled into the expression.
    P = hill_tensor(Ellipsoid(hs_a, hs_b, hs_c), TensISO{3}(hs_k))
    @test occursin("ell_F", string(get_array(P)[1, 1]))
    @test length(string(get_array(P)[1, 1])) < 1000

    # So do Carlson's integrals, exported too; a folded substitution evaluates
    # every such call.
    point = Dict(hs_a => 0.7, hs_b => 0.4, hs_c => 1.3)
    for (f, args) in ((ell_RF, (0.7, 0.4, 1.3)), (ell_RD, (0.7, 0.4, 1.3)), (ell_F, (0.7, 0.4)), (ell_E, (0.7, 0.4)))
        expr = f((hs_a, hs_b, hs_c)[1:length(args)]...)
        @test occursin(string(nameof(f)), string(expr))
        @test Symbolics.value(Symbolics.substitute(expr, point; fold = Val(true))) ≈ f(args...) rtol = 1.0e-15
    end
end
