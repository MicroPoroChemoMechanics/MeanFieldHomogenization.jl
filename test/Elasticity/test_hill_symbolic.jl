using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
import Symbolics
import ForwardDiff

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

@testset "second_hill_tensor of a symbolic sphere" begin
    # Q = C − C:P:C of a sphere, held to its closed form
    # 12kμ/(3k+4μ) 𝕁 + 2μ(9k+8μ)/(5(3k+4μ)) 𝕂, and K − K·P·K = (2k/3) 𝟏 at order 2.
    Q = second_hill_tensor(Ellipsoid(hs_a, hs_a, hs_a), TensISO{3}(3hs_k, 2hs_μ))
    @test eltype(get_array(Q)) <: Symbolics.Num
    k, μ = HS_POINT[4], HS_POINT[5]
    Q_ref = TensISO{3}(12k * μ / (3k + 4μ), 2μ * (9k + 8μ) / (5 * (3k + 4μ)))
    @test _hs_eval(Q) ≈ get_array(Q_ref) rtol = 1.0e-13
    Q2 = second_hill_tensor(Ellipsoid(hs_a, hs_a, hs_a), TensISO{3}(hs_k))
    @test _hs_eval(Q2) ≈ (2k / 3) * I(3) rtol = 1.0e-13
end

@testset "derivatives of the elliptic integrals, symbolically" begin
    # Each registered rule, expanded and evaluated, against ForwardDiff through
    # the AGM and Carlson arithmetic of the numerical methods.
    Symbolics.@variables m φ x y z
    D(v) = Symbolics.Differential(v)
    num(e, vals) = Symbolics.value(Symbolics.substitute(Symbolics.expand_derivatives(e), vals; fold = Val(true)))
    for m₀ in (0.1, 0.5, 0.9)
        @test num(D(m)(ell_K(m)), Dict(m => m₀)) ≈ ForwardDiff.derivative(ell_K, m₀) rtol = 1.0e-12
        @test num(D(m)(ell_E(m)), Dict(m => m₀)) ≈ ForwardDiff.derivative(ell_E, m₀) rtol = 1.0e-12
        for φ₀ in (0.3, 1.1)
            v = Dict(φ => φ₀, m => m₀)
            @test num(D(φ)(ell_F(φ, m)), v) ≈ ForwardDiff.derivative(p -> ell_F(p, m₀), φ₀) rtol = 1.0e-12
            @test num(D(m)(ell_F(φ, m)), v) ≈ ForwardDiff.derivative(q -> ell_F(φ₀, q), m₀) rtol = 1.0e-11
            @test num(D(φ)(ell_E(φ, m)), v) ≈ ForwardDiff.derivative(p -> ell_E(p, m₀), φ₀) rtol = 1.0e-12
            @test num(D(m)(ell_E(φ, m)), v) ≈ ForwardDiff.derivative(q -> ell_E(φ₀, q), m₀) rtol = 1.0e-11
        end
    end
    p₀ = (0.7, 0.4, 1.3)
    v = Dict(x => p₀[1], y => p₀[2], z => p₀[3])
    for (i, s) in enumerate((x, y, z)), f in (ell_RF, ell_RD)
        g(t) = f(ntuple(j -> j == i ? t : p₀[j], 3)...)
        @test num(D(s)(f(x, y, z)), v) ≈ ForwardDiff.derivative(g, p₀[i]) rtol = 1.0e-11
    end
end

@testset "the Hill tensor of a symbolic triaxial ellipsoid, differentiated" begin
    # ∂P/∂b of the closed form, expanded through the elliptic integrals, against
    # the derivative of the numerical Hill tensor.
    # On the nine coefficients of the orthotropic tensor, substituted: the 81
    # components, compiled, cost more memory than the check is worth.
    P = hill_tensor(Ellipsoid(hs_a, hs_b, hs_c), TensISO{3}(3hs_k, 2hs_μ))
    point = Dict(zip(HS_VARS, HS_POINT))
    dP = [Symbolics.value(Symbolics.substitute(Symbolics.expand_derivatives(Symbolics.Differential(hs_b)(x)), point; fold = Val(true))) for x in get_data(P)]
    ref = ForwardDiff.derivative(b -> collect(get_data(hill_tensor(Ellipsoid(1.0, b, 0.3), TensISO{3}(15.0, 4.0)))), 0.6)
    @test dP ≈ ref rtol = 1.0e-10 atol = 1.0e-13
end

@testset "symbolic routes: TI coaxial spheroid, layered sphere, iterative schemes" begin
    # A spheroid in a coaxial transversely isotropic matrix: the closed form of
    # the integrals, taken on the type, against the numerical Hill tensor.
    Cti(k, μ) = ti_from_constants(k, μ, k / 3, 2k, μ / 2, [0, 0, 1])
    # Its five Walpole coefficients, substituted: compiling the 81 components
    # of so long an expression costs more memory than the check is worth. The
    # roots of the acoustic polynomial may be complex conjugate, so that the
    # closed form is the analytic continuation through complex intermediates:
    # it is evaluated at complex values, and its imaginary part vanishes.
    point = Dict(zip(HS_VARS, complex.(HS_POINT)))
    function walpole_at(P)
        w = [Symbolics.value(Symbolics.substitute(ℓ, point; fold = Val(true))) for ℓ in get_data(P)]
        @test maximum(abs ∘ imag, w) <= 1.0e-12 * maximum(abs, w)
        return real.(w)
    end
    P = hill_tensor(Ellipsoid(hs_a, hs_a, hs_c), Cti(hs_k, hs_μ))
    @test walpole_at(P) ≈ collect(get_data(hill_tensor(Ellipsoid(1.0, 1.0, 0.3), Cti(5.0, 2.0)))) rtol = 1.0e-12
    # Prolate too, where the roots fall on the other side of 1.
    Pp = hill_tensor(Ellipsoid(hs_c, hs_c, hs_a), Cti(hs_k, hs_μ))
    @test walpole_at(Pp) ≈ collect(get_data(hill_tensor(Ellipsoid(0.3, 0.3, 1.0), Cti(5.0, 2.0)))) rtol = 1.0e-12
    # A layered sphere of symbolic radii and moduli.
    sphere(a, b, k, μ) = LayeredSphere((a, b), (TensISO{3}(6k, 4μ), TensISO{3}(9k, 8μ)))
    N = property_contribution(sphere(hs_c, hs_a, hs_k, hs_μ), TensISO{3}(3hs_k, 2hs_μ))
    Nn = [Symbolics.value(Symbolics.substitute(x, Dict(zip(HS_VARS, HS_POINT)); fold = Val(true))) for x in get_data(N)]
    @test Nn ≈ collect(get_data(property_contribution(sphere(0.3, 1.0, 5.0, 2.0), TensISO{3}(15.0, 4.0)))) rtol = 1.0e-12
    # The iterative schemes refuse symbolic moduli, by name and with the way out.
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1, 1, 1), Dict(:C => TensISO{3}(3hs_k, 2hs_μ)); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1, 1, 1), Dict(:C => TensISO{3}(6hs_k, 4hs_μ)); fraction = 0.2)
    for sch in (SelfConsistent(), AsymmetricSelfConsistent(), DifferentialScheme())
        @test_throws "explicit scheme" homogenize(rve, sch, :C)
    end
    # The path of the differential scheme too, which integrates the same ODE.
    @test_throws "DifferentialScheme iterates" differential_path(rve, DifferentialScheme(), :C)
    @test homogenize(rve, MoriTanaka(), :C) isa TensISO{4, 3, Symbolics.Num}
    # Mori-Tanaka with a crack family: the estimate inverts a fourth-order
    # tensor of symbolic components (TensND ≥ 0.11.2), held to the numbers.
    function cracked(k, μ, d)
        r = RVE()
        add_phase!(r, :M, Ellipsoid(1, 1, 1), Dict(:C => TensISO{3}(3k, 2μ)); fraction = :rest)
        add_phase!(r, :F, PennyCrack(1.0), Dict(:C => TensISO{3}(0, 0)); density = d)
        return homogenize(r, MoriTanaka(), :C)
    end
    Ccr = cracked(hs_k, hs_μ, hs_c)
    Ccr_at = [Symbolics.value(Symbolics.substitute(x, Dict(zip(HS_VARS, HS_POINT)); fold = Val(true))) for x in canonical_components(Ccr)]
    @test Ccr_at ≈ canonical_components(cracked(5.0, 2.0, 0.3)) rtol = 1.0e-12
end

@testset "symbolic routes: anisotropic conduction, cylinder in an anisotropic matrix" begin
    at_point(x) = Symbolics.value(Symbolics.substitute(x, Dict(zip(HS_VARS, HS_POINT)); fold = Val(true)))
    # A conductivity diagonal in the inclusion's frame: the fictitious
    # ellipsoid of semi-axes aᵢ / √Kᵢᵢ, against the numerical route
    # (eigendecomposition, SVD and Newton potential at numbers).
    Kti(kt, kn) = TensTI{2}(kt, kn, [0, 0, 1])
    for (ell, elln) in (
            (Ellipsoid(hs_a, hs_a, hs_c), Ellipsoid(1.0, 1.0, 0.3)),
            (Ellipsoid(hs_a, hs_b, hs_c), Ellipsoid(1.0, 0.6, 0.3)),
            (Cylinder(hs_a, hs_c), Cylinder(1.0, 0.3)),
        )
        P = hill_tensor(ell, Kti(hs_k, hs_μ))
        @test at_point.(canonical_components(P)) ≈ canonical_components(hill_tensor(elln, Kti(5.0, 2.0))) rtol = 1.0e-12
    end
    # A conductivity whose axes are not the inclusion's is refused.
    @test_throws "diagonal in the ellipsoid's frame" hill_tensor(Ellipsoid(hs_a, hs_b, hs_c), TensTI{2}(hs_k, hs_μ, [0, 1, 1]))
    # The cylinder in an anisotropic elastic matrix is a quadrature: refused.
    Cti = ti_from_constants(hs_k, hs_μ, hs_k / 3, 2hs_k, hs_μ / 2, [0, 0, 1])
    @test_throws "adaptive quadrature" hill_tensor(Cylinder(hs_a, hs_a), Cti)
end
