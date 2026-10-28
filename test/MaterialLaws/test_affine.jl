# The affine formulation: linear phases give the linear scheme exactly; on a
# laminate, whose phases carry uniform fields, it is the exact solution, found
# here independently by the continuity of the tractions; with Mori–Tanaka and
# an ellipsoidal inclusion, the phase strains it returns are those of the
# Eshelby problem of the linear thermoelastic comparison composite; the
# tangents and the derivatives in the parameters are consistent.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
using ForwardDiff

const MP_AF = MeanFieldHomogenization.MaterialLaws

_power_af(k, σ₀, m) = PowerLawElasticity(; k, σ₀, ε₀ = 1.0e-3, m)
_stress_af(law, ε) = MP_AF._fluxes(law, ε, SVector{0, Float64}())
_tangent_af(law, ε) = ForwardDiff.jacobian(e -> _stress_af(law, e), ε)

function _rve_af(shape = Ellipsoid(1.0); M = TensISO{3}(1.0, 1.0), I = TensISO{3}(1.0, 1.0))
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => M); fraction = :rest)
    add_phase!(rve, :I, shape, Dict(:C => I); fraction = 0.3)
    return rve
end

function _laminate_af()
    lam = Laminate(; normal = (0, 0, 1))
    add_layer!(lam, :A, Dict(:C => TensISO{3}(1.0, 1.0)); fraction = 0.4)
    add_layer!(lam, :B, Dict(:C => TensISO{3}(1.0, 1.0)); fraction = 0.6)
    return lam
end

const E_AF = SVector(1.0e-3, -2.0e-4, 3.0e-4, 1.0e-4, 0, -2.0e-4)

@testset "linear phases: the linear Mori–Tanaka estimate" begin
    # m = 1: linear elasticity of bulk modulus k and shear modulus σ₀/(3ε₀).
    c = AffineComposite(_rve_af(), (M = _power_af(50.0e3, 100.0, 1.0), I = _power_af(100.0e3, 400.0, 1.0)))
    σ, 𝕂, _ = point_update(c, E_AF, initial_state(c), 1.0)
    rve = _rve_af(; M = TensISO{3}(3 * 50.0e3, 2 * 100.0 / 3.0e-3), I = TensISO{3}(3 * 100.0e3, 2 * 400.0 / 3.0e-3))
    C = KM(homogenize(rve, MoriTanaka(), :C))
    @test 𝕂 ≈ C rtol = 1.0e-12
    @test σ ≈ C * E_AF rtol = 1.0e-12
end

# The exact laminate: uniform strains in the layers, ε_A = E + f_B (a ⊗ˢ n) and
# ε_B = E − f_A (a ⊗ˢ n), the jump a such that the tractions σ·n agree.
function _exact_laminate_af(lawA, lawB, E; fA = 0.4, fB = 0.6)
    symn(a) = SVector(0, 0, a[3], a[2] / sqrt(2), a[1] / sqrt(2), 0)
    traction(σ) = SVector(σ[5] / sqrt(2), σ[4] / sqrt(2), σ[3])
    F(a) = traction(_stress_af(lawA, E + fB * symn(a))) - traction(_stress_af(lawB, E - fA * symn(a)))
    a = zero(SVector{3, eltype(E)})
    for _ in 1:50
        a -= ForwardDiff.jacobian(F, a) \ F(a)
    end
    return fA * _stress_af(lawA, E + fB * symn(a)) + fB * _stress_af(lawB, E - fA * symn(a)), maximum(abs, F(a))
end

@testset "a laminate: the exact solution" begin
    laws = (A = _power_af(50.0e3, 100.0, 0.3), B = _power_af(100.0e3, 400.0, 0.3))
    c = AffineComposite(_laminate_af(), laws; scheme = Laminated())
    σ, _, _ = point_update(c, E_AF, initial_state(c), 1.0)
    Σ, jump = _exact_laminate_af(laws.A, laws.B, E_AF)
    @test jump < 1.0e-10 * maximum(abs, Σ)
    @test σ ≈ Σ rtol = 1.0e-12
    @test check_material_interface(c; verbose = false)
    @inferred point_update(c, E_AF, initial_state(c), 1.0)
    # The derivative in a parameter of a phase, against that of the exact solution.
    dσ = ForwardDiff.derivative(s -> point_update(AffineComposite(_laminate_af(), (A = _power_af(50.0e3, s, 0.3), B = laws.B); scheme = Laminated()), E_AF, initial_state(c), 1.0)[1], 100.0)
    dΣ = ForwardDiff.derivative(s -> first(_exact_laminate_af(_power_af(50.0e3, s, 0.3), laws.B, E_AF)), 100.0)
    @test dσ ≈ dΣ rtol = 1.0e-10
end

@testset "Mori–Tanaka: the Eshelby problem of the comparison composite" begin
    shape = Ellipsoid(2.0, 1.0, 0.5)
    laws = (M = _power_af(50.0e3, 100.0, 0.3), I = _power_af(100.0e3, 400.0, 0.3))
    c = AffineComposite(_rve_af(shape), laws)
    σ, _, st = point_update(c, E_AF, initial_state(c), 1.0)
    εM = st.y
    εI = (E_AF - 0.7 * εM) / 0.3
    # The inclusion of tangent L_I and prestress τ_I in the matrix of tangent
    # L_M and prestress τ_M, under the matrix strain: its strain is uniform,
    # (𝕀 + ℙ(L_I − L_M)) ε_I = ε_M − ℙ(τ_I − τ_M), and the mixture rule closes it.
    LM, LI = _tangent_af(laws.M, εM), _tangent_af(laws.I, εI)
    τM, τI = _stress_af(laws.M, εM) - LM * εM, _stress_af(laws.I, εI) - LI * εI
    ℙ = MP_AF._mandel(TensND.change_tens_canon(hill_tensor(shape, MP_AF._symmetric_tensor(LM))))
    B = inv(I + ℙ * (LI - LM))
    @test (I + (0.3 / 0.7) * B) \ (B * (E_AF / 0.7 - ℙ * (τI - τM))) ≈ εI rtol = 1.0e-12
    @test σ ≈ 0.7 * _stress_af(laws.M, εM) + 0.3 * _stress_af(laws.I, εI) rtol = 1.0e-14
    @inferred point_update(c, E_AF, initial_state(c), 1.0)
end

@testset "the consistent tangent of Mori–Tanaka" begin
    # Hill tensors of the anisotropic tangents to a tight tolerance, so that the
    # finite difference of the check is not dominated by the quadrature.
    laws = (M = _power_af(50.0e3, 100.0, 0.3), I = _power_af(100.0e3, 400.0, 0.3))
    c = AffineComposite(_rve_af(), laws; reltol = 1.0e-12, abstol = 1.0e-14)
    @test check_material_interface(c; verbose = false)
end

@testset "refusals and limits" begin
    laws = (M = _power_af(50.0e3, 100.0, 0.3), I = _power_af(100.0e3, 400.0, 0.3))
    @test_throws ArgumentError AffineComposite(_rve_af(), (M = laws.M,))
    @test_throws ArgumentError AffineComposite(_rve_af(), (M = laws.M, X = laws.I))
    @test_throws ArgumentError AffineComposite(_rve_af(), (M = laws.M, I = J2Plasticity(; E = 70.0e3, ν = 0.3, σ₀ = 100.0, H = 1.0e3)))
    # Under a pure volume change the phases carry no deviatoric strain, where a
    # power law of exponent below one has no finite tangent: no linearization.
    c = AffineComposite(_rve_af(), laws)
    @test_throws LocalIntegrationError point_update(c, SVector(1.0e-3, 1.0e-3, 1.0e-3, 0, 0, 0), initial_state(c), 1.0)
    # At rest nothing is solved.
    σ, _, _ = point_update(c, zero(SVector{6}), initial_state(c), 1.0)
    @test iszero(σ)
    @test MP_AF._with_external(c, (;)) === c
    @test MP_AF._with_external(c, (T = 1.0,)) isa AffineComposite
end
