# The engine of MaterialPoints: the consistent tangent against dual numbers
# carried through the whole step, derivatives with respect to parameters (first
# and nested) against references solved independently, inference and
# allocations, symbolic laws, laws written by a user, and the refusals.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
using ForwardDiff
import Symbolics

const MP_ENGINE = MeanFieldHomogenization.MaterialPoints
const SQ2_ENG = sqrt(2.0)
_mve(a, b, c, d = 0.0, e = 0.0, f = 0.0) = SVector(a, b, c, SQ2_ENG * d, SQ2_ENG * e, SQ2_ENG * f)

# One law per regime: plastic, viscoplastic, viscoelastic, nonlinear elastic,
# coupled; with a step and a previous state that put it in that regime.
function _regimes()
    j2 = J2Plasticity(;
        E = 200.0e3, ν = 0.3, σ₀ = 250.0, H = 2.0e3, Hkin = 4.0e3,
        hardening = p -> 250.0 + 2.0e3 * p + 80.0 * (1 - exp(-200.0 * p))
    )
    norton = NortonViscoplasticity(; E = 200.0e3, ν = 0.3, K = 500.0, n = 3.0)
    maxwell = MaxwellViscoelasticity(; E = 200.0e3, ν = 0.3, τ = 2.0)
    power = PowerLawElasticity(; k = 1.0e5, σ₀ = 100.0, ε₀ = 1.0e-3, m = 0.3)
    biot = BiotPoroelasticity(; E = 200.0e3, ν = 0.3, b = 0.8, M = 1.0e4)
    ε = _mve(3.0e-3, -1.0e-3, -0.5e-3, 4.0e-4, -2.0e-4, 6.0e-4)
    return (
        (j2, ε, 0.0), (norton, ε, 0.5), (maxwell, ε, 0.3), (power, ε, 0.0),
        (biot, vcat(ε, SVector(3.0)), 0.0),
    )
end

@testset "the tangent is the derivative of the whole step" begin
    for (law, g, Δt) in _regimes()
        prev = initial_state(law)
        # A first step, so that the previous state is not virgin.
        _, _, prev = point_update(law, g / 2, prev, Δt)
        F, 𝕂, _ = point_update(law, g, prev, Δt)
        # Duals carried through the predictor, the Newton iterations and the
        # derivative steps: an independent computation of the same Jacobian.
        J = ForwardDiff.jacobian(h -> point_update(law, h, prev, Δt)[1], g)
        @test 𝕂 ≈ J rtol = 1.0e-9
        @test F ≈ point_update(law, g, prev, Δt)[1]
    end
end

# ── Derivatives with respect to the parameters ───────────────────────────────

# The radial return of linear isotropic hardening in closed form: σ and the
# consistent tangent, as functions of (E, ν, σ₀, H) written here.
function _rr_stress(ε, E, ν, σ₀, H)
    μ, k = E / (2 * (1 + ν)), E / (3 * (1 - 2ν))
    one6 = SVector(1.0, 1, 1, 0, 0, 0)
    e = ε - (sum(ε[1:3]) / 3) * one6
    s = 2μ * e
    q = sqrt(1.5 * sum(abs2, s))
    Δp = (q - σ₀) / (3μ + H)
    return k * sum(ε[1:3]) * one6 + s - 2μ * Δp * 1.5 * s / q
end

@testset "derivatives with respect to the parameters" begin
    ε = _mve(3.0e-3, -1.0e-3, -0.5e-3, 4.0e-4)
    law(p) = J2Plasticity(; E = p[1], ν = p[2], σ₀ = p[3], H = p[4])
    p₀ = SVector(200.0e3, 0.3, 250.0, 2.0e3)
    # Stress: the Jacobian with respect to the four parameters.
    Jσ = ForwardDiff.jacobian(p -> point_update(law(p), ε, initial_state(law(p₀)), 0.0)[1], p₀)
    Jref = ForwardDiff.jacobian(p -> _rr_stress(ε, p...), p₀)
    @test Jσ ≈ Jref rtol = 1.0e-9
    # Tangent: the derivative of the consistent tangent with respect to E.
    d𝕂 = ForwardDiff.derivative(E -> point_update(law(SVector(E, 0.3, 250.0, 2.0e3)), ε, initial_state(law(p₀)), 0.0)[2], 200.0e3)
    d𝕂ref = ForwardDiff.derivative(E -> ForwardDiff.jacobian(e -> _rr_stress(e, E, 0.3, 250.0, 2.0e3), ε), 200.0e3)
    @test d𝕂 ≈ d𝕂ref rtol = 1.0e-8
    # Nested: the second derivative with respect to the yield stress of a law
    # whose hardening saturates, against a scalar Newton written here.
    sat(σ₀) = J2Plasticity(; E = 200.0e3, ν = 0.3, σ₀, hardening = p -> σ₀ + 150.0 * (1 - exp(-400.0 * p)))
    εu = _mve(4.0e-3, -2.0e-3, -2.0e-3)
    σ11(σ₀) = point_update(sat(σ₀), εu, initial_state(sat(250.0)), 0.0)[1][1]
    function σ11ref(σ₀)
        # Isochoric extension: σ_eq trial = 3μ e, Δp solves 3μ(e − Δp) = R(Δp).
        μ = 200.0e3 / 2.6
        e = 4.0e-3 - (0.0) / 3
        Δp = zero(σ₀)
        for _ in 1:60
            r = 3μ * (e - Δp) - (σ₀ + 150.0 * (1 - exp(-400.0 * Δp)))
            Δp += r / (3μ + 150.0 * 400.0 * exp(-400.0 * Δp))
        end
        return 2μ * (e - Δp)                      # s₁₁ = 2μ (e₁₁ − Δεᵖ₁₁), Δεᵖ₁₁ = Δp
    end
    @test σ11(250.0) ≈ σ11ref(250.0) rtol = 1.0e-11
    @test ForwardDiff.derivative(σ11, 250.0) ≈ ForwardDiff.derivative(σ11ref, 250.0) rtol = 1.0e-9
    @test ForwardDiff.derivative(s -> ForwardDiff.derivative(σ11, s), 250.0) ≈
        ForwardDiff.derivative(s -> ForwardDiff.derivative(σ11ref, s), 250.0) rtol = 1.0e-7
    # A dual time step: the derivative of a Maxwell step with respect to Δt.
    mx = MaxwellViscoelasticity(; E = 200.0e3, ν = 0.3, τ = 2.0)
    st = point_update(mx, ε, initial_state(mx), 0.0)[3]
    dσdΔt = ForwardDiff.derivative(Δt -> point_update(mx, ε, st, Δt)[1][1], 0.4)
    σ₀11 = point_update(mx, ε, st, 0.0)[1][1]
    @test dσdΔt ≈ ForwardDiff.derivative(Δt -> σ₀11 / (1 + Δt / 2.0), 0.4) rtol = 1.0e-10
end

@testset "inference, no allocation" begin
    for (law, g, Δt) in _regimes()
        prev = initial_state(law)
        @inferred point_update(law, g, prev, Δt)
        point_update(law, g, prev, Δt)
        @test (@allocated point_update(law, g, prev, Δt)) == 0
    end
end

# ── Symbolic numbers ─────────────────────────────────────────────────────────

@testset "symbolic laws without internal variable" begin
    Symbolics.@variables k μ e1 e2 e3
    iso(g, α, p) = p.k / 2 * trace_mandel(g.ε)^2 + p.μ * sum(abs2, deviator(g.ε))
    law = PotentialLaw(; energy = iso, p = (; k, μ))
    F, 𝕂, _ = point_update(law, SVector(e1, e2, e3, 0, 0, 0), initial_state(law), 0)
    @test isequal(Symbolics.simplify(𝕂[1, 2] - (k - 2μ / 3)), 0)
    @test isequal(Symbolics.simplify(𝕂[4, 4] - 2μ), 0)
    val = Dict(k => 10.0, μ => 3.0, e1 => 1.0e-3, e2 => -2.0e-4, e3 => 5.0e-4)
    num = PotentialLaw(; energy = iso, p = (k = 10.0, μ = 3.0))
    Fn = point_update(num, SVector(1.0e-3, -2.0e-4, 5.0e-4, 0, 0, 0), initial_state(num), 0.0)[1]
    @test Symbolics.value.(Symbolics.substitute.(F, Ref(val); fold = Val(true))) ≈ Fn rtol = 1.0e-13
    # A law written with a matrix product: Symbolics 7.47's ForwardDiff extension
    # calls `calc_muladd_xy`/`calc_muladd_xz` without importing them, so
    # `muladd` with a `Num` among duals fails upstream. Passes once it is fixed.
    Symbolics.@variables E b M
    @test_broken point_update(BiotPoroelasticity(; E, ν = 0.3, b, M), SVector(e1, e2, 0, 0, 0, 0, 1), initial_state(BiotPoroelasticity(; E = 1.0, ν = 0.3, b = 1.0, M = 1.0)), 0) isa Tuple
    # An iterative law refuses symbols, before evaluating anything.
    j2 = J2Plasticity(; E = 200.0e3, ν = 0.3, σ₀ = 250.0)
    @test_throws ArgumentError point_update(j2, SVector{6}(fill(e1, 6)), initial_state(j2), 0)
    @test_throws ArgumentError point_update(J2Plasticity(; E, ν = 0.3, σ₀ = 250.0), _mve(1.0e-3, 0, 0), initial_state(j2), 0.0)
end

# ── Laws written by a user ───────────────────────────────────────────────────

@testset "a damage law with a threshold" begin
    ℂ = MP_ENGINE._iso_mandel(30.0e3, 0.2)
    Y(ε, p) = sum(ε .* (p.ℂ * ε)) / 2
    target(ε, p) = 1 - sqrt(p.Y₀ / Y(ε, p))
    law = ResidualLaw(;
        name = :ToyDamage,
        stress = (ε, y, p) -> (1 - y.d) * (p.ℂ * ε),
        residual = (y, g, prev, Δt, p) -> (d = y.d - target(g.ε, p),),
        elastic = (g, prev, Δt, p) -> target(g.ε, p) <= prev.y.d,
        layout = (d = 1,), p = (ℂ = ℂ, Y₀ = 1.0e-4),
    )
    @test state_layout(law) == (d = 1,)
    @test law_parameters(law).Y₀ == 1.0e-4
    @test kinematics(law) === SmallStrain()
    @test sprint(show, law) == "ToyDamage (residual route; ε → σ; internal variables d)"
    # Uniaxial strain up, down, up again: d follows the threshold, then stays.
    es = [0.0, 2.0e-4, 4.0e-4, 2.0e-4, 0.0, 3.0e-4, 5.0e-4]
    ts = range(0, 1; length = length(es))
    r = point_response(law, LoadingPath(ts; ε11 = t -> es[round(Int, t * (length(es) - 1)) + 1], ε22 = 0, ε33 = 0))
    Yu(e) = ℂ[1, 1] * e^2 / 2
    dref = accumulate(max, [e == 0 ? 0.0 : max(0.0, 1 - sqrt(1.0e-4 / Yu(e))) for e in es])
    @test r[:d] ≈ dref atol = 1.0e-14
    @test r[:σ11] ≈ (1 .- dref) .* ℂ[1, 1] .* es rtol = 1.0e-12
end

@testset "a coupled law with a vector field" begin
    # Fourier conduction as a potential: ψ = −½ k |∇T|², q = −∂ψ/∂∇T … written
    # with the sign: q = −k ∇T.
    law = PotentialLaw(;
        energy = (g, α, p) -> p.k * sum(abs2, g.∇T) / 2, gradients = (∇T = 3,),
        flux_names = (q = 3,), signs = (q = -1,), p = (k = 2.5,)
    )
    ∇T = SVector(1.0, -2.0, 0.5)
    r = material_response(law, (∇T = ∇T,), initial_state(law), 0.0)
    @test r.fluxes.q ≈ -2.5 * ∇T
    @test r.tangents.q∇T ≈ -2.5 * I(3)
    @test probe_gradients(law, nothing).∇T == SVector(1.0e-3, 1.0e-3, 1.0e-3)
    rp = point_response(law, LoadingPath(0:0.5:1; ∇T1 = t -> t, ∇T2 = 0, q3 = 0))
    @test rp[:q1][end] ≈ -2.5 && rp[:∇T3][end] == 0.0
    # A tensor flux against a scalar gradient block, read through the contract.
    biot = BiotPoroelasticity(; E = 200.0e3, ν = 0.3, b = 0.8, M = 1.0e4)
    @test probe_gradients(biot, from_mandel(_mve(1.0e-3, 0, 0))).p == 1.0e-3
    @test check_material_interface(biot; verbose = false)
end

@testset "refusals" begin
    s(ε, y, p) = ε
    res(y, g, prev, Δt, p) = (a = y.a,)
    @test_throws ArgumentError ResidualLaw(; residual = res, layout = (a = 1,))
    @test_throws ArgumentError ResidualLaw(; stress = s, fluxes = (g, y, p) -> (σ = g.ε,), residual = res, layout = (a = 1,))
    @test_throws ArgumentError ResidualLaw(; stress = s, residual = res, layout = (a = 1,), gradients = (ε = 6, p = 1), flux_names = (σ = 6, φ = 1))
    @test_throws ArgumentError ResidualLaw(; stress = s, residual = res, layout = (a = 0,))
    @test_throws ArgumentError ResidualLaw(; stress = s, residual = res, layout = (a = 1,), gradients = (ε = 6,), flux_names = (σ = 3,))
    @test_throws DimensionMismatch ResidualLaw(; stress = s, residual = res, layout = (a = 1,), y₀ = [1.0, 2.0])
    @test ResidualLaw(; stress = s, residual = res, layout = (a = 1, b = 6), y₀ = (a = 2, b = SVector{6}(1:6))).y₀ ==
        SVector{7}(2.0, 1, 2, 3, 4, 5, 6)
    @test ResidualLaw(; stress = s, residual = res, layout = (a = 1,), y₀ = [3]).y₀ == SVector(3.0)
    @test_throws ArgumentError PotentialLaw(; energy = (g, α, p) -> 0.0, layout = (α = 1,))
    @test_throws ArgumentError PotentialLaw(; energy = (g, α, p) -> 0.0, signs = (τ = 1,))
    @test_throws ArgumentError PotentialLaw(; energy = (g, α, p) -> 0.0, signs = (σ = 2,))
    # A residual of the wrong length.
    bad = ResidualLaw(; stress = (ε, y, p) -> ε, residual = (y, g, prev, Δt, p) -> SVector(1.0, 2.0), layout = (a = 1,))
    @test_throws DimensionMismatch point_update(bad, _mve(1.0e-3, 0, 0), initial_state(bad), 0.0)
    @test_throws DimensionMismatch point_update(bad, SVector(1.0, 2.0), initial_state(bad), 0.0)
    # The local algorithm.
    @test_throws ArgumentError LocalNewton(; abstol = 0)
    @test_throws ArgumentError LocalNewton(; reltol = -1)
    @test_throws ArgumentError LocalNewton(; maxiters = 0)
    # No convergence, and a singular Jacobian: the error the driver catches.
    stiff = ResidualLaw(; stress = (ε, y, p) -> ε, residual = (y, g, prev, Δt, p) -> (a = y.a^3 - 2,), layout = (a = 1,), y₀ = [10.0])
    err = (@test_throws LocalIntegrationError point_update(stiff, _mve(1.0e-3, 0, 0), initial_state(stiff), 0.0; algorithm = LocalNewton(; maxiters = 2))).value
    @test err isa LocalIntegrationError && err.iterations == 2
    @test occursin("no convergence", sprint(showerror, err))
    flat = ResidualLaw(; stress = (ε, y, p) -> ε, residual = (y, g, prev, Δt, p) -> (a = zero(y.a) + 1,), layout = (a = 1,))
    @test_throws LocalIntegrationError point_update(flat, _mve(1.0e-3, 0, 0), initial_state(flat), 0.0)
end
