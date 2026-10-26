# Single-crystal viscoplasticity: the slip systems against their Miller indices
# written here, the cubic elasticity, steady creep against the Schmid factors
# computed here in two orientations, the creep rate once the hardening has
# saturated, and the consistent tangent against dual numbers through the step.

using Test
using MeanFieldHomogenization
using TensND
using StaticArrays
using LinearAlgebra
using ForwardDiff

# The FCC systems {111}<110>, by their Miller indices.
const FCC_MILLER = [
    ((1, 1, 1), (0, 1, -1)), ((1, 1, 1), (1, 0, -1)), ((1, 1, 1), (1, -1, 0)),
    ((-1, 1, 1), (0, 1, -1)), ((-1, 1, 1), (1, 0, 1)), ((-1, 1, 1), (1, 1, 0)),
    ((1, -1, 1), (0, 1, 1)), ((1, -1, 1), (1, 0, -1)), ((1, -1, 1), (1, 1, 0)),
    ((1, 1, -1), (0, 1, 1)), ((1, 1, -1), (1, 0, 1)), ((1, 1, -1), (1, -1, 0)),
]
_unit(v) = SVector{3, Float64}(v) / norm(SVector{3, Float64}(v))

# The same plane and the same line, whatever the signs.
_same_system(a, b) = abs(dot(a[1], b[1])) ≈ 1 && abs(dot(a[2], b[2])) ≈ 1

@testset "slip systems" begin
    fcc = slip_systems(:fcc)
    @test length(fcc) == 12
    @test all(((n, m),) -> norm(n) ≈ 1 && norm(m) ≈ 1 && abs(dot(n, m)) < 1.0e-14, fcc)
    @test all(s -> count(t -> _same_system(s, t), fcc) == 1, [(_unit(n), _unit(m)) for (n, m) in FCC_MILLER])
    bcc = slip_systems(:bcc)
    @test length(bcc) == 12
    @test all(((n, m),) -> abs(dot(n, m)) < 1.0e-14 && count(!iszero, round.(n; digits = 12)) == 2, bcc)
    @test_throws ArgumentError slip_systems(:hcp)
end

const C11_CR, C12_CR, C44_CR = 168.0e3, 121.0e3, 75.0e3      # a copper-like cubic crystal, MPa
_crystal(; frame = TensND.CanonicalBasis{3, Float64}(), kw...) = CrystalViscoplasticity(;
    C₁₁ = C11_CR, C₁₂ = C12_CR, C₄₄ = C44_CR, K = 20.0, n = 5.0, τ₀ = 15.0, frame, kw...
)

@testset "cubic elasticity" begin
    for frame in (TensND.CanonicalBasis{3, Float64}(), TensND.RotatedBasis(0.3, 1.1, -0.4))
        law = _crystal(; frame)
        ε = SVector(2.0e-5, -1.0e-5, 0.5e-5, 0, 0, 1.0e-5)
        σ, 𝕂, st = point_update(law, ε, initial_state(law), 0.0)
        @test 𝕂 ≈ mandel(tens_cubic(C11_CR, C12_CR, C44_CR, frame), TensND.CanonicalBasis{3, Float64}()) rtol = 1.0e-12
        @test iszero(st.y)
    end
end

# Axial strain rate under a constant uniaxial stress σ along the unit vector a of
# the crystal frame, with no hardening: each system slips at the rate its Schmid
# factor gives, steadily.
function _steady_axial_rate(a, σ; K = 20.0, n = 5.0, τ₀ = 15.0)
    rate = 0.0
    for (nm, mm) in FCC_MILLER
        nv, mv = _unit(nm), _unit(mm)
        f = dot(a, nv) * dot(a, mv)                 # Schmid factor
        τ = σ * f
        rate += max((abs(τ) - τ₀) / K, 0)^n * sign(τ) * f
    end
    return rate
end

@testset "steady creep against the Schmid factors" begin
    σ = 60.0
    # Along a cube axis, then along a direction of no symmetry: the crystal is
    # rotated by `frame`, the load stays along the global e₁.
    for frame in (TensND.CanonicalBasis{3, Float64}(), TensND.RotatedBasis(0.7, 0.4, 0.2))
        law = _crystal(; frame)
        R = SMatrix{3, 3}(TensND.vecbasis(frame))
        a = transpose(R) * SVector(1.0, 0, 0)           # e₁ in the crystal frame
        ts = range(0, 10; length = 6)
        r = point_response(law, LoadingPath(ts; σ11 = σ))
        S = inv(mandel(tens_cubic(C11_CR, C12_CR, C44_CR, frame), TensND.CanonicalBasis{3, Float64}()))
        @test r[:ε11] ≈ [S[1, 1] * σ + _steady_axial_rate(a, σ) * t for t in ts] rtol = 1.0e-9
    end
end

@testset "creep once the hardening has saturated" begin
    # Along [001] eight systems carry the same resolved stress σ/√6; with all the
    # interactions equal (latent = 1) and a fast saturation, each critical stress
    # tends to τ₀ + 8Q and the creep rate to its steady value at that threshold.
    σ, Q = 90.0, 1.0
    law = _crystal(; Q, b = 400.0)
    r = point_response(law, LoadingPath(range(0, 4; length = 41); σ11 = σ))
    late = (r[:ε11][end] - r[:ε11][end - 1]) / (r[:t][end] - r[:t][end - 1])
    τ = σ / sqrt(6)
    @test late ≈ 8 * ((τ - 15.0 - 8Q) / 20.0)^5 / sqrt(6) rtol = 1.0e-6
    @test all(≈(r[:v][end][1]), filter(>(0), r[:v][end]))     # the active systems alike
end

@testset "the tangent is the derivative of the whole step" begin
    law = _crystal(; frame = TensND.RotatedBasis(0.7, 0.4, 0.2), Q = 2.0, b = 50.0)
    st = point_update(law, SVector(3.0e-4, -1.0e-4, -1.0e-4, 0, 0, 0), initial_state(law), 0.5)[3]
    ε = SVector(6.0e-4, -2.0e-4, -2.5e-4, 1.0e-4, 0, 2.0e-4)
    σ, 𝕂, _ = point_update(law, ε, st, 0.5)
    @test 𝕂 ≈ ForwardDiff.jacobian(e -> point_update(law, e, st, 0.5)[1], ε) rtol = 1.0e-9
    # And the derivative with respect to a parameter, against a central difference
    # of the step (a check of the derivative, not a computation the law uses).
    f(τ₀) = point_update(_crystal(; frame = TensND.RotatedBasis(0.7, 0.4, 0.2), τ₀, Q = 2.0, b = 50.0), ε, st, 0.5)[1][1]
    @test ForwardDiff.derivative(f, 15.0) ≈ (f(15.0 + 1.0e-4) - f(15.0 - 1.0e-4)) / 2.0e-4 rtol = 1.0e-5
    @test_throws DomainError _crystal(; n = 0.5)
end
