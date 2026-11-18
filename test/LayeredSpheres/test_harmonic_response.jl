using Test
using MeanFieldHomogenization
using TensND
using ForwardDiff

# =============================================================================
#  Response of a layered sphere to a harmonic of any degree (conduction).
#
#  References, all independent of the transfer-matrix recurrence:
#   - homogeneous sphere, perfect interface: the two continuity conditions
#     solved by hand, B/A = -n (k₁ - k₀) R^(2n+1) / (n k₁ + (n+1) k₀);
#   - Kapitza resistance ρ: the same with the temperature drop ρ qₙ, i.e. the
#     apparent conductivity k₁ / (1 + n ρ k₁ / R) of the sphere at degree n;
#   - surface conductance kˢ: apparent conductivity k₁ + (n + 1) kˢ / R;
#   - coated sphere: the 4×4 system of the four interface conditions.
# =============================================================================

_homogeneous(n, k₁, k₀, R) = -n * (k₁ - k₀) * R^(2n + 1) / (n * k₁ + (n + 1) * k₀)

# Four continuity conditions of a core (k₁, r₁) in a shell (k₂, r₂) in a
# matrix k₀, unknowns (a₁, a₂, b₂, B) for a unit incident amplitude.
function _coated_reference(n, k₁, k₂, k₀, r₁, r₂)
    M = [
        r₁^n -r₁^n -r₁^(-(n + 1)) 0.0
        n * k₁ * r₁^(n - 1) -n * k₂ * r₁^(n - 1) (n + 1) * k₂ * r₁^(-(n + 2)) 0.0
        0.0 r₂^n r₂^(-(n + 1)) -r₂^(-(n + 1))
        0.0 n * k₂ * r₂^(n - 1) -(n + 1) * k₂ * r₂^(-(n + 2)) (n + 1) * k₀ * r₂^(-(n + 2))
    ]
    return (M \ [0.0, 0.0, r₂^n, n * k₀ * r₂^(n - 1)])[4]
end

@testset "harmonic_response — homogeneous sphere, any degree" begin
    k₀, k₁, R = 2.0, 7.0, 0.8
    K₀, K₁ = TensISO{3}(k₀), TensISO{3}(k₁)
    s = LayeredSphere((R,), (K₁,))
    for n in 1:8
        @test harmonic_response(s, K₀, n) ≈ _homogeneous(n, k₁, k₀, R) rtol = 1.0e-13
    end
    @test @inferred(harmonic_response(s, K₀, 3)) isa Float64
    @test_throws ArgumentError harmonic_response(s, K₀, 0)
end

@testset "harmonic_response — degree 1 is the dipole of the uniform gradient" begin
    # p = -3 V k₀ (k₁ - k₀)/(k₁ + 2k₀) G and T_dis = B r⁻² cosθ with B = B/A · G.
    k₀, k₁, R = 1.5, 0.2, 1.3
    s = LayeredSphere((0.6 * R, R), (TensISO{3}(k₁), TensISO{3}(4.0)))
    K₀ = TensISO{3}(k₀)
    N = property_contribution(s, K₀)
    # contribution tensor per unit volume ↔ dipole: N = -3 k₀ (B/A) / R³
    @test N[1, 1] ≈ -3 * k₀ * harmonic_response(s, K₀, 1) / R^3 rtol = 1.0e-12
end

@testset "harmonic_response — imperfect interfaces" begin
    k₀, k₁, R = 2.0, 7.0, 0.8
    K₀, K₁ = TensISO{3}(k₀), TensISO{3}(k₁)
    ρ, kˢ = 0.35, 1.7
    kap = LayeredSphere((R,), (K₁,); interfaces = (KapitzaInterface(ρ),))
    sur = LayeredSphere((R,), (K₁,); interfaces = (SurfaceConductiveInterface(kˢ),))
    for n in 1:6
        @test harmonic_response(kap, K₀, n) ≈
            _homogeneous(n, k₁ / (1 + n * ρ * k₁ / R), k₀, R) rtol = 1.0e-13
        @test harmonic_response(sur, K₀, n) ≈
            _homogeneous(n, k₁ + (n + 1) * kˢ / R, k₀, R) rtol = 1.0e-13
    end
end

@testset "harmonic_response — coated sphere against the 4×4 system" begin
    k₁, k₂, k₀, r₁, r₂ = 9.0, 0.4, 2.0, 0.5, 0.9
    s = LayeredSphere((r₁, r₂), (TensISO{3}(k₁), TensISO{3}(k₂)))
    for n in 1:6
        @test harmonic_response(s, TensISO{3}(k₀), n) ≈
            _coated_reference(n, k₁, k₂, k₀, r₁, r₂) rtol = 1.0e-12
    end
end

@testset "harmonic_response — impermeable core and shell" begin
    k₀, R = 2.0, 0.8
    K₀ = TensISO{3}(k₀)
    core = LayeredSphere((R,), (TensISO{3}(0.0),))
    # a shielded core changes nothing: the shell alone decides
    shell = LayeredSphere((0.5 * R, R), (TensISO{3}(5.0), TensISO{3}(0.0)))
    for n in 1:5
        @test harmonic_response(core, K₀, n) ≈ _homogeneous(n, 0.0, k₀, R) rtol = 1.0e-13
        @test harmonic_response(shell, K₀, n) ≈ _homogeneous(n, 0.0, k₀, R) rtol = 1.0e-13
    end
end

@testset "harmonic_response — ForwardDiff on contrast, radius and interface" begin
    k₀, k₁, R, ρ, n = 2.0, 7.0, 0.8, 0.35, 3
    f(k) = harmonic_response(LayeredSphere((R,), (TensISO{3}(k),)), TensISO{3}(k₀), n)
    g(r) = harmonic_response(LayeredSphere((r,), (TensISO{3}(k₁),)), TensISO{3}(k₀), n)
    h(x) = harmonic_response(
        LayeredSphere((R,), (TensISO{3}(k₁),); interfaces = (KapitzaInterface(x),)),
        TensISO{3}(k₀), n
    )
    m(k) = harmonic_response(LayeredSphere((R,), (TensISO{3}(k₁),)), TensISO{3}(k), n)
    @test ForwardDiff.derivative(f, k₁) ≈
        ForwardDiff.derivative(k -> _homogeneous(n, k, k₀, R), k₁) rtol = 1.0e-12
    @test ForwardDiff.derivative(g, R) ≈ (2n + 1) * _homogeneous(n, k₁, k₀, R) / R rtol = 1.0e-12
    @test ForwardDiff.derivative(h, ρ) ≈ ForwardDiff.derivative(
        x -> _homogeneous(n, k₁ / (1 + n * x * k₁ / R), k₀, R), ρ
    ) rtol = 1.0e-12
    @test ForwardDiff.derivative(m, k₀) ≈
        ForwardDiff.derivative(k -> _homogeneous(n, k₁, k, R), k₀) rtol = 1.0e-12
    # nested: second derivative with respect to the contrast
    @test ForwardDiff.derivative(k -> ForwardDiff.derivative(f, k), k₁) ≈
        ForwardDiff.derivative(
        k -> ForwardDiff.derivative(x -> _homogeneous(n, x, k₀, R), k), k₁
    ) rtol = 1.0e-11
end
