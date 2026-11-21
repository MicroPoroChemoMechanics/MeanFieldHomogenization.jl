# =============================================================================
#  test_graded_layers.jl — layers that follow a radial profile.
#
#  References, independent of the layered sphere:
#   1. a shell whose moduli follow a power of the radius between a core and the
#      matrix: the radial equation is of Euler type, with two powers of r as
#      solutions, in elasticity (hydrostatic loading) and in conduction;
#   2. the zone of Lutz & Zimmerman (2005), whose moduli tend to those of the
#      matrix as (a/r)^b: a Frobenius series in a/r.
#  The response of the layers tends to them as the square of their thickness.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

# Bulk part n_K of the stiffness contribution, and the conductivity one.
_gl_bulk(N) = (c = canonical_components(N); sum(c[i, i, j, j] for i in 1:3, j in 1:3) / 9)
_gl_cond(N) = canonical_components(N)[1, 1]

# Core (K_c) of radius 1, shell 1 < r < R with λ = λ_s r^β and μ = μ_s r^β, matrix
# (K_m, μ_m): u = A r, B₁ r^p₁ + B₂ r^p₂, r + g/r²; σ_rr = (λ + 2μ) u′ + 2λ u/r.
# Returns n_K = -(g/R³)(3K_m + 4μ_m)/3 per unit volume of the sphere of radius R.
function _gl_power_shell_bulk(K_c, λ_s, μ_s, β, R, K_m, μ_m)
    M = λ_s + 2μ_s
    Δ = sqrt((1 + β)^2 + 4 * (2 - 2β * λ_s / M))
    p = ((-(1 + β) + Δ) / 2, (-(1 + β) - Δ) / 2)
    σ(q, r) = r^β * (M * q + 2λ_s) * r^(q - 1)
    S = [
        1 -1 -1 0
        3K_c -σ(p[1], 1) -σ(p[2], 1) 0
        0 R^p[1] R^p[2] -1 / R^2
        0 σ(p[1], R) σ(p[2], R) 4μ_m / R^3
    ]
    g = (S \ [0, 0, R, 3K_m])[4]
    return -(g / R^3) * (3K_m + 4μ_m) / 3
end

# Conduction: T = F(r) cos θ with F = A r, B₁ r^p₁ + B₂ r^p₂, r + g/r²; the flux
# is k F′. Returns n_k = -3 k_m g/R³.
function _gl_power_shell_cond(k_c, k_s, β, R, k_m)
    Δ = sqrt((1 + β)^2 + 8)
    p = ((-(1 + β) + Δ) / 2, (-(1 + β) - Δ) / 2)
    q(e, r) = k_s * r^β * e * r^(e - 1)
    S = [
        1 -1 -1 0
        k_c -q(p[1], 1) -q(p[2], 1) 0
        0 R^p[1] R^p[2] -1 / R^2
        0 q(p[1], R) q(p[2], R) 2k_m / R^3
    ]
    g = (S \ [0, 0, R, k_m])[4]
    return -3k_m * g / R^3
end

# Lutz & Zimmerman (2005): λ(r) = λ_m + (λ_if - λ_m)(a/r)^b and μ likewise
# outside a core of radius a. With s = a/r, u = A₁ r Σ C_{jb} s^{jb} + A₂ r Σ
# C_{jb+3} s^{jb+3}; the equation of equilibrium gives
#   C_{n+b} = -{ΔM [n² + (b-3) n - b] - 2b Δλ} / ((n+b)(n+b-3) M_m) C_n .
# Returns f = -A₂/A₁, for which u ≃ A₁ (r - f a³/r²) far from the core.
function _gl_lutz_zimmerman(K_i, λ_if, μ_if, λ_m, μ_m, b; terms = 400)
    M_m, M_if = λ_m + 2μ_m, λ_if + 2μ_if
    ΔM, Δλ = M_if - M_m, λ_if - λ_m
    ratio(n) = -(ΔM * (n^2 + (b - 3) * n - b) - 2b * Δλ) / ((n + b) * (n + b - 3) * M_m)
    S₁ = S₁′ = S₂ = S₂′ = zero(ratio(0))
    c₁ = c₂ = one(ratio(0))
    for j in 0:terms                        # a recurrence on the coefficients
        S₁ += c₁; S₁′ += j * b * c₁; S₂ += c₂; S₂′ += (j * b + 3) * c₂
        c₁ *= ratio(j * b); c₂ *= ratio(j * b + 3)
    end
    K_if = λ_if + 2μ_if / 3
    return (3 * (K_i - K_if) * S₁ + M_if * S₁′) / (3 * (K_i - K_if) * S₂ + M_if * S₂′)
end

@testset "graded_layers" begin

    @testset "the layers" begin
        profile(r) = TensISO{3}(2 - 1 / r)
        radii, moduli = graded_layers(profile, 1.0, 2.0, 4)
        @test radii == (1.25, 1.5, 1.75, 2.0)
        @test moduli == map(profile, (1.125, 1.375, 1.625, 1.875))
        # the radii given, as a tuple or as a vector
        @test graded_layers(profile, (1.0, 1.25, 1.5, 1.75, 2.0)) == (radii, moduli)
        @test graded_layers(profile, [1.0, 1.25, 1.5, 1.75, 2.0]) == (radii, moduli)
        @test graded_layers(profile, (1.0, 1.1, 2.0))[1] == (1.1, 2.0)
        @inferred graded_layers(profile, (1.0, 1.5, 2.0))
        @inferred graded_layers(profile, (ForwardDiff.Dual(1.0, 1.0), 1.5, 2.0))
        # integers, and a single layer
        @test graded_layers(profile, 1, 2, 1) == ((2.0,), (profile(1.5),))
        @test_throws ArgumentError graded_layers(profile, 1.0, 2.0, 0)
        @test_throws ArgumentError graded_layers(profile, (1.0,))
        @test_throws ArgumentError graded_layers(profile, ())
        # in a sphere and in a cylinder
        sphere = LayeredSphere((1.0, radii...), (TensISO{3}(5.0), moduli...))
        @test layer_count(sphere) == 5
        @test layer_modulus(sphere, 3) == profile(1.375)
        @test LayeredCylinder((1.0, radii...), (TensISO{3}(5.0), moduli...)) isa LayeredCylinder
    end

    @testset "a shell that follows a power of the radius" begin
        K_c, μ_c, K_m, μ_m, R = 6.0, 3.0, 1.5, 0.8, 1.6
        λ_s, μ_s, β = 0.9, 0.5, 2.5
        C₀ = iso_stiffness(K_m, μ_m)
        shell(r) = iso_stiffness((λ_s + 2μ_s / 3) * r^β, μ_s * r^β)
        function layered(n)
            radii, moduli = graded_layers(shell, 1.0, R, n)
            return _gl_bulk(property_contribution(LayeredSphere((1.0, radii...), (iso_stiffness(K_c, μ_c), moduli...)), C₀, C₀))
        end
        exact = _gl_power_shell_bulk(K_c, λ_s, μ_s, β, R, K_m, μ_m)
        e = [layered(n) - exact for n in (10, 20, 40)]
        @test abs(e[3]) < 2.0e-4 * K_m
        @test 3.7 < e[1] / e[2] < 4.3           # the square of the thickness
        @test 3.7 < e[2] / e[3] < 4.3
        @test (4 * layered(40) - layered(20)) / 3 ≈ exact atol = 1.0e-6 * K_m

        # conduction
        k_c, k_s, k_m = 8.0, 0.4, 1.3
        k₀ = TensISO{3}(k_m)
        function conducting(n, β = β, R = R)
            radii, moduli = graded_layers(r -> TensISO{3}(k_s * r^β), 1.0, R, n)
            return _gl_cond(property_contribution(LayeredSphere((1.0, radii...), (TensISO{3}(k_c), moduli...)), k₀, k₀))
        end
        exact = _gl_power_shell_cond(k_c, k_s, β, R, k_m)
        e = [conducting(n) - exact for n in (10, 20, 40)]
        @test abs(e[3]) < 2.0e-4 * k_m
        @test 3.7 < e[1] / e[2] < 4.3
        @test (4 * conducting(40) - conducting(20)) / 3 ≈ exact atol = 1.0e-6 * k_m

        # derivatives with respect to the exponent of the profile and to the
        # outer radius, the two discretizations extrapolated as the value is
        extrapolated(f) = (4 * f(40) - f(20)) / 3
        dβ = extrapolated(n -> ForwardDiff.derivative(x -> conducting(n, x), β))
        @test dβ ≈ ForwardDiff.derivative(x -> _gl_power_shell_cond(k_c, k_s, x, R, k_m), β) atol = 1.0e-5 * k_m
        dR = extrapolated(n -> ForwardDiff.derivative(x -> conducting(n, β, x), R))
        @test dR ≈ ForwardDiff.derivative(x -> _gl_power_shell_cond(k_c, k_s, β, x, k_m), R) atol = 1.0e-5 * k_m
    end

    @testset "the zone of Lutz & Zimmerman (2005)" begin
        # the core five times stiffer than the matrix, ν = 1/4 everywhere, b = 10
        K_m, ν, b = 1.0, 0.25, 10
        μ_m = 3K_m * (1 - 2ν) / (2 * (1 + ν))
        K_i, μ_i = 5K_m, 5μ_m
        C₀ = iso_stiffness(K_m, μ_m)
        # the profile is cut where it differs from the matrix by 5⁻¹⁰, and the
        # layers are of equal steps in 1/r, thinner where the profile is steep
        r_out = 5.0
        edges(n) = 1 ./ range(1, 1 / r_out; length = n + 1)
        for D in (0.5, -0.5)                  # zone softer, or stiffer, than the matrix
            K_if, μ_if = (1 - D) * K_m, (1 - D) * μ_m
            zone(r) = iso_stiffness(K_m + (K_if - K_m) / r^b, μ_m + (μ_if - μ_m) / r^b)
            function f(n)
                radii, moduli = graded_layers(zone, edges(n))
                n_K = _gl_bulk(property_contribution(LayeredSphere((1.0, radii...), (iso_stiffness(K_i, μ_i), moduli...)), C₀, C₀))
                return 3 * n_K * r_out^3 / (3K_m + 4μ_m)
            end
            exact = _gl_lutz_zimmerman(K_i, K_if - 2μ_if / 3, μ_if, K_m - 2μ_m / 3, μ_m, b)
            @test (4 * f(120) - f(60)) / 3 ≈ exact rtol = 2.0e-5
            # conduction, by the correspondence of the article: λ → 0, 2μ and 3K → k
            k₀ = TensISO{3}(1.0)
            function fk(n)
                radii, moduli = graded_layers(r -> TensISO{3}(1 - D / r^b), edges(n))
                return _gl_cond(property_contribution(LayeredSphere((1.0, radii...), (TensISO{3}(5.0), moduli...)), k₀, k₀)) * r_out^3 / 3
            end
            exact = _gl_lutz_zimmerman(5 / 3, 0.0, (1 - D) / 2, 0.0, 1 / 2, b)
            @test (4 * fk(120) - fk(60)) / 3 ≈ exact rtol = 2.0e-5
        end
        # a zone identical to the matrix: the sphere alone
        @test _gl_lutz_zimmerman(K_i, K_m - 2μ_m / 3, μ_m, K_m - 2μ_m / 3, μ_m, b) ≈ 3 * (K_i - K_m) / (3K_i + 4μ_m)
    end
end
