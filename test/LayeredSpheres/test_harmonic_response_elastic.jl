using Test
using MeanFieldHomogenization
using TensND
using ForwardDiff
using LinearAlgebra

# =============================================================================
#  Elastic response of a layered sphere to a harmonic of any degree.
#
#  Reference independent of the transfer matrices: the interface conditions
#  themselves, checked point by point on the sphere. The field of the matrix is
#  rebuilt from the Cartesian form of the Lamé solutions and the response under
#  test; the field of the inclusion is left free (three amplitudes); stresses
#  come from ForwardDiff. Several points give more equations than unknowns, so
#  a residual at rounding level means the response is the right one.
# =============================================================================

_hre_jac(f, x) = ForwardDiff.jacobian(v -> real.(f(v)), x) + im * ForwardDiff.jacobian(v -> imag.(f(v)), x)
function _hre_traction(f, x, μ, ν)
    J = _hre_jac(f, x)
    n = x / norm(x)
    return 2 * μ * ((J + transpose(J)) / 2 * n + ν / (1 - 2ν) * tr(J) * n)
end

# Residual of the interface conditions of a homogeneous sphere of radius R
# under the regular solution (j, t, s): compliances (sn, st) of a spring
# interface, zero for a perfect one.
function _hre_residual(B, j, t, s, R, μ₀, ν₀, μ₁, ν₁; sn = 0.0, st = 0.0)
    N² = harmonic_norm(t, s)^2
    families = t == 0 ? (3,) : (1, 2, 3)
    sent = t == 0 ? (1,) : (1, 2, 3)
    outer(x) = lame_regular(j, t, s, x, ν₀) + sum(B[i, j] / N² * lame_irregular(i, t, s, x, ν₀) for i in sent)
    directions = ([0.36, 0.48, 0.8], [-0.6, 0.0, 0.8], [0.0, -1.0, 0.0], [0.8, 0.36, -0.48], [-0.48, -0.6, -0.64])
    rows, rhs = Vector{ComplexF64}[], ComplexF64[]
    for d in directions
        x = R * d
        τ₀ = _hre_traction(outer, x, μ₀, ν₀)
        inner = [lame_regular(i, t, s, x, ν₁) for i in families]
        τ₁ = [_hre_traction(v -> lame_regular(i, t, s, v, ν₁), x, μ₁, ν₁) for i in families]
        # the jump of displacement a traction τ gives across the spring
        jump(τ) = sn * dot(d, τ) * d + st * (τ - dot(d, τ) * d)
        for c in 1:3
            push!(rows, [τ₁[i][c] for i in eachindex(families)]); push!(rhs, τ₀[c])
            push!(rows, [inner[i][c] + jump(τ₁[i])[c] for i in eachindex(families)]); push!(rhs, outer(x)[c])
        end
    end
    A = permutedims(reduce(hcat, rows))
    d = A \ rhs
    return norm(A * d - rhs) / norm(rhs)
end

const _HRE_C(μ, ν) = TensISO{3}(2μ * (1 + ν) / (1 - 2ν), 2μ)        # (3k, 2μ)

@testset "elastic harmonic response — interface conditions, perfect interface" begin
    R, μ₀, ν₀, μ₁, ν₁ = 0.8, 1.0, 0.3, 7.0, 0.2
    sphere = LayeredSphere((R,), (_HRE_C(μ₁, ν₁),))
    @test _hre_residual(harmonic_response(sphere, _HRE_C(μ₀, ν₀), 0), 3, 0, 0, R, μ₀, ν₀, μ₁, ν₁) < 1.0e-10
    for (t, s) in ((1, 0), (2, 1), (3, -2), (5, 3)), j in 1:3
        B = harmonic_response(sphere, _HRE_C(μ₀, ν₀), t)
        @test _hre_residual(B, j, t, s, R, μ₀, ν₀, μ₁, ν₁) < 1.0e-9
    end
    # rigid motions send nothing back
    B₁ = harmonic_response(sphere, _HRE_C(μ₀, ν₀), 1)
    @test norm(B₁[:, 1:2]) < 1.0e-12 * norm(B₁)
    @test_throws ArgumentError harmonic_response(sphere, _HRE_C(μ₀, ν₀), -1)
end

@testset "elastic harmonic response — spring interface" begin
    R, μ₀, ν₀, μ₁, ν₁ = 0.8, 1.0, 0.3, 7.0, 0.2
    sn, st = 0.15, 0.4
    sphere = LayeredSphere((R,), (_HRE_C(μ₁, ν₁),); interfaces = (SpringInterface(; sn, st),))
    @test _hre_residual(harmonic_response(sphere, _HRE_C(μ₀, ν₀), 0), 3, 0, 0, R, μ₀, ν₀, μ₁, ν₁; sn, st) < 1.0e-10
    for (t, s) in ((1, 1), (2, 0), (4, -3)), j in 1:3
        B = harmonic_response(sphere, _HRE_C(μ₀, ν₀), t)
        @test _hre_residual(B, j, t, s, R, μ₀, ν₀, μ₁, ν₁; sn, st) < 1.0e-9
    end
end

@testset "elastic harmonic response — layers, limits and derivatives" begin
    R, μ₀, ν₀ = 0.8, 1.0, 0.3
    C₀ = _HRE_C(μ₀, ν₀)
    # a coating of the core material, or of the matrix material up to a larger
    # radius, changes nothing; nor does a sphere of the matrix send anything
    plain = LayeredSphere((R,), (_HRE_C(7.0, 0.2),))
    split = LayeredSphere((0.5, R), (_HRE_C(7.0, 0.2), _HRE_C(7.0, 0.2)))
    padded = LayeredSphere((R, 1.3), (_HRE_C(7.0, 0.2), C₀))
    for t in (0, 1, 2, 4)
        B = harmonic_response(plain, C₀, t)
        @test harmonic_response(split, C₀, t) ≈ B rtol = 1.0e-10
        @test harmonic_response(padded, C₀, t) ≈ B rtol = 1.0e-10
        @test norm(harmonic_response(LayeredSphere((R,), (C₀,)), C₀, t)) < 1.0e-12
    end
    # degree 0: the center of dilatation of a sphere (kushch2013, Eq. (4.59))
    k₀, k₁ = 2μ₀ * (1 + ν₀) / (3 * (1 - 2ν₀)), 2 * 7.0 * (1 + 0.2) / (3 * (1 - 2 * 0.2))
    γ₀ = (4ν₀ - 2) / 3
    @test harmonic_response(plain, C₀, 0)[1, 3] ≈ γ₀ * R^3 * (3k₁ - 3k₀) / (4μ₀ + 3k₁) rtol = 1.0e-12
    # a void and a rigid sphere are the two limits of the contrast
    void = harmonic_response(LayeredSphere((R,), (TensISO{3}(0.0, 0.0),)), C₀, 2)
    soft = harmonic_response(LayeredSphere((R,), (_HRE_C(1.0e-9, 0.2),)), C₀, 2)
    @test void ≈ soft rtol = 1.0e-7
    # an incompressible sphere
    stiff = harmonic_response(LayeredSphere((R,), (TensISO{3}(Inf, 2 * 7.0),)), C₀, 2)
    @test stiff ≈ harmonic_response(LayeredSphere((R,), (_HRE_C(7.0, 0.5 - 1.0e-10),)), C₀, 2) rtol = 1.0e-6
    # derivative with respect to the shear modulus of the sphere and to its radius
    f(μ) = harmonic_response(LayeredSphere((R,), (_HRE_C(μ, 0.2),)), C₀, 3)[3, 1]
    g(r) = harmonic_response(LayeredSphere((r,), (_HRE_C(7.0, 0.2),)), C₀, 3)[3, 1]
    h = 1.0e-5
    @test ForwardDiff.derivative(f, 7.0) ≈ (f(7.0 + h) - f(7.0 - h)) / 2h rtol = 1.0e-7
    @test ForwardDiff.derivative(g, R) ≈ (g(R + h) - g(R - h)) / 2h rtol = 1.0e-7
end
