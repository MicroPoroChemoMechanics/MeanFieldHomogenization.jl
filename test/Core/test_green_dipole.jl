using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

const _MFHC = MeanFieldHomogenization.Core

@testset "Kelvin Green gradient / dipole far field (isotropic)" begin

    C₀ = iso_stiffness(10.0, 6.0)              # k = 10, μ = 6
    E, ν = _MFHC.extract_iso_moduli(C₀)
    μ = E / (2 * (1 + ν))

    Π = [
        2.0 -0.5 0.3
        -0.5 1.0 0.7
        0.3 0.7 -1.5
    ]
    xs = ([1.0, 0.0, 0.0], [0.3, -1.2, 2.0], [-2.5, 0.4, -0.9])

    @testset "closed form == contraction of ∇G" begin
        # `dipole_displacement_iso` collapses ∂G_ij/∂x_k · Π_jk analytically;
        # it must reproduce the explicit 3×3×3 contraction for symmetric Π.
        for x in xs
            G = green_gradient_iso(C₀, x)
            u_full = [sum(G[i, j, k] * Π[j, k] for j in 1:3, k in 1:3) for i in 1:3]
            u = dipole_displacement_iso(C₀, x, Π)
            @test u ≈ u_full rtol = 1.0e-12
        end
    end

    @testset "∇G symmetry in (i,j) and singularity" begin
        for x in xs
            G = green_gradient_iso(C₀, x)
            for k in 1:3, i in 1:3, j in 1:3
                @test G[i, j, k] ≈ G[j, i, k] rtol = 1.0e-12
            end
        end
        @test_throws DomainError green_gradient_iso(C₀, [0.0, 0.0, 0.0])
        @test_throws DomainError dipole_displacement_iso(C₀, [0.0, 0.0, 0.0], Π)
    end

    @testset "1/r² decay and parity" begin
        x = [0.3, -1.2, 2.0]
        u = dipole_displacement_iso(C₀, x, Π)
        for λ in (2.0, 5.0, 17.0)
            @test dipole_displacement_iso(C₀, λ * x, Π) ≈ u / λ^2 rtol = 1.0e-12
        end
        # ∇G is odd in x ⟹ so is the dipole field.
        @test dipole_displacement_iso(C₀, -x, Π) ≈ -u rtol = 1.0e-12
    end

    @testset "hydrostatic Π = p·1 gives a purely radial field" begin
        # A center of dilatation: u = c·n̂/r², with c = -2(1-2ν)p/(16πμ(1-ν)).
        p = 3.0
        for x in xs
            r = norm(x)
            n̂ = x / r
            u = dipole_displacement_iso(C₀, x, p * Matrix(I, 3, 3))
            c = -2 * (1 - 2ν) * p / (16 * π * μ * (1 - ν))
            @test u ≈ (c / r^2) * n̂ rtol = 1.0e-12
        end
    end

    @testset "equilibrium div σ = 0 away from the origin" begin
        # The Green function is the fundamental solution of the Navier
        # equation, so its dipole contraction is a genuine elastic field
        # everywhere but at the source.  Checked through AD on the closed form.
        u_of(x) = dipole_displacement_iso(C₀, x, Π)
        λ = E * ν / ((1 + ν) * (1 - 2ν))
        for x0 in xs
            # div σ_i = (λ + μ) ∂_i(div u) + μ Δu_i
            divu(x) = tr(ForwardDiff.jacobian(u_of, x))
            grad_divu = ForwardDiff.gradient(divu, x0)
            lap = [
                tr(ForwardDiff.hessian(x -> u_of(x)[i], x0))
                    for i in 1:3
            ]
            resid = (λ + μ) * grad_divu + μ * lap
            scale = norm(u_of(x0)) / norm(x0)^2
            @test norm(resid) ≤ 1.0e-8 * max(scale, one(scale))
        end
    end

    @testset "type genericity (ForwardDiff.Dual through the moduli)" begin
        x = [0.3, -1.2, 2.0]
        f(k) = dipole_displacement_iso(iso_stiffness(k, 6.0), x, Π)[1]
        d = ForwardDiff.derivative(f, 10.0)
        fd = (f(10.0 + 1.0e-6) - f(10.0 - 1.0e-6)) / 2.0e-6
        @test d ≈ fd rtol = 1.0e-6
    end
end
