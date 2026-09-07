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

# ─── The transport counterpart ───────────────────────────────────────────────

@testset "conduction Green gradient and dipole" begin
    k₀ = 2.5
    K₀ = TensND.TensISO{3}(k₀)
    pts = ([0.3, -0.7, 1.1], [1.0, 0.0, 0.0], [-2.0, 3.0, -1.5], [0.01, 0.02, -0.03])

    @testset "against the closed form" begin
        @test extract_iso_conductivity(K₀) == k₀
        for x in pts
            r = norm(x)
            @test collect(green_gradient_iso2(K₀, x)) ≈ -x ./ (4π * k₀ * r^3)
            # A bare scalar is accepted too, and agrees.
            @test green_gradient_iso2(k₀, x) ≈ green_gradient_iso2(K₀, x)
        end
        @test_throws DomainError green_gradient_iso2(K₀, [0.0, 0.0, 0.0])
        @test_throws DomainError dipole_temperature_iso(K₀, [0.0, 0.0, 0.0], [1.0, 0, 0])
    end

    @testset "the dipole field" begin
        M = [1.0, 2.0, -0.5]
        for x in pts
            r = norm(x)
            @test dipole_temperature_iso(K₀, x, M) ≈ -dot(M, x) / (4π * k₀ * r^3)
            # It is the gradient contracted with the moment, by definition.
            @test dipole_temperature_iso(K₀, x, M) ≈ dot(green_gradient_iso2(K₀, x), M)
        end
        # Linear in the moment, and inverse in the conductivity.
        x = pts[1]
        @test dipole_temperature_iso(K₀, x, 3 .* M) ≈ 3 * dipole_temperature_iso(K₀, x, M)
        @test dipole_temperature_iso(TensND.TensISO{3}(2k₀), x, M) ≈
            dipole_temperature_iso(K₀, x, M) / 2
        # And it decays as 1/r², which is what makes it the leading correction
        # to a truncated cell.
        @test dipole_temperature_iso(K₀, 2 .* x, M) / dipole_temperature_iso(K₀, x, M) ≈ 1 / 4
        @test dipole_temperature_iso(K₀, 10 .* x, M) / dipole_temperature_iso(K₀, x, M) ≈
            1 / 100
    end

    @testset "it is harmonic away from the source" begin
        # A free oracle: the dipole field must satisfy Laplace's equation, and
        # nothing in its derivation was asked to enforce that.
        M = [0.7, -1.3, 0.4]
        for x in pts[1:3]
            H = ForwardDiff.hessian(y -> dipole_temperature_iso(K₀, y, M), x)
            @test abs(LinearAlgebra.tr(H)) < 1.0e-8 * norm(H)
        end
    end

    @testset "differentiation" begin
        M = [1.0, 2.0, -0.5]
        x = pts[1]
        # In the conductivity: T ∝ 1/k₀, so ∂T/∂k₀ = -T/k₀.
        f = k -> dipole_temperature_iso(k, x, M)
        @test ForwardDiff.derivative(f, k₀) ≈ -f(k₀) / k₀
        # In position: the gradient of the dipole temperature is finite and the
        # field is homogeneous of degree -2, so x·∇T = -2T (Euler).
        g = ForwardDiff.gradient(y -> dipole_temperature_iso(K₀, y, M), x)
        @test dot(x, g) ≈ -2 * dipole_temperature_iso(K₀, x, M)
    end
end
