using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using StaticArrays
using Random

# =============================================================================
#  test_green_plane_aniso.jl — Green operator of an anisotropic elastic medium
#  of the plane.
#
#  The operator is a finite part, integrated by parts in the source. The
#  references below do not go through that integration:
#
#   1. The first gradient of the Green function is a principal value,
#        ∂ₗG_ij = −1/(4π²) p.v.∫ K⁻¹_ij ξ_l/(ξ·x) dφ ,
#      written here on its own as a regular integral; its derivative by
#      `ForwardDiff` is the second gradient.
#   2. The Green function is the response to a unit force: the tractions of
#      C : grad(G·a) on a circle around the source have the resultant −a, and
#      C_ijkl ∂ⱼ∂ₗ G_km = 0 away from the source.
#   3. An isotropic stiffness typed as a general tensor comes back with the
#      closed form of the isotropic operator.
#   4. Homogeneity of degree −2, and the operator of a rotated stiffness at a
#      rotated point is the rotated operator.
# =============================================================================

const GPA = MeanFieldHomogenization.Core

"∂ₗG_ij(x) as a principal value on the half circle, by Gauss-Legendre on (0, π/2)."
function gpa_gradient(C, x; nodes = 64)
    r = sqrt(x[1]^2 + x[2]^2)
    n = SVector(x[1] / r, x[2] / r)
    t = SVector(-n[2], n[1])
    ψ, w = GPA.gauss_legendre_nodes(nodes, 0.0, π / 2)
    function q(u)                                  # K⁻¹_ij ξ_l at the angle u from the pole
        ξ = cos(u) * t - sin(u) * n
        K = SMatrix{2, 2}([sum(ξ[a] * C[a, i, j, b] * ξ[b] for a in 1:2, b in 1:2) for i in 1:2, j in 1:2])
        K⁻¹ = inv(K)
        return SArray{Tuple{2, 2, 2}}([K⁻¹[i, j] * ξ[l] for i in 1:2, j in 1:2, l in 1:2])
    end
    # ξ·x = −r sin u: −1/(4π²) · 2 · p.v.∫ q/(−r sin u) du = 1/(2π² r) ∫₀^{π/2} [q(u) − q(−u)]/sin u du
    return sum(w[m] * (q(ψ[m]) - q(-ψ[m])) / sin(ψ[m]) for m in eachindex(ψ)) / (2 * π^2 * r)
end

"∂ₖ∂ₗG_ij(x), the derivative of the principal value above."
gpa_hessian(C, x; nodes = 64) =
    reshape(ForwardDiff.jacobian(y -> vec(gpa_gradient(C, y; nodes)), SVector{2}(x)), Size(2, 2, 2, 2))

gpa_operator(H) = [-(H[i, k, j, l] + H[j, k, i, l] + H[i, l, j, k] + H[j, l, i, k]) / 4 for i in 1:2, j in 1:2, k in 1:2, l in 1:2]

gpa_components(C₀) = SArray{Tuple{2, 2, 2, 2}}(canonical_components(C₀))

"A strongly orthotropic stiffness of the plane, on a frame that is not that of the axes."
gpa_ortho() = TensOrtho(40.0, 6.0, 2.5, 3.0, Basis(0.37))
"A stiffness of the plane without symmetry (positive definite Kelvin-Mandel matrix)."
gpa_general() = from_mandel([30.0 4.0 3.0; 4.0 9.0 -2.0; 3.0 -2.0 7.0])

@testset "Green operator of an anisotropic plane" begin
    rng = Xoshiro(20261127)
    points = ([0.8, -0.3], [-1.1, 0.4], [0.2, 1.7])

    @testset "isotropic stiffness typed as a general tensor" begin
        Ciso = iso_stiffness(Val(2), 5.0, 2.0)
        Cgen = Tens(get_array(Ciso))
        for x in points
            Γ = get_array(green_operator_aniso(Cgen, x))
            @test Γ ≈ get_array(green_operator_iso(Ciso, x)) rtol = 1.0e-12
            @test get_array(green_operator(Cgen, x)) ≈ Γ          # the dispatcher takes this path
        end
    end

    @testset "against the derivative of the principal value" begin
        for C₀ in (gpa_ortho(), gpa_general(), cubic_from_constants(20.0, 12.0, 9.0, Basis(1.1)))
            C = gpa_components(C₀)
            for x in points
                H = GPA._plane_green_hessian(C, x, 64)
                Href = gpa_hessian(C, x)
                @test H ≈ Href rtol = 1.0e-10
                @test get_array(green_operator_aniso(C₀, x; nodes = 64)) ≈ gpa_operator(Href) rtol = 1.0e-10
            end
        end
    end

    @testset "unit force: equilibrium and resultant" begin
        for C₀ in (gpa_ortho(), gpa_general())
            C = gpa_components(C₀)
            for x in points
                H = GPA._plane_green_hessian(C, x, 64)
                # C_ijkl ∂ⱼ∂ₗ G_km = 0 away from the source
                div = [sum(C[i, j, k, l] * H[k, m, j, l] for j in 1:2, k in 1:2, l in 1:2) for i in 1:2, m in 1:2]
                @test norm(div) < 1.0e-10 * norm(C) * norm(H)
            end
            # The tractions on a circle of radius ρ balance the force: resultant −δ_im.
            ρ, N = 0.7, 200
            R = zeros(2, 2)
            for a in 1:N
                φ = 2π * (a - 0.5) / N
                ν = [cos(φ), sin(φ)]
                dG = gpa_gradient(C, ρ * ν)
                R += [sum(C[i, j, k, l] * dG[k, m, l] * ν[j] for j in 1:2, k in 1:2, l in 1:2) for i in 1:2, m in 1:2] * (2π * ρ / N)
            end
            @test R ≈ -I atol = 1.0e-10
        end
    end

    @testset "symmetries, homogeneity, rotation" begin
        C₀ = gpa_ortho()
        x = [0.6, 0.9]
        Γ = get_array(green_operator_aniso(C₀, x))
        @test Γ ≈ permutedims(Γ, (2, 1, 3, 4)) && Γ ≈ permutedims(Γ, (1, 2, 4, 3))
        @test Γ ≈ permutedims(Γ, (3, 4, 1, 2)) rtol = 1.0e-10             # major symmetry of a stiffness
        @test get_array(green_operator_aniso(C₀, 3.0 .* x)) ≈ Γ ./ 9
        @test get_array(green_operator_aniso(C₀, -x)) ≈ Γ
        # The stiffness and the point turned together: the operator turned.
        θ = 0.83
        R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
        turn(A) = [sum(R[i, a] * R[j, b] * R[k, c] * R[l, d] * A[a, b, c, d] for a in 1:2, b in 1:2, c in 1:2, d in 1:2) for i in 1:2, j in 1:2, k in 1:2, l in 1:2]
        Cturned = Tens(turn(Array(get_array(Tens(get_array(C₀))))))
        @test get_array(green_operator_aniso(Cturned, R * x)) ≈ turn(Γ) rtol = 1.0e-10
        # A tensor written in a rotated basis is the same tensor.
        @test get_array(green_operator_aniso(express(Tens(get_array(C₀)), Basis(0.5)), x)) ≈ Γ rtol = 1.0e-12
    end

    @testset "quadrature, element types, derivatives" begin
        C₀, x = gpa_ortho(), [0.8, -0.3]
        Γref = get_array(green_operator_aniso(C₀, x; nodes = 96))
        errors = [norm(get_array(green_operator_aniso(C₀, x; nodes)) - Γref) / norm(Γref) for nodes in (8, 16, 32, 48)]
        @test issorted(errors; rev = true) && errors[3] < 1.0e-8 && errors[4] < 1.0e-12
        @test eltype(green_operator_aniso(C₀, x)) === Float64
        @test (@inferred GPA._green_operator_aniso(C₀, SVector{2}(x), 32)) isa SArray{Tuple{2, 2, 2, 2}, Float64}
        @test_throws DomainError green_operator_aniso(C₀, [0.0, 0.0])
        # With respect to a modulus: the derivative of the reference, through its own path.
        Γ₁₁(c) = get_array(green_operator_aniso(TensOrtho(c, 6.0, 2.5, 3.0, Basis(0.37)), x; nodes = 64))[1, 1, 1, 1]
        ref₁₁(c) = gpa_operator(gpa_hessian(gpa_components(TensOrtho(c, 6.0, 2.5, 3.0, Basis(0.37))), x))[1, 1, 1, 1]
        @test ForwardDiff.derivative(Γ₁₁, 40.0) ≈ ForwardDiff.derivative(ref₁₁, 40.0) rtol = 1.0e-8
        # With respect to the point: homogeneity of degree −2, x·grad Γ = −2Γ (Euler).
        J = ForwardDiff.jacobian(y -> vec(GPA._green_operator_aniso(C₀, y, 48)), SVector{2}(x))
        @test J * x ≈ -2 .* vec(get_array(green_operator_aniso(C₀, x; nodes = 48))) rtol = 1.0e-9
    end

    @testset "interaction tensor in an anisotropic plane" begin
        # Two discs far apart: the interaction tensor tends to the Green operator
        # times the area of the source, whatever the reference.
        C₀ = gpa_ortho()
        a, d = 0.05, [3.0, 1.0]
        T = get_array(interaction_tensor(Ellipsoid(a, a), Ellipsoid(a, a), d, C₀))
        @test T ≈ π * a^2 .* get_array(green_operator_aniso(C₀, d)) rtol = 1.0e-2
    end
end
