# =============================================================================
#  test_toroidal_inclusions.jl — the insulating torus.
#
#  References are independent of the dipole from which the code reads its
#  tensor:
#   1. the thin ring, a wire across the gradient (2) or half along it (3/2);
#   2. the field rebuilt from the amplitudes: it is the remote field far away,
#      and its normal derivative vanishes on the torus;
#   3. the mean gradient over the pore as an integral of the temperature over
#      its surface;
#   4. the amplitudes of Radi & Sevostianov (2016), a closed form along the axis
#      and an elimination without pivoting across it, and the closed form of
#      the surface integral;
#   5. the same torus on a grid of Gaussians;
#   6. the homogeneity of degree 0 in (R, r) and extended precision for AD.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

const TI = MeanFieldHomogenization.ToroidalInclusions

_tor_iso(x) = TensISO{3}(x)
_tor_A(R, r) = TI._insulating_torus_concentration(acosh(R / r))
_tor_trapz(f, n = 2000) = (2π / n) * sum(f(-π + 2π * k / n) for k in 0:(n - 1))

# The series around the torus α₀ (focal radius 1) for a unit remote gradient
# along the axis (`Val(:sine)`) or across it (`Val(:cosine)`, the factor cos ϕ
# left out): the remote part, on the functions of the second kind, and the
# perturbation, on those of the first kind, as two functions of (α, β).
function _tor_series(parity::Val, α₀)
    N = TI._series_length(α₀)
    c = 4 * sqrt(2) / π
    m, s = parity isa Val{:sine} ? (0, [c * n for n in 0:(N + 1)]) : (1, [-(n == 0 ? c / 2 : c) for n in 0:(N + 1)])
    b = TI._neumann_amplitudes(parity, m, s, α₀)
    n₀ = TI._first_index(parity)
    wave(n, β) = parity isa Val{:sine} ? sin(n * β) : cos(n * β)
    factor(α, β) = sqrt(cosh(α) - cos(β))
    remote(α, β) = factor(α, β) * sum(s[n + 1] * toroidal_Q(N, m, α)[n + 1] * wave(n, β) for n in n₀:N)
    perturbation(α, β) = factor(α, β) * sum(b[n - n₀ + 1] * toroidal_P(N, m, α)[n + 1] * wave(n, β) for n in n₀:N)
    return remote, perturbation
end

# The temperature itself, the remote part in closed form: x₃ = sin β/δ along
# the axis, ϱ = sinh α/δ across it.
function _tor_field(parity::Val, α₀)
    _, perturbation = _tor_series(parity, α₀)
    remote(α, β) = (parity isa Val{:sine} ? sin(β) : sinh(α)) / (cosh(α) - cos(β))
    return (α, β) -> remote(α, β) + perturbation(α, β)
end

@testset "Toroidal inclusions" begin

    @testset "the thin ring" begin
        for λ in (1.0e-2, 1.0e-3, 1.0e-4)
            Aₜ, Aₙ = _tor_A(1.0, λ)
            @test abs(Aₜ - 3 / 2) < 5 * λ^2
            @test abs(Aₙ - 2) < 5 * λ^2
        end
        # the size does not matter
        @test collect(_tor_A(3.0, 0.6)) ≈ collect(_tor_A(1.0, 0.2)) rtol = 1.0e-13
    end

    @testset "the field: remote part and condition on the torus" begin
        for α₀ in (0.4, 1.3, 2.5)
            Tz, Tx = _tor_field(Val(:sine), α₀), _tor_field(Val(:cosine), α₀)
            # the series of the remote part is x₃, or x₁ = ϱ cos ϕ; it converges
            # as e^{-nα}, slower than the perturbation
            z, _ = _tor_series(Val(:sine), α₀)
            x, _ = _tor_series(Val(:cosine), α₀)
            for (α, β) in ((α₀, 0.7), (α₀ + 0.5, -2.1), (α₀ + 2, 3.0))
                δ = cosh(α) - cos(β)
                @test z(α, β) ≈ sin(β) / δ rtol = 1.0e-5 atol = 1.0e-7
                @test x(α, β) ≈ sinh(α) / δ rtol = 1.0e-5
            end
            # the perturbation vanishes far from the torus
            @test abs(Tz(1.0e-3, 2.0e-3) - sin(2.0e-3) / (cosh(1.0e-3) - cos(2.0e-3))) < 1.0e-4
            # no flux through the torus. The series is cut where the amplitudes
            # of the perturbation, which decay as e^{-2nα₀}, reach the rounding;
            # the flux of the remote terms left out decays as e^{-nα₀} only.
            for β in (-2.6, -0.7, 0.3, 1.9, 3.0)
                scale = abs(Tz(α₀, β)) + abs(Tx(α₀, β)) + 1
                @test abs(ForwardDiff.derivative(α -> Tz(α, β), α₀)) < 1.0e-6 * scale
                @test abs(ForwardDiff.derivative(α -> Tx(α, β), α₀)) < 1.0e-6 * scale
            end
        end
    end

    @testset "the tensor as a surface integral of the temperature" begin
        # ⟨grad T⟩ over the pore is (1/V) ∮ T n dS, n = -e^α outward and
        # dS = sinh α₀/δ² dϕ dβ for a focal radius 1.
        for α₀ in (0.4, 1.3, 2.5)
            sh, ch = sinh(α₀), cosh(α₀)
            V = 2π^2 * ch / sh^3
            Tz, Tx = _tor_field(Val(:sine), α₀), _tor_field(Val(:cosine), α₀)
            δ(β) = ch - cos(β)
            Aₙ = 2π / V * _tor_trapz(β -> Tz(α₀, β) * (sh * sin(β) / δ(β)) * sh / δ(β)^2)
            Aₜ = π / V * _tor_trapz(β -> Tx(α₀, β) * (-(1 - ch * cos(β)) / δ(β)) * sh / δ(β)^2)
            A = TI._insulating_torus_concentration(α₀)
            @test A[2] ≈ Aₙ rtol = 1.0e-9
            @test A[1] ≈ Aₜ rtol = 1.0e-9
        end
    end

    @testset "the published amplitudes and the surface integral in closed form" begin
        # Radi & Sevostianov (2016) write the perturbation with the amplitudes
        # Bₙ = -bₙ/c (along the axis) and εₙCₙ = -2bₙ/c (across it), c = 4√2/π.
        c = 4 * sqrt(2) / π
        for λ in (0.05, 0.2, 0.5, 0.8, 0.95)
            α₀ = acosh(1 / λ)
            sh, ch = sinh(α₀), cosh(α₀)
            N = TI._series_length(α₀)
            Aₜ, Aₙ = TI._insulating_torus_concentration(α₀)

            # along the axis: their closed form (3.17)-(3.18)
            P, dP, dQ = toroidal_P(N + 1, 0, α₀), toroidal_dP(N + 1, 0, α₀), toroidal_dQ(N + 1, 0, α₀)
            ρ = dQ ./ dP
            ξ = sum(m^2 / (4m^2 - 1) * ρ[m + 1] for m in 1:(N + 1)) / (ρ[1] / 2 - sum(ρ[m + 1] / (4m^2 - 1) for m in 1:(N + 1)))
            B = [(ξ + n^2) / (2n - 1) * ρ[n + 1] - 2 * sum((ξ + m^2) / (4m^2 - 1) * ρ[m + 1] for m in n:(N + 1)) for n in 1:N]
            b = TI._neumann_amplitudes(Val(:sine), 0, [c * n for n in 0:(N + 1)], α₀)
            @test -b / c ≈ B atol = 1.0e-12
            # the integral over the torus, term by term
            @test Aₙ ≈ 1 - 8 * sqrt(2) / (3π) * sh^4 / ch * sum(n * b[n] * P[n + 1] * dQ[n + 1] for n in 1:N) rtol = 1.0e-11

            # across the axis: their elimination (4.9)-(4.11), Cₙ - aₙCₙ₊₁ - cₙCₙ₋₁ = -dₙ
            P¹, dP¹ = toroidal_P(N + 1, 1, α₀), toroidal_dP(N + 1, 1, α₀)
            Q¹, dQ¹ = toroidal_Q(N + 1, 1, α₀), toroidal_dQ(N + 1, 1, α₀)
            D = [P¹[n + 1] * sh + 2 * dP¹[n + 1] * ch for n in 0:N]
            a = [(n == 0 ? 2 : 1) * dP¹[n + 2] / D[n + 1] for n in 0:N]
            l = [n == 0 ? 0.0 : dP¹[n] / D[n + 1] for n in 0:N]
            d = [(Q¹[n + 1] * sh + 2 * dQ¹[n + 1] * ch - (n == 0 ? 2 * dQ¹[2] : dQ¹[n + 2] + dQ¹[n])) / D[n + 1] for n in 0:N]
            β, Y, C = zeros(N + 1), zeros(N + 1), zeros(N + 1)
            Y[1] = d[1]
            for n in 1:N
                β[n + 1] = l[n + 1] * a[n] / (1 - β[n])
                Y[n + 1] = (d[n + 1] + l[n + 1] * Y[n]) / (1 - β[n + 1])
            end
            C[N + 1] = -Y[N + 1]
            for n in (N - 1):-1:0
                C[n + 1] = -Y[n + 1] + a[n + 1] / (1 - β[n + 1]) * C[n + 2]
            end
            b¹ = TI._neumann_amplitudes(Val(:cosine), 1, [-(n == 0 ? c / 2 : c) for n in 0:(N + 1)], α₀)
            @test -b¹ / c ≈ [n == 0 ? C[1] / 2 : C[n + 1] for n in 0:N] atol = 1.0e-11
            @test Aₜ ≈ 1 + 2 * sqrt(2) / (3π) * sh^4 / ch * sum(b¹[n + 1] * P¹[n + 1] * (2 * dQ¹[n + 1] + ch / sh * Q¹[n + 1]) for n in 0:N) rtol = 1.0e-11
        end
    end

    @testset "truncation and number types" begin
        for α₀ in (0.05, 0.4, 2.5)
            N = TI._series_length(α₀)
            c = 4 * sqrt(2) / π
            b = TI._neumann_amplitudes(Val(:sine), 0, [c * n for n in 0:(N + 1)], α₀)
            b₂ = TI._neumann_amplitudes(Val(:sine), 0, [c * n for n in 0:(2N + 1)], α₀)
            @test dot(1:N, b) ≈ dot(1:(2N), b₂) rtol = 1.0e-12
        end
        setprecision(BigFloat, 256) do
            Ab = _tor_A(big"1.0", big"0.3")
            @test eltype(Ab) == BigFloat
            @test collect(_tor_A(1.0, 0.3)) ≈ Float64.(collect(Ab)) rtol = 1.0e-12
        end
        @test eltype(_tor_A(1.0f0, 0.3f0)) == Float32
        @inferred TI._insulating_torus_concentration(1.3)
        @inferred TI._insulating_torus_concentration(ForwardDiff.Dual(1.3, 1.0))
    end

    @testset "the inclusion and its contract" begin
        K₀ = _tor_iso(2.5)
        t = InsulatingTorus(2.0, 0.5)
        @test t isa InsulatingTorus{Float64}
        @test InsulatingTorus(2, 0.5) isa InsulatingTorus{Float64}
        @test dimension(t) == 3
        @test shape_trait(t) isa ToroidalShape
        @test !is_homogeneous_inclusion(t)
        @test inclusion_basis(t) isa TensND.AbstractBasis
        @test occursin("InsulatingTorus", sprint(show, t))
        A = gradient_gradient_concentration(t, _tor_iso(0.0), K₀)
        Aₜ, Aₙ = _tor_A(2.0, 0.5)
        @test canonical_components(A) ≈ [Aₜ 0 0; 0 Aₜ 0; 0 0 Aₙ]
        @inferred gradient_gradient_concentration(t, _tor_iso(0.0), K₀)
        # the property of the phase is not read, nor the modulus of the matrix
        @test canonical_components(gradient_gradient_concentration(t, _tor_iso(7.0), _tor_iso(0.1))) ≈ canonical_components(A)
        @test iszero(canonical_components(flux_gradient_concentration(t, _tor_iso(0.0), K₀)))
        # contributions of a pore: N = -K₀·A and R = A·K₀⁻¹
        @test canonical_components(property_contribution(t, _tor_iso(0.0), K₀)) ≈ -2.5 * canonical_components(A)
        @test canonical_components(inverse_property_contribution(t, _tor_iso(0.0), K₀)) ≈ canonical_components(A) / 2.5
        # an axis given by Euler angles or by a basis
        tilted = InsulatingTorus(2.0, 0.5; euler_angles = (0.4, 1.1, 0.0))
        Q = Matrix(TensND.vecbasis(inclusion_basis(tilted)))
        @test canonical_components(gradient_gradient_concentration(tilted, _tor_iso(0.0), K₀)) ≈ Q * Diagonal([Aₜ, Aₜ, Aₙ]) * Q'
        same = InsulatingTorus(2.0, 0.5; basis = inclusion_basis(tilted))
        @test canonical_components(gradient_gradient_concentration(same, _tor_iso(0.0), K₀)) ≈ Q * Diagonal([Aₜ, Aₜ, Aₙ]) * Q'
    end

    @testset "against the grid of Gaussians" begin
        K₀ = _tor_iso(1.0)
        # a thinner tube needs a finer grid
        for (R, r, n) in ((1.0, 0.5, 24), (1.0, 0.25, 32))
            A = canonical_components(gradient_gradient_concentration(InsulatingTorus(R, r), _tor_iso(0.0), K₀))
            Rg = canonical_components(inverse_property_contribution(GaussianInclusion(Torus(R, r); n), _tor_iso(1.0e-4), K₀))
            @test Rg[1, 1] ≈ A[1, 1] rtol = 6.0e-2
            @test Rg[3, 3] ≈ A[3, 3] rtol = 6.0e-2
        end
    end

    @testset "in a scheme" begin
        K₀, f = _tor_iso(2.0), 0.08
        t = InsulatingTorus(1.0, 0.3)
        Aₜ, Aₙ = _tor_A(1.0, 0.3)
        rve = RVE()
        add_phase!(rve, :matrix, Ellipsoid(1.0), Dict(:K => K₀); fraction = :rest)
        add_phase!(rve, :pores, t, Dict(:K => K₀); fraction = f)
        @test canonical_components(homogenize(rve, Dilute(), :K)) ≈ 2.0 * (I - f * Diagonal([Aₜ, Aₜ, Aₙ]))
        Kmt = canonical_components(homogenize(rve, MoriTanaka(), :K))
        @test Kmt ≈ 2.0 * (1 - f) * inv((1 - f) * I + f * Diagonal([Aₜ, Aₜ, Aₙ]))
        # randomly oriented pores
        iso = RVE()
        add_phase!(iso, :matrix, Ellipsoid(1.0), Dict(:K => K₀); fraction = :rest)
        add_phase!(iso, :pores, t, Dict(:K => K₀); fraction = f, symmetrize = IsoSymmetrize())
        @test canonical_components(homogenize(iso, Dilute(), :K)) ≈ 2.0 * (1 - f * (2Aₜ + Aₙ) / 3) * I
        @test homogenize(iso, SelfConsistent(), :K) isa TensND.AbstractTens
    end

    @testset "derivatives" begin
        # degree 0 in (R, r): R ∂A/∂R + r ∂A/∂r = 0
        R, r = 1.7, 0.6
        dR = ForwardDiff.derivative(x -> collect(_tor_A(x, r)), R)
        dr = ForwardDiff.derivative(x -> collect(_tor_A(R, x)), r)
        @test R .* dR ≈ -r .* dr rtol = 1.0e-9
        # against a difference in extended precision
        setprecision(BigFloat, 256) do
            h = big"1.0e-25"
            ref = (collect(_tor_A(big(R), big(r) + h)) - collect(_tor_A(big(R), big(r) - h))) / (2h)
            @test dr ≈ Float64.(ref) rtol = 1.0e-9
        end
        # through the inclusion and a scheme
        K₀ = _tor_iso(2.0)
        k33(x) = canonical_components(gradient_gradient_concentration(InsulatingTorus(R, x), _tor_iso(0.0), K₀))[3, 3]
        @test ForwardDiff.derivative(k33, r) ≈ dr[2] rtol = 1.0e-12
        # second derivative, nested duals
        @test ForwardDiff.derivative(y -> ForwardDiff.derivative(k33, y), r) isa Float64
    end

    @testset "what is refused" begin
        t = InsulatingTorus(2.0, 0.5)
        @test_throws DomainError InsulatingTorus(1.0, 1.0)
        @test_throws DomainError InsulatingTorus(1.0, 2.0)
        @test_throws DomainError InsulatingTorus(1.0, 0.0)
        Kti = TensND.TensTI{2}(1.0, 2.0, (0.0, 0.0, 1.0))
        @test_throws ArgumentError gradient_gradient_concentration(t, _tor_iso(0.0), Kti)
        @test_throws ArgumentError flux_gradient_concentration(t, _tor_iso(0.0), Kti)
        C₀ = iso_stiffness(3.0, 1.0)
        @test_throws ArgumentError gradient_gradient_concentration(t, C₀, C₀)
        @test_throws ArgumentError flux_gradient_concentration(t, C₀, C₀)
    end
end
