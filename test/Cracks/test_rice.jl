# =============================================================================
#  test_rice.jl — Rice's formula: the contribution of a crack from its
#  intensity factors integrated along a growth path.
#
#   1. A penny grown from a point, through `RiceCrack`, against the closed-form
#      `PennyCrack` — elasticity and conduction, in every scheme.
#   2. An elliptic crack grown self-similarly, with the intensity factors of
#      `sif` (Kassir & Sih through the COD tensors), against `cod_tensor`:
#      isotropic and transversely isotropic matrices, an oblique axis included.
#   3. Guard rails.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using StaticArrays
using ForwardDiff

const RC_E, RC_ν, RC_k = 10.0, 0.25, 3.0
const RC_C = MeanFieldHomogenization.iso_stiffness_E_nu(RC_E, RC_ν)
const RC_K = TensISO{3}(RC_k)

_rc_penny(a = 1.0) = RiceCrack(
    a;
    K_I = (ρ, ν) -> 2sqrt(ρ / π),
    K_II = (ρ, ν) -> 4sqrt(ρ / π) / (2 - ν),
    K_III = (ρ, ν) -> 4 * (1 - ν) * sqrt(ρ / π) / (2 - ν),
)

_rc_comp(t, ℬ) = Array(TensND.components(t, ℬ))

@testset "RiceCrack — a penny grown from a point" begin
    for a in (1.0, 2.5)
        rc = _rc_penny(a)
        pc = PennyCrack(a)
        @test _rc_comp(cod_tensor(rc, RC_C), pc.basis) ≈ _rc_comp(cod_tensor(pc, RC_C), pc.basis) rtol = 1.0e-10
        @test cod_tensor(rc, RC_K) ≈ cod_tensor(pc, RC_K) rtol = 1.0e-10
        @test TensND.canonical_components(inverse_property_contribution(rc, RC_C)) ≈
            TensND.canonical_components(inverse_property_contribution(pc, RC_C)) rtol = 1.0e-10
    end
    rc = _rc_penny()
    @test MeanFieldHomogenization.shape_trait(rc) === CircularFronts
    @test crack_density_factor(rc) ≈ 4π / 3
    @test semi_major(rc) == semi_minor(rc) == 1.0
    @test occursin("RiceCrack", sprint(show, rc))
    @test TensND.canonical_components(MeanFieldHomogenization.shape_tensor(rc)) ≈ Diagonal([1.0, 1.0, 0.0])

    # the interface corrections are those of any crack
    K = TensISO{3}(2.0)
    @test _rc_comp(cod_tensor(rc, RC_C; K_interface = K), rc.basis) ≈
        _rc_comp(cod_tensor(PennyCrack(1.0), RC_C; K_interface = K), rc.basis) rtol = 1.0e-10
    @test cod_tensor(rc, RC_K; α_interface = 0.7) ≈ cod_tensor(PennyCrack(1.0), RC_K; α_interface = 0.7) rtol = 1.0e-10

    @testset "schemes" begin
        ε = 0.1
        for (P, key) in ((RC_C, :C), (RC_K, :K)), scheme in (Dilute(), MoriTanaka(), SelfConsistent())
            rves = map((rc, PennyCrack(1.0))) do geom
                rve = RVE(; distribution_shape = Ellipsoid(1.0))
                add_phase!(rve, :M, Ellipsoid(1.0), Dict(key => P); fraction = :rest)
                add_phase!(rve, :F, geom, Dict(key => P); density = ε, symmetrize = IsoSymmetrize())
                rve
            end
            @test TensND.canonical_components(homogenize(rves[1], scheme, key)) ≈ TensND.canonical_components(homogenize(rves[2], scheme, key)) rtol = 1.0e-9
        end
    end

    @testset "derivative in the Poisson ratio" begin
        # B_tt = 16(1 − ν²)/(3πE(1 − ν/2)) for a penny
        f(ν) = TensND.canonical_components(cod_tensor(rc, MeanFieldHomogenization.iso_stiffness_E_nu(RC_E, ν)))[1, 1]
        g(ν) = 16 * (1 - ν^2) / (3π * RC_E * (1 - ν / 2))
        @test ForwardDiff.derivative(f, RC_ν) ≈ ForwardDiff.derivative(g, RC_ν) rtol = 1.0e-9
    end
end

# The front of an elliptic crack grown self-similarly, λ ∈ [0, 1]: the point
# λ(a cos θ ℓ + b sin θ m) sweeps dA = λ a b dθ dλ, and the intensity factors scale
# as √λ, so the matrices M(θ) of the final crack serve the whole growth.
function _rc_ellipse_front(crack, P₀, nθ; kw...)
    ℬ = crack_basis(crack)
    l̂, m̂, n̂ = (TensND.tens_basis(ℬ, i) for i in 1:3)
    a, b = semi_major(crack), semi_minor(crack)
    return map(0:(nθ - 1)) do k
        θ = 2π * k / nθ
        y = cos(θ) * l̂ + sin(θ) * m̂
        ν = normalize(SVector(cos(θ) / a, sin(θ) / b, 0.0))
        w = 2π / nθ * a * b
        if P₀ isa TensND.AbstractTens{4}
            cols = map(1:3) do i
                e = TensND.tens_basis(ℬ, i)
                Σ = i == 3 ? n̂ ⊗ n̂ : e ⊗ n̂ + n̂ ⊗ e
                K, _ = sif(crack, P₀, Σ; y₀ = y, kw...)
                SVector{3}(TensND.components(K, ℬ))
            end
            (w, ν, hcat(cols...))
        else
            (w, ν, sif(crack, P₀, n̂; y₀ = y, kw...))
        end
    end
end
_rc_scaled(data) = λ -> map(d -> (λ * d[1], d[2], sqrt(λ) * d[3]), data)

@testset "sif of an elliptic crack: Irwin's closed form" begin
    # K_I = σ √(πb) ϱ^{1/2} / E(1 − η²), ϱ² = η² cos² θ + sin² θ, for a crack of
    # semi-axes not equal to one (the intensity factor scales as √b)
    a, b = 2.0, 0.6
    crack = EllipticCrack(a, b)
    l̂, m̂, n̂ = (TensND.tens_basis(crack_basis(crack), i) for i in 1:3)
    η = b / a
    for θ in (0.0, 0.7, π / 2)
        _, (KI, _, _) = sif(crack, RC_C, n̂ ⊗ n̂; y₀ = cos(θ) * l̂ + sin(θ) * m̂)
        ϱ = sqrt(η^2 * cos(θ)^2 + sin(θ)^2)
        @test KI ≈ sqrt(π * b) * sqrt(ϱ) / MeanFieldHomogenization.Elliptic.ell_E(1 - η^2) rtol = 1.0e-12
        @test sif(crack, RC_K, n̂; y₀ = cos(θ) * l̂ + sin(θ) * m̂) ≈ KI rtol = 1.0e-12
    end
end

@testset "rice_cod_tensor — an elliptic crack grown self-similarly" begin
    nθ = 32
    @testset "isotropic, η = $η" for η in (1.0, 0.5, 0.2)
        crack = EllipticCrack(1.0, η)
        ℬ = crack_basis(crack)
        ℓ = cbrt(semi_major(crack) * semi_minor(crack)^2)    # ε = N a b²
        nθη = ceil(Int, nθ / η)     # the front turns faster as the crack flattens
        front = _rc_scaled(_rc_ellipse_front(crack, RC_C, nθη))
        B = rice_cod_tensor(front, 0.0, 1.0, RC_C; size = ℓ, basis = ℬ)
        @test _rc_comp(B, ℬ) ≈ _rc_comp(cod_tensor(crack, RC_C), ℬ) rtol = 1.0e-9
        frontT = _rc_scaled(_rc_ellipse_front(crack, RC_K, nθη))
        b = rice_cod_tensor(frontT, 0.0, 1.0, RC_K; size = ℓ, basis = ℬ)
        @test b ≈ cod_tensor(crack, RC_K) rtol = 1.0e-9
    end

    @testset "transversely isotropic, oblique axis" begin
        C = TensND.TensTI{4}(30.0, 8.0, 5.0, 12.0, 4.0, normalize([1.0, 0.4, 0.7]))
        crack = EllipticCrack(1.0, 0.6; euler_angles = (0.3, 0.8, -0.5))
        ℬ = crack_basis(crack)
        ℓ = cbrt(semi_major(crack) * semi_minor(crack)^2)
        front = _rc_scaled(_rc_ellipse_front(crack, C, nθ; reltol = 1.0e-10, abstol = 1.0e-14))
        B = rice_cod_tensor(front, 0.0, 1.0, C; size = ℓ, basis = ℬ, reltol = 1.0e-10)
        Bref = cod_tensor(crack, C; reltol = 1.0e-10, abstol = 1.0e-14)
        @test _rc_comp(B, ℬ) ≈ _rc_comp(Bref, ℬ) rtol = 1.0e-7
    end
end

@testset "Rice — guard rails" begin
    rc = _rc_penny()
    aniso = TensND.TensTI{4}(30.0, 8.0, 5.0, 12.0, 4.0, [0.0, 0.0, 1.0])
    @test_throws ArgumentError cod_tensor(rc, aniso)
    @test_throws ArgumentError cod_tensor(rc, TensND.TensTI{2}(1.0, 2.0, [0.0, 0.0, 1.0]))
    @test_throws "IsoSymmetrize" cod_tensor(rc, aniso)
    # an unknown shape family has no growth-path normalization
    @test_throws ArgumentError MeanFieldHomogenization.Cracks._rice_size_power(Int)
end
