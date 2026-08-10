using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

# =============================================================================
#  test_surface_stiffness.jl — the equivalent particle of Dormieux, Lemarchand
#  & Brisard (2016): a nanoinclusion plus its Gurtin-Murdoch interface.
#
#  Coverage:
#   1. Spherical limit X = 1 reproduces the intrinsic form of their Eq. (74),
#      2(6μs+κs)/(5R) 𝕂 + (4κs/R) 𝕁.
#   2. Asymptotic oblate limit X → 0, their Eq. (76): (3/c)(κs E₁ + μs E₃).
#   3. Asymptotic prolate limit X → ∞, their Eq. (78), including the ratio
#      C₁₁₁₁/C₁₁₂₂ = 3 they compare with atomistic simulations.
#   4. Transverse-isotropy consistency C₁₂₁₂ = (C₁₁₁₁ - C₁₁₂₂)/2.
#   5. Continuity through X = 1, where the closed form has a removable
#      singularity and the implementation switches to a series.
#   6. Size scaling: ℂ^int ∝ 1/size under homothety (their Eq. 50).
#   7. `equivalent_particle` feeding an ordinary Mori-Tanaka estimate — the
#      point of the paper — and the vanishing-interface limit.
#   8. Orientation, error paths, ForwardDiff.
# =============================================================================

const RTOL_SS = 1.0e-10

_κs() = 1.3
_μs() = 0.7

# ℂ^int = α 𝕂 + β 𝕁 written out in components, for the spherical reference.
function _iso_ref(α, β)
    d = Matrix{Float64}(I, 3, 3)
    return [
        β * d[i, j] * d[k, l] / 3 +
            α * ((d[i, k] * d[j, l] + d[i, l] * d[j, k]) / 2 - d[i, j] * d[k, l] / 3)
            for i in 1:3, j in 1:3, k in 1:3, l in 1:3
    ]
end

@testset "Surface stiffness — spherical limit, Dormieux et al. Eq. (74)" begin
    κs, μs = _κs(), _μs()
    for R in (0.5, 2.0, 10.0)
        A = get_array(surface_stiffness(Ellipsoid(R), κs, μs))
        ref = _iso_ref(2 * (6μs + κs) / (5R), 4κs / R)
        @test maximum(abs.(A .- ref)) < RTOL_SS * maximum(abs.(ref))
        # And the component form quoted in the paper's text.
        @test A[1, 1, 1, 1] ≈ 8 * (κs + μs) / (5R) rtol = RTOL_SS
        @test A[1, 1, 2, 2] ≈ 2 * (3κs - 2μs) / (5R) rtol = RTOL_SS
        @test A[1, 3, 1, 3] ≈ (κs + 6μs) / (5R) rtol = RTOL_SS
    end
end

@testset "Surface stiffness — asymptotic oblate limit, Eq. (76)" begin
    # ℂ^int → (3/c)(κs E₁ + μs E₃): in components, c·C₁₁₁₁ → 3(κs+μs)/2,
    # c·C₁₁₂₂ → 3(κs-μs)/2, c·C₁₂₁₂ → 3μs/2, and everything involving the
    # polar axis vanishes — a platelet stiffens only in its own plane.
    κs, μs = _κs(), _μs()
    c = 1.0e-6
    A = get_array(surface_stiffness(Ellipsoid(1.0, 1.0, c), κs, μs))
    @test c * A[1, 1, 1, 1] ≈ 3 * (κs + μs) / 2 rtol = 1.0e-7
    @test c * A[1, 1, 2, 2] ≈ 3 * (κs - μs) / 2 rtol = 1.0e-7
    @test c * A[1, 2, 1, 2] ≈ 3μs / 2 rtol = 1.0e-7
    @test c * abs(A[3, 3, 3, 3]) < 1.0e-6
    @test c * abs(A[1, 1, 3, 3]) < 1.0e-6
end

@testset "Surface stiffness — asymptotic prolate limit, Eq. (78)" begin
    # Nanofiber / nanotube limit.  With the symmetry axis along e₁ the paper's
    # C₃₃₃₃ (axial) is our C₁₁₁₁, and its transverse ratio C₁₁₁₁/C₁₁₂₂ = 3 is
    # the value it compares with the atomistic result of Saether et al. (2003).
    κs, μs = _κs(), _μs()
    a = 1.0
    A = get_array(surface_stiffness(Ellipsoid(1.0e6 * a, a, a), κs, μs))
    @test A[1, 1, 1, 1] ≈ 3π * (κs + μs) / (4a) rtol = 1.0e-5
    @test A[2, 2, 2, 2] / A[2, 2, 3, 3] ≈ 3.0 rtol = 1.0e-8
    @test A[2, 2, 2, 2] ≈ 9π * (κs + μs) / (32a) rtol = 1.0e-5
    @test A[2, 2, 3, 3] ≈ 3π * (κs + μs) / (32a) rtol = 1.0e-5
end

@testset "Surface stiffness — transverse-isotropy consistency" begin
    # The paper states C₁₂₁₂ = (C₁₁₁₁ - C₁₁₂₂)/2 as a bracketed identity; it
    # must hold at every aspect ratio, not only in the limits.
    κs, μs = _κs(), _μs()
    for X in (0.05, 0.5, 0.9, 1.0, 1.1, 3.0, 20.0)
        ell = X < 1 ? Ellipsoid(1.0, 1.0, X) : (X > 1 ? Ellipsoid(X, 1.0, 1.0) : Ellipsoid(1.0))
        A = get_array(surface_stiffness(ell, κs, μs))
        # In-plane indices depend on where the symmetry axis sits.
        (i, j, k) = X > 1 ? (2, 3, 1) : (1, 2, 3)
        @test A[i, j, i, j] ≈ (A[i, i, i, i] - A[i, i, j, j]) / 2 rtol = 1.0e-9
    end
end

@testset "Surface stiffness — continuity through the spherical case" begin
    # X = 1 is a removable singularity of the closed form: both terms diverge
    # as (X²-1)⁻² and cancel.  The implementation switches to a series there,
    # and the switch must be invisible.
    κs, μs = _κs(), _μs()
    C(X) = get_array(
        surface_stiffness(
            X < 1 ? Ellipsoid(1.0, 1.0, X) : (X > 1 ? Ellipsoid(X, 1.0, 1.0) : Ellipsoid(1.0)),
            κs, μs
        )
    )[1, 1, 1, 1]
    C1 = C(1.0)
    for δ in (1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2)
        @test C(1 - δ) ≈ C1 rtol = 20δ
        @test C(1 + δ) ≈ C1 rtol = 20δ
    end
    # Monotone approach from both sides, no spurious jump at the switch.
    @test abs(C(1 - 1.0e-5) - C1) < abs(C(1 - 1.0e-3) - C1)
    @test abs(C(1 + 1.0e-5) - C1) < abs(C(1 + 1.0e-3) - C1)
end

@testset "Surface stiffness — size scaling (Eq. 50)" begin
    # Scaling a particle by ρ divides its interface stiffness by ρ: the
    # stiffening is a size effect, and vanishes for large particles.
    κs, μs = _κs(), _μs()
    A = get_array(surface_stiffness(Ellipsoid(2.0, 1.0, 1.0), κs, μs))
    for ρ in (2.0, 10.0)
        B = get_array(surface_stiffness(Ellipsoid(2ρ, ρ, ρ), κs, μs))
        @test maximum(abs.(ρ .* B .- A)) < RTOL_SS * maximum(abs.(A))
    end
end

@testset "Surface stiffness — orientation follows the spheroid axis" begin
    κs, μs = _κs(), _μs()
    # Two ways to declare the same prolate spheroid, with different axes:
    # `Ellipsoid(3, 1, 1)` stores its semi-axes in descending order and carries
    # the symmetry axis along e₁, while `Spheroid(3)` builds a basis whose
    # third axis is the symmetry axis. The interface tensor must follow, so
    # the two differ by the permutation 1 ↔ 3.
    A = get_array(surface_stiffness(Ellipsoid(3.0, 1.0, 1.0), κs, μs))
    B = get_array(surface_stiffness(Spheroid(3.0), κs, μs))
    @test A[1, 1, 1, 1] ≈ B[3, 3, 3, 3] rtol = 1.0e-9      # axial
    @test A[2, 2, 2, 2] ≈ B[1, 1, 1, 1] rtol = 1.0e-9      # transverse
    @test A[1, 2, 1, 2] ≈ B[3, 2, 3, 2] rtol = 1.0e-9      # axial shear
    # Tilting a `Spheroid` by the first ZYZ angle — the inclination from e₃ —
    # brings its axis onto e₁ and recovers the `Ellipsoid` components.
    D = get_array(surface_stiffness(Spheroid(3.0; euler_angles = (π / 2, 0, 0)), κs, μs))
    @test maximum(abs.(D .- A)) < 1.0e-9 * maximum(abs.(A))
    # An oblate spheroid stores its short axis last, and keeps e₃ as the axis.
    E = get_array(surface_stiffness(Ellipsoid(1.0, 1.0, 0.3), κs, μs))
    F = get_array(surface_stiffness(Spheroid(0.3), κs, μs))
    @test maximum(abs.(E .- F)) < 1.0e-9 * maximum(abs.(E))
end

@testset "equivalent_particle — plugs into an ordinary scheme" begin
    # The point of the paper: no new scheme is needed, the interface is
    # absorbed into the particle stiffness.
    κs, μs = _κs(), _μs()
    C_m = TensISO{3}(3 * 1.0, 2 * 0.4)
    C_i = TensISO{3}(3 * 4.0, 2 * 2.0)
    f, R = 0.15, 0.01                       # 10 nm particles
    sph = Ellipsoid(R)
    C_eq = equivalent_particle(C_i, sph, κs, μs)
    @test get_array(C_eq) ≈ get_array(C_i) .+ get_array(surface_stiffness(sph, κs, μs)) rtol = 1.0e-14

    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_m))
    add_phase!(rve, :nano, sph, Dict(:C => C_eq); fraction = f)
    C_nano = get_array(homogenize(rve, MoriTanaka(), :C))

    rve0 = RVE(:M)
    add_matrix!(rve0, Ellipsoid(1.0), Dict(:C => C_m))
    add_phase!(rve0, :bulk, sph, Dict(:C => C_i); fraction = f)
    C_bulk = get_array(homogenize(rve0, MoriTanaka(), :C))
    # A positive surface energy stiffens the composite, and the effect is a
    # size effect: it must shrink as the particle grows.
    @test C_nano[1, 2, 1, 2] > C_bulk[1, 2, 1, 2]
    big = Ellipsoid(1.0)
    rve1 = RVE(:M)
    add_matrix!(rve1, Ellipsoid(1.0), Dict(:C => C_m))
    add_phase!(rve1, :nano, big, Dict(:C => equivalent_particle(C_i, big, κs, μs)); fraction = f)
    C_big = get_array(homogenize(rve1, MoriTanaka(), :C))
    @test abs(C_big[1, 2, 1, 2] - C_bulk[1, 2, 1, 2]) <
        abs(C_nano[1, 2, 1, 2] - C_bulk[1, 2, 1, 2])
    # No interface ⇒ the plain particle.
    @test get_array(equivalent_particle(C_i, sph, 0.0, 0.0)) ≈ get_array(C_i) atol = 1.0e-14
end

@testset "Surface stiffness — error paths and ForwardDiff" begin
    κs, μs = _κs(), _μs()
    # A triaxial ellipsoid has no transversely isotropic interface tensor.
    @test_throws ArgumentError surface_stiffness(Ellipsoid(3.0, 2.0, 1.0), κs, μs)
    # Differentiable in the surface moduli and in the particle size.
    fκ = k -> get_array(surface_stiffness(Ellipsoid(2.0, 1.0, 1.0), k, μs))[1, 1, 1, 1]
    @test ForwardDiff.derivative(fκ, κs) ≈ (fκ(κs + 1.0e-7) - fκ(κs - 1.0e-7)) / 2.0e-7 rtol = 1.0e-6
    fa = a -> get_array(surface_stiffness(Ellipsoid(2a, a, a), κs, μs))[1, 1, 1, 1]
    @test ForwardDiff.derivative(fa, 1.0) ≈ (fa(1.0 + 1.0e-7) - fa(1.0 - 1.0e-7)) / 2.0e-7 rtol = 1.0e-5
end
