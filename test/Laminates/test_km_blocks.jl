# =============================================================================
#  test_km_blocks.jl — the laminate kernels (`src/Core/laminate_algebra.jl`),
#  the layer below the laminate cell. The Kelvin-Mandel block algebra they are
#  written on (partition, blocks, Schur inverse, plane pseudo-inverses, acoustic
#  tensor) is TensND's and tested there (`test/test_km_blocks.jl`).
#
#  Coverage:
#   1. `flat_hill` against the closed-form flat-inclusion Hill tensor of an
#      isotropic matrix, and `plane_pinv` against `LinearAlgebra.pinv`.
#   2. The Hill identities `ℙ:ℂ:ℙ = ℙ` and `ℚ:ℙ = 0`.
#   3. `laminate_stiffness` : N = 1 degeneracy, identical layers, the exact
#      Backus (1962) bilayer closed form, in-plane rotation invariance.
#   4. Concentration : `Σ f 𝔸 = Σ f 𝔹 = 𝕀`, the block structure of `𝔸`, and
#      `⟨ℂ:𝔸⟩ = ℂ_hom`.
#   5. `laminate_conductivity` : the exact series / parallel laws.
#   6. ForwardDiff through the kernel.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using StaticArrays
using ForwardDiff
using Random

const MFHC = MeanFieldHomogenization.Core
const ATOL_KM = 1.0e-12

# A fully triclinic, symmetric-positive-definite Kelvin-Mandel matrix.
function _rand_km6(rng)
    A = randn(rng, 6, 6)
    return SMatrix{6, 6}(A * A' + 8I)
end

_iso_km(k, μ) = SMatrix{6, 6}(KM(TensISO{3}(3k, 2μ)))

@testset "flat_hill / plane_pinv" begin
    k, μ = 2.0, 0.8
    λ = k - 2μ / 3
    P = MFHC.flat_hill(_iso_km(k, μ))
    # Flat-inclusion Hill tensor of an isotropic matrix, normal e₃:
    #   P_3333 = 1/(λ+2μ),  P_2323 = P_1313 = 1/(4μ),  all else 0.
    @test P[3, 3] ≈ 1 / (λ + 2μ) atol = ATOL_KM
    @test P[4, 4] ≈ 2 * (1 / (4μ)) atol = ATOL_KM     # Mandel: [4,4] = 2·P_2323
    @test P[5, 5] ≈ 2 * (1 / (4μ)) atol = ATOL_KM
    @test iszero(TensND.km_ip_block(P))

    rng = MersenneTwister(31)
    M = _rand_km6(rng)
    # `plane_pinv` reproduces the Moore-Penrose pseudo-inverse of the
    # OP-supported part — the SVD-free path validated against the SVD.
    Mop = TensND.km_op_embed(TensND.km_op_block(M))
    @test MFHC.plane_pinv(M) ≈ pinv(Matrix(Mop)) atol = 1.0e-10
    # ⟨ℙ⟩† : ⟨ℙ⟩ = Πᴼ
    P = MFHC.flat_hill(M)
    @test MFHC.plane_pinv(P) ≈ Mop atol = 1.0e-10
end

@testset "Hill identities ℙ:ℂ:ℙ = ℙ and ℚ:ℙ = 0" begin
    rng = MersenneTwister(37)
    for _ in 1:5
        M = _rand_km6(rng)
        P = MFHC.flat_hill(M)
        Q = M - M * P * M
        @test P * M * P ≈ P atol = 1.0e-10
        @test maximum(abs, Q * P) < 1.0e-10
        # ℚ operates within in-plane tensors only
        @test maximum(abs, Q[MFHC.KM_OP, :]) < 1.0e-10
        @test maximum(abs, Q[:, MFHC.KM_OP]) < 1.0e-10
    end
end

@testset "laminate_stiffness — degeneracies" begin
    rng = MersenneTwister(47)
    Z = zero(SMatrix{6, 6, Float64})
    M = _rand_km6(rng)
    # N = 1 : the single layer is returned unchanged.
    @test MFHC.laminate_stiffness((M,), (1.0,), Z, Z) ≈ M atol = 1.0e-10
    # Identical layers, unequal fractions.
    @test MFHC.laminate_stiffness((M, M, M), (0.2, 0.5, 0.3), Z, Z) ≈ M atol = 1.0e-10
end

@testset "laminate_stiffness — Backus (1962) bilayer closed form" begin
    k1, μ1, k2, μ2 = 2.0, 0.8, 0.5, 0.2
    λ1, λ2 = k1 - 2μ1 / 3, k2 - 2μ2 / 3
    f = (0.3, 0.7)
    Z = zero(SMatrix{6, 6, Float64})
    Ch = MFHC.laminate_stiffness((_iso_km(k1, μ1), _iso_km(k2, μ2)), f, Z, Z)

    avg(g) = f[1] * g(λ1, μ1) + f[2] * g(λ2, μ2)
    r33 = 1 / avg((l, m) -> 1 / (l + 2m))
    rλ = avg((l, m) -> l / (l + 2m))

    @test Ch[3, 3] ≈ r33 atol = ATOL_KM                                   # C₃₃₃₃
    @test Ch[4, 4] / 2 ≈ 1 / avg((l, m) -> 1 / m) atol = ATOL_KM          # C₂₃₂₃
    @test Ch[6, 6] / 2 ≈ avg((l, m) -> m) atol = ATOL_KM                  # C₁₂₁₂
    @test Ch[1, 3] ≈ r33 * rλ atol = ATOL_KM                              # C₁₁₃₃
    @test Ch[1, 1] ≈ avg((l, m) -> 4m * (l + m) / (l + 2m)) + r33 * rλ^2 atol = ATOL_KM
    @test Ch[1, 2] ≈ avg((l, m) -> 2m * l / (l + 2m)) + r33 * rλ^2 atol = ATOL_KM

    # Isotropic layers ⇒ the result is EXACTLY transversely isotropic about n.
    @test Ch[1, 1] ≈ Ch[2, 2] atol = ATOL_KM
    @test Ch[1, 3] ≈ Ch[2, 3] atol = ATOL_KM
    @test Ch[4, 4] ≈ Ch[5, 5] atol = ATOL_KM
    @test Ch[6, 6] ≈ Ch[1, 1] - Ch[1, 2] atol = ATOL_KM
    @test Ch ≈ Ch' atol = ATOL_KM
    for (i, j) in ((1, 4), (1, 5), (1, 6), (2, 4), (2, 5), (2, 6), (3, 6), (4, 5), (4, 6), (5, 6))
        @test abs(Ch[i, j]) < ATOL_KM
    end
end

@testset "laminate_stiffness — in-plane rotation invariance" begin
    # Rotating (ℓ, m) about n by φ must rotate the answer by that same φ:
    # the kernel privileges neither in-plane direction.
    rng = MersenneTwister(53)
    C6s = ntuple(_ -> _rand_km6(rng), 3)
    f = (0.2, 0.45, 0.35)
    Z = zero(SMatrix{6, 6, Float64})
    Ch = MFHC.laminate_stiffness(C6s, f, Z, Z)

    φ = 0.83
    Rz = SMatrix{3, 3}(cos(φ), sin(φ), 0.0, -sin(φ), cos(φ), 0.0, 0.0, 0.0, 1.0)
    Qz = TensND.KM_rotation(Rz)
    C6r = ntuple(i -> Qz' * C6s[i] * Qz, 3)
    @test MFHC.laminate_stiffness(C6r, f, Z, Z) ≈ Qz' * Ch * Qz atol = 1.0e-10
end

@testset "laminate concentration — sum rules and block structure" begin
    rng = MersenneTwister(59)
    C6s = ntuple(_ -> _rand_km6(rng), 3)
    f = (0.2, 0.45, 0.35)
    Z = zero(SMatrix{6, 6, Float64})
    Ch = MFHC.laminate_stiffness(C6s, f, Z, Z)

    As = [MFHC.laminate_strain_concentration(C6s[i], Ch) for i in 1:3]
    Bs = [MFHC.laminate_stress_concentration(C6s[i], Ch) for i in 1:3]

    @test sum(f[i] * As[i] for i in 1:3) ≈ I atol = 1.0e-10
    @test sum(f[i] * Bs[i] for i in 1:3) ≈ I atol = 1.0e-10
    # The macroscopic in-plane strain reaches every layer unchanged.
    for A in As
        @test TensND.km_ip_block(A) ≈ I atol = ATOL_KM
        @test maximum(abs, A[MFHC.KM_IP, MFHC.KM_OP]) < ATOL_KM
    end
    # ⟨ℂ:𝔸⟩ = ℂ_hom
    @test sum(f[i] * C6s[i] * As[i] for i in 1:3) ≈ Ch atol = 1.0e-10
end

@testset "laminate_conductivity — exact series / parallel laws" begin
    f = (0.2, 0.45, 0.35)
    Z3 = zero(SMatrix{3, 3, Float64})
    ks = (2.0, 0.3, 1.0)
    K3s = ntuple(i -> SMatrix{3, 3}(Diagonal(fill(ks[i], 3))), 3)
    Kh = MFHC.laminate_conductivity(K3s, f, Z3, Z3)
    @test Kh[3, 3] ≈ 1 / sum(f[i] / ks[i] for i in 1:3) atol = ATOL_KM   # series
    @test Kh[1, 1] ≈ sum(f[i] * ks[i] for i in 1:3) atol = ATOL_KM       # parallel
    @test Kh[2, 2] ≈ Kh[1, 1] atol = ATOL_KM

    # Anisotropic layers: the series law holds on the NORMAL component alone.
    rng = MersenneTwister(61)
    K3a = ntuple(_ -> (A = randn(rng, 3, 3); SMatrix{3, 3}(A * A' + 3I)), 3)
    Kha = MFHC.laminate_conductivity(K3a, f, Z3, Z3)
    @test 1 / Kha[3, 3] ≈ sum(f[i] / K3a[i][3, 3] for i in 1:3) atol = ATOL_KM
    As = [MFHC.laminate_gradient_concentration(K3a[i], Kha) for i in 1:3]
    Bs = [MFHC.laminate_flux_concentration(K3a[i], Kha) for i in 1:3]
    @test sum(f[i] * As[i] for i in 1:3) ≈ I atol = 1.0e-10
    @test sum(f[i] * Bs[i] for i in 1:3) ≈ I atol = 1.0e-10
    @test MFHC.laminate_conductivity((K3a[1],), (1.0,), Z3, Z3) ≈ K3a[1] atol = 1.0e-10
end

@testset "laminate kernel — ForwardDiff" begin
    rng = MersenneTwister(67)
    C6s = ntuple(_ -> _rand_km6(rng), 3)
    f = (0.2, 0.45, 0.35)
    Z = zero(SMatrix{6, 6, Float64})
    g(x) = MFHC.laminate_stiffness((C6s[1] * x, C6s[2], C6s[3]), f, Z, Z)[3, 3]
    h = 1.0e-6
    @test ForwardDiff.derivative(g, 1.0) ≈ (g(1 + h) - g(1 - h)) / (2h) rtol = 1.0e-6

    # ... and through a volume fraction
    gf(x) = MFHC.laminate_stiffness(C6s, (x, 0.45, 0.55 - x), Z, Z)[1, 1]
    @test ForwardDiff.derivative(gf, 0.2) ≈ (gf(0.2 + h) - gf(0.2 - h)) / (2h) rtol = 1.0e-6
end
