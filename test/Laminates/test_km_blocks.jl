# =============================================================================
#  test_km_blocks.jl — in-plane / out-of-plane Kelvin-Mandel block algebra
#  (`src/Core/laminate_algebra.jl`), the layer below the laminate cell.
#
#  Coverage:
#   1. The IP/OP partition and the embed ∘ block round trips.
#   2. `acoustic_tensor` against a naive `K[i,j] = C[3,i,3,j]` loop — on a
#      TRICLINIC stiffness, the only case that catches the (3,2,1) index
#      reversal between the OP Mandel slots and the acoustic indices.
#   3. `flat_hill` against the closed-form flat-inclusion Hill tensor of an
#      isotropic matrix, and `plane_pinv` against `LinearAlgebra.pinv`
#      (validating the SVD-free replacement *against* the SVD).
#   4. The Hill identities `ℙ:ℂ:ℙ = ℙ` and `ℚ:ℙ = 0`.
#   5. `_inv_km6` (block elimination, no pivoting) against `inv`.
#   6. `_bond6` against `TensND.KM(C, basis)`.
#   7. `laminate_stiffness` : N = 1 degeneracy, identical layers, the exact
#      Backus (1962) bilayer closed form, in-plane rotation invariance.
#   8. Localization : `Σ f 𝔸 = Σ f 𝔹 = 𝕀`, the block structure of `𝔸`, and
#      `⟨ℂ:𝔸⟩ = ℂ_hom`.
#   9. `laminate_conductivity` : the exact series / parallel laws.
#  10. ForwardDiff through the kernel.
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

@testset "Mandel IP/OP partition" begin
    @test MFHC.KM_IP == SVector(1, 2, 6)
    @test MFHC.KM_OP == SVector(3, 4, 5)

    rng = MersenneTwister(11)
    M = _rand_km6(rng)

    # embed ∘ block is the projection onto each subspace; the two projections
    # are complementary on a block-diagonal input and always idempotent.
    Bop = MFHC._op_block(M)
    @test MFHC._op_block(MFHC._op_embed(Bop)) ≈ Bop atol = ATOL_KM
    Bip = MFHC._ip_block(M)
    @test MFHC._ip_block(MFHC._ip_embed(Bip)) ≈ Bip atol = ATOL_KM
    # an OP-embedded matrix has a strictly zero in-plane block
    @test iszero(MFHC._ip_block(MFHC._op_embed(Bop)))
    @test iszero(MFHC._op_block(MFHC._ip_embed(Bip)))
end

@testset "acoustic_tensor — triclinic guard on the (3,2,1) reversal" begin
    rng = MersenneTwister(23)
    M = _rand_km6(rng)
    arr = Array(inv_KM(Matrix(M)))
    # K_ab = n_i C_iajb n_j with n = e₃, i.e. C[3,a,3,b]
    K_naive = SMatrix{3, 3}([arr[3, i, 3, j] for i in 1:3, j in 1:3])
    @test MFHC.acoustic_tensor(M) ≈ K_naive atol = ATOL_KM
    # ... and it is genuinely non-diagonal here, so the reversal is exercised
    @test !isapprox(K_naive, Diagonal(K_naive); atol = 1.0e-6)

    # isotropic sanity: K = diag(μ, μ, λ+2μ)
    k, μ = 2.0, 0.8
    λ = k - 2μ / 3
    @test MFHC.acoustic_tensor(_iso_km(k, μ)) ≈ Diagonal([μ, μ, λ + 2μ]) atol = ATOL_KM
end

@testset "flat_hill / plane_pinv" begin
    k, μ = 2.0, 0.8
    λ = k - 2μ / 3
    P = MFHC.flat_hill(_iso_km(k, μ))
    # Flat-inclusion Hill tensor of an isotropic matrix, normal e₃:
    #   P_3333 = 1/(λ+2μ),  P_2323 = P_1313 = 1/(4μ),  all else 0.
    @test P[3, 3] ≈ 1 / (λ + 2μ) atol = ATOL_KM
    @test P[4, 4] ≈ 2 * (1 / (4μ)) atol = ATOL_KM     # Mandel: [4,4] = 2·P_2323
    @test P[5, 5] ≈ 2 * (1 / (4μ)) atol = ATOL_KM
    @test iszero(MFHC._ip_block(P))

    rng = MersenneTwister(31)
    M = _rand_km6(rng)
    # `plane_pinv` reproduces the Moore-Penrose pseudo-inverse of the
    # OP-supported part — the SVD-free path validated against the SVD.
    Mop = MFHC._op_embed(MFHC._op_block(M))
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

@testset "_inv_km6 — block elimination without pivoting" begin
    rng = MersenneTwister(41)
    for _ in 1:5
        M = _rand_km6(rng)
        @test MFHC._inv_km6(M) * M ≈ I atol = 1.0e-10
        @test MFHC._inv_km6(M) ≈ inv(Matrix(M)) atol = 1.0e-10
    end
end

@testset "_bond6 against TensND" begin
    b = TensND.RotatedBasis(0.37, 0.91, -0.55)
    R = SMatrix{3, 3}(Matrix(TensND.vecbasis(b, :cov)))
    Q = MFHC._bond6(R)
    @test Q' * Q ≈ I atol = 1.0e-12

    rng = MersenneTwister(43)
    M = _rand_km6(rng)
    Ct = Tens(inv_KM(Matrix(M)))                       # canonical basis
    # Components in the frame `b` are Q' M Q.
    @test Q' * M * Q ≈ Matrix(KM(Ct, b)) atol = 1.0e-10
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
    Qz = MFHC._bond6(Rz)
    C6r = ntuple(i -> Qz' * C6s[i] * Qz, 3)
    @test MFHC.laminate_stiffness(C6r, f, Z, Z) ≈ Qz' * Ch * Qz atol = 1.0e-10
end

@testset "laminate localization — sum rules and block structure" begin
    rng = MersenneTwister(59)
    C6s = ntuple(_ -> _rand_km6(rng), 3)
    f = (0.2, 0.45, 0.35)
    Z = zero(SMatrix{6, 6, Float64})
    Ch = MFHC.laminate_stiffness(C6s, f, Z, Z)

    As = [MFHC.laminate_strain_localization(C6s[i], Ch) for i in 1:3]
    Bs = [MFHC.laminate_stress_localization(C6s[i], Ch) for i in 1:3]

    @test sum(f[i] * As[i] for i in 1:3) ≈ I atol = 1.0e-10
    @test sum(f[i] * Bs[i] for i in 1:3) ≈ I atol = 1.0e-10
    # The macroscopic in-plane strain reaches every layer unchanged.
    for A in As
        @test MFHC._ip_block(A) ≈ I atol = ATOL_KM
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
    As = [MFHC.laminate_gradient_localization(K3a[i], Kha) for i in 1:3]
    Bs = [MFHC.laminate_flux_localization(K3a[i], Kha) for i in 1:3]
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
