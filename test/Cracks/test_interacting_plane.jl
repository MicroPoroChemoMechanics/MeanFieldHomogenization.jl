# =============================================================================
#  test_interacting_plane.jl — stress intensity factors of interacting cracks
#  of a plane by the method of average tractions (`sif` on `center => crack`).
#
#  References, independent of the code under test:
#  - one crack: the factor of an isolated crack, √(πb) Σ·n;
#  - two collinear cracks: the closed form (6) of Kachanov (1987), with the
#    complete elliptic integrals, and the columns of his Table 1
#    (data/literature/kachanov1987.json);
#  - the transmission factor between two collinear cracks of the periodic row,
#    Λ⁽ᵏ⁾ of the same article (p. 27), read on the mean traction 1/(1 − Λ);
#  - invariance under a rotation of the whole body;
#  - an anisotropic plane: an isotropic stiffness given as a generic tensor
#    recovers Westergaard's fields; an isolated crack has √(πb) Σ·n whatever
#    the anisotropy; collinear cracks are those of the isotropic plane (Mauge
#    and Kachanov 1994); the rotation of the body, of its stiffness included.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using JSON
using MeanFieldHomogenization.Elliptic: ell_K, ell_E

const C0_IP = TensISO{2}(2.0, 1.0)
_ip_tension(θ) = (R = [cos(θ) -sin(θ); sin(θ) cos(θ)]; Tens(R * [0.0 0.0; 0.0 1.0] * transpose(R)))

# Kachanov (1987), eq. (6): two collinear cracks (−1, −k) and (k, 1).
function _ip_eq6(k)
    Λ = sqrt(2(1 + k)) / (1 + sqrt(k)) - 1
    m = 1 - k^2
    K, E = ell_K(m), ell_E(m)
    outer = 1 + (2E - k * (k + 1) * K - π / 2 * (1 - k)) / ((1 - Λ) * π * (1 - k))
    inner = 1 + (-2E + (k + 1) * K - π / 2 * (1 - k)) / ((1 - Λ) * π * (1 - k))
    return (; inner, outer, mean = 1 / (1 - Λ))
end
_ip_pair(k; θ = 0.0) = (
    c = (1 + k) / 2; l = (1 - k) / 2; R = [cos(θ) -sin(θ); sin(θ) cos(θ)];
    [R * [-c, 0.0] => RibbonCrack(l; angle = θ), R * [c, 0.0] => RibbonCrack(l; angle = θ)]
)

@testset "interacting cracks of a plane: one crack" begin
    Σ = Tens([0.3 0.4; 0.4 1.1])
    crack = RibbonCrack(0.7; angle = 0.6)
    r = only(sif([[0.2, -0.5] => crack], C0_IP, Σ))
    _, (KI, KII) = sif(crack, C0_IP, Σ)
    @test r.plus[1] ≈ KI && r.minus[1] ≈ KI
    @test r.plus[2] ≈ KII && r.minus[2] ≈ KII
end

@testset "interacting cracks of a plane: two collinear cracks" begin
    lit = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kachanov1987.json"))
    for row in lit["tables"]["two_collinear"]
        k = row["k"]
        K0 = sqrt(π * (1 - k) / 2)
        r = sif(_ip_pair(k), C0_IP, _ip_tension(0.0))
        ref = _ip_eq6(k)
        @test r[2].minus[1] / K0 ≈ ref.inner rtol = 1.0e-10           # inner tip of the right crack
        @test r[2].plus[1] / K0 ≈ ref.outer rtol = 1.0e-10
        @test r[1].plus[1] ≈ r[2].minus[1] rtol = 1.0e-10              # symmetry
        @test r[1].traction[1] ≈ ref.mean rtol = 1.0e-10
        @test abs(r[2].minus[2]) < 1.0e-10                             # no sliding
        # the column of the article, printed with three decimals
        @test r[2].minus[1] / K0 ≈ row["inner_method"] atol = 6.0e-4
        @test r[2].plus[1] / K0 ≈ row["outer_method"] atol = 6.0e-4
    end
    # the same body turned by an angle, under the turned load
    r0, r1 = sif(_ip_pair(0.05), C0_IP, _ip_tension(0.0)), sif(_ip_pair(0.05; θ = 0.9), C0_IP, _ip_tension(0.9))
    @test r1[2].minus[1] ≈ r0[2].minus[1] rtol = 1.0e-10
    @test abs(r1[2].minus[2]) < 1.0e-10
    # the derivative with respect to the spacing, against that of eq. (6)
    f(k) = sif(_ip_pair(k), C0_IP, _ip_tension(0.0))[2].minus[1] / sqrt(π * (1 - k) / 2)
    @test ForwardDiff.derivative(f, 0.1) ≈ ForwardDiff.derivative(k -> _ip_eq6(k).inner, 0.1) rtol = 1.0e-7
end

@testset "interacting cracks of a plane: transmission factor of the periodic row" begin
    l = 0.5
    for (ζ, n) in ((1.5, 1), (1.5, 3), (2.0, 2))
        h = 2l * ζ
        r = sif([[0.0, 0.0] => RibbonCrack(l; angle = 0.0), [n * h, 0.0] => RibbonCrack(l; angle = 0.0)], C0_IP, _ip_tension(0.0))
        x = n * ζ
        Λ = x * (sqrt(1 + 1 / x) - sqrt(1 - 1 / x)) - 1
        @test r[1].traction[1] ≈ 1 / (1 - Λ) rtol = 1.0e-10
    end
end

@testset "interacting cracks of an anisotropic plane" begin
    C_gen = Tens(Array(get_array(TensND.express(C0_IP, TensND.CanonicalBasis{2, Float64}()))))
    C_ort = from_mandel([10.0 2.0 0.0; 2.0 3.0 0.0; 0.0 0.0 3.0])
    Σ = Tens([0.2 0.3; 0.3 1.0])
    cfg = [[0.0, 0.0] => RibbonCrack(0.5; angle = 0.3), [0.4, 0.9] => RibbonCrack(0.4; angle = 1.1)]
    a, b = sif(cfg, C0_IP, Σ), sif(cfg, C_gen, Σ)
    @test all(a[k].plus ≈ b[k].plus && a[k].minus ≈ b[k].minus for k in 1:2)
    one_ = only(sif([[0.3, 0.1] => RibbonCrack(0.6; angle = 0.7)], C_ort, Σ))
    _, (KI, KII) = sif(RibbonCrack(0.6; angle = 0.7), C0_IP, Σ)
    @test one_.plus ≈ [KI, KII] && one_.minus ≈ [KI, KII]
    for θ in (0.0, 0.6)
        @test sif(_ip_pair(0.1; θ), C_ort, _ip_tension(θ))[2].minus ≈ sif(_ip_pair(0.1; θ), C0_IP, _ip_tension(θ))[2].minus atol = 1.0e-12
    end
    # stacked cracks see the anisotropy
    stack = [[0.0, 0.0] => RibbonCrack(0.5; angle = 0.0), [0.0, 0.4] => RibbonCrack(0.5; angle = 0.0)]
    @test abs(sif(stack, C_ort, _ip_tension(0.0))[1].plus[1] - sif(stack, C0_IP, _ip_tension(0.0))[1].plus[1]) > 1.0e-3
    # the whole body turned, its stiffness included
    θ = 0.8
    R = [cos(θ) -sin(θ); sin(θ) cos(θ)]
    C_rot = Tens(Array(get_array(TensND.express(C_ort, TensND.RotatedBasis(-θ)))))
    turned = [R * [0.0, 0.0] => RibbonCrack(0.5; angle = 0.3 + θ), R * [0.4, 0.9] => RibbonCrack(0.4; angle = 1.1 + θ)]
    r0, r1 = sif(cfg, C_ort, Σ), sif(turned, C_rot, Tens(R * Matrix(canonical_components(Σ)) * transpose(R)))
    @test all(r0[k].plus ≈ r1[k].plus && r0[k].minus ≈ r1[k].minus for k in 1:2)
    # dual numbers through the roots of the anisotropic material, against a
    # centered difference as the reference of the test
    g(e) = sif(stack, from_mandel([e 2.0 0.0; 2.0 3.0 0.0; 0.0 0.0 3.0]), _ip_tension(0.0))[1].plus[1]
    h = 1.0e-4
    @test ForwardDiff.derivative(g, 10.0) ≈ (g(10.0 + h) - g(10.0 - h)) / (2h) rtol = 1.0e-6
end

@testset "interacting cracks of a plane: sliding, and a crack out of the plane" begin
    # two parallel cracks under shear: the stack shields, sliding only
    pair = [[0.0, 0.0] => RibbonCrack(0.5; angle = 0.0), [0.0, 0.3] => RibbonCrack(0.5; angle = 0.0)]
    r = sif(pair, C0_IP, Tens([0.0 1.0; 1.0 0.0]))
    @test 0 < abs(r[1].plus[2]) < sqrt(π * 0.5)
    # a crack is a segment of the plane
    @test_throws ArgumentError sif([[0.0, 0.0] => RibbonCrack(0.5; euler_angles = (0.3, 0.0, 0.0))], C0_IP, Tens([0.0 1.0; 1.0 0.0]))
end
