using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using JSON

# =============================================================================
#  Elastic confocal spheroid with a spring interface.
#
#  References: the sphere with the same spring (`LayeredSphere`, closed form),
#  the perfect interface as the limit of a stiff spring, and the Maxwell
#  estimates of Kushch (2018), Table 2, for oblate and prolate spheroids.
# =============================================================================

const _SPS = MeanFieldHomogenization.LayeredSpheroids
_sps_C(μ, ν) = TensISO{3}(2μ * (1 + ν) / (1 - 2ν), 2μ)
_sps_mandel(A) = mandel(A, CanonicalBasis{3, Float64}())

@testset "spring spheroid — limits" begin
    C₀, C₁ = _sps_C(1.0, 0.3), _sps_C(20.0, 0.2)
    for (axis, disk) in ((0.4, 1.0), (1.0, 0.4))
        perfect = LayeredSpheroid((axis,), (disk,), (C₁,))
        stiff = LayeredSpheroid((axis,), (disk,), (C₁,); interfaces = (SpringInterface(1.0e10, 1.0e10),))
        A = _sps_mandel(gradient_gradient_concentration(stiff, C₁, C₀))
        @test A ≈ _sps_mandel(gradient_gradient_concentration(perfect, C₁, C₀)) rtol = 1.0e-7
        N = _sps_mandel(property_contribution(stiff, C₁, C₀))
        @test N ≈ _sps_mandel(property_contribution(perfect, C₁, C₀)) rtol = 1.0e-7
    end
    # a nearly spherical spheroid against the sphere with the same spring
    R, ε = 1.0, 1.0e-4
    spring = SpringInterface(2.0, 0.5)
    sphere = LayeredSphere((R,), (C₁,); interfaces = (spring,))
    spheroid = LayeredSpheroid((R * (1 + ε),), (R,), (C₁,); interfaces = (spring,))
    Ns = _sps_mandel(property_contribution(sphere, C₁, C₀))
    Nd = _sps_mandel(property_contribution(spheroid, C₁, C₀))
    @test norm(Nd - Ns) ≤ 10 * ε * norm(Ns)
end

@testset "spring spheroid — convergence and symmetry" begin
    C₀, C₁ = _sps_C(1.0, 0.3), _sps_C(100.0, 0.2)
    s = LayeredSpheroid((1 / 3,), (1.0,), (C₁,); interfaces = (SpringInterface(10.0, 10.0),))
    N = _sps_mandel(property_contribution(s, C₁, C₀))
    # the exact contribution is major-symmetric; the truncation shows as asymmetry
    @test norm(N - transpose(N)) ≤ 1.0e-6 * norm(N)
    N8 = _sps_mandel(property_contribution(s, C₁, C₀; D = 8))
    @test norm(N8 - N) ≤ 1.0e-3 * norm(N)
    @test norm(N8 - transpose(N8)) > norm(N - transpose(N))
    # two layers: an inner spring opens inside the particle, an outer one outside it
    inner = LayeredSpheroid((0.3, 0.5), (0.8, sqrt(0.8)), (C₁, _sps_C(5.0, 0.25)); interfaces = (SpringInterface(3.0, 1.0), PerfectInterface{Float64}()))
    r = _SPS.spheroid_layer_strain_concentration(inner, C₀)
    sum_layers = sum(layer_volume_fraction(inner, k) * _sps_mandel(r.layers[k]) for k in 1:2)
    @test norm(_sps_mandel(r.A) - sum_layers) > 1.0e-3 * norm(sum_layers)
    @test_throws ArgumentError _SPS.spheroid_strain_concentration(
        LayeredSpheroid((0.4,), (1.0,), (C₁,); interfaces = (MembraneInterface(1.0, 1.0),)), C₀
    )
end

@testset "spring spheroid — Kushch (2018), Table 2" begin
    lit = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2018.json"))
    q(k) = lit["quantities"][k]["value"]
    C₀ = _sps_C(1.0, q("nu0"))
    C₁ = _sps_C(q("mu_ratio"), q("nu1"))
    c = q("fraction")
    for row in lit["tables"]["maxwell_aligned"]
        e = row["e"]
        e == 1 && continue                  # the sphere: `LayeredSphere`
        D = row["D"] / max(1.0, e)          # D̃ = D max(l₁, l₃)/μ₀ with l₁ = 1
        spheroid = LayeredSpheroid((e,), (1.0,), (C₁,); interfaces = (SpringInterface(D, D),))
        asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid(1.0, 1.0, e) |> _ -> Ellipsoid(1 / cbrt(c), 1 / cbrt(c), e / cbrt(c))))
        add_matrix!(asm, Dict(:C => C₀))
        add_particle!(asm, :p, (0.0, 0.0, 0.0), spheroid, Dict(:C => C₁))
        M = _sps_mandel(homogenize(asm, MoriTanaka(), :C))
        @test M[1, 1] ≈ row["C11"] atol = 2.0e-4
        @test M[1, 2] ≈ row["C12"] atol = 2.0e-4
        @test M[1, 3] ≈ row["C13"] atol = 2.0e-4
        @test M[3, 3] ≈ row["C33"] atol = 2.0e-4
        @test M[4, 4] / 2 ≈ row["C44"] atol = 2.0e-4
    end
end

@testset "spring spheroid — derivative with respect to the stiffness" begin
    C₀, C₁ = _sps_C(1.0, 0.3), _sps_C(20.0, 0.2)
    n33(k) = _sps_mandel(property_contribution(LayeredSpheroid((0.5,), (1.0,), (C₁,); interfaces = (SpringInterface(k, 2k),)), C₁, C₀; D = 8))[3, 3]
    h = 1.0e-5
    @test ForwardDiff.derivative(n33, 3.0) ≈ (n33(3.0 + h) - n33(3.0 - h)) / 2h rtol = 1.0e-6
end
