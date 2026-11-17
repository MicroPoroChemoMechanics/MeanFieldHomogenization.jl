using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using JSON

# =============================================================================
#  Multipole expansion method — penny-shaped cracks.
#
#  References: one insulating crack in a ball closed by Maxwell's scheme, whose
#  contribution −(8/3) k₀ a³/V e₃⊗e₃ is exact; the two ways of translating a
#  harmonic agree where both hold; Kushch (2013), Tables 6.3 and 6.4, for cracks
#  on orthogonal and body-centered lattices.
# =============================================================================

const _MCR = MeanFieldHomogenization.Assemblies
const _MCR_K0 = Dict(:K => TensISO{3}(1.0))
_mcr_literature() = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2013.json"))
_mcr_k33(asm, nmax) = canonical_components(homogenize(asm, MultipoleExpansion(; nmax), :K))[3, 3]

function _mcr_single(a, R)
    asm = ParticleAssembly(; boundary = MixedBC(R))
    add_matrix!(asm, _MCR_K0)
    add_particle!(asm, :c, (0.0, 0.0, 0.0), EllipticCrack(a, a), Dict{Symbol, Any}())
    return asm
end

# k₀ + σ/(1 − σ/(3k₀)) with σ = −(8/3) k₀ a³/V: the dilute contribution of the
# crack closed by Maxwell's scheme on a ball of volume V.
function _mcr_maxwell(a, R)
    σ = -8 * a^3 / (3 * (4π * R^3 / 3))
    return 1 + σ / (1 - σ / 3)
end

@testset "MEM cracks — one crack in a ball is Maxwell's estimate at any degree" begin
    for nmax in (1, 4)
        K = canonical_components(homogenize(_mcr_single(0.7, 2.0), MultipoleExpansion(; nmax), :K))
        @test K[3, 3] ≈ _mcr_maxwell(0.7, 2.0) rtol = 1.0e-12
        @test K[1, 1] ≈ 1 && K[2, 2] ≈ 1
        @test norm(K - Diagonal(diag(K))) ≤ 1.0e-14
    end
    # the derivative along the radius of the crack
    @test ForwardDiff.derivative(a -> _mcr_k33(_mcr_single(a, 2.0), 2), 0.7) ≈
        ForwardDiff.derivative(a -> _mcr_maxwell(a, 2.0), 0.7) rtol = 1.0e-10
end

@testset "MEM cracks — the near and far translations agree where both hold" begin
    K₀ = TensISO{3}(1.0)
    nmax = 6
    crack = _MCR._mem_spheroid_data(EllipticCrack(0.5, 0.5), K₀, nmax)
    spheroid = _MCR._mem_spheroid_data(LayeredSpheroid((0.2,), (0.5,), (TensISO{3}(10.0),)), K₀, nmax)
    # the crack reads the odd harmonics only, those of k − |l| odd
    odd = [_MCR._mem_index(k, l) for k in 1:nmax for l in (-k):k if isodd(k - abs(l))]
    for p in (crack, spheroid), R in ([0.6, 1.0, 1.2], [0.0, 0.0, 1.6], [1.7, 0.0, 0.3])
        near = _MCR._mem_eta_near(crack, p, R, nmax)
        far = _MCR._mem_eta_far(crack, p, irregular_solid_harmonics(R, 2 * (nmax + _MCR._MEM_FAR_TERMS)), nmax)
        @test norm(near[odd, :] - far[odd, :]) ≤ 1.0e-9 * norm(far[odd, :])
        @test iszero(near[setdiff(axes(near, 1), odd), :])
    end
end

@testset "MEM cracks — overlaps and refusals" begin
    function cell(particles...)
        asm = ParticleAssembly(; boundary = PeriodicBox((3.0, 3.0, 0.5)))
        add_matrix!(asm, _MCR_K0)
        for (k, (x, g)) in enumerate(particles)
            add_particle!(asm, Symbol(:p, k), x, g, g isa EllipticCrack ? Dict{Symbol, Any}() : Dict(:K => TensISO{3}(2.0)))
        end
        return asm
    end
    disk(a) = EllipticCrack(a, a)
    # normal e₁: the columns of the basis are ℓ, m, n
    upright(a) = EllipticCrack(a, a, [0.0 0.0 1.0; 0.0 1.0 0.0; -1.0 0.0 0.0])
    # parallel cracks in distinct planes never touch, however close
    asm = cell(((0.0, 0.0, 0.0), disk(1.0)), ((0.5, 0.0, 0.1), disk(1.0)))
    @test validate_assembly(asm) === asm
    @test particle_volume_fraction(asm, :p1) == 0
    # coplanar ones do when their disks overlap
    @test_throws ArgumentError validate_assembly(cell(((0.0, 0.0, 0.0), disk(0.7)), ((1.2, 0.0, 0.0), disk(0.7))))
    @test validate_assembly(cell(((0.0, 0.0, 0.0), disk(0.7)), ((1.5, 0.0, 0.0), disk(0.7)))) isa ParticleAssembly
    # a crack across a spheroid, and two crossing cracks
    @test_throws ArgumentError validate_assembly(cell(((0.0, 0.0, 0.0), disk(0.7)), ((0.5, 0.0, 0.0), Ellipsoid(0.3, 0.3, 0.2))))
    @test_throws ArgumentError validate_assembly(cell(((0.0, 0.0, 0.0), disk(0.7)), ((0.3, 0.0, 0.0), upright(0.7))))
    @test validate_assembly(cell(((0.0, 0.0, 0.0), disk(0.7)), ((0.0, 0.0, 0.3), Ellipsoid(0.3, 0.3, 0.2)))) isa ParticleAssembly
    @test validate_assembly(cell(((0.0, 0.0, 0.0), disk(0.7)), ((1.5, 0.0, 0.0), upright(0.7)))) isa ParticleAssembly
    # the expansion takes penny-shaped cracks of normal e₃
    @test_throws ArgumentError homogenize(cell(((0.0, 0.0, 0.0), upright(0.7))), MultipoleExpansion(; nmax = 2), :K)
    @test_throws ArgumentError homogenize(cell(((0.0, 0.0, 0.0), EllipticCrack(0.7, 0.4))), MultipoleExpansion(; nmax = 2), :K)
    # the schemes of one polarization per inclusion have no use for a crack
    asm = cell(((0.0, 0.0, 0.0), disk(0.7)))
    @test_throws ArgumentError homogenize(asm, ClusterModel(), :K)
    @test_throws ArgumentError homogenize(asm, EquivalentInclusion(), :K)
end

# Kushch (2013), Tables 6.3 and 6.4: cracks of unit radius on the lattice of
# periods (a₁, a₁, a₃), at the origin, and at (a₁/2, a₁/2, a₃/2) for the
# body-centered array.
function _mcr_lattice(a₁, a₃; centered = false)
    asm = ParticleAssembly(; boundary = PeriodicBox((a₁, a₁, a₃)))
    add_matrix!(asm, _MCR_K0)
    add_particle!(asm, :a, (0.0, 0.0, 0.0), EllipticCrack(1.0, 1.0), Dict{Symbol, Any}())
    centered && add_particle!(asm, :b, (a₁ / 2, a₁ / 2, a₃ / 2), EllipticCrack(1.0, 1.0), Dict{Symbol, Any}())
    return asm
end

@testset "MEM cracks — Kushch (2013), Tables 6.3 and 6.4" begin
    lit = _mcr_literature()["tables"]
    # printed with three decimals, recovered to one unit of the third
    for (key, centered, ratio, column) in (
            ("conductivity_cracks_so", false, 1.0, "a3_over_a1_1"),
            ("conductivity_cracks_so", false, 0.5, "a3_over_a1_1_2"),
            ("conductivity_cracks_bc", true, 2.0, "a3_over_a1_2"),
            ("conductivity_cracks_bc", true, 1.0, "a3_over_a1_1"),
        )
        for row in lit[key]
            row["a1"] in (5.0, 3.0) || continue
            a₁ = row["a1"]
            @test _mcr_k33(_mcr_lattice(a₁, ratio * a₁; centered), 8) ≈ row[column] atol = 1.0e-3
        end
    end
    # Table 6.3 prints 0.932 at a₁ = 4, a₃/a₁ = 1/2, the digits of 0.923
    # transposed: the body-centered array of the same crack density 1/32
    # (Table 6.4, a₁ = 4, a₃/a₁ = 1) gives 0.919, the density governing the
    # conductivity of well separated cracks
    so = _mcr_k33(_mcr_lattice(4.0, 2.0), 8)
    bc = only(row["a3_over_a1_1"] for row in lit["conductivity_cracks_bc"] if row["a1"] == 4.0)
    @test so ≈ 0.923 atol = 1.0e-3
    @test abs(so - bc) < 5.0e-3
end

# ─── Elasticity ──────────────────────────────────────────────────────────────

const _MCR_NU0 = 0.25
_mcr_iso(κ, μ) = TensISO{3}(3κ, 2μ)
const _MCR_C0 = Dict(:C => _mcr_iso(2 * (1 + _MCR_NU0) / (3 * (1 - 2 * _MCR_NU0)), 1.0))
_mcr_mandel(C) = TensND.mandel(C, TensND.CanonicalBasis{3, Float64}())

@testset "MEM cracks, elasticity — one crack in a ball is Maxwell's estimate" begin
    a, R = 0.7, 2.0
    asm = ParticleAssembly(; boundary = MixedBC(R))
    add_matrix!(asm, _MCR_C0)
    add_particle!(asm, :c, (0.0, 0.0, 0.0), EllipticCrack(a, a), Dict{Symbol, Any}())
    # the cracked RVE of the same density, by the crack code of the package
    rve = RVE(; distribution_shape = Ellipsoid(1.0))
    add_phase!(rve, :M, Ellipsoid(1.0), _MCR_C0; fraction = :rest)
    add_phase!(rve, :F, PennyCrack(1.0), _MCR_C0; density = a^3 / (4π * R^3 / 3))
    ref = _mcr_mandel(homogenize(rve, Maxwell()))
    for nmax in (2, 4)
        @test _mcr_mandel(homogenize(asm, MultipoleExpansion(; nmax), :C)) ≈ ref rtol = 1.0e-12
    end
    # the radius of a crack is a plain number in elasticity
    asmd = ParticleAssembly(; boundary = MixedBC(R))
    add_matrix!(asmd, _MCR_C0)
    add_particle!(asmd, :c, (0.0, 0.0, 0.0), EllipticCrack(ForwardDiff.Dual(a, 1.0), a), Dict{Symbol, Any}())
    @test_throws ArgumentError homogenize(asmd, MultipoleExpansion(; nmax = 2), :C)
end

@testset "MEM cracks, elasticity — the near and far readings agree where both hold" begin
    ν₀ = _MCR_NU0
    κ₀ = 2 * (1 + ν₀) / (3 * (1 - 2ν₀))
    nmax = 4
    crack = _MCR._mem_elastic_spheroid(EllipticCrack(1.0, 1.0), :c, κ₀, 1.0, ν₀, nmax)
    spheroid = _MCR._mem_elastic_spheroid(LayeredSpheroid((0.3,), (1.0,), (_mcr_iso(5.0, 3.0),)), :s, κ₀, 1.0, ν₀, nmax)
    norms = [solid_harmonic_norm(k, m) for k in 0:(2 * crack.K + 2) for m in (-k):k]
    n = _MCR._mem_eblock(nmax)
    # the crack reads only the traction on its plane, so its received
    # coefficients are known up to what it does not answer: compare what it
    # sends; the residual is the truncation of the near reading at this degree
    for (q, p) in ((crack, crack), (spheroid, crack), (crack, spheroid)), r in ([0.0, 0.0, 6.0], [6.5, 0.0, 0.0], [3.0, 2.0, 5.0])
        near = _MCR._mem_elastic_near(q, p, r, nmax)
        M = zeros(ComplexF64, n, n)
        _MCR._mem_couple_far!(M, 1:n, 1:n, MixedBC(100.0), (irregular_solid_harmonics(r, nmax + p.K), r), q, p, ν₀, norms, nmax)
        @test norm(q.Ẑ * (near + M)) ≤ 3.0e-2 * norm(q.Ẑ * M)
    end
end

# Kushch (2013), Tables 7.7 to 7.10: E*₃₃/E₀ = 1/(E₀ D₃₃₃₃) and μ*₁₃/μ₀ = 1/(4 D₁₃₁₃)
# of cracks of unit radius on the lattice of periods (a, a, a/ā), ν₀ = 0.25.
function _mcr_elastic_lattice(a, ā; centered = false)
    a₃ = a / ā
    asm = ParticleAssembly(; boundary = PeriodicBox((a, a, a₃)))
    add_matrix!(asm, _MCR_C0)
    add_particle!(asm, :a, (0.0, 0.0, 0.0), EllipticCrack(1.0, 1.0), Dict{Symbol, Any}())
    centered && add_particle!(asm, :b, (a / 2, a / 2, a₃ / 2), EllipticCrack(1.0, 1.0), Dict{Symbol, Any}())
    D = inv(_mcr_mandel(homogenize(asm, MultipoleExpansion(; nmax = 4), :C)))
    E₀ = 2 * (1 + _MCR_NU0)
    return 1 / (E₀ * D[3, 3]), 1 / (2 * D[5, 5])
end

@testset "MEM cracks, elasticity — Kushch (2013), Tables 7.7 to 7.10" begin
    lit = _mcr_literature()["tables"]
    for (centered, ā, column, a) in ((false, 1.0, "a_over_a3_1", 5.0), (false, 4.0, "a_over_a3_4", 3.0), (true, 1.0, "a_over_a3_1", 3.0))
        E, μ = _mcr_elastic_lattice(a, ā; centered)
        tag = centered ? "bc" : "so"
        @test E ≈ only(r[column] for r in lit["young_cracks_$tag"] if r["a"] == a) atol = 1.0e-3
        @test μ ≈ only(r[column] for r in lit["shear_cracks_$tag"] if r["a"] == a) atol = 1.0e-3
    end
end
