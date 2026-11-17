using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using JSON

# =============================================================================
#  Multipole expansion method — elasticity, finite and periodic assemblies of spheres.
# =============================================================================

_mel_C(μ, ν) = TensISO{3}(2μ * (1 + ν) / (1 - 2ν), 2μ)        # (3k, 2μ)
_mel_mandel(C) = mandel(C, CanonicalBasis{3, Float64}())
_mel_rel(a, b) = norm(_mel_mandel(a) - _mel_mandel(b)) / norm(_mel_mandel(b))

const _MEL_C0 = _mel_C(1.0, 0.3)

@testset "MEM elasticity — one sphere is Maxwell and Mori-Tanaka at any degree" begin
    for C₁ in (_mel_C(100.0, 0.2), _mel_C(0.05, 0.4), TensISO{3}(0.0, 0.0)), nmax in (2, 4)
        asm = ParticleAssembly(; boundary = MixedBC(2.0))
        add_matrix!(asm, Dict(:C => _MEL_C0))
        add_particle!(asm, :p, (0.3, -0.2, 0.1), Ellipsoid(1.1), Dict(:C => C₁))
        C = homogenize(asm, MultipoleExpansion(; nmax), :C)
        @test _mel_rel(C, homogenize(asm, MoriTanaka(), :C)) < 1.0e-11
    end
end

# Simple cubic block of n³ equal spheres of total fraction c inside the ball of
# the same fraction (the finite-cluster model of kushch2013, appendix A.3).
function _mel_cubic_cluster(n, c, C₁; a = 1.0)
    R = a * cbrt(3c / (4π))
    asm = ParticleAssembly(; boundary = MixedBC(R * cbrt(n^3 / c)))
    add_matrix!(asm, Dict(:C => _MEL_C0))
    o = (n - 1) / 2
    for i in 1:n, j in 1:n, k in 1:n
        add_particle!(
            asm, Symbol(:p, i, j, k), (a * (i - 1 - o), a * (j - 1 - o), a * (k - 1 - o)),
            Ellipsoid(R), Dict(:C => C₁)
        )
    end
    return asm
end

@testset "MEM elasticity — finite cluster of Kushch (2013), appendix A.3" begin
    # 27 nearly rigid spheres on a 3×3×3 cubic block, fraction 0.2, ν = 0.3:
    # k*/k₀ = 1.4064, μ₁*/μ₀ = 1.6568 (C₁₁₁₁ − C₁₁₂₂)/2, μ₂*/μ₀ = 1.4719 (C₁₂₁₂)
    q = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2013.json"))["quantities"]
    value(name) = q[name]["value"]
    ν = value("fcm_poisson_ratio")
    @test _MEL_C0 == _mel_C(1.0, ν)
    asm = _mel_cubic_cluster(3, value("fcm_volume_fraction"), _mel_C(value("fcm_shear_contrast"), ν))
    k₀ = 2 * (1 + ν) / (3 * (1 - 2ν))
    M = _mel_mandel(homogenize(asm, MultipoleExpansion(; nmax = 5), :C))
    @test (M[1, 1] + 2 * M[1, 2]) / (3 * k₀) ≈ value("fcm_bulk_modulus") atol = 1.0e-4
    @test (M[1, 1] - M[1, 2]) / 2 ≈ value("fcm_shear_modulus_1") atol = 6.0e-4
    @test M[4, 4] / 2 ≈ value("fcm_shear_modulus_2") atol = 2.0e-4
    # cubic symmetry of the block, which a one-site estimate does not see
    @test M[1, 1] ≈ M[2, 2] ≈ M[3, 3] && M[4, 4] ≈ M[5, 5] ≈ M[6, 6]
    @test M ≈ transpose(M) rtol = 1.0e-13
    @test abs((M[1, 1] - M[1, 2]) / 2 - M[4, 4] / 2) > 0.1
    @test _mel_mandel(homogenize(asm, :mem, :C)) ≈ _mel_mandel(homogenize(asm, MultipoleExpansion(), :C))
end

# Three unequal spheres of unequal stiffnesses, in the ball of radius 3.
function _mel_three_spheres(centers; C = (_mel_C(20.0, 0.2), TensISO{3}(0.0, 0.0), _mel_C(0.2, 0.4)), R = (0.5, 0.4, 0.6))
    asm = ParticleAssembly(; boundary = MixedBC(3.0))
    add_matrix!(asm, Dict(:C => _MEL_C0))
    for i in 1:3
        add_particle!(asm, Symbol(:p, i), centers[i], Ellipsoid(R[i]), Dict(:C => C[i]))
    end
    return asm
end

const _MEL_CENTERS = ([0.0, 0.0, 0.0], [1.0, 0.2, -0.1], [-0.3, 1.2, 0.4])

@testset "MEM elasticity — frame indifference and silent spheres" begin
    scheme = MultipoleExpansion(; nmax = 3)
    C = homogenize(_mel_three_spheres(_MEL_CENTERS), scheme, :C)
    θ, φ = 0.7, -1.2
    Q = [cos(φ) -sin(φ) 0; sin(φ) cos(φ) 0; 0 0 1] * [1 0 0; 0 cos(θ) -sin(θ); 0 sin(θ) cos(θ)]
    Cq = homogenize(_mel_three_spheres(map(c -> Q * c, _MEL_CENTERS)), scheme, :C)
    # the rotated arrangement has the rotated stiffness: compared on the energy of
    # a strain E and of the strain Qᵀ·E·Q seen from the first arrangement
    E = [0.3 -0.2 0.5; -0.2 1.1 0.4; 0.5 0.4 -0.7]
    energy(C, E) = (e = Tens(E); e ⊡ (C ⊡ e))
    @test energy(Cq, E) ≈ energy(C, transpose(Q) * E * Q) rtol = 1.0e-10
    @test !(energy(C, E) ≈ energy(C, transpose(Q) * E * Q))
    # a sphere of the matrix stiffness sends nothing back
    asm = _mel_three_spheres(_MEL_CENTERS; C = (_mel_C(20.0, 0.2), _MEL_C0, _mel_C(0.2, 0.4)))
    ref = ParticleAssembly(; boundary = MixedBC(3.0))
    add_matrix!(ref, Dict(:C => _MEL_C0))
    add_particle!(ref, :p1, _MEL_CENTERS[1], Ellipsoid(0.5), Dict(:C => _mel_C(20.0, 0.2)))
    add_particle!(ref, :p3, _MEL_CENTERS[3], Ellipsoid(0.6), Dict(:C => _mel_C(0.2, 0.4)))
    @test _mel_rel(homogenize(asm, scheme, :C), homogenize(ref, scheme, :C)) < 1.0e-12
end

@testset "MEM elasticity — layers and interfaces: one sphere against the layered sphere" begin
    # One sphere in a ball: the scheme is Maxwell's, whose contribution tensor is
    # that of the layered sphere under a uniform strain (the bulk and shear
    # recurrences), obtained here from the degrees 0 and 2 of the response at any
    # degree. Spring, membrane and coating are compared with it.
    core, shell = _mel_C(30.0, 0.2), _mel_C(0.3, 0.35)
    spheres = (
        LayeredSphere((0.9,), (core,); interfaces = (SpringInterface(; sn = 0.2, st = 0.5),)),
        LayeredSphere((0.9,), (core,); interfaces = (MembraneInterface(1.5, 0.8),)),
        LayeredSphere((0.5, 0.9), (core, shell)),
        LayeredSphere(
            (0.5, 0.9), (core, shell);
            interfaces = (SpringInterface(; sn = 0.1, st = 0.3), MembraneInterface(0.7, 1.2))
        ),
    )
    for sphere in spheres
        asm = ParticleAssembly(; boundary = MixedBC(2.0))
        add_matrix!(asm, Dict(:C => _MEL_C0))
        add_particle!(asm, :p, (0.0, 0.0, 0.0), sphere, Dict{Symbol, Any}())
        f = (0.9 / 2.0)^3
        N = property_contribution(sphere, _MEL_C0)
        maxwell = _MEL_C0 + (f * N) ⊡ inv(one(_MEL_C0) - hill_tensor(Ellipsoid(1.0), _MEL_C0) ⊡ (f * N))
        @test _mel_rel(homogenize(asm, MultipoleExpansion(; nmax = 3), :C), maxwell) < 1.0e-10
    end
end

# C₁₁₂₂/2μ₀, C₁₁₁₁/2μ₀ and C₁₂₁₂/μ₀ of a cubic array, the columns of kushch2013, Tables 5.1 to 5.3.
function _mel_cubic_moduli(asm, nmax)
    C = homogenize(asm, MultipoleExpansion(; nmax), :C)
    M = _mel_mandel(C)
    return (M[1, 2] / 2, M[1, 1] / 2, M[6, 6] / 2)
end

@testset "MEM elasticity — periodic arrays of Kushch (2013), Tables 5.1 to 5.3 and appendix A.4" begin
    lit = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2013.json"))
    ν = lit["quantities"]["fcm_poisson_ratio"]["value"]           # ν₀ = ν₁ = 0.3 in chapter 5 too
    M = Dict(:C => _mel_C(1.0, ν))
    for (kind, name) in ((:sc, "stiffness_sc"), (:bcc, "stiffness_bcc"), (:fcc, "stiffness_fcc")), row in lit["tables"][name]
        c = row["c"]
        c ≤ 0.4 || continue
        for μ₁ in (0, 10, 1000)
            kind === :sc && c == 0.4 && μ₁ == 1000 && continue     # resolved below at a higher degree
            asm = cubic_lattice(kind, M, Dict(:C => _mel_C(float(μ₁), ν)); fraction = c)
            ref = (row["c1122_contrast_$μ₁"], row["c1111_contrast_$μ₁"], row["c1212_contrast_$μ₁"])
            @test all(abs.(_mel_cubic_moduli(asm, 8) .- ref) .< 6.5e-4)
        end
    end
    # appendix A.4: simple cubic array, c = 0.4, μ₁/μ₀ = 1000, five decimals
    q = lit["quantities"]
    asm = cubic_lattice(:sc, M, Dict(:C => _mel_C(1000.0, ν)); fraction = 0.4)
    C = _mel_mandel(homogenize(asm, MultipoleExpansion(; nmax = 12), :C))
    @test C[1, 3] / 2 ≈ q["ruc_c1133"]["value"] rtol = 2.0e-5
    @test C[3, 3] / 2 ≈ q["ruc_c3333"]["value"] rtol = 2.0e-5
    @test C[6, 6] / 2 ≈ q["ruc_c1212"]["value"] rtol = 2.0e-5
    # cubic symmetry: the three directions and the three shears are alike
    @test C[1, 1] ≈ C[2, 2] rtol = 1.0e-12
    @test C[1, 1] ≈ C[3, 3] rtol = 1.0e-12
    @test C[4, 4] ≈ C[5, 5] rtol = 1.0e-12
    @test C[4, 4] ≈ C[6, 6] rtol = 1.0e-12
    @test norm(C[1:3, 4:6]) < 1.0e-12 * norm(C)
    # the same array described by a larger cell: 8 spheres in the cube of side 2
    one_cell = cubic_lattice(:sc, M, Dict(:C => _mel_C(10.0, ν)); fraction = 0.3, period = 1.0)
    R = particle_geometry(one_cell, :p1).semi_axes[1]
    big = ParticleAssembly(; boundary = PeriodicBox(2.0))
    add_matrix!(big, M)
    for i in 0:1, j in 0:1, k in 0:1
        add_particle!(big, Symbol(:p, i, j, k), (0.3 + i, -0.2 + j, 0.1 + k), Ellipsoid(R), Dict(:C => _mel_C(10.0, ν)))
    end
    @test _mel_rel(homogenize(big, MultipoleExpansion(; nmax = 4), :C), homogenize(one_cell, MultipoleExpansion(; nmax = 4), :C)) < 1.0e-10
end

@testset "MEM elasticity — derivatives" begin
    h = 1.0e-5
    function pair(z, μ)
        a = ParticleAssembly(; boundary = MixedBC(3.0))
        add_matrix!(a, Dict(:C => _MEL_C0))
        add_particle!(a, :a, [0.0, 0.0, z], Ellipsoid(0.8), Dict(:C => _mel_C(μ, 0.2)))
        add_particle!(a, :b, [0.0, 0.3, -0.9], Ellipsoid(0.7), Dict(:C => _mel_C(0.1, 0.4)))
        return mandel(homogenize(a, MultipoleExpansion(; nmax = 3), :C), CanonicalBasis{3, typeof(z * μ)}())[3, 3]
    end
    @test ForwardDiff.derivative(z -> pair(z, 12.0), 0.9) ≈ (pair(0.9 + h, 12.0) - pair(0.9 - h, 12.0)) / 2h rtol = 1.0e-6
    @test ForwardDiff.derivative(μ -> pair(0.9, μ), 12.0) ≈ (pair(0.9, 12.0 + h) - pair(0.9, 12.0 - h)) / 2h rtol = 1.0e-6
    # periodic array: the stiffness is homogeneous of degree 1 in the moduli and
    # does not depend on the size of the cell at a fixed fraction
    function lattice(λ, a)
        asm = cubic_lattice(:bcc, Dict(:C => λ * _MEL_C0), Dict(:C => λ * _mel_C(7.0, 0.2)); fraction = 0.3, period = a)
        return mandel(homogenize(asm, MultipoleExpansion(; nmax = 3), :C), CanonicalBasis{3, typeof(λ * a)}())
    end
    @test ForwardDiff.derivative(λ -> lattice(λ, 1.0), 1.0) ≈ lattice(1.0, 1.0) rtol = 1.0e-10
    @test norm(ForwardDiff.derivative(a -> lattice(1.0, a), 1.3)) < 1.0e-10 * norm(lattice(1.0, 1.3))
end

@testset "MEM elasticity — refusals" begin
    asm = ParticleAssembly(; boundary = MixedBC(3.0))
    add_matrix!(asm, Dict(:C => _MEL_C0))
    add_particle!(asm, :p, (0.0, 0.0, 0.0), Ellipsoid(0.5), Dict(:C => _mel_C(5.0, 0.2)))
    @test_throws ArgumentError concentration_tensors(asm, MultipoleExpansion(), :C)
    @test_throws ArgumentError multipole_coefficients(asm, :C)
    # spheroids are taken in conduction only
    spheroid = ParticleAssembly(; boundary = MixedBC(3.0))
    add_matrix!(spheroid, Dict(:C => _MEL_C0))
    add_particle!(spheroid, :p, (0.0, 0.0, 0.0), Ellipsoid(0.5, 0.5, 0.3), Dict(:C => _mel_C(5.0, 0.2)))
    @test_throws ArgumentError homogenize(spheroid, MultipoleExpansion(), :C)
    # the dipoles of elasticity are of degree 2
    @test_throws ArgumentError homogenize(asm, MultipoleExpansion(; nmax = 1), :C)
    # an incompressible matrix
    incompressible = ParticleAssembly(; boundary = MixedBC(3.0))
    add_matrix!(incompressible, Dict(:C => TensISO{3}(Inf, 2.0)))
    add_particle!(incompressible, :p, (0.0, 0.0, 0.0), Ellipsoid(0.5), Dict(:C => _mel_C(5.0, 0.2)))
    @test_throws ArgumentError homogenize(incompressible, MultipoleExpansion(), :C)
end
