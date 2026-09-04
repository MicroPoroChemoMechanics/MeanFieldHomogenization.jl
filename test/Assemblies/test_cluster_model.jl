using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

# =============================================================================
#  test_cluster_model.jl — the cluster model of Molinari & El Mouden (1996).
#
#  Coverage:
#   1. Appendix C, EXACT: a cluster reduced to its own receiver reproduces
#      Mori-Tanaka algebraically — one family and two families.
#   2. The isotropic part of 𝕋̄ vanishes, so the effective BULK modulus of any
#      cubic array equals the Mori-Tanaka one exactly, at every fraction and
#      for every lattice.
#   3. Cubic symmetry of the effective stiffness of a cubic array.
#   4. Convergence in the cluster radius, and the plateau beyond ≈ 2 periods
#      reported in their Fig. 3.
#   5. Sign of the correction: Mori-Tanaka OVERestimates the shear modulus of
#      a simple-cubic array of stiff spheres (their Fig. 6).
#   6. Effect of the spatial distribution: SC, BCC and FCC differ at equal
#      volume fraction (their Fig. 16).
#   7. Single-phase assembly returns the matrix property; Voigt/Reuss bracketing.
#   8. Conduction (`:K`) runs through the same kernel.
#   9. ForwardDiff through a particle radius and a particle coordinate.
# =============================================================================

const RTOL_EXACT = 1.0e-12
const ATOL_EXACT = 1.0e-12

_Cm() = TensISO{3}(3 * 1.0, 2 * 0.4)          # k = 1, μ = 0.4
_Ci() = TensISO{3}(3 * 10.0, 2 * 6.0)
_Cvoid() = TensISO{3}(3 * 1.0e-9, 2 * 1.0e-9)
_Crigid() = TensISO{3}(3 * 1.0e4, 2 * 1.0e4)

_mu(C) = get_array(C)[1, 2, 1, 2]
_kappa(C) = (A = get_array(C); sum(A[i, i, j, j] for i in 1:3, j in 1:3) / 9)

function _mt_rve(f, Ci; Cm = _Cm())
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => Cm); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:C => Ci); fraction = f)
    return rve
end

@testset "Cluster model — Molinari App. C: empty cluster ≡ Mori-Tanaka" begin
    # This is an algebraic identity, not a numerical coincidence: with no
    # neighbor in the cluster every 𝕋̄ vanishes and the block system becomes
    # the Mori-Tanaka one term by term.  Anything looser than machine
    # precision here means a bug in the assembly of 𝕄.
    for f in (0.05, 0.25, 0.45)
        asm = cubic_lattice(:sc, Dict(:C => _Cm()), Dict(:C => _Ci()); fraction = f, cutoff = 0.0)
        C_cl = get_array(homogenize(asm, ClusterModel(), :C))
        C_mt = get_array(homogenize(_mt_rve(f, _Ci()), MoriTanaka(), :C))
        @test maximum(abs.(C_cl .- C_mt)) < RTOL_EXACT * maximum(abs.(C_mt))
    end
    # Two families — the case the appendix actually treats.
    asm = cubic_lattice(
        :bcc, Dict(:C => _Cm()), [Dict(:C => _Ci()), Dict(:C => _Cvoid())];
        fraction = 0.2, cutoff = 0.0
    )
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => _Cm()); fraction = :rest)
    add_phase!(rve, :I1, Ellipsoid(1.0), Dict(:C => _Ci()); fraction = 0.1)
    add_phase!(rve, :I2, Ellipsoid(1.0), Dict(:C => _Cvoid()); fraction = 0.1)
    C_cl = get_array(homogenize(asm, ClusterModel(), :C))
    C_mt = get_array(homogenize(rve, MoriTanaka(), :C))
    @test maximum(abs.(C_cl .- C_mt)) < RTOL_EXACT * maximum(abs.(C_mt))
end

@testset "Cluster model — cubic arrays keep the Mori-Tanaka bulk modulus" begin
    # The interaction tensor has a strictly vanishing isotropic part, hence so
    # does 𝕋̄, hence the spherical part of the problem never sees the spatial
    # distribution.  Exact, for every lattice and every fraction.
    for kind in (:sc, :bcc, :fcc), f in (0.1, 0.3, 0.45)
        asm = cubic_lattice(kind, Dict(:C => _Cm()), Dict(:C => _Ci()); fraction = f, cutoff = 3.0)
        κ_cl = _kappa(homogenize(asm, ClusterModel(), :C))
        κ_mt = _kappa(homogenize(_mt_rve(f, _Ci()), MoriTanaka(), :C))
        @test κ_cl ≈ κ_mt rtol = 1.0e-11
    end
end

@testset "Cluster model — cubic symmetry of a cubic array" begin
    asm = cubic_lattice(:sc, Dict(:C => _Cm()), Dict(:C => _Ci()); fraction = 0.3, cutoff = 3.0)
    A = get_array(homogenize(asm, ClusterModel(), :C))
    @test A[1, 1, 1, 1] ≈ A[2, 2, 2, 2] rtol = 1.0e-10
    @test A[1, 1, 1, 1] ≈ A[3, 3, 3, 3] rtol = 1.0e-10
    @test A[1, 1, 2, 2] ≈ A[1, 1, 3, 3] rtol = 1.0e-10
    @test A[1, 2, 1, 2] ≈ A[1, 3, 1, 3] rtol = 1.0e-10
    # A cubic array is NOT isotropic: the two shear moduli differ.
    @test !isapprox(A[1, 2, 1, 2], (A[1, 1, 1, 1] - A[1, 1, 2, 2]) / 2; rtol = 1.0e-3)
end

@testset "Cluster model — convergence in the cluster radius" begin
    # Molinari & El Mouden's Fig. 3: the estimate is flat beyond R_c ≈ 2
    # periods, and equals Mori-Tanaka below one period (no neighbor yet).
    f = 0.3
    μ_mt = _mu(homogenize(_mt_rve(f, _Crigid()), MoriTanaka(), :C))
    vals = [
        _mu(
            homogenize(
                cubic_lattice(
                    :sc, Dict(:C => _Cm()), Dict(:C => _Crigid());
                    fraction = f, cutoff = c
                ), ClusterModel(), :C
            )
        ) for c in (0.5, 2.0, 3.0, 4.0)
    ]
    @test vals[1] ≈ μ_mt rtol = RTOL_EXACT          # no neighbor inside 0.5 period
    plateau = vals[2:end]
    @test maximum(plateau) - minimum(plateau) < 0.06 * sum(plateau) / length(plateau)
    # `cluster_radius` on the scheme overrides the assembly's own cutoff.
    asm = cubic_lattice(:sc, Dict(:C => _Cm()), Dict(:C => _Crigid()); fraction = f, cutoff = 3.0)
    @test _mu(homogenize(asm, ClusterModel(; cluster_radius = 0.0), :C)) ≈ μ_mt rtol = RTOL_EXACT
end

@testset "Cluster model — Mori-Tanaka overestimates a simple-cubic array" begin
    # Fig. 6 of the paper: for rigid spheres on a simple-cubic lattice the
    # Mori-Tanaka estimate of the shear modulus is above the cluster one, and
    # is therefore NOT a bound for this (non-isotropic) distribution.
    for f in (0.2, 0.4)
        asm = cubic_lattice(
            :sc, Dict(:C => _Cm()), Dict(:C => _Crigid()); fraction = f, cutoff = 3.0
        )
        μ_cl = _mu(homogenize(asm, ClusterModel(), :C))
        μ_mt = _mu(homogenize(_mt_rve(f, _Crigid()), MoriTanaka(), :C))
        @test μ_cl < μ_mt
    end
end

@testset "Cluster model — the spatial distribution matters" begin
    # Fig. 16: at equal porosity the three cubic arrangements give different
    # shear moduli, the simple-cubic one being the softest.
    f = 0.3
    μ = Dict(
        kind => _mu(
            homogenize(
                cubic_lattice(
                    kind, Dict(:C => _Cm()), Dict(:C => _Cvoid());
                    fraction = f, cutoff = 3.0
                ), ClusterModel(), :C
            )
        ) for kind in (:sc, :bcc, :fcc)
    )
    @test μ[:sc] < μ[:bcc]
    @test μ[:sc] < μ[:fcc]
    @test !isapprox(μ[:sc], μ[:bcc]; rtol = 1.0e-2)
end

@testset "Cluster model — degenerate and bracketing cases" begin
    # Particles identical to the matrix: the effective property is the matrix.
    asm = cubic_lattice(:sc, Dict(:C => _Cm()), Dict(:C => _Cm()); fraction = 0.3, cutoff = 3.0)
    C = get_array(homogenize(asm, ClusterModel(), :C))
    @test maximum(abs.(C .- get_array(_Cm()))) < 1.0e-10 * maximum(abs.(get_array(_Cm())))
    # Voigt / Reuss bracketing on the shear modulus.
    f = 0.3
    asm = cubic_lattice(:sc, Dict(:C => _Cm()), Dict(:C => _Ci()); fraction = f, cutoff = 3.0)
    μ_cl = _mu(homogenize(asm, ClusterModel(), :C))
    rve = _mt_rve(f, _Ci())
    @test _mu(homogenize(rve, Reuss(), :C)) ≤ μ_cl ≤ _mu(homogenize(rve, Voigt(), :C))
end

@testset "Cluster model — conduction" begin
    K_m, K_i = TensISO{3}(1.0), TensISO{3}(20.0)
    f = 0.25
    # Empty cluster ⇒ Mori-Tanaka, in conduction too: the kernel is written
    # once and dispatches on the tensor order alone.
    asm = cubic_lattice(:sc, Dict(:K => K_m), Dict(:K => K_i); fraction = f, cutoff = 0.0)
    K_cl = get_array(homogenize(asm, ClusterModel(), :K))
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:K => K_m); fraction = :rest)
    add_phase!(rve, :I, Ellipsoid(1.0), Dict(:K => K_i); fraction = f)
    K_mt = get_array(homogenize(rve, MoriTanaka(), :K))
    @test maximum(abs.(K_cl .- K_mt)) < RTOL_EXACT * maximum(abs.(K_mt))
    # With a cluster the array stays cubic, hence isotropic at second order.
    asm = cubic_lattice(:sc, Dict(:K => K_m), Dict(:K => K_i); fraction = f, cutoff = 3.0)
    K = get_array(homogenize(asm, ClusterModel(), :K))
    @test K[1, 1] ≈ K[2, 2] rtol = 1.0e-10
    @test K[1, 1] ≈ K[3, 3] rtol = 1.0e-10
    @test abs(K[1, 2]) < 1.0e-10 * K[1, 1]
end

@testset "Cluster model — ForwardDiff" begin
    # Sensitivity to a particle radius, through the whole N-body solve.
    fμ = a -> begin
        asm = cubic_lattice(
            :sc, Dict(:C => _Cm()), Dict(:C => _Ci()); radius = a, cutoff = 2.0,
            T = typeof(a)
        )
        _mu(homogenize(asm, ClusterModel(), :C))
    end
    d_ad = ForwardDiff.derivative(fμ, 0.3)
    d_fd = (fμ(0.3 + 1.0e-6) - fμ(0.3 - 1.0e-6)) / 2.0e-6
    @test isfinite(d_ad)
    @test d_ad ≈ d_fd rtol = 1.0e-5

    # Sensitivity to a particle coordinate — the degree of freedom no other
    # cell of the package has.
    base = ParticleAssembly(; boundary = PeriodicBox(1.0; cutoff = 2.0))
    add_matrix!(base, Dict(:C => _Cm()))
    add_particle!(base, :p1, (0.0, 0.0, 0.0), Ellipsoid(0.2), Dict(:C => _Ci()))
    add_particle!(base, :p2, (0.5, 0.5, 0.5), Ellipsoid(0.2), Dict(:C => _Ci()))
    fx = x -> _mu(homogenize(set_param(base, center_param(:p2, 1), x), ClusterModel(), :C))
    d_ad = ForwardDiff.derivative(fx, 0.5)
    d_fd = (fx(0.5 + 1.0e-6) - fx(0.5 - 1.0e-6)) / 2.0e-6
    @test isfinite(d_ad)
    @test d_ad ≈ d_fd rtol = 1.0e-5 atol = 1.0e-9
    @test get_param(base, center_param(:p2, 1)) == 0.5
    @test get_param(base, radius_param(:p1)) == 0.2
end
