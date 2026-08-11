using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

# =============================================================================
#  test_as_rve.jl — `RVE(asm)`, the bridge that makes every one-site scheme
#  available on a `ParticleAssembly`.
#
#  Coverage:
#   1. The conversion itself: phase names, derived fractions, matrix.
#   2. Every one-site scheme on an assembly == the same scheme on the RVE a
#      user would have built by hand. This is the whole contract.
#   3. Size-independence: the assembly's particle radius sets the FRACTION,
#      not the shape, so a hand-built RVE with a unit sphere must agree.
#   4. N identical phases == one phase of the summed fraction — the identity
#      that justifies one-phase-per-particle, checked on the NONLINEAR schemes
#      where it is not obvious (self-consistent, differential).
#   5. Conduction, 2D, and `MixedBC` go through unchanged.
#   6. `cluster_radius = 0` ≡ Mori-Tanaka, now expressible in one line on a
#      single cell — the sharpest check that the two paths agree.
#   7. The N-body schemes still read the positions (the bridge must not
#      swallow them).
#   8. Keyword forwarding: `matrix_geometry`, `distribution_shape` (PCW).
#   9. ForwardDiff through the bridge — the dual-valued fraction path.
#  10. Multiscale: a `Homogenized` particle property resolved through it.
#  11. Error paths: `Laminated` refused, naming the right cell.
# =============================================================================

_BR_Cm() = TensISO{3}(3 * 1.0, 2 * 0.4)          # k = 1, μ = 0.4 ⇒ ν = 0.3
_BR_Ci() = TensISO{3}(3 * 10.0, 2 * 6.0)
_BR_Km() = TensISO{3}(1.0)
_BR_Ki() = TensISO{3}(20.0)

_BR_μ(C) = get_array(C)[1, 2, 1, 2]

# The RVE a user would have written by hand for the same microstructure.
function _BR_hand_rve(f, C_m, C_i; geom = Ellipsoid(1.0))
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => C_m))
    add_phase!(rve, :I, geom, Dict(:C => C_i); fraction = f)
    return rve
end

const _BR_ONE_SITE = (
    Voigt(), Reuss(), Dilute(), DiluteDual(), MoriTanaka(), Maxwell(),
    SelfConsistent(), AsymmetricSelfConsistent(), DifferentialScheme(),
)

@testset "RVE(asm) — the conversion" begin
    asm = cubic_lattice(:sc, Dict(:C => _BR_Cm()), Dict(:C => _BR_Ci()); fraction = 0.3)
    rve = RVE(asm)
    m = asm.matrix_name
    @test rve isa RVE
    # One phase per particle, keeping the particle's own name, plus the matrix.
    @test rve.matrix_name === m
    @test sort(setdiff(rve.phase_names, [m])) == sort(particle_names(asm))
    # The fractions carried over are the assembly's DERIVED ones.
    @test matrix_volume_fraction(rve) ≈ matrix_volume_fraction(asm) rtol = 1.0e-12
    @test 1 - matrix_volume_fraction(rve) ≈ inclusion_volume_fraction(asm) rtol = 1.0e-12
    # The matrix keeps its properties and gets a ball by default.
    @test rve.phases[m].properties[:C] == _BR_Cm()
    @test rve.phases[m].geometry isa Ellipsoid{3}
end

@testset "RVE(asm) — every one-site scheme matches a hand-built RVE" begin
    # The assembly's spheres have radius 0.25 in a unit box; the hand-built
    # RVE uses a UNIT sphere. They must still agree exactly, because the Hill
    # tensor is size-independent: in an assembly the radius sets the volume
    # FRACTION, and that fraction is what the one-site schemes read.
    asm = cubic_lattice(:sc, Dict(:C => _BR_Cm()), Dict(:C => _BR_Ci()); fraction = 0.3)
    f = inclusion_volume_fraction(asm)
    ref = _BR_hand_rve(f, _BR_Cm(), _BR_Ci())
    for s in _BR_ONE_SITE
        @testset "$(nameof(typeof(s)))" begin
            a = get_array(homogenize(asm, s, :C))
            b = get_array(homogenize(ref, s, :C))
            @test maximum(abs.(a .- b)) < 1.0e-12 * maximum(abs.(b))
        end
    end
end

@testset "RVE(asm) — N identical phases ≡ one phase of the summed fraction" begin
    # This is what licenses one-phase-per-particle. Obvious for the linear
    # schemes, NOT obvious for the fixed-point and ODE ones — so those are the
    # ones checked here.
    C_m, C_i = _BR_Cm(), _BR_Ci()
    asm = ParticleAssembly(; boundary = PeriodicBox(1.0))
    add_matrix!(asm, Dict(:C => C_m))
    for (k, c) in enumerate(
            ((0.25, 0.25, 0.25), (0.75, 0.25, 0.25), (0.25, 0.75, 0.25), (0.75, 0.75, 0.75))
        )
        add_particle!(asm, Symbol("p", k), c, Ellipsoid(0.15), Dict(:C => C_i))
    end
    @test length(RVE(asm).phase_names) == 5          # four particles + matrix
    f = inclusion_volume_fraction(asm)
    ref = _BR_hand_rve(f, C_m, C_i)
    for s in (MoriTanaka(), SelfConsistent(), AsymmetricSelfConsistent(), DifferentialScheme())
        a = get_array(homogenize(asm, s, :C))
        b = get_array(homogenize(ref, s, :C))
        @test maximum(abs.(a .- b)) < 1.0e-10 * maximum(abs.(b))
    end
end

@testset "RVE(asm) — conduction" begin
    asm = cubic_lattice(:sc, Dict(:K => _BR_Km()), Dict(:K => _BR_Ki()); fraction = 0.25)
    f = inclusion_volume_fraction(asm)
    ref = RVE(:M)
    add_matrix!(ref, Ellipsoid(1.0), Dict(:K => _BR_Km()))
    add_phase!(ref, :I, Ellipsoid(1.0), Dict(:K => _BR_Ki()); fraction = f)
    for s in (MoriTanaka(), SelfConsistent(), DifferentialScheme())
        a = get_array(homogenize(asm, s, :K))
        b = get_array(homogenize(ref, s, :K))
        @test maximum(abs.(a .- b)) < 1.0e-10 * maximum(abs.(b))
    end
end

@testset "RVE(asm) — 2D, and MixedBC" begin
    # `MixedBC` measures |Ω| by the SVE, so the fractions come out against the
    # SVE area rather than a box; and the matrix must get a DISK, not a ball.
    #
    # Only `Voigt`/`Reuss` are checked numerically here, because the rest of
    # the one-site family is 3D-only on an `RVE` (`_mt_dispatch`,
    # `_phase_stiffness_contribution` and `_sc_step_dispatch` have no
    # `AbstractTens{4,2}` methods). That limitation is the RVE's, and the
    # bridge inherits it exactly — which is what the last block asserts.
    C_m = TensISO{2}(3 * 1.0, 2 * 0.4)
    C_i = TensISO{2}(3 * 10.0, 2 * 6.0)
    asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid(10.0, 10.0)))
    add_matrix!(asm, Dict(:C => C_m))
    add_particle!(asm, :a, (-3.0, 0.0), Ellipsoid(1.0, 1.0), Dict(:C => C_i))
    add_particle!(asm, :b, (3.0, 0.0), Ellipsoid(1.0, 1.0), Dict(:C => C_i))

    rve = RVE(asm)
    @test rve.phases[asm.matrix_name].geometry isa Ellipsoid{2}   # a disk, not a ball
    f = inclusion_volume_fraction(asm)
    @test f ≈ 2 * π * 1.0^2 / (π * 10.0^2) rtol = 1.0e-12         # 2 disks in the SVE

    ref = RVE(:M)
    add_matrix!(ref, Ellipsoid(1.0, 1.0), Dict(:C => C_m))
    add_phase!(ref, :I, Ellipsoid(1.0, 1.0), Dict(:C => C_i); fraction = f)
    for s in (Voigt(), Reuss())
        a = get_array(homogenize(asm, s, :C))
        b = get_array(homogenize(ref, s, :C))
        @test maximum(abs.(a .- b)) < 1.0e-12 * maximum(abs.(b))
    end
    # The bridge adds no limitation of its own: a scheme that fails on the
    # hand-built 2D RVE fails identically through the assembly.
    @test_throws MethodError homogenize(ref, MoriTanaka(), :C)
    @test_throws MethodError homogenize(asm, MoriTanaka(), :C)
    # …while the N-body schemes, which have their own 2D kernels, do work.
    @test _BR_μ(homogenize(asm, EquivalentInclusion(), :C)) > 0
end

@testset "RVE(asm) — cluster_radius = 0 ≡ Mori-Tanaka, on ONE cell" begin
    # Molinari & El Mouden, App. C. Before the bridge this identity needed a
    # second, hand-built cell to state; now both sides are the same object,
    # which removes the last way of getting the comparison wrong.
    for f in (0.1, 0.3)
        asm = cubic_lattice(:sc, Dict(:C => _BR_Cm()), Dict(:C => _BR_Ci()); fraction = f)
        a = get_array(homogenize(asm, ClusterModel(; cluster_radius = 0.0), :C))
        b = get_array(homogenize(asm, MoriTanaka(), :C))
        @test maximum(abs.(a .- b)) < 1.0e-12 * maximum(abs.(b))
    end
end

@testset "RVE(asm) — the N-body schemes still read the positions" begin
    # Guard against the bridge silently swallowing ClusterModel /
    # EquivalentInclusion: with a real cluster the two answers must DIFFER.
    asm = cubic_lattice(:sc, Dict(:C => _BR_Cm()), Dict(:C => _BR_Ci()); fraction = 0.3, cutoff = 3.0)
    μ_cluster = _BR_μ(homogenize(asm, ClusterModel(), :C))
    μ_mt = _BR_μ(homogenize(asm, MoriTanaka(), :C))
    @test !isapprox(μ_cluster, μ_mt; rtol = 1.0e-3)
    μ_eim = _BR_μ(homogenize(asm, EquivalentInclusion(), :C))
    @test μ_eim ≈ μ_cluster rtol = 1.0e-10          # same system, same cutoff
end

@testset "RVE(asm) — keyword forwarding" begin
    asm = cubic_lattice(:sc, Dict(:C => _BR_Cm()), Dict(:C => _BR_Ci()); fraction = 0.3)
    # `matrix_geometry` reaches the conversion …
    rve = RVE(asm; matrix_geometry = Ellipsoid(3.0, 1.0, 1.0))
    @test collect(rve.phases[asm.matrix_name].geometry.semi_axes) == [3.0, 1.0, 1.0]
    # … and through `homogenize`. It is read by the schemes that localize the
    # matrix like any other phase, so the self-consistent answer must move
    # while Mori-Tanaka (which never looks at it) must not.
    sc_ball = _BR_μ(homogenize(asm, SelfConsistent(), :C))
    sc_fiber = _BR_μ(homogenize(asm, SelfConsistent(), :C; matrix_geometry = Ellipsoid(20.0, 1.0, 1.0)))
    @test !isapprox(sc_ball, sc_fiber; rtol = 1.0e-6)
    mt_ball = _BR_μ(homogenize(asm, MoriTanaka(), :C))
    mt_fiber = _BR_μ(homogenize(asm, MoriTanaka(), :C; matrix_geometry = Ellipsoid(20.0, 1.0, 1.0)))
    @test mt_ball ≈ mt_fiber rtol = 1.0e-14
    # `distribution_shape` is what PCW needs, and an assembly has none.
    pcw = homogenize(asm, PonteCastanedaWillis(), :C; distribution_shape = Ellipsoid(1.0))
    @test _BR_μ(pcw) > 0
end

@testset "RVE(asm) — ForwardDiff through the bridge" begin
    # A radius is the assembly's own degree of freedom, and here it reaches a
    # one-site scheme only through the fraction — which is exactly the path
    # that must promote to `Dual`.
    C_m, C_i = _BR_Cm(), _BR_Ci()
    build = a -> begin
        asm = ParticleAssembly(; boundary = PeriodicBox(1.0))
        add_matrix!(asm, Dict(:C => C_m))
        add_particle!(asm, :p1, (0.5, 0.5, 0.5), Ellipsoid(a), Dict(:C => C_i))
        return _BR_μ(homogenize(asm, MoriTanaka(), :C))
    end
    g = ForwardDiff.derivative(build, 0.3)
    fd = (build(0.3 + 1.0e-6) - build(0.3 - 1.0e-6)) / 2.0e-6
    @test g ≈ fd rtol = 1.0e-6
    @test g > 0                                     # a bigger particle ⇒ stiffer

    # …and through the exported lens, which is the supported way in.
    asm = cubic_lattice(:sc, Dict(:C => C_m), Dict(:C => C_i); fraction = 0.3)
    d = derivative(
        asm, MoriTanaka(), radius_param(first(particle_names(asm)));
        indexer = C -> get_array(C)[1, 2, 1, 2]
    )
    @test isfinite(d) && d > 0
end

@testset "RVE(asm) — multiscale through the bridge" begin
    # A particle whose property is itself a homogenized cell must resolve
    # lazily through the one-site path exactly as it does through the N-body
    # one — same nested cache, same result as computing it in two steps.
    C_m, C_i = _BR_Cm(), _BR_Ci()
    inner = RVE(:m)
    add_matrix!(inner, Ellipsoid(1.0), Dict(:C => C_m))
    add_phase!(inner, :f, Ellipsoid(1.0), Dict(:C => C_i); fraction = 0.4)

    asm = ParticleAssembly(; boundary = PeriodicBox(1.0))
    add_matrix!(asm, Dict(:C => C_m))
    add_particle!(
        asm, :p1, (0.5, 0.5, 0.5), Ellipsoid(0.3),
        Dict(:C => Homogenized(inner, MoriTanaka()))
    )
    got = get_array(homogenize(asm, MoriTanaka(), :C))

    C_in = homogenize(inner, MoriTanaka(), :C)
    two_step = ParticleAssembly(; boundary = PeriodicBox(1.0))
    add_matrix!(two_step, Dict(:C => C_m))
    add_particle!(two_step, :p1, (0.5, 0.5, 0.5), Ellipsoid(0.3), Dict(:C => C_in))
    want = get_array(homogenize(two_step, MoriTanaka(), :C))
    @test maximum(abs.(got .- want)) ≈ 0.0 atol = 1.0e-14
end

@testset "RVE(asm) — error paths" begin
    asm = cubic_lattice(:sc, Dict(:C => _BR_Cm()), Dict(:C => _BR_Ci()); fraction = 0.3)
    # `Laminated` needs an ordered stack; the message must name the cell the
    # caller actually built, not the RVE the bridge would have made.
    err = try
        homogenize(asm, Laminated(), :C)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("ParticleAssembly", err.msg)
    @test !occursin("RVE", err.msg)
end
