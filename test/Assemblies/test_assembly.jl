using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Random

# =============================================================================
#  test_assembly.jl — the `ParticleAssembly` cell and its generators.
#
#  Coverage:
#   1. Two-step construction, accessors, derived volume fractions.
#   2. `validate_assembly`: missing matrix, missing particle, mixed dimensions,
#      overfilled cell, overlapping particles.
#   3. The `AbstractHomogenizationCell` contract (member names, raw property
#      access, immutable property update).
#   4. `cubic_lattice`: site counts, packing limits, family assignment,
#      fractions reproduced exactly from the requested value.
#   5. `random_assembly`: non-overlap in the periodic metric and inside an SVE,
#      reproducibility from a seeded RNG.
#   6. Families: representative selection and aggregated fractions.
# =============================================================================

const ATOL_F = 1.0e-12

_Cm() = TensISO{3}(3.0, 0.8)
_Ci() = TensISO{3}(30.0, 12.0)

function _simple_assembly(; cutoff = 3.0)
    asm = ParticleAssembly(; boundary = PeriodicBox(1.0; cutoff = cutoff))
    add_matrix!(asm, Dict(:C => _Cm()))
    add_particle!(asm, :p1, (0.0, 0.0, 0.0), Ellipsoid(0.2), Dict(:C => _Ci()))
    add_particle!(asm, :p2, (0.5, 0.5, 0.5), Ellipsoid(0.2), Dict(:C => _Ci()))
    return asm
end

@testset "ParticleAssembly — construction and accessors" begin
    asm = _simple_assembly()
    @test particle_names(asm) == [:p1, :p2]
    @test particle_center(asm, :p2) == [0.5, 0.5, 0.5]
    @test particle_geometry(asm, :p1) isa Ellipsoid
    @test particle_property(asm, :p1, :C) == _Ci()
    @test matrix_property(asm, :C) == _Cm()
    # Volume fractions are derived, never stored.
    v = 4π * 0.2^3 / 3
    @test particle_volume(asm, :p1) ≈ v rtol = 1.0e-14
    @test assembly_volume(asm) ≈ 1.0 rtol = 1.0e-14
    @test particle_volume_fraction(asm, :p1) ≈ v rtol = 1.0e-14
    @test inclusion_volume_fraction(asm) ≈ 2v rtol = 1.0e-14
    @test matrix_volume_fraction(asm) ≈ 1 - 2v rtol = 1.0e-14
    @test validate_assembly(asm) === asm
end

@testset "ParticleAssembly — the cell contract" begin
    asm = _simple_assembly()
    @test MeanFieldHomogenization.Core.cell_member_names(asm) == [:matrix, :p1, :p2]
    @test MeanFieldHomogenization.Core.cell_container_property(asm, :p1, :C) == _Ci()
    # Property updates rebuild rather than mutate.
    C_new = TensISO{3}(99.0, 33.0)
    asm2 = MeanFieldHomogenization.Core.cell_set_property(asm, :p1, :C, C_new)
    @test particle_property(asm2, :p1, :C) == C_new
    @test particle_property(asm, :p1, :C) == _Ci()
end

@testset "ParticleAssembly — validation" begin
    # No matrix
    a1 = ParticleAssembly(; boundary = PeriodicBox(1.0))
    add_particle!(a1, :p, (0.0, 0.0, 0.0), Ellipsoid(0.1), Dict(:C => _Ci()))
    @test_throws ArgumentError validate_assembly(a1)
    # No particle
    a2 = ParticleAssembly(; boundary = PeriodicBox(1.0))
    add_matrix!(a2, Dict(:C => _Cm()))
    @test_throws ArgumentError validate_assembly(a2)
    # Overlapping particles — the interaction kernel is undefined there, so
    # this must fail at validation rather than deep inside a lattice sum.
    a3 = ParticleAssembly(; boundary = PeriodicBox(1.0))
    add_matrix!(a3, Dict(:C => _Cm()))
    add_particle!(a3, :p1, (0.0, 0.0, 0.0), Ellipsoid(0.3), Dict(:C => _Ci()))
    add_particle!(a3, :p2, (0.4, 0.0, 0.0), Ellipsoid(0.3), Dict(:C => _Ci()))
    @test_throws ArgumentError validate_assembly(a3)
    # A cell the particles fill entirely
    a4 = ParticleAssembly(; boundary = PeriodicBox(1.0))
    add_matrix!(a4, Dict(:C => _Cm()))
    add_particle!(a4, :p1, (0.0, 0.0, 0.0), Ellipsoid(0.7), Dict(:C => _Ci()))
    @test_throws ArgumentError validate_assembly(a4)
    # Duplicate name, and a center of the wrong length
    a5 = _simple_assembly()
    @test_throws ArgumentError add_particle!(a5, :p1, (0.1, 0.1, 0.1), Ellipsoid(0.1), Dict(:C => _Ci()))
    @test_throws ArgumentError add_particle!(a5, :p3, (0.1, 0.1), Ellipsoid(0.1), Dict(:C => _Ci()))
end

@testset "cubic_lattice — motifs, fractions and families" begin
    for (kind, nsites) in ((:sc, 1), (:bcc, 2), (:fcc, 4))
        asm = cubic_lattice(kind, Dict(:C => _Cm()), Dict(:C => _Ci()); fraction = 0.2)
        @test length(particle_names(asm)) == nsites
        @test inclusion_volume_fraction(asm) ≈ 0.2 rtol = 1.0e-12
        # All sites of a Bravais lattice are equivalent: one family, one unknown.
        @test length(family_labels(asm)) == 1
        @test validate_assembly(asm) === asm
    end
    # Packing limits are enforced against the requested fraction.
    @test max_packing_fraction(:sc) ≈ π / 6 rtol = 1.0e-14
    @test max_packing_fraction(:bcc) ≈ sqrt(3) * π / 8 rtol = 1.0e-14
    @test max_packing_fraction(:fcc) ≈ sqrt(2) * π / 6 rtol = 1.0e-14
    @test_throws ArgumentError cubic_lattice(:sc, Dict(:C => _Cm()), Dict(:C => _Ci()); fraction = 0.6)
    @test_throws ArgumentError cubic_lattice(:hcp, Dict(:C => _Cm()), Dict(:C => _Ci()); fraction = 0.2)
    # A two-material motif gets two families — Molinari's BCC array of
    # alternating voids and rigid spheres.
    C_void = TensISO{3}(1.0e-9, 1.0e-9)
    asm = cubic_lattice(
        :bcc, Dict(:C => _Cm()), [Dict(:C => _Ci()), Dict(:C => C_void)]; fraction = 0.2
    )
    @test length(family_labels(asm)) == 2
    @test particle_family(asm, :p1) != particle_family(asm, :p2)
end

@testset "random_assembly — non-overlap and reproducibility" begin
    # Periodic box: overlap must be tested in the periodic metric, so a
    # particle near a face may not overlap its own image across the boundary.
    rng = Random.MersenneTwister(42)
    asm = random_assembly(
        8, Dict(:C => _Cm()), Dict(:C => _Ci());
        fraction = 0.15, period = 1.0, rng = rng, cycles = 300
    )
    @test length(particle_names(asm)) == 8
    @test inclusion_volume_fraction(asm) ≈ 0.15 rtol = 1.0e-12
    a = particle_geometry(asm, :p1).semi_axes[1]
    for i in 1:8, j in (i + 1):8
        ci = particle_center(asm, Symbol(:p, i))
        cj = particle_center(asm, Symbol(:p, j))
        d = sqrt(
            sum(
                let t = abs(ci[k] - cj[k])
                    min(t, 1.0 - t)^2
                end for k in 1:3
            )
        )
        @test d > 2a
    end
    # Same seed, same microstructure.
    asm2 = random_assembly(
        8, Dict(:C => _Cm()), Dict(:C => _Ci());
        fraction = 0.15, period = 1.0, rng = Random.MersenneTwister(42), cycles = 300
    )
    @test particle_center(asm2, :p5) == particle_center(asm, :p5)
end

@testset "random_assembly — inside an ellipsoidal SVE" begin
    # Under mixed boundary conditions the particles live *inside* the SVE and
    # must fit entirely: this is Brisard's geometry, not a periodic cell.
    rng = Random.MersenneTwister(7)
    R = 6.0
    asm = random_assembly(
        12, Dict(:C => _Cm()), Dict(:C => _Ci());
        radius = 1.0, rng = rng, cycles = 300, boundary = MixedBC(Ellipsoid(R))
    )
    @test validate_assembly(asm) === asm
    @test assembly_volume(asm) ≈ 4π * R^3 / 3 rtol = 1.0e-14
    for nm in particle_names(asm)
        c = particle_center(asm, nm)
        @test norm(c) ≤ R - 1.0 + 1.0e-12          # fits entirely inside
    end
    # 2D variant — the geometry of Brisard's Table 1.
    asm2 = random_assembly(
        10, Dict(:C => TensISO{2}(3.0, 0.8)), Dict(:C => TensISO{2}(30.0, 12.0));
        radius = 1.0, dim = 2, rng = Random.MersenneTwister(3), cycles = 300,
        boundary = MixedBC(Ellipsoid(8.0, 8.0))
    )
    @test length(particle_names(asm2)) == 10
    @test validate_assembly(asm2) === asm2
end

@testset "Families — representatives and aggregated fractions" begin
    asm = ParticleAssembly(; boundary = PeriodicBox(1.0))
    add_matrix!(asm, Dict(:C => _Cm()))
    add_particle!(asm, :a1, (0.0, 0.0, 0.0), Ellipsoid(0.1), Dict(:C => _Ci()); family = 1)
    add_particle!(asm, :a2, (0.5, 0.0, 0.0), Ellipsoid(0.1), Dict(:C => _Ci()); family = 1)
    add_particle!(asm, :b1, (0.0, 0.5, 0.0), Ellipsoid(0.15), Dict(:C => _Ci()); family = 2)
    @test family_labels(asm) == [1, 2]
    @test MeanFieldHomogenization.Assemblies._family_representative(asm, 1) === :a1
    @test MeanFieldHomogenization.Assemblies._family_fraction(asm, 1) ≈
        2 * 4π * 0.1^3 / 3 rtol = 1.0e-12
    @test MeanFieldHomogenization.Assemblies._family_fraction(asm, 2) ≈
        4π * 0.15^3 / 3 rtol = 1.0e-12
end
