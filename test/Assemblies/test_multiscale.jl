using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra

# =============================================================================
#  test_multiscale.jl — a `ParticleAssembly` inside the declarative multiscale
#  seam, in both directions.
#
#  The nesting machinery is generic: `Homogenized` resolves through
#  `cell_member_names` / `cell_container_property` / `cell_set_property`, which
#  an assembly implements like every other cell. These tests pin that it really
#  is generic, rather than working by accident on the two pre-existing cells.
#
#  Coverage:
#   1. Assembly as the INNER cell of an RVE — identical to the manual two-step.
#   2. Assembly as the OUTER cell, a particle being itself homogenized.
#   3. Three levels through an ANISOTROPIC intermediate: the cluster model on a
#      cubic array returns a cubic tensor, so chaining it at all requires the
#      anisotropic Green operator. This is the case that raised before
#      `Core/green_aniso.jl` landed.
#   4. `has_nested_property` introspection and the raw-vs-resolving accessors.
#   5. Nested sensitivity through the scales (`nested` lens on a modulus).
#   6. Lenses that do not apply to an assembly report why, and name the one
#      that does.
# =============================================================================

_Cm() = TensISO{3}(3 * 1.0, 2 * 0.4)
_Ci() = TensISO{3}(3 * 10.0, 2 * 6.0)
_Cf() = TensISO{3}(3 * 40.0, 2 * 25.0)

_idx(C) = get_array(C)[1, 2, 1, 2]

@testset "Multiscale — assembly as the inner cell of an RVE" begin
    inner = cubic_lattice(:sc, Dict(:C => _Cm()), Dict(:C => _Ci()); fraction = 0.3, cutoff = 2.0)
    outer = RVE(:M)
    add_matrix!(outer, Ellipsoid(1.0), Dict(:C => Homogenized(inner, ClusterModel())))
    add_phase!(outer, :F, Ellipsoid(1.0), Dict(:C => _Cf()); fraction = 0.2)
    C_nested = get_array(homogenize(outer, MoriTanaka(), :C))

    # The same thing done by hand, in two steps.
    ref = RVE(:M)
    add_matrix!(ref, Ellipsoid(1.0), Dict(:C => homogenize(inner, ClusterModel(), :C)))
    add_phase!(ref, :F, Ellipsoid(1.0), Dict(:C => _Cf()); fraction = 0.2)
    @test maximum(abs.(C_nested .- get_array(homogenize(ref, MoriTanaka(), :C)))) ≈ 0.0 atol = 1.0e-14
end

@testset "Multiscale — assembly as the outer cell" begin
    sub = RVE(:S)
    add_matrix!(sub, Ellipsoid(1.0), Dict(:C => _Ci()))
    add_phase!(sub, :n, Ellipsoid(1.0), Dict(:C => _Cf()); fraction = 0.25)

    asm = ParticleAssembly(; boundary = PeriodicBox(1.0; cutoff = 2.0))
    add_matrix!(asm, Dict(:C => _Cm()))
    add_particle!(
        asm, :p1, (0.0, 0.0, 0.0), Ellipsoid(0.3),
        Dict(:C => Homogenized(sub, MoriTanaka()))
    )
    C_nested = get_array(homogenize(asm, ClusterModel(), :C))

    flat = ParticleAssembly(; boundary = PeriodicBox(1.0; cutoff = 2.0))
    add_matrix!(flat, Dict(:C => _Cm()))
    add_particle!(
        flat, :p1, (0.0, 0.0, 0.0), Ellipsoid(0.3),
        Dict(:C => homogenize(sub, MoriTanaka(), :C))
    )
    @test maximum(abs.(C_nested .- get_array(homogenize(flat, ClusterModel(), :C)))) ≈ 0.0 atol = 1.0e-14
end

@testset "Multiscale — three levels through an anisotropic intermediate" begin
    # The cluster model on a cubic array returns a *cubic* tensor, not an
    # isotropic one — the two shear constants differ. Using it as the reference
    # medium of a second assembly therefore exercises the anisotropic Green
    # operator; before it existed, this chain raised an `ArgumentError`.
    lvl1 = cubic_lattice(:sc, Dict(:C => _Cm()), Dict(:C => _Ci()); fraction = 0.2, cutoff = 1.5)
    C1 = get_array(homogenize(lvl1, ClusterModel(), :C))
    @test !isapprox(C1[1, 2, 1, 2], (C1[1, 1, 1, 1] - C1[1, 1, 2, 2]) / 2; rtol = 1.0e-3)

    lvl2 = ParticleAssembly(; boundary = PeriodicBox(1.0; cutoff = 1.5))
    add_matrix!(lvl2, Dict(:C => Homogenized(lvl1, ClusterModel())))
    add_particle!(lvl2, :q, (0.0, 0.0, 0.0), Ellipsoid(0.25), Dict(:C => _Cf()))
    C2 = get_array(homogenize(lvl2, ClusterModel(), :C))
    @test all(isfinite, C2)
    @test C2[1, 1, 1, 1] > C1[1, 1, 1, 1]        # a stiff particle stiffens it

    lvl3 = RVE(:T)
    add_matrix!(lvl3, Ellipsoid(1.0), Dict(:C => Homogenized(lvl2, ClusterModel())))
    add_phase!(lvl3, :v, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0e-9, 1.0e-9)); fraction = 0.05)
    C3 = get_array(homogenize(lvl3, MoriTanaka(), :C))
    @test all(isfinite, C3)
    @test C3[1, 1, 1, 1] < C2[1, 1, 1, 1]        # adding porosity softens it
end

@testset "Multiscale — introspection and the raw accessor" begin
    inner = cubic_lattice(:sc, Dict(:C => _Cm()), Dict(:C => _Ci()); fraction = 0.2, cutoff = 1.0)
    outer = RVE(:M)
    add_matrix!(outer, Ellipsoid(1.0), Dict(:C => Homogenized(inner, ClusterModel())))
    add_phase!(outer, :F, Ellipsoid(1.0), Dict(:C => _Cf()); fraction = 0.2)
    @test MeanFieldHomogenization.Core.has_nested_property(outer, :C)

    asm = ParticleAssembly(; boundary = PeriodicBox(1.0; cutoff = 1.0))
    add_matrix!(asm, Dict(:C => _Cm()))
    add_particle!(asm, :p, (0.0, 0.0, 0.0), Ellipsoid(0.2), Dict(:C => _Ci()))
    @test !MeanFieldHomogenization.Core.has_nested_property(asm, :C)

    # The raw accessor must NOT resolve — that is what type inspections rely on.
    @test MeanFieldHomogenization.Core.cell_container_property(outer, :M, :C) isa Homogenized
    @test matrix_property(outer, :C) isa TensND.AbstractTens    # the resolving one does

    # An assembly nested under an *iterative* outer scheme: the memoization of
    # `Homogenized` is what keeps the inner cell from being re-homogenized on
    # every iteration, so this has to stay finite and cheap.
    @test all(isfinite, get_array(homogenize(outer, SelfConsistent(; maxiters = 40), :C)))
end

@testset "Multiscale — nested sensitivity through the scales" begin
    inner = cubic_lattice(:sc, Dict(:C => _Cm()), Dict(:C => _Ci()); fraction = 0.3, cutoff = 2.0)
    outer = RVE(:M)
    add_matrix!(outer, Ellipsoid(1.0), Dict(:C => Homogenized(inner, ClusterModel())))
    add_phase!(outer, :F, Ellipsoid(1.0), Dict(:C => _Cf()); fraction = 0.2)

    # A modulus of the assembly's own matrix, two scales down.
    lens = nested(:M, :C, property(:matrix, :C, :μ))
    x₀ = get_param(outer, lens)
    d_ad = derivative(outer, MoriTanaka(), lens; indexer = _idx)
    d_fd = (
        _idx(homogenize(set_param(outer, lens, x₀ + 1.0e-6), MoriTanaka(), :C)) -
            _idx(homogenize(set_param(outer, lens, x₀ - 1.0e-6), MoriTanaka(), :C))
    ) / 2.0e-6
    @test isfinite(d_ad)
    @test d_ad ≈ d_fd rtol = 1.0e-6
end

@testset "Multiscale — lenses that do not apply to an assembly" begin
    asm = cubic_lattice(:sc, Dict(:C => _Cm()), Dict(:C => _Ci()); fraction = 0.2)
    # An assembly stores no amount and no distribution shape; the message must
    # say so and name the lens that does the job.
    for lens in (amount(:p1), geometry(:p1, :semi_axes), shape_param(:semi_axes))
        e = try
            get_param(asm, lens)
            nothing
        catch err
            err
        end
        @test e isa ArgumentError
        @test occursin("ParticleAssembly", sprint(showerror, e))
    end
    # The ones that do apply.
    @test get_param(asm, radius_param(:p1)) == particle_geometry(asm, :p1).semi_axes[1]
    @test get_param(asm, center_param(:p1, 1)) == particle_center(asm, :p1)[1]
    # `property` addresses the *stored* TensND datum, so on a `TensISO{4}` the
    # `:μ` selector reaches `β = 2μ` — the package-wide convention, shared with
    # the `RVE` lens.
    @test get_param(asm, property(:p1, :C, :μ)) ≈ 12.0 rtol = 1.0e-12
    @test get_param(asm, property(:matrix, :C, :K)) ≈ 3.0 rtol = 1.0e-12
end
