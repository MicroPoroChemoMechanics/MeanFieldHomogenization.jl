using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using JSON

# =============================================================================
#  Multipole expansion method — conduction of assemblies of ellipsoids.
#
#  References: one ellipsoid in a domain of its own shape is Mori-Tanaka at any
#  degree; the Maxwell estimates of an ellipsoid with a Kapitza interface and
#  the translation coefficients of Kushch (2020), Tables 2 and 3; the near and
#  far translations against each other; one lattice described by two cells.
# =============================================================================

const _MEL = MeanFieldHomogenization.Assemblies
const _MEL_K0 = Dict(:K => TensISO{3}(1.0))
_mel_lit() = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2020.json"))
_mel_k(asm, nmax) = diag(canonical_components(homogenize(asm, MultipoleExpansion(; nmax), :K)))

@testset "MEM ellipsoids — one ellipsoid is Mori-Tanaka at any degree" begin
    for (semi, k₁) in (((1.5, 1.0, 0.5), 10.0), ((0.5, 1.5, 1.0), 4.0)), nmax in (1, 3)
        asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid((semi ./ cbrt(0.5))...)))
        add_matrix!(asm, _MEL_K0)
        add_particle!(asm, :p, (0.0, 0.0, 0.0), Ellipsoid(semi...), Dict(:K => TensISO{3}(k₁)))
        # to the accuracy of the quadrature of the Lamé functions of the second kind
        @test _mel_k(asm, nmax) ≈ diag(canonical_components(homogenize(asm, MoriTanaka(), :K))) rtol = 1.0e-9
    end
end

@testset "MEM ellipsoids — Kapitza interface, Kushch (2020), Table 2" begin
    lit = _mel_lit()
    semi = Tuple(lit["quantities"]["semi_axes"]["value"])
    c = lit["quantities"]["fraction"]["value"]
    k₁ = lit["quantities"]["contrast"]["value"]
    for row in lit["tables"]["maxwell_kapitza"]
        row["nmax"] ≤ 3 || continue
        for (key, hc) in (("hc_1e-4", 1.0e-4), ("hc_1", 1.0), ("hc_1e4", 1.0e4))
            asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid((semi ./ cbrt(c))...)))
            add_matrix!(asm, _MEL_K0)
            add_particle!(asm, :p, (0.0, 0.0, 0.0), InterfacedEllipsoid(semi..., KapitzaInterface(1 / hc)), Dict(:K => TensISO{3}(k₁)))
            @test _mel_k(asm, row["nmax"])[1] ≈ row[key] atol = 1.0e-7
        end
    end
    # a perfect interface carried by the type is the plain ellipsoid
    plain = ParticleAssembly(; boundary = MixedBC(Ellipsoid((semi ./ cbrt(c))...)))
    add_matrix!(plain, _MEL_K0)
    add_particle!(plain, :p, (0.0, 0.0, 0.0), Ellipsoid(semi...), Dict(:K => TensISO{3}(k₁)))
    carried = ParticleAssembly(; boundary = MixedBC(Ellipsoid((semi ./ cbrt(c))...)))
    add_matrix!(carried, _MEL_K0)
    add_particle!(carried, :p, (0.0, 0.0, 0.0), InterfacedEllipsoid(semi..., PerfectInterface{Float64}()), Dict(:K => TensISO{3}(k₁)))
    @test _mel_k(carried, 2) ≈ _mel_k(plain, 2) rtol = 1.0e-13
end

@testset "MEM ellipsoids — translation, Kushch (2020), Table 3" begin
    lit = _mel_lit()
    semi = Tuple(lit["quantities"]["semi_axes"]["value"])
    d = _MEL._mem_ellipsoid_data(semi, 2.0, 7, true)
    i₁, i₇ = _MEL._mem_index(1, -1), _MEL._mem_index(7, -7)   # the first harmonic of the degrees 1 and 7
    for row in lit["tables"]["translation"]
        R = [0.0, 0.0, row["alpha_over_a1"] * semi[1]]
        η = _MEL._mem_eta_near(d, d, R, 7)
        # Kushch's η_nmkl: receiver (n, m) first, sender (k, l) second
        @test real(η[i₁, i₁]) ≈ row["eta_1111"] rtol = 2.0e-5
        @test real(η[i₁, i₇]) ≈ row["eta_1171"] rtol = 2.0e-5
        @test real(η[i₇, i₇]) ≈ row["eta_7171"] rtol = 2.0e-5
    end
    # the far form, through the spherical harmonics, agrees beyond the threshold
    for R in ([0.3, -0.2, 7.5], [6.0, 5.0, 1.0])
        near = _MEL._mem_eta_near(d, d, R, 7)
        far = _MEL._mem_eta_far(d, d, irregular_harmonics(R, _MEL._mem_far_degree(d, 7)), 7)
        @test norm(far - near) ≤ 1.0e-8 * norm(near)
    end
end

@testset "MEM ellipsoids — periodic cells and finite pairs" begin
    P = Dict(:K => TensISO{3}(20.0))
    semi = (1.5, 1.0, 0.5)
    s = cbrt(0.3 * 6 / (4π / 3 * prod(semi)))
    one_cell = ParticleAssembly(; boundary = PeriodicBox((3.0, 2.0, 1.0)))
    add_matrix!(one_cell, _MEL_K0)
    add_particle!(one_cell, :p, (0.0, 0.0, 0.0), Ellipsoid((s .* semi)...), P)
    two_cells = ParticleAssembly(; boundary = PeriodicBox((3.0, 2.0, 2.0)))
    add_matrix!(two_cells, _MEL_K0)
    for k in 0:1
        add_particle!(two_cells, Symbol(:p, k), (0.1, -0.2, 1.0k + 0.05), Ellipsoid((s .* semi)...), P)
    end
    @test _mel_k(two_cells, 2) ≈ _mel_k(one_cell, 2) rtol = 1.0e-9
    # a finite pair, near and far
    pair(z) = begin
        asm = ParticleAssembly(; boundary = MixedBC(12.0))
        add_matrix!(asm, _MEL_K0)
        add_particle!(asm, :a, (0.0, 0.0, 0.0), Ellipsoid(semi...), P)
        add_particle!(asm, :b, (0.0, 0.0, z), Ellipsoid(semi...), P)
        _mel_k(asm, 2)
    end
    # closer along e₃: more conducting along e₃, less across, the pair shielding itself
    @test pair(1.2)[3] > pair(8.0)[3]
    @test pair(1.2)[1] < pair(8.0)[1]
    @test pair(8.0) ≈ pair(8.0 + 1.0e-9) rtol = 1.0e-8
end

@testset "MEM ellipsoids — derivative and refusals" begin
    semi = (1.5, 1.0, 0.5)
    k(k₁) = begin
        asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid((semi ./ cbrt(0.3))...)))
        add_matrix!(asm, _MEL_K0)
        add_particle!(asm, :p, (0.0, 0.0, 0.0), InterfacedEllipsoid(semi..., KapitzaInterface(0.5)), Dict(:K => TensISO{3}(k₁)))
        canonical_components(homogenize(asm, MultipoleExpansion(; nmax = 2), :K))[1, 1]
    end
    h = 1.0e-5
    @test ForwardDiff.derivative(k, 7.0) ≈ (k(7.0 + h) - k(7.0 - h)) / 2h rtol = 1.0e-6
    # with respect to the Kapitza resistance, the conductivities staying plain numbers
    kρ(ρ) = begin
        asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid((semi ./ cbrt(0.3))...)))
        add_matrix!(asm, _MEL_K0)
        add_particle!(asm, :p, (0.0, 0.0, 0.0), InterfacedEllipsoid(semi..., KapitzaInterface(ρ)), Dict(:K => TensISO{3}(7.0)))
        canonical_components(homogenize(asm, MultipoleExpansion(; nmax = 2), :K))[1, 1]
    end
    @test ForwardDiff.derivative(kρ, 0.5) ≈ (kρ(0.5 + h) - kρ(0.5 - h)) / 2h rtol = 1.0e-6
    base() = (a = ParticleAssembly(; boundary = MixedBC(6.0)); add_matrix!(a, _MEL_K0); a)
    tilted = base()
    add_particle!(tilted, :p, (0.0, 0.0, 0.0), Ellipsoid(1.5, 1.0, 0.5, [0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 1.0] * [cos(0.3) 0 sin(0.3); 0 1 0; -sin(0.3) 0 cos(0.3)]), Dict(:K => TensISO{3}(5.0)))
    @test_throws ArgumentError homogenize(tilted, MultipoleExpansion(), :K)
    crossed = base()
    add_particle!(crossed, :a, (0.0, 0.0, 0.0), Ellipsoid(1.5, 1.0, 0.5), Dict(:K => TensISO{3}(5.0)))
    add_particle!(crossed, :b, (4.0, 0.0, 0.0), Ellipsoid(1.0, 1.5, 0.5), Dict(:K => TensISO{3}(5.0)))
    @test_throws ArgumentError homogenize(crossed, MultipoleExpansion(), :K)
    mixed = base()
    add_particle!(mixed, :a, (0.0, 0.0, 0.0), Ellipsoid(1.5, 1.0, 0.5), Dict(:K => TensISO{3}(5.0)))
    add_particle!(mixed, :b, (4.0, 0.0, 0.0), Ellipsoid(0.5), Dict(:K => TensISO{3}(5.0)))
    @test_throws ArgumentError homogenize(mixed, MultipoleExpansion(), :K)
    for intf in (SurfaceConductiveInterface(1.0), SpringInterface(1.0, 1.0))
        bad = base()
        add_particle!(bad, :p, (0.0, 0.0, 0.0), InterfacedEllipsoid(1.5, 1.0, 0.5, intf), Dict(:K => TensISO{3}(5.0)))
        @test_throws ArgumentError homogenize(bad, MultipoleExpansion(), :K)
    end
    aniso = base()
    add_particle!(aniso, :p, (0.0, 0.0, 0.0), Ellipsoid(1.5, 1.0, 0.5), Dict(:K => TensTI{2}(1.0, 2.0, (0.0, 0.0, 1.0))))
    @test_throws ArgumentError homogenize(aniso, MultipoleExpansion(), :K)
    interfaced = base()
    add_particle!(interfaced, :p, (0.0, 0.0, 0.0), InterfacedEllipsoid(1.5, 1.0, 0.5, KapitzaInterface(0.5)), Dict(:K => TensISO{3}(5.0)))
    @test_throws ArgumentError homogenize(interfaced, MoriTanaka(), :K)
    @test_throws ArgumentError homogenize(interfaced, ClusterModel(), :K)
    @test_throws ArgumentError InterfacedEllipsoid(1.5, 1.0, 0.0, KapitzaInterface(0.5))
    dual = base()
    add_particle!(dual, :p, (0.0, 0.0, 0.0), Ellipsoid(ForwardDiff.Dual(1.5, 1.0), 1.0, 0.5), Dict(:K => TensISO{3}(5.0)))
    @test_throws ArgumentError homogenize(dual, MultipoleExpansion(), :K)
    # with respect to a position, against a centered difference
    kz(z) = begin
        asm = ParticleAssembly(; boundary = MixedBC(12.0))
        add_matrix!(asm, _MEL_K0)
        add_particle!(asm, :a, (0.0, 0.0, 0.0), Ellipsoid(semi...), Dict(:K => TensISO{3}(5.0)))
        add_particle!(asm, :b, (0.0, 0.0, z), Ellipsoid(semi...), Dict(:K => TensISO{3}(5.0)))
        canonical_components(homogenize(asm, MultipoleExpansion(; nmax = 2), :K))[3, 3]
    end
    @test ForwardDiff.derivative(kz, 1.6) ≈ (kz(1.6 + h) - kz(1.6 - h)) / 2h rtol = 1.0e-5
    @test MeanFieldHomogenization.Core.dimension(InterfacedEllipsoid(1.5, 1.0, 0.5, KapitzaInterface(0.5))) == 3
    @test assembly_volume(interfaced) ≈ 4π * 6.0^3 / 3
end
