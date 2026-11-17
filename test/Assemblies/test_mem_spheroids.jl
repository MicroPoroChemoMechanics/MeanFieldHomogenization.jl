using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using JSON

# =============================================================================
#  Multipole expansion method — conduction of assemblies of spheroids.
#
#  References: one spheroid in a domain of its own shape is Mori-Tanaka at any
#  degree; the two ways of translating a harmonic agree where both hold; the
#  tables of Kushch et al. (2015) for oblate spheroids on proportional lattices,
#  with perfect, low-conducting and highly conducting interfaces.
# =============================================================================

const _MSP = MeanFieldHomogenization.Assemblies
const _MSP_K0 = Dict(:K => TensISO{3}(1.0))
_msp_literature() = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2015.json"))

# The proportional lattices of Kushch et al. (2015): periods (1, 1, e), sites in
# units of the periods, spheroids of semi-axes (l₁, l₁, e l₁) at the fraction c.
const _MSP_SITES = Dict(
    :sp => [(0.0, 0.0, 0.0)],
    :bcp => [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
    :fcp => [(0.0, 0.0, 0.0), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0), (0.0, 0.5, 0.5)],
)
_msp_l1(kind, c) = cbrt(3c / (4π * length(_MSP_SITES[kind])))
function _msp_lattice(kind, c; e = 1 / 3, k₁ = 1000.0, interface = PerfectInterface{Float64}())
    l₁ = _msp_l1(kind, c)
    asm = ParticleAssembly(; boundary = PeriodicBox((1.0, 1.0, e)))
    add_matrix!(asm, _MSP_K0)
    for (i, x) in enumerate(_MSP_SITES[kind])
        s = LayeredSpheroid((e * l₁,), (l₁,), (TensISO{3}(k₁),); interfaces = (interface,))
        add_particle!(asm, Symbol(:p, i), (x[1], x[2], e * x[3]), s, Dict{Symbol, Any}())
    end
    return asm
end
_msp_k(asm, nmax) = (K = canonical_components(homogenize(asm, MultipoleExpansion(; nmax), :K)); (K[1, 1], K[3, 3]))

@testset "MEM spheroids — one spheroid is Mori-Tanaka at any degree" begin
    for (axis, disk) in ((0.6, 0.3), (0.2, 0.5)), nmax in (1, 3)
        asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid(2disk, 2disk, 2axis)))
        add_matrix!(asm, _MSP_K0)
        add_particle!(asm, :p, (0.0, 0.0, 0.0), Ellipsoid(disk, disk, axis), Dict(:K => TensISO{3}(8.0)))
        K = canonical_components(homogenize(asm, MultipoleExpansion(; nmax), :K))
        @test K ≈ canonical_components(homogenize(asm, MoriTanaka(), :K)) rtol = 1.0e-12
    end
end

@testset "MEM spheroids — the near and far translations agree where both hold" begin
    K₀ = TensISO{3}(1.0)
    for (axis, disk) in ((0.138, 0.415), (0.5, 0.25))
        d = _MSP._mem_spheroid_data(LayeredSpheroid((axis,), (disk,), (TensISO{3}(1000.0),)), K₀, 6)
        # the far series converges as (2|c|/|R|)^{2r}: points beyond 1.8 × 2|c|
        for R in ([0.6, 1.0, 1.2], [0.0, 0.0, 1.6])
            near = _MSP._mem_eta_near(d, d, R, 6)
            far = _MSP._mem_eta_far(d, d, irregular_harmonics(R, 2 * (6 + _MSP._MEM_FAR_TERMS)), 6)
            @test norm(near - far) ≤ 1.0e-9 * norm(far)
        end
    end
    # a finite pair: the near form below the threshold, the far one above, continuous
    pair(z) = begin
        asm = ParticleAssembly(; boundary = MixedBC(6.0))
        add_matrix!(asm, _MSP_K0)
        add_particle!(asm, :a, (0.0, 0.0, 0.0), Ellipsoid(0.5, 0.5, 0.2), Dict(:K => TensISO{3}(20.0)))
        add_particle!(asm, :b, (z, 0.0, 0.0), Ellipsoid(0.5, 0.5, 0.2), Dict(:K => TensISO{3}(20.0)))
        _msp_k(asm, 4)
    end
    rn = _MSP._MEM_NEAR * 2 * sqrt(0.5^2 - 0.2^2)
    @test all(pair(rn * (1 - 1.0e-9)) .≈ pair(rn * (1 + 1.0e-9)))
end

@testset "MEM spheroids — Kushch et al. (2015), convergence on the simple lattice, Table 3" begin
    lit = _msp_literature()
    e = lit["quantities"]["aspect_ratio"]["value"]
    k₁ = lit["quantities"]["contrast_lc"]["value"]
    for row in lit["tables"]["convergence_sp"]
        row["tmax"] ≤ 5 || continue
        asm = _msp_lattice(:sp, 0.5; e, k₁)
        k11, k33 = _msp_k(asm, row["tmax"])
        @test k11 ≈ row["k11_hc_infinity"] atol = 6.0e-4
        @test k33 ≈ row["k33_hc_infinity"] atol = 6.0e-4
        hc = 1.0                                       # hc_tilde = l₁ hc/k₀, ρ = 1/hc
        k11, k33 = _msp_k(_msp_lattice(:sp, 0.5; e, k₁, interface = KapitzaInterface(_msp_l1(:sp, 0.5) / hc)), row["tmax"])
        @test k11 ≈ row["k11_hc_1"] atol = 6.0e-4
        @test k33 ≈ row["k33_hc_1"] atol = 6.0e-4
    end
end

@testset "MEM spheroids — Kushch et al. (2015), lattices and interfaces, Tables 4 to 8" begin
    lit = _msp_literature()
    e = lit["quantities"]["aspect_ratio"]["value"]
    cases = (
        ("lattices_perfect", lit["quantities"]["contrast_lc"]["value"], c -> PerfectInterface{Float64}()),
        ("lattices_lc_10", lit["quantities"]["contrast_lc"]["value"], c -> KapitzaInterface(c / 10)),
        ("lattices_hc_1", lit["quantities"]["contrast_hc"]["value"], c -> SurfaceConductiveInterface(c)),
    )
    for (table, k₁, interface) in cases, (kind, c, nmax) in ((:sp, 0.1, 5), (:bcp, 0.2, 5), (:fcp, 0.1, 3))
        row = only(r for r in lit["tables"][table] if r["c"] == c)
        k11, k33 = _msp_k(_msp_lattice(kind, c; e, k₁, interface = interface(_msp_l1(kind, c))), nmax)
        @test k11 ≈ row["k11_$(kind)"] atol = 1.5e-3
        @test k33 ≈ row["k33_$(kind)"] atol = 1.5e-3
    end
end

@testset "MEM spheroids — derivatives" begin
    # with respect to the contrast, on a periodic lattice, against a centered difference
    k(k₁) = _msp_k(_msp_lattice(:sp, 0.2; k₁), 3)[2]
    h = 1.0e-5
    @test ForwardDiff.derivative(k, 10.0) ≈ (k(10.0 + h) - k(10.0 - h)) / 2h rtol = 1.0e-6
end

@testset "MEM spheroids — refusals" begin
    tilted = ParticleAssembly(; boundary = MixedBC(3.0))
    add_matrix!(tilted, _MSP_K0)
    add_particle!(tilted, :p, (0.0, 0.0, 0.0), Ellipsoid(0.5, 0.2, 0.2), Dict(:K => TensISO{3}(5.0)))
    @test_throws ArgumentError homogenize(tilted, MultipoleExpansion(), :K)
    triaxial = ParticleAssembly(; boundary = MixedBC(3.0))
    add_matrix!(triaxial, _MSP_K0)
    add_particle!(triaxial, :p, (0.0, 0.0, 0.0), Ellipsoid(0.5, 0.3, 0.2), Dict(:K => TensISO{3}(5.0)))
    @test_throws ArgumentError homogenize(triaxial, MultipoleExpansion(), :K)
    tilted_layered = ParticleAssembly(; boundary = MixedBC(3.0))
    add_matrix!(tilted_layered, _MSP_K0)
    add_particle!(tilted_layered, :p, (0.0, 0.0, 0.0), LayeredSpheroid((0.5,), (0.2,), (TensISO{3}(5.0),); axis = (1.0, 0.0, 0.0)), Dict{Symbol, Any}())
    @test_throws ArgumentError homogenize(tilted_layered, MultipoleExpansion(), :K)
    anisotropic = ParticleAssembly(; boundary = MixedBC(3.0))
    add_matrix!(anisotropic, _MSP_K0)
    add_particle!(anisotropic, :p, (0.0, 0.0, 0.0), Ellipsoid(0.5, 0.5, 0.2), Dict(:K => TensTI{2}(1.0, 2.0, (0.0, 0.0, 1.0))))
    @test_throws ArgumentError homogenize(anisotropic, MultipoleExpansion(), :K)
    mixed = ParticleAssembly(; boundary = MixedBC(4.0))
    add_matrix!(mixed, _MSP_K0)
    add_particle!(mixed, :a, (0.0, 0.0, 0.0), Ellipsoid(0.5, 0.5, 0.2), Dict(:K => TensISO{3}(5.0)))
    add_particle!(mixed, :b, (2.0, 0.0, 0.0), Ellipsoid(0.4), Dict(:K => TensISO{3}(5.0)))
    @test_throws ArgumentError homogenize(mixed, MultipoleExpansion(), :K)
    spheroids = _msp_lattice(:sp, 0.1)
    @test_throws ArgumentError concentration_tensors(spheroids, MultipoleExpansion(; nmax = 1), :K)
    @test_throws ArgumentError multipole_coefficients(spheroids, :K; scheme = MultipoleExpansion(; nmax = 1))
end

@testset "rectangular periodic cells and the overlap of ellipsoids" begin
    box = PeriodicBox((1.0, 2.0, 0.5))
    @test box_periods(box) == (1.0, 2.0, 0.5)
    @test box.cutoff == 6.0
    @test_throws ArgumentError PeriodicBox((1.0, -2.0, 0.5))
    asm = ParticleAssembly(; boundary = PeriodicBox((1.0, 1.0, 1 / 3)))
    add_matrix!(asm, _MSP_K0)
    add_particle!(asm, :p, (0.0, 0.0, 0.0), Ellipsoid(0.3, 0.3, 0.1), Dict(:K => TensISO{3}(5.0)))
    @test assembly_volume(asm) ≈ 1 / 3
    @test _msp_k(asm, 1)[1] > 1
    # the schemes that truncate the image sums need a cubic cell
    @test_throws ArgumentError homogenize(asm, ClusterModel(), :K)
    # two oblate spheroids stacked along their axis: apart, their bounding balls
    # overlap; pressed together, they overlap
    stacked(z) = begin
        a = ParticleAssembly(; boundary = MixedBC(3.0))
        add_matrix!(a, _MSP_K0)
        add_particle!(a, :a, (0.0, 0.0, 0.0), Ellipsoid(0.5, 0.5, 0.1), Dict(:K => TensISO{3}(5.0)))
        add_particle!(a, :b, (0.0, 0.0, z), LayeredSpheroid((0.1,), (0.5,), (TensISO{3}(5.0),)), Dict{Symbol, Any}())
        validate_assembly(a)
    end
    @test stacked(0.21) isa ParticleAssembly
    @test_throws ArgumentError stacked(0.19)
    ball(x) = begin
        a = ParticleAssembly(; boundary = MixedBC(3.0))
        add_matrix!(a, _MSP_K0)
        add_particle!(a, :a, (0.0, 0.0, 0.0), Ellipsoid(0.5, 0.5, 0.1), Dict(:K => TensISO{3}(5.0)))
        add_particle!(a, :b, (x, 0.0, 0.0), Ellipsoid(0.3), Dict(:K => TensISO{3}(5.0)))
        validate_assembly(a)
    end
    @test ball(0.81) isa ParticleAssembly
    @test_throws ArgumentError ball(0.79)
end
