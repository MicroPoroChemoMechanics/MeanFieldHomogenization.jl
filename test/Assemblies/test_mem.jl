using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff

# =============================================================================
#  Multipole expansion method — conduction, finite assemblies of spheres.
# =============================================================================

const _MEM_K0 = TensISO{3}(1.0)

# Simple cubic block of n³ equal spheres of total fraction c inside the ball of
# the same fraction (the finite-cluster model of kushch2013, appendix A.1).
function _mem_cubic_cluster(n, c, K₁; a = 1.0)
    R = a * cbrt(3c / (4π))
    R_eff = R * cbrt(n^3 / c)
    asm = ParticleAssembly(; boundary = MixedBC(R_eff))
    add_matrix!(asm, Dict(:K => _MEM_K0))
    o = (n - 1) / 2
    for i in 1:n, j in 1:n, k in 1:n
        add_particle!(
            asm, Symbol(:p, i, j, k), (a * (i - 1 - o), a * (j - 1 - o), a * (k - 1 - o)),
            Ellipsoid(R), Dict(:K => K₁)
        )
    end
    return asm
end

@testset "MEM conduction — one sphere is Maxwell at any degree" begin
    for k₁ in (10.0, 0.1, 1.0), tmax in (1, 4)
        asm = ParticleAssembly(; boundary = MixedBC(2.0))
        add_matrix!(asm, Dict(:K => _MEM_K0))
        add_particle!(asm, :p, (0.3, -0.2, 0.1), Ellipsoid(1.1), Dict(:K => TensISO{3}(k₁)))
        f = (1.1 / 2.0)^3
        β = (k₁ - 1) / (k₁ + 2)
        K = homogenize(asm, MultipoleExpansion(; tmax), :K)
        @test canonical_components(K) ≈ (1 + 2f * β) / (1 - f * β) * I(3) rtol = 1.0e-13
        @test canonical_components(K) ≈ canonical_components(homogenize(asm, MoriTanaka(), :K)) rtol = 1.0e-13
    end
end

@testset "MEM conduction — finite cluster of Kushch (2013), appendix A.1" begin
    # 27 equal spheres on a 3×3×3 cubic block, fraction 0.2, contrast 10
    asm = _mem_cubic_cluster(3, 0.2, TensISO{3}(10.0))
    K = canonical_components(homogenize(asm, MultipoleExpansion(; tmax = 7), :K))
    @test K[1, 1] ≈ 1.53345 atol = 5.0e-6
    @test K ≈ K[1, 1] * I(3) rtol = 1.0e-12
    # the dipoles alone already differ from the one-site estimate
    K₁ = canonical_components(homogenize(asm, MultipoleExpansion(; tmax = 1), :K))[1, 1]
    K_mt = canonical_components(homogenize(asm, MoriTanaka(), :K))[1, 1]
    @test K_mt < K₁ < K[1, 1]
    @test canonical_components(homogenize(asm, :mem, :K))[1, 1] ≈
        canonical_components(homogenize(asm, MultipoleExpansion(), :K))[1, 1]
end

# Four unequal spheres of unequal conductivities, in the ball of radius 3.
function _mem_four_spheres(centers; k = (8.0, 0.2, 3.0, 15.0), R = (0.5, 0.35, 0.6, 0.4))
    asm = ParticleAssembly(; boundary = MixedBC(3.0))
    add_matrix!(asm, Dict(:K => _MEM_K0))
    for i in 1:4
        add_particle!(asm, Symbol(:p, i), centers[i], Ellipsoid(R[i]), Dict(:K => TensISO{3}(k[i])))
    end
    return asm
end

const _MEM_CENTERS = ([0.0, 0.0, 0.0], [1.0, 0.2, -0.1], [-0.3, 1.2, 0.4], [0.2, -0.5, 1.1])

@testset "MEM conduction — frame indifference and silent particles" begin
    K = canonical_components(homogenize(_mem_four_spheres(_MEM_CENTERS), MultipoleExpansion(; tmax = 4), :K))
    @test K ≈ transpose(K) rtol = 1.0e-13
    @test !(K ≈ K[1, 1] * I(3))                        # an anisotropic arrangement
    θ, φ = 0.7, -1.2
    Q = [cos(φ) -sin(φ) 0; sin(φ) cos(φ) 0; 0 0 1] * [1 0 0; 0 cos(θ) -sin(θ); 0 sin(θ) cos(θ)]
    Kq = canonical_components(
        homogenize(_mem_four_spheres(map(c -> Q * c, _MEM_CENTERS)), MultipoleExpansion(; tmax = 4), :K)
    )
    @test Kq ≈ Q * K * transpose(Q) rtol = 1.0e-11
    # a sphere of the matrix conductivity sends nothing back
    asm = _mem_four_spheres(_MEM_CENTERS; k = (8.0, 0.2, 1.0, 15.0))
    ref = ParticleAssembly(; boundary = MixedBC(3.0))
    add_matrix!(ref, Dict(:K => _MEM_K0))
    for (i, k, R) in ((1, 8.0, 0.5), (2, 0.2, 0.35), (4, 15.0, 0.4))
        add_particle!(ref, Symbol(:p, i), _MEM_CENTERS[i], Ellipsoid(R), Dict(:K => TensISO{3}(k)))
    end
    @test canonical_components(homogenize(asm, MultipoleExpansion(; tmax = 3), :K)) ≈
        canonical_components(homogenize(ref, MultipoleExpansion(; tmax = 3), :K)) rtol = 1.0e-13
end

@testset "MEM conduction — layered spheres and interfaces as particles" begin
    k₁, R, ρ = 6.0, 0.8, 0.4
    # one sphere behind a Kapitza resistance: Maxwell with the apparent conductivity
    asm = ParticleAssembly(; boundary = MixedBC(2.0))
    add_matrix!(asm, Dict(:K => _MEM_K0))
    sphere = LayeredSphere((R,), (TensISO{3}(k₁),); interfaces = (KapitzaInterface(ρ),))
    add_particle!(asm, :p, (0.0, 0.0, 0.0), sphere, Dict{Symbol, Any}())
    @test particle_volume_fraction(asm, :p) ≈ (R / 2.0)^3
    kₐ = k₁ / (1 + ρ * k₁ / R)
    f, β = (R / 2.0)^3, (kₐ - 1) / (kₐ + 2)
    @test canonical_components(homogenize(asm, MultipoleExpansion(; tmax = 3), :K)) ≈
        (1 + 2f * β) / (1 - f * β) * I(3) rtol = 1.0e-13
    # two particles: a perfect one-layer sphere is the ball of the same property
    function pair(geometry, props)
        a = ParticleAssembly(; boundary = MixedBC(3.0))
        add_matrix!(a, Dict(:K => _MEM_K0))
        add_particle!(a, :a, (0.0, 0.0, 0.9), geometry(0.8), props)
        add_particle!(a, :b, (0.0, 0.3, -0.9), geometry(0.7), props)
        return canonical_components(homogenize(a, MultipoleExpansion(; tmax = 5), :K))
    end
    layered = pair(r -> LayeredSphere((r,), (TensISO{3}(k₁),)), Dict{Symbol, Any}())
    @test layered ≈ pair(Ellipsoid, Dict(:K => TensISO{3}(k₁))) rtol = 1.0e-13
    # and a resistance lowers the conductivity it would have without it
    resisting = pair(
        r -> LayeredSphere((r,), (TensISO{3}(k₁),); interfaces = (KapitzaInterface(ρ),)),
        Dict{Symbol, Any}()
    )
    @test tr(resisting) < tr(layered)
end

@testset "MEM conduction — periodic arrays" begin
    M = Dict(:K => _MEM_K0)
    P(k) = Dict(:K => TensISO{3}(k))
    Kzz(asm, tmax) = canonical_components(homogenize(asm, MultipoleExpansion(; tmax), :K))[3, 3]
    # the dipoles alone on a cubic array are Mori-Tanaka, exactly
    for (kind, c) in ((:sc, 0.3), (:bcc, 0.4), (:fcc, 0.5))
        asm = cubic_lattice(kind, M, P(10.0); fraction = c)
        K = canonical_components(homogenize(asm, MultipoleExpansion(; tmax = 1), :K))
        @test K ≈ canonical_components(homogenize(asm, MoriTanaka(), :K)) rtol = 1.0e-12
    end
    # Kushch & Sevostianov (2014), Table 2: simple cubic array, three decimals
    for (c, k₁, ref) in ((0.1, 0.0, 0.857), (0.3, 10.0, 1.888), (0.4, 100.0, 3.133), (0.2, 1000.0, 1.753))
        @test Kzz(cubic_lattice(:sc, M, P(k₁); fraction = c), 9) ≈ ref atol = 5.0e-4
    end
    # Kushch (2013), Tables 3.1 to 3.3: body- and face-centered arrays
    @test Kzz(cubic_lattice(:bcc, M, P(1.0e8); fraction = 0.65), 21) ≈ 9.027 atol = 2.0e-3
    # the same array described by a larger cell: 8 spheres in the cube of side 2
    one_cell = cubic_lattice(:sc, M, P(10.0); fraction = 0.3, period = 1.0)
    R = particle_geometry(one_cell, :p1).semi_axes[1]
    big = ParticleAssembly(; boundary = PeriodicBox(2.0))
    add_matrix!(big, M)
    for i in 0:1, j in 0:1, k in 0:1
        add_particle!(big, Symbol(:p, i, j, k), (0.3 + i, -0.2 + j, 0.1 + k), Ellipsoid(R), P(10.0))
    end
    @test Kzz(big, 5) ≈ Kzz(one_cell, 5) rtol = 1.0e-10
    @test box_periods(big.boundary) == (2.0, 2.0, 2.0)
end

@testset "MEM conduction — derivatives" begin
    M = Dict(:K => _MEM_K0)
    # contrast, periodic dipoles: the Maxwell formula differentiated
    c = 0.3
    maxwell(k) = (1 + 2c * (k - 1) / (k + 2)) / (1 - c * (k - 1) / (k + 2))
    periodic(k) = canonical_components(
        homogenize(cubic_lattice(:sc, M, Dict(:K => TensISO{3}(k)); fraction = c), MultipoleExpansion(; tmax = 1), :K)
    )[1, 1]
    @test ForwardDiff.derivative(periodic, 10.0) ≈ ForwardDiff.derivative(maxwell, 10.0) rtol = 1.0e-11
    @test ForwardDiff.derivative(k -> ForwardDiff.derivative(periodic, k), 10.0) ≈
        ForwardDiff.derivative(k -> ForwardDiff.derivative(maxwell, k), 10.0) rtol = 1.0e-10
    # position and radius of one sphere of a pair, against a centered difference
    function pair(z, r)
        a = ParticleAssembly(; boundary = MixedBC(3.0))
        add_matrix!(a, M)
        add_particle!(a, :a, [0.0, 0.0, z], Ellipsoid(r), Dict(:K => TensISO{3}(12.0)))
        add_particle!(a, :b, [0.0, 0.3, -0.9], Ellipsoid(0.7), Dict(:K => TensISO{3}(0.1)))
        return canonical_components(homogenize(a, MultipoleExpansion(; tmax = 4), :K))[3, 3]
    end
    h = 1.0e-5
    @test ForwardDiff.derivative(z -> pair(z, 0.8), 0.9) ≈ (pair(0.9 + h, 0.8) - pair(0.9 - h, 0.8)) / 2h rtol = 1.0e-7
    @test ForwardDiff.derivative(r -> pair(0.9, r), 0.8) ≈ (pair(0.9, 0.8 + h) - pair(0.9, 0.8 - h)) / 2h rtol = 1.0e-7
    # the position of a sphere of a periodic cell, through the lattice sums
    function cell(x)
        a = ParticleAssembly(; boundary = PeriodicBox(1.0))
        add_matrix!(a, M)
        add_particle!(a, :a, [0.0, 0.0, 0.0], Ellipsoid(0.25), Dict(:K => TensISO{3}(12.0)))
        add_particle!(a, :b, [x, 0.45, 0.5], Ellipsoid(0.2), Dict(:K => TensISO{3}(0.1)))
        return canonical_components(homogenize(a, MultipoleExpansion(; tmax = 3), :K))[1, 1]
    end
    @test ForwardDiff.derivative(cell, 0.4) ≈ (cell(0.4 + h) - cell(0.4 - h)) / 2h rtol = 1.0e-6
end

@testset "MEM — refusals" begin
    M = Dict(:K => _MEM_K0)
    base() = (a = ParticleAssembly(; boundary = MixedBC(3.0)); add_matrix!(a, M); a)
    asm = base()
    add_particle!(asm, :p, (0.0, 0.0, 0.0), Ellipsoid(0.5), Dict(:K => TensISO{3}(5.0)))
    @test_throws ArgumentError homogenize(asm, MultipoleExpansion(; tmax = 0), :K)
    # a spheroid, an anisotropic sphere, an anisotropic matrix
    spheroid = base()
    add_particle!(spheroid, :p, (0.0, 0.0, 0.0), Ellipsoid(0.5, 0.5, 0.3), Dict(:K => TensISO{3}(5.0)))
    @test_throws ArgumentError homogenize(spheroid, MultipoleExpansion(), :K)
    aniso = base()
    add_particle!(aniso, :p, (0.0, 0.0, 0.0), Ellipsoid(0.5), Dict(:K => Tens([1.0 0 0; 0 2.0 0; 0 0 3.0])))
    @test_throws ArgumentError homogenize(aniso, MultipoleExpansion(), :K)
    matrix = ParticleAssembly(; boundary = MixedBC(3.0))
    add_matrix!(matrix, Dict(:K => Tens([1.0 0 0; 0 2.0 0; 0 0 3.0])))
    add_particle!(matrix, :p, (0.0, 0.0, 0.0), Ellipsoid(0.5), Dict(:K => TensISO{3}(5.0)))
    @test_throws ArgumentError homogenize(matrix, MultipoleExpansion(), :K)
    # a plane assembly
    plane = ParticleAssembly(; boundary = MixedBC(Ellipsoid(3.0, 3.0)))
    add_matrix!(plane, Dict(:K => TensISO{2}(1.0)))
    add_particle!(plane, :p, (0.0, 0.0), Ellipsoid(0.5, 0.5), Dict(:K => TensISO{2}(5.0)))
    @test_throws ArgumentError homogenize(plane, MultipoleExpansion(), :K)
    # the schemes of one polarization per inclusion have no use for a layered sphere
    layered = base()
    add_particle!(layered, :p, (0.0, 0.0, 0.0), LayeredSphere((0.5,), (TensISO{3}(5.0),)), Dict{Symbol, Any}())
    @test_throws ArgumentError homogenize(layered, ClusterModel(), :K)
    @test_throws ArgumentError homogenize(layered, EquivalentInclusion(), :K)
    # a dense system the machine cannot hold is refused before it is allocated
    check = MeanFieldHomogenization.Assemblies._check_system_size
    @test check(10, ComplexF64, 1, 2) === nothing
    @test_throws ArgumentError check(100_000, ComplexF64, 4000, 4; available = 2^30)
end

@testset "MEM conduction — concentration tensors" begin
    M = Dict(:K => _MEM_K0)
    rel(a, b) = norm(canonical_components(a) - canonical_components(b)) / norm(canonical_components(b))
    Id = TensISO{3}(1.0)
    # a finite assembly with a plain ball, a resisting sphere, a surface-conducting
    # one and a ball of the matrix conductivity; and a periodic cell
    finite = ParticleAssembly(; boundary = MixedBC(3.0))
    add_matrix!(finite, M)
    add_particle!(finite, :ball, _MEM_CENTERS[1], Ellipsoid(0.5), Dict(:K => TensISO{3}(8.0)))
    add_particle!(
        finite, :kapitza, _MEM_CENTERS[2],
        LayeredSphere((0.35,), (TensISO{3}(6.0),); interfaces = (KapitzaInterface(0.3),)), Dict{Symbol, Any}()
    )
    add_particle!(
        finite, :surface, _MEM_CENTERS[3],
        LayeredSphere((0.3, 0.6), (TensISO{3}(0.2), TensISO{3}(4.0)); interfaces = (PerfectInterface{Float64}(), SurfaceConductiveInterface(0.7))),
        Dict{Symbol, Any}()
    )
    add_particle!(finite, :silent, _MEM_CENTERS[4], Ellipsoid(0.4), Dict(:K => _MEM_K0))
    periodic = ParticleAssembly(; boundary = PeriodicBox(1.0))
    add_matrix!(periodic, M)
    add_particle!(periodic, :a, [0.0, 0.0, 0.0], Ellipsoid(0.25), Dict(:K => TensISO{3}(12.0)))
    add_particle!(periodic, :b, [0.4, 0.45, 0.5], Ellipsoid(0.2), Dict(:K => TensISO{3}(0.1)))
    for asm in (finite, periodic)
        scheme = MultipoleExpansion(; tmax = 4)
        ct = concentration_tensors(asm, scheme, :K)
        @test first(keys(ct)) === :matrix
        @test rel(ct.effective, homogenize(asm, scheme, :K)) < 1.0e-13
        # the averages over the phases are the macroscopic ones
        @test rel(sum(ct.weights[n] * ct.strain[n] for n in keys(ct)), Id) < 1.0e-12
        @test rel(sum(ct.weights[n] * ct.stress[n] for n in keys(ct)), ct.effective) < 1.0e-12
        # the matrix, whose averages are what the particles leave, obeys its own
        # law: the dipoles and the flux averages of the spheres are consistent
        @test rel(ct.stress[:matrix], _MEM_K0 ⋅ ct.strain[:matrix]) < 1.0e-11
    end
    ct = concentration_tensors(finite, MultipoleExpansion(; tmax = 4), :K)
    # a homogeneous ball obeys its law too, a silent one included
    @test rel(ct.stress[:ball], TensISO{3}(8.0) ⋅ ct.strain[:ball]) < 1.0e-12
    @test rel(ct.stress[:silent], ct.strain[:silent]) < 1.0e-12
    @test !(canonical_components(ct.strain[:silent]) ≈ I(3))      # yet it feels its neighbors
    @test haskey(ct.properties, :ball) && !haskey(ct.properties, :kapitza)
    # one ball in a ball: the concentration of Mori-Tanaka
    one_ball = ParticleAssembly(; boundary = MixedBC(2.0))
    add_matrix!(one_ball, M)
    add_particle!(one_ball, :p, (0.0, 0.0, 0.0), Ellipsoid(1.1), Dict(:K => TensISO{3}(10.0)))
    f, a = (1.1 / 2.0)^3, 3 / (10.0 + 2)
    @test rel(concentration_tensors(one_ball, MultipoleExpansion(), :K).strain[:p], TensISO{3}(a / (1 - f + f * a))) < 1.0e-12
end

@testset "MEM conduction — coefficients of the expansion" begin
    # one ball: the dipole of the uniform gradient and nothing else
    k₁, R = 10.0, 1.1
    asm = ParticleAssembly(; boundary = MixedBC(2.0))
    add_matrix!(asm, Dict(:K => _MEM_K0))
    add_particle!(asm, :p, (0.0, 0.0, 0.0), Ellipsoid(R), Dict(:K => TensISO{3}(k₁)))
    c = multipole_coefficients(asm; scheme = MultipoleExpansion(; tmax = 3))
    b₁ = -(k₁ - 1) * R^3 / (k₁ + 2)
    @test c[:p, 1, 0] ≈ [0, 0, b₁]                      # A₁₀ = b₁ G₃
    @test c[:p, 1, 1] ≈ [b₁ / 2, -im * b₁ / 2, 0]       # A₁₁ = b₁ (G₁ − i G₂)/2
    @test c[:p, 1, -1] ≈ [-b₁ / 2, -im * b₁ / 2, 0]
    @test all(iszero, c[:p, 2, 1]) && all(iszero, c[:p, 3, -2])
    @test c.names == [:p] && c.tmax == 3
    @test sprint(show, c) == "MultipoleCoefficients(1 particles, tmax = 3)"
    @test_throws KeyError c[:q, 1, 0]
    @test_throws BoundsError c[:p, 4, 0]
    @test_throws BoundsError c[:p, 1, 2]
    # two perfectly conducting balls: the temperature rebuilt from the series
    # is uniform on the surface of each, whatever the remote gradient
    pair = ParticleAssembly(; boundary = MixedBC(3.0))
    add_matrix!(pair, Dict(:K => _MEM_K0))
    centers = Dict(:a => [0.0, 0.1, 1.0], :b => [0.2, 0.0, -1.0])
    radii = Dict(:a => 0.6, :b => 0.5)
    for n in (:a, :b)
        add_particle!(pair, n, centers[n], Ellipsoid(radii[n]), Dict(:K => TensISO{3}(1.0e12)))
    end
    tmax = 12
    cp = multipole_coefficients(pair; scheme = MultipoleExpansion(; tmax))
    function temperature(x, i)
        T = x[i]
        for n in (:a, :b)
            Y = irregular_harmonics(x - centers[n], tmax)
            T += real(sum(cp[n, t, s][i] * Y[t, s] for t in 1:tmax for s in (-t):t))
        end
        return T
    end
    directions = ([0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0], [0, -1.0, 0], [0.6, 0, 0.8], [0, 0.8, -0.6])
    for n in (:a, :b), i in 1:3
        Ts = [temperature(centers[n] + radii[n] * d, i) for d in directions]
        @test maximum(Ts) - minimum(Ts) < 1.0e-6
    end
end
