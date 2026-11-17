using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using JSON

# =============================================================================
#  Multipole expansion method — assemblies of spheroids in elasticity.
#
#  References: one spheroid in a domain of its own shape is Mori-Tanaka
#  (Eshelby's solution), exactly from the degree 2; the near and far forms of
#  a pair against each other; one lattice described by two cells; the
#  stiffness of Kushch (2013), Table 7.6.
# =============================================================================

const _MSE = MeanFieldHomogenization.Assemblies
_mse_iso(κ, μ) = TensISO{3}(3κ, 2μ)
_mse_mandel(t) = TensND.mandel(t, TensND.CanonicalBasis{3, eltype(TensND.get_array(t))}())
_mse_C(asm, scheme) = _mse_mandel(homogenize(asm, scheme, :C))
const _MSE_C0 = _mse_iso(1.0, 0.6)

function _mse_one(semi, P; boundary = MixedBC(Ellipsoid((2 .* semi)...)), C₀ = _MSE_C0)
    asm = ParticleAssembly(; boundary)
    add_matrix!(asm, Dict(:C => C₀))
    add_particle!(asm, :p, (0.0, 0.0, 0.0), Ellipsoid(semi...), Dict(:C => P))
    return asm
end

@testset "MEM elastic spheroids — one spheroid is Mori-Tanaka from the degree 2" begin
    for (semi, P) in (((1.0, 1.0, 2.0), _mse_iso(10.0, 6.0)), ((1.0, 1.0, 0.2), _mse_iso(10.0, 6.0)), ((1.0, 1.0, 0.5), _mse_iso(0.0, 0.0)))
        asm = _mse_one(semi, P)
        @test _mse_C(asm, MultipoleExpansion(; nmax = 2)) ≈ _mse_C(asm, MoriTanaka()) rtol = 1.0e-12
    end
end

@testset "MEM elastic spheroids — a spring interface against the layered spheroid" begin
    # the response by least squares on a meridian against the transfer of the
    # Papkovich-Neuber potentials of `LayeredSpheroid`, two independent routes
    for (kn, kt) in ((2.0, 3.0), (0.2, 0.5))
        P = _mse_iso(10.0, 6.0)
        asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid(2.0, 2.0, 1.0)))
        add_matrix!(asm, Dict(:C => _MSE_C0))
        add_particle!(asm, :p, (0.0, 0.0, 0.0), LayeredSpheroid((0.5,), (1.0,), (P,); interfaces = (SpringInterface(kn, kt),)), Dict(:C => P))
        @test _mse_C(asm, MultipoleExpansion(; nmax = 8)) ≈ _mse_C(asm, MoriTanaka()) rtol = 1.0e-6
    end
end

@testset "MEM elastic spheroids — a pair, near and far; a lattice, one and two cells" begin
    pair(z) = begin
        asm = ParticleAssembly(; boundary = MixedBC(12.0))
        add_matrix!(asm, Dict(:C => _MSE_C0))
        add_particle!(asm, :a, (0.0, 0.0, 0.0), Ellipsoid(1.0, 1.0, 0.5), Dict(:C => _mse_iso(10.0, 6.0)))
        add_particle!(asm, :b, (0.3, 0.0, z), Ellipsoid(1.0, 1.0, 0.5), Dict(:C => _mse_iso(10.0, 6.0)))
        _mse_C(asm, MultipoleExpansion(; nmax = 4))
    end
    # the near form below 2.5 times the sum of the focal distances, √3, the far one above
    rn = 2.5 * 2 * sqrt(0.75)
    # the centers 0.3 apart across the axis: z of the threshold
    z = sqrt(rn^2 - 0.3^2)
    # the two forms truncate differently; they agree to that truncation
    @test pair(z - 1.0e-6) ≈ pair(z + 1.0e-6) rtol = 1.0e-8
    P = _mse_iso(10.0, 6.0)
    one_cell = _mse_one((1.0, 1.0, 0.5), P; boundary = PeriodicBox((3.0, 3.0, 2.0)))
    two_cells = ParticleAssembly(; boundary = PeriodicBox((3.0, 3.0, 4.0)))
    add_matrix!(two_cells, Dict(:C => _MSE_C0))
    for k in 0:1
        add_particle!(two_cells, Symbol(:p, k), (0.1, -0.2, 2.0k + 0.05), Ellipsoid(1.0, 1.0, 0.5), Dict(:C => P))
    end
    @test _mse_C(two_cells, MultipoleExpansion(; nmax = 3)) ≈ _mse_C(one_cell, MultipoleExpansion(; nmax = 3)) rtol = 1.0e-10
end

@testset "MEM elastic spheroids — Kushch (2013), Table 7.6" begin
    lit = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2013.json"))
    # ν₀ = ν₁ = 0.3, μ₀ = 1; spheroids (1, 1, ε) on a lattice of periods proportional to them
    iso(μ) = _mse_iso(2μ * 1.3 / 1.2, μ)
    for row in lit["tables"]["stiffness_spheroids"]
        ε, μ₁ = row["aspect_ratio"], row["contrast"]
        ε < 1 || continue
        semi = (1.0, 1.0, ε)
        asm = _mse_one(semi, iso(μ₁); boundary = PeriodicBox(cbrt(4π / 3 / 0.1) .* semi), C₀ = iso(1.0))
        # a flat cavity converges more slowly in the degree
        nmax = ε == 0.2 && μ₁ == 0 ? 8 : 6
        @test _mse_C(asm, MultipoleExpansion(; nmax))[3, 3] ≈ row["c_0.1"] atol = 6.0e-4
    end
end

@testset "MEM elastic spheroids — derivatives" begin
    h = 1.0e-5
    # with respect to the shear modulus of the spheroid, and to that of the matrix on a lattice
    k(μ₁) = _mse_C(_mse_one((1.0, 1.0, 0.5), _mse_iso(10.0, μ₁)), MultipoleExpansion(; nmax = 2))[3, 3]
    @test ForwardDiff.derivative(k, 6.0) ≈ (k(6.0 + h) - k(6.0 - h)) / 2h rtol = 1.0e-6
    g(μ₀) = _mse_C(_mse_one((1.0, 1.0, 0.5), _mse_iso(10.0, 6.0); boundary = PeriodicBox((3.0, 3.0, 2.0)), C₀ = _mse_iso(1.0, μ₀)), MultipoleExpansion(; nmax = 2))[1, 3]
    @test ForwardDiff.derivative(g, 0.6) ≈ (g(0.6 + h) - g(0.6 - h)) / 2h rtol = 1.0e-5
    # with respect to a position, near and far
    kz(z) = begin
        asm = ParticleAssembly(; boundary = MixedBC(10.0))
        add_matrix!(asm, Dict(:C => _MSE_C0))
        add_particle!(asm, :a, (0.0, 0.0, 0.0), Ellipsoid(1.0, 1.0, 0.5), Dict(:C => _mse_iso(10.0, 6.0)))
        add_particle!(asm, :b, (0.2, 0.0, z), Ellipsoid(1.0, 1.0, 0.5), Dict(:C => _mse_iso(10.0, 6.0)))
        _mse_C(asm, MultipoleExpansion(; nmax = 2))[3, 3]
    end
    for z in (1.3, 5.0)
        @test ForwardDiff.derivative(kz, z) ≈ (kz(z + h) - kz(z - h)) / 2h rtol = 1.0e-5
    end
end

@testset "MEM elastic spheroids — refusals" begin
    one_in(geometry, P = _mse_iso(10.0, 6.0)) = begin
        asm = ParticleAssembly(; boundary = MixedBC(6.0))
        add_matrix!(asm, Dict(:C => _MSE_C0))
        add_particle!(asm, :p, (0.0, 0.0, 0.0), geometry, Dict(:C => P))
        asm
    end
    # an anisotropic spheroid, semi-axes carrying a derivative
    @test_throws ArgumentError homogenize(one_in(Ellipsoid(1.0, 1.0, 0.5), TensTI{4}(10.0, 6.0, 8.0, 5.0, 2.0, (0.0, 0.0, 1.0))), MultipoleExpansion(), :C)
    @test_throws ArgumentError homogenize(one_in(Ellipsoid(1.0, 1.0, ForwardDiff.Dual(0.5, 1.0))), MultipoleExpansion(), :C)
    # a layered spheroid, and one behind a spring
    layered = ParticleAssembly(; boundary = MixedBC(6.0))
    add_matrix!(layered, Dict(:C => _MSE_C0))
    add_particle!(layered, :p, (0.0, 0.0, 0.0), LayeredSpheroid((sqrt(0.06), 0.5), (0.9, 1.0), (_mse_iso(10.0, 6.0), _mse_iso(5.0, 3.0))), Dict{Symbol, Any}())
    @test_throws ArgumentError homogenize(layered, MultipoleExpansion(), :C)
    kapitza = ParticleAssembly(; boundary = MixedBC(6.0))
    add_matrix!(kapitza, Dict(:C => _MSE_C0))
    add_particle!(kapitza, :p, (0.0, 0.0, 0.0), LayeredSpheroid((0.5,), (1.0,), (_mse_iso(10.0, 6.0),); interfaces = (KapitzaInterface(1.0),)), Dict{Symbol, Any}())
    @test_throws ArgumentError homogenize(kapitza, MultipoleExpansion(), :C)
end
