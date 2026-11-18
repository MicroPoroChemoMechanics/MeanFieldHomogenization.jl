using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using JSON

# =============================================================================
#  Multipole expansion method — anisotropic constituents in conduction.
#
#  References: one particle in a domain of its own shape is Mori-Tanaka, which
#  MFH computes with the Hill tensors of anisotropic media; an isotropic
#  conductivity written as a plain tensor gives the isotropic route; the
#  anisotropy parameter of Kushch & Sevostianov (2014), Table 5.
# =============================================================================

const _MAN_LIT = joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2014.json")
_man_K(asm, s) = canonical_components(homogenize(asm, s, :K))
_man_rotation(θ) = [cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1] * [1 0 0; 0 cos(0.7) -sin(0.7); 0 sin(0.7) cos(0.7)]
function _man_one(K₀, shape, P, domain)
    asm = ParticleAssembly(; boundary = MixedBC(domain))
    add_matrix!(asm, Dict(:K => K₀))
    add_particle!(asm, :p, (0.0, 0.0, 0.0), shape, Dict(:K => P))
    return asm
end

@testset "MEM anisotropic — one particle is Mori-Tanaka" begin
    Q = _man_rotation(0.4)
    K₁ = Tens(Q * Diagonal([5.0, 2.0, 8.0]) * transpose(Q))
    cases = (
        # a rotated orthotropic sphere, spheroid and ellipsoid in an isotropic matrix
        (TensISO{3}(1.0), Ellipsoid(1.0), K₁, Ellipsoid(2.0)),
        (TensISO{3}(1.0), Ellipsoid(1.0, 1.0, 0.5), K₁, Ellipsoid(2.0, 2.0, 1.0)),
        (TensISO{3}(1.0), Ellipsoid(1.5, 1.0, 0.5), K₁, Ellipsoid(3.0, 2.0, 1.0)),
        # an isotropic sphere in a transversely isotropic matrix: a spheroid once mapped
        (TensTI{2}(1.0, 4.0, (0.0, 0.0, 1.0)), Ellipsoid(1.0), TensISO{3}(10.0), Ellipsoid(2.0)),
        # both anisotropic, the matrix orthotropic
        (Tens(Matrix(Diagonal([1.0, 2.0, 3.0]))), Ellipsoid(1.5, 1.0, 0.5), K₁, Ellipsoid(3.0, 2.0, 1.0)),
    )
    for (K₀, shape, P, domain) in cases, nmax in (1, 3)
        asm = _man_one(K₀, shape, P, domain)
        # to the accuracy of the quadrature of the Lamé functions for the ellipsoids
        @test _man_K(asm, MultipoleExpansion(; nmax)) ≈ _man_K(asm, MoriTanaka()) rtol = 1.0e-9
    end
end

@testset "MEM anisotropic — an isotropic tensor takes the isotropic route" begin
    pair(P, shape) = begin
        asm = ParticleAssembly(; boundary = MixedBC(8.0))
        add_matrix!(asm, Dict(:K => TensISO{3}(1.0)))
        add_particle!(asm, :a, (0.0, 0.0, 0.0), shape, Dict(:K => P))
        add_particle!(asm, :b, (0.3, 0.2, 2.3), shape, Dict(:K => P))
        asm
    end
    plain = Tens(Matrix(10.0I, 3, 3))
    for shape in (Ellipsoid(1.0), Ellipsoid(1.0, 1.0, 0.6))
        @test _man_K(pair(plain, shape), MultipoleExpansion(; nmax = 4)) ≈
            _man_K(pair(TensISO{3}(10.0), shape), MultipoleExpansion(; nmax = 4)) rtol = 1.0e-12
    end
    # an isotropic matrix written as a plain tensor
    iso = _man_one(TensISO{3}(1.0), Ellipsoid(1.0, 1.0, 0.5), TensISO{3}(4.0), Ellipsoid(2.0, 2.0, 1.0))
    plain_matrix = _man_one(Tens(Matrix(1.0I, 3, 3)), Ellipsoid(1.0, 1.0, 0.5), TensISO{3}(4.0), Ellipsoid(2.0, 2.0, 1.0))
    @test _man_K(plain_matrix, MultipoleExpansion(; nmax = 3)) ≈ _man_K(iso, MultipoleExpansion(; nmax = 3)) rtol = 1.0e-12
end

@testset "MEM anisotropic — Kushch & Sevostianov (2014), Table 5" begin
    # spheroids of semi-axes (1, 1, e), conductivity 10 k₁₁, in a matrix of
    # conductivities k₁₁ = 1 and k₃₃ = κ about e₃; periods proportional to the
    # semi-axes; ϖ = (k*₃₃/κ)/(k*₁₁/1)
    ϖ(e, κ, c) = begin
        semi = (1.0, 1.0, e)
        asm = ParticleAssembly(; boundary = PeriodicBox(cbrt(4π / 3 / c) .* semi))
        add_matrix!(asm, Dict(:K => TensTI{2}(1.0, κ, (0.0, 0.0, 1.0))))
        add_particle!(asm, :p, (0.0, 0.0, 0.0), Ellipsoid(semi...), Dict(:K => TensISO{3}(10.0)))
        k = diag(_man_K(asm, MultipoleExpansion(; nmax = 6)))
        (k[3] / κ) / k[1]
    end
    rows = JSON.parsefile(_MAN_LIT)["tables"]["anisotropy_ti"]
    # a sphere (e = 0.5, κ = 0.25), an oblate (1, 4) and a prolate spheroid (2, 0.5) once mapped
    for (e, κ) in ((0.5, 0.25), (1.0, 4.0), (2.0, 0.5)), c in ("0.15", "0.30")
        row = only(filter(r -> r["aspect_ratio"] == e && r["matrix_anisotropy"] == κ, rows))
        @test ϖ(e, κ, parse(Float64, c)) ≈ row["c_$c"] atol = 6.0e-4
    end
end

@testset "MEM anisotropic — derivatives" begin
    h = 1.0e-5
    # with respect to one component of the conductivity of an anisotropic particle
    Q = _man_rotation(0.4)
    k(x) = begin
        P = Tens(Q * Diagonal([x, 2.0, 8.0]) * transpose(Q))
        _man_K(_man_one(TensISO{3}(1.0), Ellipsoid(1.0, 1.0, 0.5), P, Ellipsoid(2.0, 2.0, 1.0)), MultipoleExpansion(; nmax = 2))[1, 2]
    end
    @test ForwardDiff.derivative(k, 5.0) ≈ (k(5.0 + h) - k(5.0 - h)) / 2h rtol = 1.0e-6
    # with respect to a position in an anisotropic matrix, two anisotropic spheres
    kz(z) = begin
        asm = ParticleAssembly(; boundary = MixedBC(8.0))
        add_matrix!(asm, Dict(:K => TensTI{2}(1.0, 2.0, (0.0, 0.0, 1.0))))
        add_particle!(asm, :a, (0.0, 0.0, 0.0), Ellipsoid(1.0), Dict(:K => Tens(Q * Diagonal([5.0, 2.0, 8.0]) * transpose(Q))))
        add_particle!(asm, :b, (0.2, 0.0, z), Ellipsoid(1.0), Dict(:K => TensISO{3}(6.0)))
        _man_K(asm, MultipoleExpansion(; nmax = 2))[3, 3]
    end
    @test ForwardDiff.derivative(kz, 3.0) ≈ (kz(3.0 + h) - kz(3.0 - h)) / 2h rtol = 1.0e-5
end

@testset "MEM anisotropic — refusals" begin
    TI = TensTI{2}(1.0, 2.0, (0.0, 0.0, 1.0))
    one_in(K₀, shape, P = TensISO{3}(5.0)) = _man_one(K₀, shape, P, Ellipsoid(3.0))
    # a matrix not orthotropic in the axes of the assembly
    tilted_matrix = TensTI{2}(1.0, 2.0, (0.0, sin(0.3), cos(0.3)))
    @test_throws ArgumentError homogenize(one_in(tilted_matrix, Ellipsoid(1.0)), MultipoleExpansion(), :K)
    # a matrix whose anisotropy carries a derivative
    dual = TensTI{2}(1.0, ForwardDiff.Dual(2.0, 1.0), (0.0, 0.0, 1.0))
    @test_throws ArgumentError homogenize(one_in(dual, Ellipsoid(1.0)), MultipoleExpansion(), :K)
    # a rotated ellipsoid, an interface, a layered sphere in an anisotropic matrix
    @test_throws ArgumentError homogenize(one_in(TI, Ellipsoid(1.0, 0.5, 0.5, _man_rotation(0.3))), MultipoleExpansion(), :K)
    @test_throws ArgumentError homogenize(one_in(TI, InterfacedEllipsoid(1.5, 1.0, 0.5, KapitzaInterface(0.5))), MultipoleExpansion(), :K)
    layered = ParticleAssembly(; boundary = MixedBC(3.0))
    add_matrix!(layered, Dict(:K => TI))
    add_particle!(layered, :p, (0.0, 0.0, 0.0), LayeredSphere((0.5, 1.0), (TensISO{3}(5.0), TensISO{3}(2.0))), Dict{Symbol, Any}())
    @test_throws ArgumentError homogenize(layered, MultipoleExpansion(), :K)
    # an anisotropic ellipsoid behind an interface; an anisotropic sphere of
    # radius carrying a derivative; a layered sphere among anisotropic ones
    aniso = Tens(Matrix(Diagonal([5.0, 2.0, 8.0])))
    @test_throws ArgumentError homogenize(
        one_in(TensISO{3}(1.0), InterfacedEllipsoid(1.5, 1.0, 0.5, KapitzaInterface(0.5)), aniso), MultipoleExpansion(), :K
    )
    @test_throws ArgumentError homogenize(one_in(TensISO{3}(1.0), Ellipsoid(ForwardDiff.Dual(1.0, 1.0)), aniso), MultipoleExpansion(), :K)
    mixed = ParticleAssembly(; boundary = MixedBC(6.0))
    add_matrix!(mixed, Dict(:K => TensISO{3}(1.0)))
    add_particle!(mixed, :a, (0.0, 0.0, 0.0), Ellipsoid(1.0), Dict(:K => aniso))
    add_particle!(mixed, :b, (3.0, 0.0, 0.0), LayeredSphere((0.5, 1.0), (TensISO{3}(5.0), TensISO{3}(2.0))), Dict{Symbol, Any}())
    @test_throws ArgumentError homogenize(mixed, MultipoleExpansion(), :K)
end
