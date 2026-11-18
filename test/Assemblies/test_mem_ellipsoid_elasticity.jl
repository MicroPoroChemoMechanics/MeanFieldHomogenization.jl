using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using ForwardDiff
using JSON

# =============================================================================
#  Multipole expansion method — assemblies of ellipsoids in elasticity.
#
#  References: one ellipsoid, perfectly bonded or empty, in a domain of its own
#  shape is Mori-Tanaka (Eshelby's solution); the Maxwell estimates with a
#  spring interface of Kushch (2019a), Table 5; the near and far forms of a
#  pair against each other.
# =============================================================================

_mee_iso(μ, ν) = TensISO{3}(2μ * (1 + ν) / (1 - 2ν), 2μ)
_mee_mandel(t) = TensND.mandel(t, TensND.CanonicalBasis{3, eltype(TensND.get_array(t))}())
_mee_C(asm, scheme) = _mee_mandel(homogenize(asm, scheme, :C))
const _MEE_C0 = _mee_iso(1.0, 0.3)

function _mee_one(particle, P; c = 0.5, semi = (1.5, 1.0, 0.5))
    asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid((semi ./ cbrt(c))...)))
    add_matrix!(asm, Dict(:C => _MEE_C0))
    add_particle!(asm, :p, (0.0, 0.0, 0.0), particle, Dict(:C => P))
    return asm
end

@testset "MEM elastic ellipsoids — one ellipsoid is Mori-Tanaka" begin
    # an inclusion, a cavity, and the axes of the assembly not ordered by size
    for (semi, P) in (((1.5, 1.0, 0.5), _mee_iso(100.0, 0.2)), ((1.5, 1.0, 0.5), TensISO{3}(0.0, 0.0)), ((0.5, 1.5, 1.0), _mee_iso(5.0, 0.25)))
        asm = _mee_one(Ellipsoid(semi...), P; semi)
        @test _mee_C(asm, MultipoleExpansion(; nmax = 2)) ≈ _mee_C(asm, MoriTanaka()) rtol = 1.0e-9
    end
end

@testset "MEM elastic ellipsoids — spring interface, Kushch (2019a), Table 5" begin
    lit = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2019a.json"))
    q(k) = lit["quantities"][k]["value"]
    semi = Tuple(q("semi_axes"))
    keys = ("C1111" => (1, 1), "C2222" => (2, 2), "C3333" => (3, 3), "C2233" => (2, 3), "C1122" => (1, 2), "C2323" => (4, 4), "C1212" => (6, 6))
    for row in lit["tables"]["maxwell_spring"]
        row["D"] isa Number && row["D"] > 0 || continue
        # the stiffness D_n = D_t = D G₀ (G₀ = 1) recovers the table
        D = row["D"]
        asm = _mee_one(InterfacedEllipsoid(semi..., SpringInterface(D, D)), _mee_iso(q("mu_ratio"), q("nu1")); c = q("fraction"), semi)
        M = _mee_C(asm, MultipoleExpansion(; nmax = 8))
        for (key, (i, j)) in keys
            @test M[i, j] / (i > 3 ? 2 : 1) ≈ row[key] atol = 2.0e-4
        end
    end
end

@testset "MEM elastic ellipsoids — Maxwell estimate, Kushch (2019b), Table 2" begin
    lit = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2019b.json"))
    q(k) = lit["quantities"][k]["value"]
    semi = Tuple(q("semi_axes"))
    row = only(filter(r -> r["N"] == 1, lit["tables"]["cluster_maxwell"]))
    M = _mee_C(_mee_one(Ellipsoid(semi...), _mee_iso(q("mu_ratio"), q("nu1")); c = q("fraction"), semi), MultipoleExpansion(; nmax = 2))
    for (key, (i, j)) in ("C1111" => (1, 1), "C2222" => (2, 2), "C3333" => (3, 3), "C2233" => (2, 3), "C1212" => (6, 6))
        # printed with three decimals or four significant digits
        @test M[i, j] / (i > 3 ? 2 : 1) ≈ row[key] rtol = 1.0e-3
    end
end

@testset "MEM elastic ellipsoids — a pair, near and far" begin
    pair(z) = begin
        asm = ParticleAssembly(; boundary = MixedBC(20.0))
        add_matrix!(asm, Dict(:C => _MEE_C0))
        add_particle!(asm, :a, (0.0, 0.0, 0.0), Ellipsoid(1.5, 1.0, 0.5), Dict(:C => _mee_iso(10.0, 0.2)))
        add_particle!(asm, :b, (0.3, 0.0, z), Ellipsoid(1.5, 1.0, 0.5), Dict(:C => _mee_iso(10.0, 0.2)))
        _mee_C(asm, MultipoleExpansion(; nmax = 3))
    end
    # the near form below 2.5 times the sum of the focal distances h₂ = √2
    rn = 2.5 * 2 * sqrt(2.0)
    # the centers 0.3 apart across the axis: z of the threshold
    z = sqrt(rn^2 - 0.3^2)
    # the two forms truncate differently; they agree to that truncation
    @test pair(z - 1.0e-6) ≈ pair(z + 1.0e-6) rtol = 1.0e-7
end

@testset "MEM elastic ellipsoids — derivatives and refusals" begin
    h = 1.0e-5
    semi = (1.5, 1.0, 0.5)
    # with respect to the spring stiffness and to the shear modulus of the ellipsoid
    k(D) = _mee_C(_mee_one(InterfacedEllipsoid(semi..., SpringInterface(D, 2D)), _mee_iso(100.0, 0.2)), MultipoleExpansion(; nmax = 2))[1, 1]
    @test ForwardDiff.derivative(k, 3.0) ≈ (k(3.0 + h) - k(3.0 - h)) / 2h rtol = 1.0e-5
    g(μ₁) = _mee_C(_mee_one(Ellipsoid(semi...), _mee_iso(μ₁, 0.2)), MultipoleExpansion(; nmax = 2))[2, 3]
    @test ForwardDiff.derivative(g, 7.0) ≈ (g(7.0 + h) - g(7.0 - h)) / 2h rtol = 1.0e-5
    # with respect to a position, near
    kz(z) = begin
        asm = ParticleAssembly(; boundary = MixedBC(20.0))
        add_matrix!(asm, Dict(:C => _MEE_C0))
        add_particle!(asm, :a, (0.0, 0.0, 0.0), Ellipsoid(semi...), Dict(:C => _mee_iso(10.0, 0.2)))
        add_particle!(asm, :b, (0.2, 0.0, z), Ellipsoid(semi...), Dict(:C => _mee_iso(10.0, 0.2)))
        _mee_C(asm, MultipoleExpansion(; nmax = 2))[3, 3]
    end
    @test ForwardDiff.derivative(kz, 1.4) ≈ (kz(1.4 + h) - kz(1.4 - h)) / 2h rtol = 1.0e-5
    # a Kapitza interface, an anisotropic stiffness, ellipsoids oriented differently
    @test_throws ArgumentError homogenize(_mee_one(InterfacedEllipsoid(semi..., KapitzaInterface(1.0)), _mee_iso(10.0, 0.2)), MultipoleExpansion(), :C)
    @test_throws ArgumentError homogenize(_mee_one(Ellipsoid(semi...), TensTI{4}(10.0, 6.0, 8.0, 5.0, 2.0, (0.0, 0.0, 1.0))), MultipoleExpansion(), :C)
    crossed = ParticleAssembly(; boundary = MixedBC(20.0))
    add_matrix!(crossed, Dict(:C => _MEE_C0))
    add_particle!(crossed, :a, (0.0, 0.0, 0.0), Ellipsoid(1.5, 1.0, 0.5), Dict(:C => _mee_iso(10.0, 0.2)))
    add_particle!(crossed, :b, (5.0, 0.0, 0.0), Ellipsoid(1.0, 1.5, 0.5), Dict(:C => _mee_iso(10.0, 0.2)))
    @test_throws ArgumentError homogenize(crossed, MultipoleExpansion(), :C)
end
