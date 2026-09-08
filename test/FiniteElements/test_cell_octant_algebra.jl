# The mirror-parity algebra of the octant cell, and the shape trait that guards
# it. Everything here is closed form: no mesh, no solve, no gmsh -- which is
# exactly why it comes first. It is the part of the reasoning that convergence
# cannot check, and the part that decides whether an octant is admissible at
# all.

using Test
using MeanFieldHomogenization
using LinearAlgebra

const _FE = MeanFieldHomogenization.FiniteElements
const _CORE = MeanFieldHomogenization.Core

"The reflection that flips axis `k`."
_mirror(k) = (R = Matrix(1.0I, 3, 3); R[k, k] = -1.0; R)

# A shape that deliberately does not opt in, to check that the guard is a guard.
struct _NotMirroredShape <: MeanFieldHomogenization.Superspheres.AbstractSuperShape{Float64} end

@testset "Octant — mirror parity and the shape trait" begin

    @testset "the parity of each load case" begin
        # `χ_k = +1` exactly when the case is even under the reflection that
        # flips axis k. Normal strains are even under all three; each shear is
        # odd under the two axes it names.
        expected = (
            (1, 1, 1), (1, 1, 1), (1, 1, 1),
            (1, -1, -1), (-1, 1, -1), (-1, -1, 1),
        )
        for i in 1:6
            @test _FE._cell_parity(_FE._cell_kelvin_basis(i)) == Int8.(expected[i])
        end
        for i in 1:3
            e = ntuple(k -> k == i ? 1.0 : 0.0, 3)
            @test _FE._cell_parity(e) == Int8.(ntuple(k -> k == i ? -1 : 1, 3))
        end
        # A vector argument must agree with the tuple one.
        @test _FE._cell_parity([0.0, 1.0, 0.0]) == _FE._cell_parity((0.0, 1.0, 0.0))

        # Four distinct classes in elasticity, three in transport. This is the
        # number of factorizations an octant pays, so it is worth pinning.
        @test length(_FE._CELL_ELASTIC_CLASSES) == 4
        @test length(_FE._CELL_COND_CLASSES) == 3
        @test Int8.((1, 1, 1)) in _FE._CELL_ELASTIC_CLASSES
    end

    @testset "the mask keeps exactly its own class" begin
        for χ in _FE._CELL_ELASTIC_CLASSES
            m = _FE._cell_mask(χ)
            @test all(x -> x == 0.0 || x == 1.0, m)
            @test sum(m) == count(
                i -> _FE._cell_parity(_FE._cell_kelvin_basis(i)) == χ, 1:6
            )
        end
        # The classes partition the six cases: every component is kept by
        # exactly one class, so nothing is counted twice and nothing is lost.
        @test sum(_FE._cell_mask(χ) for χ in _FE._CELL_ELASTIC_CLASSES) == ones(6)
        @test sum(_FE._cell_mask2(χ) for χ in _FE._CELL_COND_CLASSES) == ones(3)
    end

    @testset "the far field obeys the parity it is assigned" begin
        # u = E·x, the trivial half -- written down so the table above is under
        # test rather than merely asserted.
        for m in 1:6
            E = _FE._cell_kelvin_basis(m)
            χ = _FE._cell_parity(E)
            for k in 1:3, x in ([0.7, -1.3, 0.9], [1.1, 0.4, -2.2])
                R = _mirror(k)
                @test E * (R * x) ≈ χ[k] * (R * (E * x)) atol = 1.0e-15
            end
        end
    end

    @testset "the dipole correction obeys the SAME parity" begin
        # This is the load-bearing test of the whole octant. The corrected
        # boundary condition adds a dipole field to the remote one; if the two
        # did not transform alike, no single plane condition could serve both
        # and the correction would be incompatible with an eighth of the cell.
        #
        # From the closed form, R u(Rx; Π) = u(x; RΠR) = χ u(x; Π). It holds
        # *identically*, not to a tolerance -- the reflections only permute
        # signs of terms already present.
        μ, ν, k₀ = 0.4, 0.27, 1.7
        xs = ([0.7, -1.3, 0.9], [1.1, 0.4, -2.2], [-0.3, 0.8, 1.5])

        for m in 1:6
            Π = _FE._cell_kelvin_basis(m)
            χ = _FE._cell_parity(Π)
            for k in 1:3, x in xs
                R = _mirror(k)
                lhs = _CORE._dipole_displacement_iso(μ, ν, R * x, Π)
                rhs = χ[k] * (R * _CORE._dipole_displacement_iso(μ, ν, x, Π))
                @test lhs ≈ rhs atol = 1.0e-14
            end
        end

        for i in 1:3
            M = [k == i ? 1.0 : 0.0 for k in 1:3]
            χ = _FE._cell_parity(M)
            for k in 1:3, x in xs
                R = _mirror(k)
                @test _CORE.dipole_temperature_iso(k₀, R * x, M) ≈
                    χ[k] * _CORE.dipole_temperature_iso(k₀, x, M) atol = 1.0e-14
            end
        end
    end

    @testset "the boundary datum is consistent with the plane condition" begin
        # Where the outer sphere meets a coordinate plane, a node carries both
        # the imposed datum and the symmetry condition. They agree exactly --
        # a consequence of the parity law, not a tolerance -- which is what
        # lets the plane condition simply overwrite the datum there.
        μ, ν, k₀, R∞ = 0.4, 0.27, 1.7, 3.0
        for k in 1:3
            # A point of the sphere lying in the plane xₖ = 0.
            v = [k == 1 ? 0.0 : 0.6, k == 2 ? 0.0 : 0.64, k == 3 ? 0.0 : 0.48]
            x = R∞ .* v ./ norm(v)
            @test iszero(x[k])
            for m in 1:6
                E = _FE._cell_kelvin_basis(m)
                χ = _FE._cell_parity(E)
                for u in (E * x, _CORE._dipole_displacement_iso(μ, ν, x, E))
                    if χ[k] == 1
                        @test abs(u[k]) < 1.0e-14      # symmetry pins uₖ
                    else
                        for l in 1:3                    # antisymmetry pins the rest
                            l == k || @test abs(u[l]) < 1.0e-14
                        end
                    end
                end
            end
            for i in 1:3
                M = [j == i ? 1.0 : 0.0 for j in 1:3]
                if _FE._cell_parity(M)[k] == -1
                    @test abs(dot(M, x)) < 1.0e-14
                    @test abs(_CORE.dipole_temperature_iso(k₀, x, M)) < 1.0e-14
                end
            end
        end
    end

    @testset "the trait, and its numerical audit" begin
        for s in (
                Supersphere(1.0, 0.3), Supersphere(1.0, 1.0), Supersphere(1.0, 2.0),
                Superspheroid(1.0, 0.5, 0.4), Superspheroid(1.0, 2.0, 1.3),
            )
            @test has_coordinate_mirrors(s)
            @test check_coordinate_mirrors(s; ndirs = 200)
        end
        # The default is `false`, so a new shape has to opt in. A default of
        # `true` would turn an oversight into a wrong but plausible answer.
        @test !has_coordinate_mirrors(_NotMirroredShape())
    end
end
