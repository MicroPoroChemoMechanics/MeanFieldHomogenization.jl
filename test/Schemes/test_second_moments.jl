# =============================================================================
#  test_second_moments.jl — second moments of the local fields over a phase
#  (`second_moment`, the lens `PropertyDirection`).
#
#  References, independent of the derivative under test:
#   1. a laminate: the fields are uniform in each layer, so that the second
#      moment is the square of the mean, given by the concentration tensors of
#      the exact solution;
#   2. the composite sphere under a hydrostatic strain (Lamé's solution, a core
#      of radius a in a shell of radius b, f = a³/b³), whose bulk modulus is the
#      Mori-Tanaka estimate of spheres: with u = A r + B/r² in the shell,
#      ⟨(tr ε)²⟩ = 9A² and ⟨εᵈ:εᵈ⟩ = 6B²/f there, and the core is uniform;
#   3. Euler's identity: an estimate homogeneous of degree one in the stiffnesses
#      gives Σᵢ fᵢ ⟨ε:ℂᵢ:ε⟩ᵢ = 𝐄:ℂʰᵒᵐ:𝐄;
#   4. the sphere in conduction (Mori-Tanaka), whose gradient is uniform:
#      g = 3k₀ G / (k₁ + 2k₀ − f (k₁ − k₀)).
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Random
import ForwardDiff
import Symbolics
import SymPyPythonCall

const SM = MeanFieldHomogenization.Schemes

sm_𝕁, sm_𝕂 = TensISO{3}(1.0, 0.0), TensISO{3}(0.0, 1.0)
sm_km(t) = collect(mandel(t, TensND.CanonicalBasis{size(t, 1), Float64}()))
# The same tensor, stored in a basis drawn at random.
sm_basis(rng) = TensND.RotatedBasis((2π .* rand(rng, 3))...)
sm_moved(t, rng) = express(Tens(canonical_components(t)), sm_basis(rng))
sm_strain(rng) = (a = randn(rng, 3, 3); sm_moved(Tens(Symmetric(a + a')), rng))

function sm_two_phases(C₀, C₁, f; key = :C, shape = Ellipsoid(1.0))
    rve = RVE()
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(key => C₀); fraction = :rest)
    add_phase!(rve, :I, shape, Dict(key => C₁); fraction = f)
    return rve
end

# Lamé's solution of the composite sphere under the strain e 𝟏: the constants of
# the shell and of the core.
function sm_lame(k₀, μ₀, k₁, f, e)
    A = e * (3k₁ + 4μ₀) / (3k₁ + 4μ₀ - 3f * (k₁ - k₀))
    B = -3f * (k₁ - k₀) * A / (3k₁ + 4μ₀)
    return (A = A, B = B, core = A + B / f)
end

@testset "second moments" begin
    rng = MersenneTwister(20261124)

    @testset "a laminate: the square of the mean, layer by layer" begin
        C_A = sm_moved(iso_stiffness(10.0, 6.0) + TensTI{4}(1.0, 2.0, 0.5, 1.5, 0.7, [0.0, 0.0, 1.0]), rng)
        C_B = sm_moved(iso_stiffness(3.0, 1.0), rng)
        lam = Laminate(; normal = (0, 0, 1))
        add_layer!(lam, :A, Dict(:C => C_A, :K => sm_moved(TensISO{3}(2.0), rng)); thickness = 0.3)
        add_layer!(lam, :B, Dict(:C => C_B, :K => sm_moved(TensISO{3}(0.3), rng)); thickness = 0.7)
        E = sm_strain(rng)
        ct = concentration_tensors(lam, Laminated(), :C)
        for layer in (:A, :B)
            ε = ct.strain[layer] ⊡ E
            @test sm_km(second_moment(lam, Laminated(), layer, E)) ≈ sm_km(ε ⊗ ε) rtol = 1.0e-10
            𝔹 = sm_moved(TensTI{4}(0.3, -1.0, 0.4, 2.0, 0.5, [0.0, 1.0, 0.0]), rng)
            @test second_moment(lam, Laminated(), layer, E, 𝔹) ≈ ε ⊡ (𝔹 ⊡ ε) rtol = 1.0e-10
        end
        # Conduction: the gradient of each layer.
        G = Tens(randn(rng, 3))
        ctK = concentration_tensors(lam, Laminated(), :K)
        for layer in (:A, :B)
            g = ctK.strain[layer] ⋅ G
            @test sm_km(second_moment(lam, Laminated(), layer, G; property = :K)) ≈ sm_km(g ⊗ g) rtol = 1.0e-10
        end
    end

    @testset "the composite sphere under a hydrostatic strain" begin
        k₀, μ₀, k₁, μ₁, f, e = 10.0, 6.0, 40.0, 20.0, 0.3, 0.7
        rve = sm_two_phases(iso_stiffness(k₀, μ₀), iso_stiffness(k₁, μ₁), f)
        E = TensISO{3}(e)
        L = sm_lame(k₀, μ₀, k₁, f, e)
        @test second_moment(rve, MoriTanaka(), :M, E, 3 * sm_𝕁) ≈ 9 * L.A^2 rtol = 1.0e-12
        @test second_moment(rve, MoriTanaka(), :M, E, sm_𝕂) ≈ 6 * L.B^2 / f rtol = 1.0e-12
        @test second_moment(rve, MoriTanaka(), :I, E, 3 * sm_𝕁) ≈ 9 * L.core^2 rtol = 1.0e-12
        @test second_moment(rve, MoriTanaka(), :I, E, sm_𝕂) ≈ 0 atol = 1.0e-12
        # The whole tensor of the shell: 3A² 𝕁 + 6B²/(5f) 𝕂, by isotropy.
        @test sm_km(second_moment(rve, MoriTanaka(), :M, E)) ≈ sm_km(TensISO{3}(3 * L.A^2, 6 * L.B^2 / (5f))) rtol = 1.0e-10

        @testset "under another derivative, with respect to a modulus of the other phase" begin
            moment(k) = second_moment(sm_two_phases(iso_stiffness(k₀, μ₀), iso_stiffness(k, μ₁), f), MoriTanaka(), :M, E, sm_𝕂)
            reference(k) = 6 * sm_lame(k₀, μ₀, k, f, e).B^2 / f
            @test ForwardDiff.derivative(moment, k₁) ≈ ForwardDiff.derivative(reference, k₁) rtol = 1.0e-10
            second(g) = ForwardDiff.derivative(k -> ForwardDiff.derivative(g, k), k₁)
            @test second(moment) ≈ second(reference) rtol = 1.0e-9
            # The strain carries the derivative, the cell does not.
            @test ForwardDiff.derivative(x -> second_moment(rve, MoriTanaka(), :M, TensISO{3}(x), sm_𝕂), e) ≈
                2 * second_moment(rve, MoriTanaka(), :M, E, sm_𝕂) / e rtol = 1.0e-12
        end

        @testset "symbolic moduli" begin
            Symbolics.@variables sk₀ sμ₀ sk₁ sμ₁ sf se
            srve = sm_two_phases(iso_stiffness(sk₀, sμ₀), iso_stiffness(sk₁, sμ₁), sf)
            m = second_moment(srve, MoriTanaka(), :M, TensISO{3}(se), sm_𝕂)
            values = Dict(sk₀ => k₀, sμ₀ => μ₀, sk₁ => k₁, sμ₁ => μ₁, sf => f, se => e)
            @test Symbolics.value(Symbolics.substitute(m, values; fold = Val(true))) ≈ 6 * L.B^2 / f rtol = 1.0e-12
            # The whole tensor of a uniform strain (Voigt), in conduction: G ⊗ G.
            Symbolics.@variables g₁ g₂ g₃
            crve = sm_two_phases(TensISO{3}(sk₀), TensISO{3}(sk₁), sf; key = :K)
            M = canonical_components(second_moment(crve, Voigt(), :I, Tens([g₁, g₂, g₃]); property = :K))
            @test all(iszero ∘ Symbolics.simplify, M .- [g₁, g₂, g₃] .* [g₁ g₂ g₃])
        end

        @testset "symbolic moduli, with SymPy" begin
            sk₀, sμ₀, sk₁, sμ₁, sf, se = SymPyPythonCall.symbols("k_0 mu_0 k_1 mu_1 f e"; positive = true)
            srve = sm_two_phases(iso_stiffness(sk₀, sμ₀), iso_stiffness(sk₁, sμ₁), sf)
            m = second_moment(srve, MoriTanaka(), :M, TensISO{3}(se), sm_𝕂)
            S = sm_lame(sk₀, sμ₀, sk₁, sf, se)
            @test iszero(SymPyPythonCall.simplify(m - 6 * S.B^2 / sf))
            g = SymPyPythonCall.symbols("g_1 g_2 g_3"; real = true)
            crve = sm_two_phases(TensISO{3}(sk₀), TensISO{3}(sk₁), sf; key = :K)
            M = canonical_components(second_moment(crve, Voigt(), :I, Tens(collect(g)); property = :K))
            @test all(iszero ∘ SymPyPythonCall.simplify, M .- collect(g) .* permutedims(collect(g)))
        end
    end

    @testset "one quadratic form or the whole tensor, in every class" begin
        # A transversely isotropic matrix about an axis that is not one of the frame,
        # an anisotropic inclusion stored in a basis drawn at random.
        C₀ = iso_stiffness(10.0, 6.0) + TensTI{4}(1.0, 2.0, 0.5, 1.5, 0.7, TensND.normalize([1.0, 2.0, 2.0]))
        C₁ = sm_moved(iso_stiffness(40.0, 20.0) + TensTI{4}(4.0, 1.0, 2.0, 3.0, 5.0, [0.0, 0.0, 1.0]), rng)
        rve = sm_two_phases(C₀, C₁, 0.25; shape = Ellipsoid(1.0, 1.0, 0.4))
        E = sm_strain(rng)
        directions = (
            sm_𝕁, sm_𝕂,
            TensTI{4}(0.3, -1.0, 0.4, 2.0, 0.5, [0.0, 1.0, 0.0]),
            sm_moved(TensTI{4}(1.0, 0.0, 0.2, 0.0, 3.0, [1.0, 0.0, 0.0]), rng),
        )
        M = second_moment(rve, MoriTanaka(), :I, E)
        @test sm_km(M) ≈ sm_km(M)'
        for 𝔹 in directions
            @test second_moment(rve, MoriTanaka(), :I, E, 𝔹) ≈ M ⊙ 𝔹 rtol = 1.0e-10
        end
        # An ellipsoid under Mori-Tanaka: a uniform strain, the square of the mean.
        ε = concentration_tensors(rve, MoriTanaka(), :C).strain[:I] ⊡ E
        @test sm_km(M) ≈ sm_km(ε ⊗ ε) rtol = 1.0e-9

        # The matrix, and both phases of a self-consistent estimate, fluctuate: the
        # mean of a positive quadratic form exceeds that form on the mean strain.
        # Isotropic phases here, the estimate being transversely isotropic about
        # the axis of the spheroids.
        iso = sm_two_phases(iso_stiffness(10.0, 6.0), iso_stiffness(40.0, 20.0), 0.25; shape = Ellipsoid(1.0, 1.0, 0.4))
        sc = SelfConsistent(; abstol = 0.0, reltol = 1.0e-13)
        for (scheme, phase) in ((MoriTanaka(), :M), (sc, :M), (sc, :I))
            ε = concentration_tensors(iso, scheme, :C).strain[phase] ⊡ E
            for 𝔹 in (sm_𝕁, sm_𝕂)
                @test second_moment(iso, scheme, phase, E, 𝔹) > (1 + 1.0e-6) * (ε ⊡ (𝔹 ⊡ ε))
            end
        end
        # The energy of the cell is that of its phases.
        for cell in (rve, iso), scheme in (Voigt(), MoriTanaka(), sc)
            W = sum(volume_fraction(cell, φ) * second_moment(cell, scheme, φ, E, SM.phase_property(cell, φ, :C)) for φ in (:M, :I))
            @test W ≈ E ⊡ (homogenize(cell, scheme, :C) ⊡ E) rtol = 1.0e-9
        end
    end

    @testset "the lens: an abscissa along a tensor" begin
        rve = sm_two_phases(iso_stiffness(10.0, 6.0), iso_stiffness(40.0, 20.0), 0.3)
        𝔹 = TensTI{4}(0.3, -1.0, 0.4, 2.0, 0.5, [0.0, 1.0, 0.0])
        lens = property(:I, :C, 𝔹)
        @test lens isa PropertyDirection
        @test get_param(rve, lens) == 0
        @test sm_km(SM.phase_property(set_param(rve, lens, 0.5), :I, :C)) ≈ sm_km(iso_stiffness(40.0, 20.0)) + 0.5 * sm_km(𝔹)
        E = sm_strain(rng)
        d = derivative(rve, MoriTanaka(), lens; indexer = C -> E ⊡ (C ⊡ E))
        @test d ≈ 0.3 * second_moment(rve, MoriTanaka(), :I, E, 𝔹) rtol = 1.0e-12
        # A coefficient of a class is the abscissa along the tensor it multiplies.
        @test derivative(rve, MoriTanaka(), property(:I, :C, :β); indexer = C -> E ⊡ (C ⊡ E)) ≈
            derivative(rve, MoriTanaka(), property(:I, :C, sm_𝕂); indexer = C -> E ⊡ (C ⊡ E)) rtol = 1.0e-12
    end

    @testset "conduction: the sphere, whose gradient is uniform" begin
        k₀, k₁, f = 1.0, 2.5, 0.2
        rve = sm_two_phases(TensISO{3}(k₀), TensISO{3}(k₁), f; key = :K)
        G = Tens(randn(rng, 3))
        g = 3k₀ / (k₁ + 2k₀ - f * (k₁ - k₀)) * G
        @test sm_km(second_moment(rve, MoriTanaka(), :I, G; property = :K)) ≈ sm_km(g ⊗ g) rtol = 1.0e-10
        @test second_moment(rve, MoriTanaka(), :I, G, TensISO{3}(1.0); property = :K) ≈ g ⋅ g rtol = 1.0e-12
        # The matrix fluctuates: its whole tensor against its quadratic forms, the
        # matrix being transversely isotropic and the directions in any basis.
        K₀ = TensTI{2}(1.0, 1.7, TensND.normalize([1.0, 2.0, 2.0]))
        ani = sm_two_phases(K₀, sm_moved(TensISO{3}(k₁), rng), f; key = :K, shape = Ellipsoid(1.0, 1.0, 0.4))
        M = second_moment(ani, MoriTanaka(), :M, G; property = :K)
        a = randn(rng, 3, 3)
        for 𝑩 in (TensISO{3}(1.0), TensTI{2}(0.3, -1.0, [0.0, 1.0, 0.0]), sm_moved(Tens(Symmetric(a + a')), rng))
            @test second_moment(ani, MoriTanaka(), :M, G, 𝑩; property = :K) ≈ M ⊡ 𝑩 rtol = 1.0e-9
        end
        ḡ = concentration_tensors(ani, MoriTanaka(), :K).strain[:M] ⋅ G
        @test minimum(eigvals(Symmetric(canonical_components(M - ḡ ⊗ ḡ)))) > -1.0e-12
        @test tr(M) > (1 + 1.0e-6) * (ḡ ⋅ ḡ)
    end

    @testset "in the plane" begin
        c₀, c₁ = plane_strain_stiffness(iso_stiffness(10.0, 6.0)), plane_strain_stiffness(iso_stiffness(40.0, 20.0))
        rve = sm_two_phases(c₀, c₁, 0.3; shape = Ellipsoid(1.0; dim = 2))
        rve.phases[:M] = SM.Phase(Ellipsoid(1.0; dim = 2), rve.phases[:M].properties)
        a = randn(rng, 2, 2)
        E = Tens(Symmetric(a + a'))
        M = second_moment(rve, MoriTanaka(), :I, E)
        ε = concentration_tensors(rve, MoriTanaka(), :C).strain[:I] ⊡ E
        @test sm_km(M) ≈ sm_km(ε ⊗ ε) rtol = 1.0e-10                    # a disk: a uniform strain
        W = sum(volume_fraction(rve, φ) * second_moment(rve, MoriTanaka(), φ, E, SM.phase_property(rve, φ, :C)) for φ in (:M, :I))
        @test W ≈ E ⊡ (homogenize(rve, MoriTanaka(), :C) ⊡ E) rtol = 1.0e-10
    end

    @testset "an assembly of particles" begin
        asm = ParticleAssembly(; boundary = PeriodicBox(1.0; cutoff = 2.0))
        add_matrix!(asm, Dict(:C => iso_stiffness(10.0, 6.0)))
        add_particle!(asm, :p1, (0.0, 0.0, 0.0), Ellipsoid(0.3), Dict(:C => iso_stiffness(40.0, 20.0)))
        E = sm_strain(rng)
        members = ((:p1, iso_stiffness(40.0, 20.0)), (asm.matrix_name, iso_stiffness(10.0, 6.0)))
        W = sum(SM._moment_fraction(asm, φ) * second_moment(asm, ClusterModel(), φ, E, C) for (φ, C) in members)
        @test W ≈ E ⊡ (homogenize(asm, ClusterModel(), :C) ⊡ E) rtol = 1.0e-9
    end

    @testset "refusals" begin
        rve = sm_two_phases(iso_stiffness(10.0, 6.0), iso_stiffness(40.0, 20.0), 0.3)
        add_phase!(rve, :cracks, EllipticCrack(1.0, 1.0), Dict{Symbol, Any}(); density = 0.1)
        E = TensISO{3}(1.0)
        @test_throws ArgumentError second_moment(rve, MoriTanaka(), :cracks, E, sm_𝕂)
        # A direction of another order than the property.
        @test_throws ArgumentError second_moment(rve, MoriTanaka(), :I, E, TensISO{3}(1.0))
        @test_throws ArgumentError get_param(rve, property(:I, :C, TensISO{3}(1.0)))
    end
end
