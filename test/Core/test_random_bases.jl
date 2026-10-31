# =============================================================================
#  test_random_bases.jl — results do not depend on the basis a tensor is
#  written in.
#
#  `get_array(t)`, `t[i, j]` and `mandel(t)` read a TensND tensor in its own basis.
#  Structured tensors (`TensISO`, `TensTI`, …) answer in the canonical one, so
#  code tested with them only looks right, and is wrong as soon as a `Tens`
#  written in a rotated basis reaches it: an anisotropic Hill tensor, `tens_IA`,
#  a COD tensor, a user's property. Each check below gives the same tensor once
#  in the canonical basis and once in a basis drawn at random (every operand its
#  own), and compares canonical components, against an independent reference
#  where one exists. Round angles would hide the defects, so none is used.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Random

const _MRB = MeanFieldHomogenization
const _MRB_SEED = 20261021

"Components in the frame whose vectors are the columns of `R`."
_mrb_in_frame(A::AbstractArray{<:Any, 4}, R) = [
    sum(R[i, a] * R[j, b] * R[k, c] * R[l, d] * A[i, j, k, l] for i in 1:3, j in 1:3, k in 1:3, l in 1:3)
        for a in 1:3, b in 1:3, c in 1:3, d in 1:3
]
_mrb_in_frame(A::AbstractArray{<:Any, 2}, R) =
    [sum(R[i, a] * R[j, b] * A[i, j] for i in 1:3, j in 1:3) for a in 1:3, b in 1:3]

"The tensor of canonical components `A`, written in the rotated basis of `R`."
_mrb_rotated(A, R) = TensND.Tens(_mrb_in_frame(A, R), TensND.RotatedBasis(R))

# `AbstractTens <: AbstractArray`: the tensor method must be the more specific.
_mrb_canon(t::TensND.AbstractTens) = Array(TensND.canonical_components(t))
_mrb_canon(A::AbstractArray) = Array(A)

# Positive definite, kept away from singular as before: TensND's draws shifted
# by a multiple of the identity.
_mrb_spd3(rng) = Array(canonical_components(rand(rng, Tens{2, 3}; definite = true) + 3 * tens_Id2(Val(3), Val(Float64))))
_mrb_spd4(rng) = Array(canonical_components(rand(rng, Tens{4, 3}; definite = true) + 10 * tens_Id4(Val(3), Val(Float64))))

@testset "random bases — draw $k" for k in 1:5
    rng = Random.Xoshiro(_MRB_SEED + k)
    R = TensND.random_rotation(rng)                                   # basis of the property
    ell = Ellipsoid(2.0, 1.0, 0.5, TensND.RotatedBasis(TensND.random_rotation(rng)))

    @testset "conduction Hill tensor of an ellipsoid, anisotropic conductor" begin
        K = _mrb_spd3(rng)
        Pc = _mrb_canon(hill_tensor(ell, TensND.Tens(K)))
        Pr = _mrb_canon(hill_tensor(ell, _mrb_rotated(K, R)))
        @test Pr ≈ Pc
        # Reference: the shape tensor assembled in canonical coordinates.
        Re = Matrix(TensND.vecbasis(ell.basis, :cov))
        A = Re * Diagonal(collect(ell.semi_axes)) * transpose(Re)
        @test Pc ≈ _MRB.Conductivity._hill_order2_aniso(A, K)
    end

    @testset "tensor systems of an assembly, blocks in mixed bases" begin
        Cs = [_mrb_spd4(rng) for _ in 1:6]
        frames = [TensND.random_rotation(rng) for _ in 1:6]
        canon = [TensND.Tens(C) for C in Cs]
        mixed = [_mrb_rotated(C, F) for (C, F) in zip(Cs, frames)]
        P₀ = TensISO{3}(3 * 30.0, 2 * 20.0)
        solve(t) = _MRB.Assemblies._solve_tensor_system([[t[1], t[2]], [t[3], t[4]]], [t[5], t[6]], P₀)
        Xc, Xm = solve(canon), solve(mixed)
        @test all(a ≈ b for (a, b) in zip(Xm, Xc))
    end

    @testset "order-2 ALV Hill kernel of a rotated ellipsoid" begin
        n = 3
        α = Matrix(LowerTriangular(rand(rng, n, n))) + 4I
        K_0 = _MRB.Viscoelasticity.iso_order2_blocks_from_params(α)
        P = _MRB.Viscoelasticity.hill_kernel_order2_at(ell, K_0)
        # Reference: I^A from the elastic route, `P = I^A / k` for a conductor k.
        IA = 7.0 * _mrb_canon(hill_tensor(ell, TensISO{3}(7.0)))
        @test P ≈ kron(inv(α), IA)
    end

    @testset "COD tensor of an oriented crack with an interface stiffness" begin
        d = 1.0 .+ rand(rng, 3)
        B = TensND.Tens(Matrix(Diagonal(d)), TensND.RotatedBasis(R))
        Kc = _mrb_spd3(rng)
        b = 0.7
        Bc = R * Diagonal(d) * transpose(R)
        ref = Bc / (I + b * Kc * Bc)
        ref = (ref + transpose(ref)) / 2
        @test _mrb_canon(_MRB.Cracks._apply_interface_stiffness(B, TensND.Tens(Kc), b)) ≈ ref
        Kr = _mrb_rotated(Kc, TensND.random_rotation(rng))
        @test _mrb_canon(_MRB.Cracks._apply_interface_stiffness(B, Kr, b)) ≈ ref
    end

    @testset "ALV crack compliance in a random orientation: the elastic one under a Heaviside law" begin
        # A constant kernel C H(t - t′) gives a block-diagonal operator whose value at
        # each instant is the elastic compliance of the crack, rotated with it.
        C = TensISO{3}(3 * 5.0, 2 * 2.0)
        for crack in (
                PennyCrack(1.0; euler_angles = Tuple(2π .* rand(rng, 3))),
                EllipticCrack(1.0, 0.4; euler_angles = Tuple(2π .* rand(rng, 3))),
            )
            H̃ = inverse_property_contribution_alv(crack, heaviside_law(C), [0.0, 0.5, 1.0])
            H_el = TensND.mandel(TensND.canonical_components(inverse_property_contribution(crack, C)); symmetrize = true)
            @test all(i -> Matrix(H̃)[(6i - 5):6i, (6i - 5):6i] ≈ H_el, 1:3)
            @test iszero(Matrix(H̃)[7:12, 1:6])
        end
    end

    @testset "layered sphere: displacement under a strain in a rotated basis" begin
        s = LayeredSphere((0.5, 1.0), (TensISO{3}(180.0, 60.0), TensISO{3}(15.0, 4.0)))
        f = LayeredSphereFields(s, TensISO{3}(90.0, 40.0))
        E = randn(rng, 3, 3)
        E = (E + transpose(E)) / 2
        for x in (0.3 * normalize(randn(rng, 3)), 0.8 * normalize(randn(rng, 3)), 1.6 * normalize(randn(rng, 3)))
            @test local_displacement(f, x, _mrb_rotated(E, R)) ≈ local_displacement(f, x, TensND.Tens(E))
        end
    end

    @testset "finite-element frames" begin
        P = _mrb_spd4(rng)
        L = TensND.random_rotation(rng)                               # the local frame
        ref = _mrb_in_frame(P, L)
        @test _MRB.FiniteElements._to_local4(TensND.Tens(P), L) ≈ ref
        @test _MRB.FiniteElements._to_local4(_mrb_rotated(P, R), L) ≈ ref
        M = Matrix(mandel(rand(rng, Tens{4, 3})))           # symmetric, of either sign
        G = _MRB.FiniteElements._from_local66(M, L)
        @test _mrb_canon(G) ≈ _mrb_in_frame(TensND.get_array(TensND.from_mandel(M)), transpose(L))
    end

    @testset "self-consistent residual and differential state" begin
        C = _mrb_spd4(rng)
        tc, tr_ = TensND.Tens(C), _mrb_rotated(C, R)
        @test _MRB.Schemes._sc_residual_norm(tc, tr_) ≤ 1.0e-10 * norm(C)
        @test _MRB.Schemes._get_state(Val(:full_4), tr_) ≈ _MRB.Schemes._get_state(Val(:full_4), tc)
        n1, n2 = normalize(randn(rng, 3)), normalize(randn(rng, 3))
        a = ti_from_constants(10.0, 3.0, 2.5, 12.0, 2.0, n1)
        @test _MRB.Schemes._diff_same_layout(a, ti_from_constants(1.0, 0.3, 0.2, 1.2, 0.5, n1))
        @test !_MRB.Schemes._diff_same_layout(a, ti_from_constants(1.0, 0.3, 0.2, 1.2, 0.5, n2))
    end

    @testset "schemes: an anisotropic property in a rotated basis" begin
        K = _mrb_spd3(rng)
        function rve_cond(Ki)
            r = RVE()
            add_phase!(r, :M, Ellipsoid(1.0), Dict(:K => TensISO{3}(2.0)); fraction = :rest)
            add_phase!(r, :I, ell, Dict(:K => Ki); fraction = 0.2)
            return r
        end
        for scheme in (MoriTanaka(), SelfConsistent(), DifferentialScheme())
            @test homogenize(rve_cond(_mrb_rotated(K, R)), scheme, :K) ≈
                homogenize(rve_cond(TensND.Tens(K)), scheme, :K) rtol = 1.0e-8
        end
        C = _mrb_spd4(rng)
        function rve_el(Ci)
            r = RVE()
            add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 30.0, 2 * 20.0)); fraction = :rest)
            add_phase!(r, :I, ell, Dict(:C => Ci); fraction = 0.2)
            return r
        end
        @test homogenize(rve_el(_mrb_rotated(C, R)), MoriTanaka()) ≈
            homogenize(rve_el(TensND.Tens(C)), MoriTanaka())
    end
end
