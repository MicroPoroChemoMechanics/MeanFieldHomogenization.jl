# =============================================================================
#  test_schemes_random_bases.jl — the homogenization schemes against echoes,
#  with ellipsoids, media and the bases they are written in drawn at random.
#
#  References: `echoes_schemes_random_bases.json`, written by
#  `scripts/bench_echoes/make_schemes_random_bases.py` (echoes, NUMINT3D Hill
#  tensors at relative tolerance 1e-10, fixed points to 1e-12). Each RVE has a
#  spherical matrix (isotropic, transversely isotropic about a random axis, or
#  anisotropic) and two families of ellipsoids, a stiff and a soft one, each with
#  its own semi-axes, orientation and anisotropic medium. Schemes: dilute, dilute
#  dual, Mori-Tanaka, self-consistent and asymmetric self-consistent, in
#  elasticity and in conduction, each with the inclusions as they are and, where
#  the reference has the symmetry of the average, orientation-averaged: about every
#  axis (`symmetrize = :iso`, isotropic matrix) or about one (`TISymmetrize`,
#  isotropic or transversely isotropic matrix about that axis). Only then does
#  averaging the contributions, as MFH does, agree with averaging the
#  concentrations, as echoes does.
#
#  Every case is homogenized twice: with each medium in its canonical components,
#  and with each medium written in a basis of its own drawn at random (rotated for
#  half of them, general and non-orthonormal for the other half). Both must match
#  echoes in canonical components: the change of basis of every order of tensor,
#  in the shape, in the media and in the result, is exercised on the way.
#
#  The tolerance is the one MFH's default Hill quadrature promises (relative 1e-6),
#  loosened to 1e-5 for the fixed points, which compound it.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Random
using JSON

const _SRB_REFS = JSON.parsefile(joinpath(@__DIR__, "echoes_schemes_random_bases.json"))
const _SRB_CORE = MeanFieldHomogenization.Core

"A matrix from the nested arrays of a JSON file."
_srb_mat(rows) = [Float64(rows[i][j]) for i in eachindex(rows), j in eachindex(rows[1])]

"The tensor of canonical components `A`, written in a basis of its own drawn at random."
function _srb_in_random_basis(A, rng, general::Bool)
    R = Matrix(TensND.random_rotation(rng))              # uniform over the rotations
    # An oblique basis of bounded conditioning (at most 1.3/0.7): the test is that
    # the result does not depend on the basis, not how far round-off grows in a
    # nearly degenerate one.
    N = randn(rng, 3, 3)
    B = general ? TensND.Basis(R * (I + 0.3 * N / opnorm(N)), Val(:cov)) : TensND.RotatedBasis(R)
    return TensND.express(TensND.Tens(A), B)
end

"The canonical components of a stiffness (Kelvin-Mandel) or a conductivity."
_srb_canon(X, elastic) = elastic ? TensND.mandel(TensND.canonical_components(X); symmetrize = true) :
    Matrix(TensND.canonical_components(X))

const _SRB_SCHEMES = Dict(
    "dilute" => Dilute(), "dilute_dual" => DiluteDual(), "mori_tanaka" => MoriTanaka(),
    "self_consistent" => SelfConsistent(reltol = 1.0e-12),
    "asymmetric_self_consistent" => AsymmetricSelfConsistent(reltol = 1.0e-12),
)

"The RVE of a case, each medium passed through `medium`."
function _srb_rve(c, prop, medium)
    elastic = prop === :C
    tens(m) = elastic ? TensND.get_array(TensND.from_mandel(_srb_mat(m))) : _srb_mat(m)
    rve = RVE()
    add_phase!(
        rve, :M, Ellipsoid(1.0), Dict(prop => medium(tens(elastic ? c["C_m_km"] : c["K_m"])));
        fraction = :rest
    )
    for i in c["inclusions"]
        ell = Ellipsoid(Float64.(i["semi_axes"])...; euler_angles = Tuple(Float64.(i["euler"])))
        kw = _srb_average(c)
        add_phase!(
            rve, Symbol(i["name"]), ell, Dict(prop => medium(tens(elastic ? i["C_km"] : i["K"])));
            fraction = Float64(i["fraction"]), kw...
        )
    end
    return rve
end

"""
The relative residual of the asymmetric self-consistent fixed point at `X`, rebuilt
from `hill_tensor` and the exact orientation averages of `Core` in Kelvin-Mandel
(or 3×3) matrices: the stiffness form `X = C_M + Σ fᵣ ⟨(Cᵣ - C_M) 𝔸ᵣ(X̄)⟩`, or the
compliance form `X⁻¹ = S_M + Σ fᵣ ⟨(Sᵣ - S_M) 𝔹ᵣ(X̄)⟩` with `𝔹ᵣ = Cᵣ 𝔸ᵣ X̄⁻¹`,
whichever the scheme took; `X̄ = ⟨X⟩` is the reference the average is taken in.
"""
function _srb_asc_residual(c, X, elastic)
    tens(M) = TensND.Tens(elastic ? TensND.get_array(TensND.from_mandel(M)) : M)
    sym = _srb_average(c)
    avg(M) = haskey(sym, :symmetrize) ?
        _srb_canon(_srb_project(tens(M), sym.symmetrize), elastic) : M
    C_M = _srb_mat(elastic ? c["C_m_km"] : c["K_m"])
    Xr = avg(X)
    Id = Matrix(1.0I, size(X)...)
    stiff, compl = copy(C_M), inv(C_M)
    for i in c["inclusions"]
        ell = Ellipsoid(Float64.(i["semi_axes"])...; euler_angles = Tuple(Float64.(i["euler"])))
        Cᵣ = _srb_mat(elastic ? i["C_km"] : i["K"])
        P = _srb_canon(hill_tensor(ell, tens(Xr)), elastic)
        A = inv(Id + P * (Cᵣ - Xr))
        f = Float64(i["fraction"])
        stiff += f * avg((Cᵣ - C_M) * A)
        compl += f * avg((inv(Cᵣ) - inv(C_M)) * Cᵣ * A * inv(Xr))
    end
    return min(norm(X - stiff) / norm(X), norm(inv(X) - compl) / norm(inv(X)))
end
_srb_project(T, sym) = sym === :iso ? _SRB_CORE.isotropify(T) : _SRB_CORE.transverse_isotropify(T, sym.axis)

"The `symmetrize` keyword of the inclusions of a case."
function _srb_average(c)
    c["average"] == "none" && return (;)
    c["average"] == "iso" && return (; symmetrize = :iso)
    # The axis of the angles: the third vector of the ZYZ frame echoes builds.
    axis = _SRB_CORE._basis_axis(TensND.RotatedBasis(Float64.(c["ti_euler"])...), 3)
    return (; symmetrize = TISymmetrize(axis))
end

"The major-symmetric part, which echoes returns for every scheme (`sym_array` of its result)."
_srb_major_sym(X) = (X + transpose(X)) / 2

@testset "Schemes against echoes, random ellipsoids, media and bases" begin
    rng = Random.Xoshiro(20261023)
    for (n, c) in enumerate(_SRB_REFS["cases"])
        elastic = c["kind"] == "elastic"
        prop = elastic ? :C : :K
        ref = _srb_mat(elastic ? c["C_eff_km"] : c["K_eff"])
        tol = startswith(c["scheme"], "dilute") || c["scheme"] == "mori_tanaka" ? 1.0e-6 : 1.0e-5
        @testset "case $n: $(c["kind"]), $(c["matrix"]) matrix, $(c["scheme"])$(c["average"] == "none" ? "" : ", " * c["average"])" begin
            scheme = _SRB_SCHEMES[c["scheme"]]
            canonical = _srb_canon(homogenize(_srb_rve(c, prop, TensND.Tens), scheme, prop), elastic)
            rotated = _srb_canon(
                homogenize(_srb_rve(c, prop, A -> _srb_in_random_basis(A, rng, isodd(n))), scheme, prop), elastic
            )
            # The bases the media are written in do not show in the result.
            @test norm(rotated - canonical) ≤ 1.0e-10 * norm(canonical)
            if c["scheme"] == "asymmetric_self_consistent" && c["average"] != "iso"
                # echoes symmetrizes every iterate (`sym_array` in `evaluate_dilute`),
                # hence a fixed point of another map; MFH iterates the scheme as it
                # stands. Its result is checked as a fixed point of that map instead.
                # (An isotropic estimate has no part for the two to differ on.)
                @test _srb_asc_residual(c, canonical, elastic) ≤ 1.0e-6
                continue
            end
            @test norm(_srb_major_sym(canonical) - ref) ≤ tol * norm(ref)
        end
    end
end
