# =============================================================================
#  test_hill_random_bases.jl — Hill tensors against echoes, with the ellipsoid,
#  the reference medium and the bases they are written in all drawn at random.
#
#  References: `echoes_hill_random_bases.json`, written by
#  `scripts/bench_echoes/make_hill_random_bases.py` (echoes, NUMINT3D, relative
#  tolerance 1e-10): sphere, prolate and oblate spheroids, triaxial ellipsoids and
#  their limits (nearly spherical, very flat, needle); isotropic, transversely
#  isotropic (about a random axis, and about the spheroid's own), orthotropic,
#  cubic and anisotropic media; isotropic and anisotropic conductors.
#
#  Every ellipsoid is written in a basis drawn at random, never the canonical one.
#  Every medium is given to MFH in four forms, which must all give the same tensor:
#   1. its canonical components, in a plain `Tens`;
#   2. the same tensor written in a rotated basis drawn at random;
#   3. the same tensor written in a general, non-orthonormal basis drawn at random
#      (contravariant components, a metric);
#   4. its structured TensND type (`TensISO`, `TensTI`, `TensOrtho`, `TensCubic`),
#      built from its constants and frame, as a user writes it.
#  The Hill tensor is checked against echoes in canonical components (the
#  reference's own), and against the canonical medium's in the basis they share.
#  Derivatives are checked on two exact identities: ℙ(λℂ) = ℙ(ℂ)/λ, and ℙ unchanged
#  when the ellipsoid is scaled uniformly.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Random
using JSON
using ForwardDiff

const _HRB_REFS = JSON.parsefile(joinpath(@__DIR__, "echoes_hill_random_bases.json"))
const _HRB_CORE = MeanFieldHomogenization.Core

"A matrix from the nested arrays of a JSON file."
_hrb_mat(rows) = [Float64(rows[i][j]) for i in eachindex(rows), j in eachindex(rows[1])]

"Components in the frame whose vectors are the columns of `R` (an orthogonal `R`)."
_hrb_in_frame(A::AbstractArray{<:Any, 4}, R) = [
    sum(R[i, a] * R[j, b] * R[k, c] * R[l, d] * A[i, j, k, l] for i in 1:3, j in 1:3, k in 1:3, l in 1:3)
        for a in 1:3, b in 1:3, c in 1:3, d in 1:3
]
_hrb_in_frame(A::AbstractArray{<:Any, 2}, R) = transpose(R) * A * R

"The tensor of canonical components `A`, written in the rotated basis of `R`."
_hrb_rotated(A, R) = TensND.Tens(_hrb_in_frame(A, R), TensND.RotatedBasis(R))

"The tensor of canonical components `A`, written in a general basis near `R`."
_hrb_general(A, R, rng) = TensND.change_tens(TensND.Tens(A), TensND.Basis(R + 0.3 * randn(rng, 3, 3), Val(:cov)))

"The medium as a user writes it, in its structured TensND type."
function _hrb_structured(c)
    m = c["material"]
    m === nothing && return nothing
    p = Float64.(m["params"])
    if m["type"] == "iso"
        return c["kind"] == "elastic" ? iso_stiffness(p...) : TensISO{3}(p[1])
    end
    F = TensND.RotatedBasis(Float64.(m["euler"])...)
    m["type"] == "TI" && return tens_ti(p..., Matrix(TensND.vecbasis(F))[:, 3])
    m["type"] == "ortho" && return TensND.TensOrtho(p..., F)
    m["type"] == "cubic" && return TensND.tens_cubic(p..., F)
    error("unknown material type $(m["type"])")
end

_hrb_km(P) = _HRB_CORE.mandel66_minor(TensND.components_canon(P))
_hrb_frame(ell) = Matrix(TensND.vecbasis(ell.basis, :cov))
const _HRB_TIGHT = (abstol = 1.0e-13, reltol = 1.0e-11)

@testset "Hill tensors against echoes, random ellipsoids, media and bases" begin
    rng = Random.Xoshiro(20261021)
    for (n, c) in enumerate(_HRB_REFS["cases"])
        @testset "case $n: $(c["kind"]), $(c["shape"]), $(c["medium"])" begin
            ell = Ellipsoid(Float64.(c["semi_axes"])...; euler_angles = Tuple(Float64.(c["euler"])))
            # Never the canonical basis: the ellipsoid's own frame is drawn at random.
            @test !(ell.basis isa TensND.CanonicalBasis)
            R = TensND.random_rotation(rng)
            E = _hrb_frame(ell)
            elastic = c["kind"] == "elastic"
            A = elastic ? _HRB_CORE.array_from_mandel66(_hrb_mat(c["C_km"])) : _hrb_mat(c["K"])
            ref = elastic ? _hrb_mat(c["P_km"]) : _hrb_mat(c["P"])
            canon(P) = elastic ? _hrb_km(P) : Matrix(TensND.components_canon(P))
            in_ell_frame(P) = elastic ?
                _HRB_CORE.mandel66_minor(TensND.get_array(TensND.change_tens(P, ell.basis))) :
                Matrix(TensND.get_array(TensND.change_tens(P, ell.basis)))
            ref_ell = elastic ?
                _HRB_CORE.mandel66_minor(_hrb_in_frame(_HRB_CORE.array_from_mandel66(ref), E)) :
                _hrb_in_frame(ref, E)

            media = Any[TensND.Tens(A), _hrb_rotated(A, R), _hrb_general(A, R, rng)]
            S = _hrb_structured(c)
            if S !== nothing
                # The structured form is the same tensor: the constants and the
                # frame the generator used reproduce its components.
                @test TensND.Tens(A) ≈ S rtol = 1.0e-12
                push!(media, S)
            end

            P_canon = hill_tensor(ell, TensND.Tens(A); _HRB_TIGHT...)
            for C₀ in media
                P = hill_tensor(ell, C₀; _HRB_TIGHT...)
                @test canon(P) ≈ ref rtol = 1.0e-9
                @test P ≈ P_canon rtol = 1.0e-9           # in the basis they share
                @test in_ell_frame(P) ≈ ref_ell rtol = 1.0e-9
                # The default tolerances keep their promise (abstol 1e-8, reltol 1e-6).
                @test canon(hill_tensor(ell, C₀)) ≈ ref rtol = 1.0e-6
            end
            if elastic && c["medium"] != "isotropic"
                # The residue route, an independent algorithm (Float64 only).
                @test canon(hill_tensor(ell, TensND.Tens(A); method = :residues)) ≈ ref rtol = 1.0e-8
            end

            # Derivatives, on exact identities: ℙ(λℂ) = ℙ(ℂ)/λ gives dℙ/dλ = −ℙ at
            # λ = 1, and a uniform scaling of the ellipsoid leaves ℙ unchanged.
            C₀ = last(media)
            dλ = ForwardDiff.derivative(λ -> canon(hill_tensor(ell, λ * C₀)), 1.0)
            @test dλ ≈ -ref rtol = 1.0e-6
            axes0 = Float64.(c["semi_axes"])
            ds = ForwardDiff.derivative(
                s -> canon(hill_tensor(Ellipsoid((s .* axes0)...; euler_angles = Tuple(Float64.(c["euler"]))), C₀)),
                1.0,
            )
            @test norm(ds) ≤ 1.0e-6 * norm(ref)
        end
    end
end
