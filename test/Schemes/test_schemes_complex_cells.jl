# =============================================================================
#  test_schemes_complex_cells.jl — every homogenization scheme of echoes on complex
#  cells, in elasticity and in conduction.
#
#  References: `echoes_schemes_complex_cells.json`, written by
#  `scripts/bench_echoes/make_schemes_complex_cells.py` (echoes, NUMINT3D Hill
#  tensors at relative tolerance 1e-10, fixed points to 1e-12). One cell per matrix,
#  isotropic or transversely isotropic about a tilted axis: a family of stiff
#  anisotropic triaxial ellipsoids and one of soft prolate spheroids, each oriented
#  off the axes along its own frame, and two crack families, an ellipse and a penny,
#  of their own orientations; Maxwell and Ponte Castañeda-Willis on a tilted oblate
#  distribution ellipsoid. Schemes: Voigt, Reuss, dilute, dilute dual, Mori-Tanaka,
#  Maxwell, Ponte Castañeda-Willis, self-consistent, asymmetric self-consistent, and,
#  on the isotropic matrix, the differential scheme, whose echoes reference is the
#  Richardson extrapolation of its explicit Euler steps (second order), hence its
#  looser tolerance. A crack has the density N a b² here, N a³ in echoes.
#
#  The Voigt and Reuss bounds pass cracks over (no volume); the Reuss reference is
#  computed without them, echoes forming zero times an infinite compliance. Every
#  estimate agrees to 6e-13 (measured); the tolerance 1e-10 leaves room for the
#  quadratures of another machine.
# =============================================================================

using Test
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using JSON

const _SCC = JSON.parsefile(joinpath(@__DIR__, "echoes_schemes_complex_cells.json"))

_scc_mat(rows) = [Float64(rows[i][j]) for i in eachindex(rows), j in eachindex(rows[1])]
_scc_medium(prop, rows) = prop === :C ? TensND.Tens(TensND.get_array(TensND.from_mandel(_scc_mat(rows)))) : TensND.Tens(_scc_mat(rows))
_scc_canon(X, prop) = prop === :C ? TensND.mandel(TensND.canonical_components(X); symmetrize = true) : Matrix(TensND.canonical_components(X))

const _SCC_SCHEMES = Dict(
    "voigt" => Voigt(), "reuss" => Reuss(), "dilute" => Dilute(), "dilute_dual" => DiluteDual(),
    "mori_tanaka" => MoriTanaka(), "maxwell" => Maxwell(), "pcw" => PonteCastanedaWillis(),
    "self_consistent" => SelfConsistent(reltol = 1.0e-12),
    "asymmetric_self_consistent" => AsymmetricSelfConsistent(reltol = 1.0e-12),
    "differential" => DifferentialScheme(abstol = 1.0e-12, reltol = 1.0e-10),
)

"The cell of a case: the matrix of name `matrix`, the inclusions and the crack families."
function _scc_rve(matrix, prop)
    d = _SCC["distribution"]
    rve = RVE(; distribution_shape = Ellipsoid(Float64.(d["semi_axes"])...; euler_angles = Tuple(Float64.(d["euler"]))))
    key = prop === :C ? "C_km" : "K"
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(prop => _scc_medium(prop, _SCC["matrices"][matrix][key])); fraction = :rest)
    for i in _SCC["inclusions"]
        ell = Ellipsoid(Float64.(i["semi_axes"])...; euler_angles = Tuple(Float64.(i["euler"])))
        add_phase!(rve, Symbol(i["name"]), ell, Dict(prop => _scc_medium(prop, i[key])); fraction = Float64(i["fraction"]))
    end
    for c in _SCC["cracks"]
        a, b = Float64.(c["semi_axes"])
        crack = EllipticCrack(a, b; euler_angles = Tuple(Float64.(c["euler"])))
        add_phase!(rve, Symbol(c["name"]), crack, Dict{Symbol, Any}(); density = (b / a)^2 * Float64(c["density"]))
    end
    return rve
end

@testset "every scheme of echoes on complex cells — elasticity and conduction" begin
    for c in _SCC["cases"]
        prop = Symbol(c["prop"])
        rve, scheme = _scc_rve(c["matrix"], prop), _SCC_SCHEMES[c["scheme"]]
        # Two inclusion families of different frames: the Mori–Tanaka estimate is not
        # major-symmetric, it says so and returns its symmetric part, as echoes does.
        X = scheme isa MoriTanaka ? (@test_logs (:warn, r"no major symmetry") homogenize(rve, scheme, prop)) :
            homogenize(rve, scheme, prop)
        rtol = c["scheme"] == "differential" ? 1.0e-6 : 1.0e-10
        @test isapprox(_scc_canon(X, prop), _scc_mat(c["effective"]); rtol)
    end
end
