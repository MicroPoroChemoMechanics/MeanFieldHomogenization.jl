# =============================================================================
#  scripts/bench_echoes/alv_harness/cells.jl
#
#  The cells of the ALV harness on the MeanFieldHomogenization.jl side: the
#  closed-form kernels of `cases.py` as MFH laws, the schemes, and the RVE of a
#  case (a dictionary as `cases.py` writes it). Shared by `run_mfh.jl` and by the
#  test `test/Viscoelasticity/test_alv_echoes_cells.jl`, so that a case is built
#  the same way wherever it is checked.
# =============================================================================

module ALVHarnessCells

using MeanFieldHomogenization
using TensND
using LinearAlgebra

export mfh_law, mfh_rve, case_property, phase_relaxation, mat, SCHEMES

# ── The kernels of cases.py, as MFH laws ─────────────────────────────────────

aging(law) = tp -> 1 - law["a"] * exp(-tp / law["t_a"])

# Echoes angles (θ, φ, ψ): the local axis 3 is (sin θ cos φ, sin θ sin φ, cos θ).
local_axis(e) = (sin(e[1]) * cos(e[2]), sin(e[1]) * sin(e[2]), cos(e[1]))

# A TI tensor from its local constants (C₁₁, C₁₂, C₁₃, C₃₃, C₄₄) about the axis n:
# Walpole (axial C₃₃, transverse C₁₁ + C₁₂, coupling √2 C₁₃, in-plane shear
# C₁₁ − C₁₂, out-of-plane shear 2C₄₄), in the Kelvin-Mandel normalization.
ti_tensor(c, n) = TensTI{4}(c[4], c[1] + c[2], √2 * c[3], c[1] - c[2], 2 * c[5], n)

function mfh_law(law)
    typ = law["type"]
    if typ == "elastic_iso"
        return heaviside_law(TensISO{3}(3 * law["k"], 2 * law["mu"]))
    elseif typ == "maxwell_aging_iso"
        g = aging(law)
        return ViscoLaw(
            (t, tp) -> g(tp) * TensISO{3}(
                3 * (law["k_inf"] + law["k1"] * exp(-(t - tp) / law["tau_k"])),
                2 * (law["mu_inf"] + law["mu1"] * exp(-(t - tp) / law["tau_mu"]))
            )
        )
    elseif typ == "kelvin_aging_iso"
        g = aging(law)
        return ViscoLaw(
            (t, tp) -> TensISO{3}(
                1 / (3 * law["k0"]) + (1 - exp(-(t - tp) / law["tau_k"])) / (3 * law["k1"] * g(tp)),
                1 / (2 * law["mu0"]) + (1 - exp(-(t - tp) / law["tau_mu"])) / (2 * law["mu1"] * g(tp))
            ), :creep
        )
    elseif typ == "maxwell_aging_ti"
        g, n = aging(law), local_axis(law["euler_law"])
        A, B = ti_tensor(law["CA"], n), ti_tensor(law["CB"], n)
        return ViscoLaw((t, tp) -> g(tp) * (exp(-(t - tp) / law["tau"]) * A + B))
    elseif typ == "elastic_cond_iso"
        return heaviside_law(TensISO{3}(law["k"]))
    elseif typ == "maxwell_aging_cond_iso"
        g = aging(law)
        return ViscoLaw((t, tp) -> TensISO{3}(g(tp) * (law["k_inf"] + law["k1"] * exp(-(t - tp) / law["tau"]))))
    elseif typ == "kelvin_aging_cond_iso"
        g = aging(law)
        return ViscoLaw((t, tp) -> TensISO{3}(1 / law["k0"] + (1 - exp(-(t - tp) / law["tau"])) / (law["k1"] * g(tp))), :creep)
    elseif typ == "maxwell_aging_cond_ti"
        g, n = aging(law), local_axis(law["euler_law"])
        return ViscoLaw((t, tp) -> (g(tp) * (law["w_inf"] + exp(-(t - tp) / law["tau"]))) * TensTI{2}(law["kt"], law["ka"], n))
    end
    error("unknown law type $typ")
end

mat(rows) = [Float64(rows[i][j]) for i in eachindex(rows), j in eachindex(rows[1])]

# The relaxation operator MFH discretizes for a law, as `homogenize_alv` does.
phase_relaxation(law::AbstractViscoLaw, times) =
    (M = Matrix(VolterraOperator(law, times)); visco_mode(law) === :creep ? inv(M) : M)
phase_relaxation(R::AbstractMatrix, times) = Matrix(R)

function symmetrize(p)
    s = p["symmetrize"]
    s === nothing && return (;)
    s["sym"] == "ISO" && return (; symmetrize = :iso)
    return (; symmetrize = TISymmetrize(local_axis((s["theta"], s["phi"], 0.0))))
end

# The first phase is the matrix of the dilute and Mori-Tanaka schemes.
const SCHEMES = Dict(
    "voigt" => (c, m) -> Voigt(), "reuss" => (c, m) -> Reuss(),
    "dilute_dual" => (c, m) -> DiluteDual(m), "maxwell" => (c, m) -> Maxwell(m),
    "pcw" => (c, m) -> PonteCastanedaWillis(m),
    "differential" => (c, m) -> DifferentialScheme(abstol = 1.0e-12, reltol = 1.0e-10),
    "dilute" => (c, m) -> Dilute(m), "mori_tanaka" => (c, m) -> MoriTanaka(m),
    "self_consistent" => (c, m) -> SelfConsistent(reltol = c["epsrel"]),
    "asymmetric_self_consistent" => (c, m) -> AsymmetricSelfConsistent(m; reltol = c["epsrel"]),
)

# The phase property: a law, or the operator of a source case.
case_property(p, sources) = p["law"]["type"] == "from_case" ? sources[p["law"]["case"]] : mfh_law(p["law"])

function mfh_rve(c, sources)
    prop = c["order"] == 4 ? :C : :K
    d = get(c, "distribution", nothing)
    rve = d === nothing ? RVE() :
        RVE(; distribution_shape = Ellipsoid(Float64.(d["axes"])...; euler_angles = Tuple(Float64.(d["euler"]))))
    for (k, p) in enumerate(c["phases"])
        if get(p, "crack", false)
            # A crack family: its density is η² times Echoes' (see cases.py).
            a, b = Float64.(p["axes"][1:2])
            crack = EllipticCrack(a, b; euler_angles = Tuple(Float64.(p["euler"])))
            add_phase!(rve, Symbol(p["name"]), crack, Dict(prop => case_property(p, sources)); density = (b / a)^2 * p["density"], symmetrize(p)...)
            continue
        end
        # The first phase takes the rest (the fractions of a case sum to 1): the matrix.
        fraction = k == 1 ? :rest : p["fraction"]
        shape = p["axes"] === nothing ? Ellipsoid(1.0) :
            Ellipsoid(Float64.(p["axes"])...; euler_angles = Tuple(Float64.(p["euler"])))
        add_phase!(rve, Symbol(p["name"]), shape, Dict(prop => case_property(p, sources)); fraction, symmetrize(p)...)
    end
    return rve, prop, Symbol(c["phases"][1]["name"])
end

end # module
