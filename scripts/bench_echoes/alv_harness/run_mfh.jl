# =============================================================================
#  scripts/bench_echoes/alv_harness/run_mfh.jl
#
#  The MeanFieldHomogenization.jl side of the aging linear viscoelastic (ALV)
#  harness: every case that `run_echoes.py` computed with Echoes
#  (`results/<case>.json`, which embeds the case) is rebuilt here from the same
#  closed-form kernels and homogenized with `homogenize_alv`, then compared.
#
#  Two comparisons per case:
#    1. the discretized operator of every phase against the one Echoes used
#       (`phase_mats`): the kernels and the trapezoidal rule, before any scheme;
#    2. the effective operator against Echoes' (relative Frobenius gap).
#  A multiscale case takes the MFH result of its source case as a phase, as a
#  `VolterraOperator` (the whole chain is MFH's), and is also checked against
#  Echoes with Echoes' source matrix handed over instead.
#
#  Timings: the minimum of 5 runs after a warm-up, for `homogenize_alv` alone;
#  Echoes' is the minimum of 3 runs of `homogenize_visco` (run_echoes.py). Run
#  alone on the machine, from the package root:
#
#    julia --project=<an environment with MeanFieldHomogenization and JSON> \
#        scripts/bench_echoes/alv_harness/run_mfh.jl [case ...]
#
#  Writes `results/mfh_report.json`.
# =============================================================================

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using JSON

const HERE = @__DIR__
const RESULTS = joinpath(HERE, "results")

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
relaxation(law::AbstractViscoLaw, times) =
    (M = Matrix(VolterraOperator(law, times)); visco_mode(law) === :creep ? inv(M) : M)
relaxation(R::AbstractMatrix, times) = Matrix(R)

function symmetrize(p)
    s = p["symmetrize"]
    s === nothing && return (;)
    s["sym"] == "ISO" && return (; symmetrize = :iso)
    return (; symmetrize = TISymmetrize(local_axis((s["theta"], s["phi"], 0.0))))
end

# The first phase is the matrix of the dilute and Mori-Tanaka schemes.
const SCHEMES = Dict(
    "dilute" => (c, m) -> Dilute(m), "mori_tanaka" => (c, m) -> MoriTanaka(m),
    "self_consistent" => (c, m) -> SelfConsistent(reltol = c["epsrel"]),
)

# The phase property: a law, or the operator of a source case.
property(p, sources) = p["law"]["type"] == "from_case" ? sources[p["law"]["case"]] : mfh_law(p["law"])

function mfh_rve(c, sources)
    prop = c["order"] == 4 ? :C : :K
    rve = RVE()
    for (k, p) in enumerate(c["phases"])
        # The first phase takes the rest (the fractions of a case sum to 1): the matrix.
        fraction = k == 1 ? :rest : p["fraction"]
        shape = p["axes"] === nothing ? Ellipsoid(1.0) :
            Ellipsoid(Float64.(p["axes"])...; euler_angles = Tuple(Float64.(p["euler"])))
        add_phase!(rve, Symbol(p["name"]), shape, Dict(prop => property(p, sources)); fraction, symmetrize(p)...)
    end
    return rve, prop, Symbol(c["phases"][1]["name"])
end

rel(a, b) = norm(a - b) / norm(b)

function run_case(c, echo, sources, echoes_sources)
    times = Float64.(c["times"])
    rve, prop, m = mfh_rve(c, sources)
    sch = SCHEMES[c["scheme"]](c, m)
    # 1. The phase operators against Echoes'.
    phase_gap = maximum(c["phases"]) do p
        rel(relaxation(property(p, sources), times), mat(echo["phase_mats"][p["name"]]))
    end
    # 2. The effective operator, and its time.
    R = homogenize_alv(rve, sch, prop; times)
    t = minimum(@elapsed(homogenize_alv(rve, sch, prop; times)) for _ in 1:5)
    gap = rel(Matrix(R), mat(echo["effective"]))
    # A multiscale case also with Echoes' own source operator, to separate the scales.
    gap_echoes_source = if any(p -> p["law"]["type"] == "from_case", c["phases"])
        rve_e, _, _ = mfh_rve(c, echoes_sources)
        rel(Matrix(homogenize_alv(rve_e, sch, prop; times)), mat(echo["effective"]))
    else
        nothing   # a single-scale case
    end
    return R, Dict(
            "case" => c["name"], "class" => string(nameof(typeof(R.class))), "phase_gap" => phase_gap,
            "gap" => gap, "gap_with_echoes_source" => gap_echoes_source,
            "t_mfh" => t, "t_echoes" => echo["wall_time_s"],
        )
end

function main(names)
    files = sort(filter(f -> endswith(f, ".json") && f != "mfh_report.json", readdir(RESULTS)))
    echoes = Dict(splitext(f)[1] => JSON.parsefile(joinpath(RESULTS, f)) for f in files)
    # Sources first: a multiscale case needs the result of its source case.
    order = sort(collect(keys(echoes)); by = n -> any(p -> p["law"]["type"] == "from_case", echoes[n]["case"]["phases"]))
    isempty(names) || (order = filter(in(names), order))
    sources = Dict{String, Any}()
    echoes_sources = Dict(n => VolterraOperator(mat(e["effective"]), Float64.(e["times"])) for (n, e) in echoes)
    rows = Dict{String, Any}[]
    @printf "%-38s %-10s %9s %9s %9s %10s %10s %7s\n" "case" "class" "phases" "gap" "gap(E-src)" "t MFH" "t Echoes" "E/MFH"
    for name in order
        e = echoes[name]
        R, row = run_case(e["case"], e, sources, echoes_sources)
        sources[name] = R
        push!(rows, row)
        @printf "%-38s %-10s %9.1e %9.1e %9.1e %10.3g %10.3g %7.2f\n" name row["class"] row["phase_gap"] row["gap"] something(row["gap_with_echoes_source"], NaN) row["t_mfh"] row["t_echoes"] row["t_echoes"] / row["t_mfh"]
    end
    open(joinpath(RESULTS, "mfh_report.json"), "w") do io
        JSON.print(io, rows, 2)
    end
    return rows
end

main(ARGS)
