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
# ALV_HARNESS_N=<N>: the cases run by run_echoes.py on N instants, in results_n<N>/.
const RESULTS = joinpath(HERE, haskey(ENV, "ALV_HARNESS_N") ? "results_n$(ENV["ALV_HARNESS_N"])" : "results")

include(joinpath(@__DIR__, "cells.jl"))
using .ALVHarnessCells

rel(a, b) = norm(a - b) / norm(b)

function run_case(c, echo, sources, echoes_sources)
    times = Float64.(c["times"])
    rve, prop, m = mfh_rve(c, sources)
    sch = SCHEMES[c["scheme"]](c, m)
    # 1. The phase operators against Echoes'.
    # A crack's kernel is zero on both sides: not compared.
    phase_gap = maximum(filter(p -> !get(p, "crack", false), c["phases"])) do p
        rel(phase_relaxation(case_property(p, sources), times), mat(echo["phase_mats"][p["name"]]))
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
    # On a realistic grid, the numbers the "Performance vs Echoes" page shows.
    haskey(ENV, "ALV_HARNESS_N") && open(joinpath(HERE, "..", "figures", "alv_harness_n$(ENV["ALV_HARNESS_N"]).json"), "w") do io
        JSON.print(
            io, Dict(
                "cpu" => Sys.cpu_info()[1].model, "julia" => string(VERSION),
                "mfh" => string(pkgversion(MeanFieldHomogenization)), "tensnd" => string(pkgversion(TensND)),
                "threads" => Threads.nthreads(), "instants" => parse(Int, ENV["ALV_HARNESS_N"]),
                "protocol" => "minimum of 5 runs of `homogenize_alv` after a warm-up against the minimum of 3 runs of Echoes' `homogenize_visco`, each code alone on the machine",
                "rows" => rows,
            ), 2
        )
    end
    return rows
end

main(ARGS)
