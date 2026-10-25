# =============================================================================
#  bench_suite.jl — entry point of the optimization-campaign benchmark suite.
#
#  Run from the MeanFieldHomogenization.jl package root (instantiate once first):
#
#    julia --project=scripts/bench -e 'using Pkg; Pkg.instantiate()'
#
#    # capture the pre-optimization reference (clean tree)
#    JULIA_NUM_THREADS=1 julia --project=scripts/bench scripts/bench/bench_suite.jl \
#        --record-baseline --label=P0-baseline
#
#    # after an optimization tier
#    JULIA_NUM_THREADS=1 julia --project=scripts/bench scripts/bench/bench_suite.jl \
#        --label=P1-dedup --out=scripts/bench/results/P1.json \
#        --baseline=scripts/bench/baseline.json --gate=bitwise --repeat-suite=2
#
#    # correctness sweep only (no timing)
#    julia --project=scripts/bench scripts/bench/bench_suite.jl --verify-only
#
#  Exit codes: 0 clean · 1 checksum-gate failure · 2 control regression /
#  invalid run · 3 setup error.
# =============================================================================

import Pkg
Pkg.activate(@__DIR__; io = devnull)

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Random
import ForwardDiff as FD
# Load the weak-dependency extensions so the `:decuhr` and NonlinearSolve
# cases are exercised rather than skipped.
import DECUHR, Integrals
import NonlinearSolve

include("harness.jl")
include("fixtures.jl")
include("cases_kernels.jl")
include("cases_schemes.jl")
include("cases_alv.jl")
include("cases_tensnd.jl")
include("cases_common.jl")

# ── CLI ─────────────────────────────────────────────────────────────────────

function parse_args(argv)
    o = Dict{String, Any}(
        "label" => "adhoc", "out" => "", "baseline" => "",
        "gate" => "bitwise", "filter" => "", "samples" => 7, "warmups" => 3,
        "seconds" => 1.0, "repeat-suite" => 1, "record-baseline" => false,
        "verify-only" => false, "force" => false, "shuffle" => false,
        "tol" => 1.0e-14, "diff" => String[],
    )
    i = 1
    while i <= length(argv)
        a = argv[i]
        if a == "--diff"
            o["diff"] = [argv[i + 1], argv[i + 2]]; i += 3; continue
        elseif startswith(a, "--")
            k, _, v = partition_arg(a)
            if k in ("record-baseline", "verify-only", "force", "shuffle")
                o[k] = true
            elseif k in ("samples", "warmups", "repeat-suite")
                o[k] = parse(Int, v)
            elseif k in ("seconds", "tol")
                o[k] = parse(Float64, v)
            elseif haskey(o, k)
                o[k] = v
            else
                error("unknown option --$k")
            end
        end
        i += 1
    end
    return o
end

function partition_arg(a)
    body = a[3:end]
    j = findfirst('=', body)
    return j === nothing ? (body, false, "") : (body[1:(j - 1)], true, body[(j + 1):end])
end

const BASELINE_PATH = joinpath(@__DIR__, "baseline.json")

function main(argv)
    o = parse_args(argv)

    if !isempty(o["diff"])
        r1 = read_report(o["diff"][1]); r2 = read_report(o["diff"][2])
        res = Dict(String(k) => v for (k, v) in pairs(r2.cases))
        st = diff_reports(res, r1; gate = Symbol(o["gate"]), tol = o["tol"])
        return st.gatefail > 0 ? 1 : (st.ctrlfail > 0 ? 2 : 0)
    end

    env = env_fingerprint()
    println("Julia $(env.julia) · $(env.cpu) · threads=$(env.threads) · opt=$(env.opt_level)")
    println("commit $(env.git_commit)  clean=$(env.git_clean)")
    println("extensions: DECUHR=$(env.ext_decuhr) NonlinearSolve=$(env.ext_nonlinearsolve)\n")

    if o["verify-only"]
        return verify_only(o)
    end

    if o["record-baseline"] && !env.git_clean && !o["force"]
        @error """
        Refusing to record a baseline from a dirty working tree.
        Dirty files:
        $(join("  " .* env.git_dirty_files, "\n"))
        Pass --force to record anyway (the dirty file list is stored in the report).
        """
        return 2
    end

    settings = Dict(
        "warmups" => o["warmups"], "samples" => o["samples"], "seconds" => o["seconds"]
    )

    results = run_suite(;
        filter_pat = o["filter"], warmups = o["warmups"],
        samples = o["samples"], seconds = o["seconds"], shuffle = o["shuffle"]
    )

    noise = 0.0
    if o["repeat-suite"] > 1
        println("\n── repeat pass (noise-floor calibration) ──")
        results2 = run_suite(;
            filter_pat = o["filter"], warmups = o["warmups"],
            samples = o["samples"], seconds = o["seconds"], shuffle = o["shuffle"]
        )
        noise = noise_floor(results, results2)
        @printf("\nnoise floor (controls, p90 of |Δt|/t) = %.2f %%\n", 100 * noise)
    end

    out = isempty(o["out"]) ?
        (o["record-baseline"] ? BASELINE_PATH :
        joinpath(@__DIR__, "results", "$(o["label"]).json")) : o["out"]
    write_report(out, results; label = o["label"], settings)
    println("\nwrote $(out)  ($(length(results)) cases)")

    ref_path = isempty(o["baseline"]) ?
        (o["record-baseline"] ? "" : BASELINE_PATH) : o["baseline"]
    if !isempty(ref_path) && isfile(ref_path)
        st = diff_reports(
            results, read_report(ref_path);
            gate = Symbol(o["gate"]), noise, tol = o["tol"]
        )
        st.gatefail > 0 && return 1
        st.ctrlfail > 0 && return 2
    end
    return 0
end

"""Correctness sweep: run each case once and report its checksum, no timing."""
function verify_only(o)
    ref = isfile(BASELINE_PATH) ? read_report(BASELINE_PATH) : nothing
    bad = 0
    for c in REGISTRY
        _matches(c, o["filter"]) || continue
        c.skip_if() && continue
        chk = c.checksum(c.body(c.setup()))
        sha = checksum_sha(chk)
        status = "—"
        if ref !== nothing && haskey(ref.cases, c.id)
            r = ref.cases[c.id]
            ok, detail = gate_check(
                (checksum = (sha256 = sha, values = chk),),
                r, Symbol(o["gate"]); tol = o["tol"]
            )
            status = detail
            ok || (bad += 1)
        end
        @printf("%-44s %s\n", c.id, status)
    end
    return bad > 0 ? 1 : 0
end

exit(main(ARGS))
