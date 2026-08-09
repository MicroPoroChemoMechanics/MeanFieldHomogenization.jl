# =============================================================================
#  harness.jl — benchmark harness for the MeanFieldHomogenization / TensND optimization
#  campaign.
#
#  Why three independent channels per case (see README.md for the full
#  rationale):
#
#    * `time`     — `BenchmarkTools.@benchmark`, min + median.  Auto-tunes
#                   `evals` so a 100 ns kernel is measured against a ≥1 ms
#                   window; the hand-rolled `@elapsed` convention used by
#                   `bench_alv.jl` cannot resolve that tier.
#    * `alloc`    — a deliberately naive loop of `@allocated` on the
#                   **verbatim** thunk, no `$`-interpolation, ≥2 warm-ups and
#                   several samples in the *same* process, reporting **min and
#                   max**.  `@benchmark` requires `$`-interpolation, which
#                   hoists a closure capture out of the measured region — i.e.
#                   it is exactly the thing that would hide a closure-boxing
#                   regression.  `max ≠ min` is itself a signal (type
#                   instability), and it invalidates any "one `@allocated` on
#                   a fresh process" reading.
#    * `counters` — instrumented `Ref{Int}` work counters read around a
#                   separate clean call.  Distinguishes "faster" from "did
#                   less work": a drop in adaptive-quadrature nodes is a
#                   change of behavior, not a speed-up.
#
#  Every case also carries a `checksum` closure evaluated on **the same call
#  that is timed**, so numerical non-regression is asserted on the real code
#  path rather than on a parallel reimplementation.
# =============================================================================

using BenchmarkTools
using JSON3
using SHA
using Printf
using Statistics
using Dates

using MeanFieldHomogenization
using TensND

const MFH = MeanFieldHomogenization
const MFHC = MeanFieldHomogenization.Core

# The work counters (`src/Core/counters.jl`) are part of the campaign's
# instrumentation.  Guard on their presence so this harness can also be run
# against a pre-campaign checkout (e.g. a `git worktree` at the baseline
# commit) — which is how the baseline is measured on genuinely untouched code.
const HAS_COUNTERS = isdefined(MFHC, :reset_counters!)
_reset_counters!() = HAS_COUNTERS ? MFHC.reset_counters!() : nothing
_read_counters() = HAS_COUNTERS ? MFHC.read_counters() : NamedTuple()
_with_counting(f) = HAS_COUNTERS ? MFHC.with_integrand_counting(f) : f()
_counting_off() = !HAS_COUNTERS || !MFHC.COUNT_INTEGRAND[]

# ── Case definition ─────────────────────────────────────────────────────────

"""
    BCase(id; group, tags, setup, body, checksum, control, skip_if)

One benchmark case.

* `setup()`      → `ctx`, run once, **not** measured.
* `body(ctx)`    → result; this is the measured call.
* `checksum(res)`→ `Vector{Float64}`, the canonical numerical fingerprint.
* `control`      → `true` marks a case that must **not** move; the control
                   cases calibrate the noise floor of a run.
* `skip_if()`    → `true` skips the case (e.g. an unloaded extension).
"""
struct BCase
    id::String
    group::Symbol
    tags::Vector{Symbol}
    setup::Function
    body::Function
    checksum::Function
    control::Bool
    skip_if::Function
end

const REGISTRY = BCase[]

function bcase(
        id::AbstractString; group::Symbol, setup::Function, body::Function,
        tags = Symbol[], checksum::Function = default_checksum,
        control::Bool = false, skip_if::Function = () -> false
    )
    any(c -> c.id == id, REGISTRY) && error("duplicate benchmark id: $id")
    push!(
        REGISTRY,
        BCase(String(id), group, collect(Symbol, tags), setup, body, checksum, control, skip_if)
    )
    return nothing
end

# ── Canonical checksum ──────────────────────────────────────────────────────

"""
    default_checksum(x) -> Vector{Float64}

Flatten a benchmark result to a deterministic `Float64` vector.

`TensND` tensors go through `get_array`, whose iteration order is the
column-major order of the underlying `Tensor`/`SArray` — deterministic and
independent of the type's parameter count.  Complex values are split into
(re, im) pairs; tuples and named tuples are flattened in declaration order.
"""
default_checksum(x::Real) = Float64[x]
default_checksum(x::Complex) = Float64[real(x), imag(x)]
default_checksum(x::TensND.AbstractTens) = default_checksum(TensND.get_array(x))
default_checksum(x::AbstractArray) = _flatten(x)
default_checksum(x::Tuple) = _flatten(x)
default_checksum(x::NamedTuple) = _flatten(values(x))

function _flatten(x)
    out = Float64[]
    _flatten!(out, x)
    return out
end

_flatten!(out, x::Real) = push!(out, Float64(x))
_flatten!(out, x::Complex) = (push!(out, Float64(real(x))); push!(out, Float64(imag(x))))
function _flatten!(out, x)
    for v in x
        _flatten!(out, v)
    end
    return out
end

"""
    canonical(v) -> String

`%.17g` is exactly round-trippable for `Float64`, so the resulting string —
and hence its sha256 — is a genuine **bit**-identity fingerprint, and the
values stored alongside it in the report are exact (old reports can be
re-diffed without re-running).
"""
canonical(v::AbstractVector{Float64}) = join((@sprintf("%.17g", x) for x in v), ",")

checksum_sha(v::AbstractVector{Float64}) = bytes2hex(sha256(canonical(v)))

# ── Environment fingerprint ─────────────────────────────────────────────────

function _git(args...)
    try
        return strip(read(`git -C $(pkgdir(MeanFieldHomogenization)) $(collect(args))`, String))
    catch
        return ""
    end
end

function env_fingerprint()
    porcelain = _git("status", "--porcelain")
    return (
        git_commit = _git("rev-parse", "HEAD"),
        git_clean = isempty(porcelain),
        git_dirty_files = isempty(porcelain) ? String[] : split(porcelain, '\n'),
        julia = string(VERSION),
        cpu = Sys.CPU_NAME,
        cpu_threads = Sys.CPU_THREADS,
        threads = Threads.nthreads(),
        opt_level = Int(Base.JLOptions().opt_level),
        check_bounds = Int(Base.JLOptions().check_bounds),
        ext_decuhr = Base.get_extension(MeanFieldHomogenization, :MeanFieldHomogenizationDECUHRExt) !== nothing,
        ext_nonlinearsolve = Base.get_extension(MeanFieldHomogenization, :MeanFieldHomogenizationNonlinearSolveExt) !== nothing,
    )
end

# ── Runner ──────────────────────────────────────────────────────────────────

"""
    run_case(c; warmups, samples, seconds) -> NamedTuple

Run the five measurement steps of one case, in order, in this process.

Step 2 (counters) is the **only** step with `COUNT_INTEGRAND` enabled, and
step 5 (timing) asserts that it is back off — the timed path must never see
the counting wrapper.
"""
function run_case(
        c::BCase; warmups::Int = 3, samples::Int = 7,
        seconds::Float64 = 2.0, time_samples::Int = 10_000
    )
    ctx = c.setup()

    # 1. correctness + compilation
    r0 = c.body(ctx)
    chk = c.checksum(r0)

    # 2. work counters (separate invocation; counting enabled only here)
    _reset_counters!()
    _with_counting(() -> c.body(ctx))
    cnt = _read_counters()

    # 3. warm-up
    for _ in 1:warmups
        c.body(ctx)
    end
    GC.gc(true)

    # 4. allocation channel — verbatim thunk, no interpolation.
    #    Allocations are deterministic (min == max on every case measured so
    #    far), so a handful of samples is enough here.
    allocs = Vector{Int}(undef, samples)
    for i in 1:samples
        allocs[i] = @allocated c.body(ctx)
    end
    GC.gc(true)

    # 5. timing channel
    @assert _counting_off() "integrand counting leaked into the timed run"
    f = c.body
    # `samples` is deliberately NOT the small allocation-channel count: with
    # only a handful of samples a single GC pause lands in every one of them
    # and inflates `minimum` — measured on this suite, that turned a +4.5 %
    # change into a phantom +103 % and a +3.6 % change into a phantom −46 %.
    # Let BenchmarkTools take as many samples as the `seconds` budget allows
    # (bounded by `time_samples`), so the minimum is a robust statistic.
    # `evals` is left unset so `tune!` picks it — required for the ~ns tier.
    b = @benchmark $f($ctx) samples = time_samples seconds = seconds

    return (
        id = c.id, group = String(c.group), tags = String.(c.tags),
        control = c.control,
        time_ns_min = minimum(b).time,
        time_ns_median = median(b).time,
        bm_evals = b.params.evals, bm_samples = length(b.times),
        alloc_bytes_min = minimum(allocs), alloc_bytes_max = maximum(allocs),
        counters = cnt,
        checksum = (n = length(chk), sha256 = checksum_sha(chk), values = chk),
    )
end

function run_suite(;
        filter_pat = "", warmups = 3, samples = 7, seconds = 2.0,
        time_samples = 10_000, shuffle = false
    )
    cases = [c for c in REGISTRY if _matches(c, filter_pat)]
    shuffle && (cases = Random.shuffle(cases))
    results = Dict{String, Any}()
    for (i, c) in enumerate(cases)
        if c.skip_if()
            @info "[$i/$(length(cases))] SKIP $(c.id)"
            continue
        end
        print(rpad("[$i/$(length(cases))] $(c.id)", 60))
        flush(stdout)
        r = try
            run_case(c; warmups, samples, seconds, time_samples)
        catch err
            println(" ERROR")
            @error "case $(c.id) failed" exception = (err, catch_backtrace())
            rethrow()
        end
        @printf(
            " %10s  %10s\n", _fmt_time(r.time_ns_min), _fmt_bytes(r.alloc_bytes_min)
        )
        results[c.id] = r
    end
    return results
end

_matches(c::BCase, pat) =
    isempty(pat) || occursin(pat, c.id) ||
    String(c.group) == pat || Symbol(pat) in c.tags

# ── Report I/O ──────────────────────────────────────────────────────────────

const SCHEMA_VERSION = 1

function write_report(path, results; label, settings)
    mkpath(dirname(abspath(path)))
    payload = Dict(
        "schema_version" => SCHEMA_VERSION,
        "label" => label,
        "created" => string(now()),
        "env" => env_fingerprint(),
        "settings" => settings,
        "cases" => results,
    )
    open(path, "w") do io
        JSON3.pretty(io, payload)
    end
    return path
end

read_report(path) = JSON3.read(read(path, String))

# ── Diff ────────────────────────────────────────────────────────────────────

_fmt_time(ns) = ns < 1.0e3 ? @sprintf("%.0f ns", ns) :
    ns < 1.0e6 ? @sprintf("%.2f µs", ns / 1.0e3) :
    ns < 1.0e9 ? @sprintf("%.2f ms", ns / 1.0e6) : @sprintf("%.2f s", ns / 1.0e9)

_fmt_bytes(b) = b < 1024 ? @sprintf("%d B", b) :
    b < 1024^2 ? @sprintf("%.2f KiB", b / 1024) : @sprintf("%.2f MiB", b / 1024^2)

_pct(new, old) = old == 0 ? (new == 0 ? 0.0 : Inf) : 100 * (new - old) / old

"""
    gate_check(new_case, ref_case, gate) -> (ok, detail)

`gate = :bitwise` compares the sha256 of the canonical `%.17g` rendering —
a true bit-identity test.  `gate = :tol` uses a **scale-relative** bound,
required because component magnitudes across the suite span `C ~ O(200)`,
`A_εε ~ O(1)` and a stiff-matrix `H ~ O(1e-3)`.
"""
function gate_check(newc, refc, gate::Symbol; tol::Float64 = 1.0e-14)
    if gate === :bitwise
        return newc.checksum.sha256 == refc.checksum.sha256 ?
            (true, "BIT-OK") : (false, "BIT-FAIL")
    end
    a = collect(Float64, newc.checksum.values)
    b = collect(Float64, refc.checksum.values)
    length(a) == length(b) || return (false, "LEN-FAIL($(length(a))≠$(length(b)))")
    isempty(a) && return (true, "OK")
    scale = max(1.0, maximum(abs, b))
    d = maximum(abs, a .- b) / scale
    return d <= tol ? (true, @sprintf("OK(%.1e)", d)) : (false, @sprintf("TOL-FAIL(%.1e)", d))
end

"""
    noise_floor(reports) -> Float64

p90 of `|Δt|/t` over the **control** cases of two repeats of the same suite.
A case counts as MOVED only beyond `max(3·noise, 3 %)`; a control that moves
that far invalidates the whole run.  This is what gives the comparison a
calibrated null hypothesis instead of a bare table of numbers.
"""
function noise_floor(r1, r2)
    rel = Float64[]
    for (id, a) in r1
        haskey(r2, id) || continue
        b = r2[id]
        (a isa NamedTuple ? a.control : a["control"]) || continue
        ta = a isa NamedTuple ? a.time_ns_min : a["time_ns_min"]
        tb = b isa NamedTuple ? b.time_ns_min : b["time_ns_min"]
        ta > 0 && push!(rel, abs(tb - ta) / ta)
    end
    isempty(rel) && return 0.0
    return quantile(sort(rel), 0.9)
end

function diff_reports(new_results, ref_report; gate::Symbol = :bitwise, noise::Float64 = 0.0, tol = 1.0e-14)
    refcases = ref_report.cases
    thresh = max(3 * noise, 0.03)
    moved = 0; gatefail = 0; ctrlfail = 0; unreliable = 0

    @printf(
        "\n── %s vs %s ──  noise floor (controls, p90) = %.1f %%\n",
        "current", ref_report.label, 100 * noise
    )
    @printf(
        "%-44s %11s %8s %11s %8s  %-22s %s\n",
        "CASE", "TIME(min)", "Δ", "ALLOC", "Δ", "WORK", "CHECK"
    )

    for id in sort(collect(keys(new_results)))
        n = new_results[id]
        haskey(refcases, Symbol(id)) || (println(rpad(id, 44), "  (new case)"); continue)
        r = refcases[Symbol(id)]

        dt = _pct(n.time_ns_min, r.time_ns_min)
        da = _pct(n.alloc_bytes_min, r.alloc_bytes_min)
        ok, detail = gate_check(n, r, gate; tol)
        ok || (gatefail += 1)

        work = _work_delta(n.counters, r.counters)
        # A differing `evals` tuning means the two minima are not the same
        # statistic (min-of-1-call vs min-of-N-amortized-calls).  Measured on
        # this suite, that alone produced a phantom +63 % on a 0.6 µs control
        # case whose allocations were byte-identical.  Do not call such a case
        # moved — flag the comparison as unreliable instead.
        evals_differ = n.bm_evals != r.bm_evals
        is_moved = !evals_differ && abs(dt) / 100 > thresh
        is_moved && (moved += 1)
        n.control && is_moved && (ctrlfail += 1)
        evals_differ && (unreliable += 1)

        @printf(
            "%-44s %11s %+7.1f%% %11s %+7.1f%%  %-22s %s%s\n",
            id, _fmt_time(n.time_ns_min), dt,
            _fmt_bytes(n.alloc_bytes_min), da, work, detail,
            evals_differ ? "  ~evals$(r.bm_evals)→$(n.bm_evals)" :
                n.control && is_moved ? "  ⚠CONTROL" : ""
        )
    end

    @printf(
        "─── %d moved, %d gate failures, %d control regressions, %d unreliable (evals differ)\n",
        moved, gatefail, ctrlfail, unreliable
    )
    return (moved = moved, gatefail = gatefail, ctrlfail = ctrlfail, unreliable = unreliable)
end

function _work_delta(newc, refc)
    parts = String[]
    for k in (:hill_calls, :cod_calls, :residue_solves, :integrand_evals, :sc_iterations, :layer_recurrences)
        a = get(refc, k, 0); b = get(newc, k, 0)
        (a == 0 && b == 0) && continue
        push!(parts, a == b ? "$(_short(k)) $a" : "$(_short(k)) $a→$b")
    end
    return isempty(parts) ? "—" : join(parts, " ")
end

_short(k::Symbol) = k === :hill_calls ? "hill" : k === :cod_calls ? "cod" :
    k === :residue_solves ? "res" : k === :integrand_evals ? "nodes" :
    k === :sc_iterations ? "sc" : "lay"
