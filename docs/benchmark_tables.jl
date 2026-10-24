# Tables of the "Performance vs Echoes" page, read from the JSON files that
# `scripts/bench_echoes/benchmark.jl` and `benchmark_schemes_speed.jl` write:
# the page shows the measured numbers, never numbers typed by hand.

using JSON
using Markdown
using Printf

const BENCH_DIR = joinpath(dirname(@__DIR__), "scripts", "bench_echoes", "figures")

bench_data(name) = JSON.parsefile(joinpath(BENCH_DIR, name))

function _fmt_time(t)
    t < 1.0e-6 && return @sprintf("%.3g ns", 1.0e9 * t)
    t < 1.0e-3 && return @sprintf("%.3g µs", 1.0e6 * t)
    t < 1.0 && return @sprintf("%.3g ms", 1.0e3 * t)
    return @sprintf("%.3g s", t)
end
# A ratio below 1 (Echoes faster) in bold, so that it cannot be missed.
_fmt_ratio(r) = r < 1 ? @sprintf("**%.2g×**", r) : r < 100 ? @sprintf("%.3g×", r) :
    string(round(Int, r; sigdigits = 2), "×")

"""
    machine_line(name) -> Markdown.MD

The machine, versions and protocol a JSON file was measured with.
"""
function machine_line(name)
    d = bench_data(name)
    return Markdown.parse(
        "Measured on $(d["cpu"]), Julia $(d["julia"]), MeanFieldHomogenization $(d["mfh"]), " *
            "TensND $(d["tensnd"]), $(d["threads"]) thread; $(d["protocol"])."
    )
end

"""
    kernel_table(section) -> Markdown.MD

One section (`"hill"`, `"crack"`, `"conduction"`, `"derivative"`) of
`benchmark_kernels.json`: the case, Echoes' algorithm, both times, their ratio
and the largest relative gap between the two results.
"""
function kernel_table(section)
    io = IOBuffer()
    println(io, "| Case | Echoes | ``t(J)`` | ``t(E)`` | ``t(E)/t(J)`` | gap |")
    println(io, "| :--- | :--- | ---: | ---: | ---: | ---: |")
    for r in bench_data("benchmark_kernels.json")[section]
        @printf(
            io, "| %s | %s | %s | %s | %s | %.1e |\n", r["label"], r["algo"],
            _fmt_time(r["tJ"]), _fmt_time(r["tE"]), _fmt_ratio(r["ratio"]), r["maxrel"]
        )
    end
    return Markdown.parse(String(take!(io)))
end

"""
    scheme_table(; strength = false) -> Markdown.MD

The rows of `benchmark_schemes_speed.jl`: the like-for-like cases, or, with
`strength = true`, the three-scale strength model alone.
"""
function scheme_table(; strength::Bool = false)
    io = IOBuffer()
    println(io, "| Case | Scheme | ``t(J)`` | ``t(E)`` | ``t(E)/t(J)`` | gap |")
    println(io, "| :--- | :--- | ---: | ---: | ---: | ---: |")
    for r in bench_data("benchmark_schemes_speed.json")["rows"]
        startswith(r["case"], "three-scale") == strength || continue
        @printf(
            io, "| %s | %s | %s | %s | %s | %.1e |\n", r["case"], replace(r["scheme"], "_" => " "),
            _fmt_time(r["t_mfh"]), _fmt_time(r["t_echoes"]), _fmt_ratio(r["ratio"]), r["gap"]
        )
    end
    return Markdown.parse(String(take!(io)))
end
