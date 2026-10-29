# =============================================================================
#  ttfx.jl — the cost of starting: loading the package and the first call of
#  each common path, in fresh Julia processes.
#
#  A first call compiles what the session has not compiled yet; its time is
#  what a user waits for in a new script, and what a precompile workload or a
#  lighter dependency removes. Each measure is the median of `--runs` fresh
#  processes; `--out` writes them as JSON.
#
#    julia --project=scripts/bench scripts/bench/ttfx.jl --runs=3 --out=ttfx.json
# =============================================================================

using JSON

# What one fresh process measures. Each path builds its inputs inside the
# timed call, as a user does in a new script.
const CHILD = raw"""
t₀ = time()
using MeanFieldHomogenization, TensND
const LOAD = time() - t₀
first_call(f) = (t = time(); f(); time() - t)
C₀ = iso_stiffness(30.0, 10.0)
two_phase(shape, scheme) = homogenize(
    (r = RVE(); add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => C₀); fraction = :rest);
     add_phase!(r, :I, shape, Dict(:C => iso_stiffness(150.0, 80.0)); fraction = 0.3); r),
    scheme, :C)
function ortho()
    KM = zeros(6, 6); KM[1:3, 1:3] .= [200 30 20; 30 50 25; 20 25 80]; KM[4, 4] = KM[5, 5] = KM[6, 6] = 80.0
    return TensND.from_mandel(KM, CanonicalBasis{3, Float64}())
end
cracked(C) = (r = RVE(); add_phase!(r, :M, Ellipsoid(1.0), Dict(:C => C); fraction = :rest);
              add_phase!(r, :Cr, PennyCrack(1.0), Dict(:C => C); density = 0.1); r)
conduction() = homogenize(
    (r = RVE(); add_phase!(r, :M, Ellipsoid(1.0), Dict(:K => TensISO{3}(1.0)); fraction = :rest);
     add_phase!(r, :I, Spheroid(5.0), Dict(:K => TensISO{3}(10.0)); fraction = 0.2); r),
    MoriTanaka(), :K)
paths = [
    "hill.sphere.iso" => () -> hill_tensor(Ellipsoid(1.0), C₀),
    "hill.spheroid.iso" => () -> hill_tensor(Spheroid(5.0), C₀),
    "mt.spheres" => () -> two_phase(Ellipsoid(1.0), MoriTanaka()),
    "asc.spheres" => () -> two_phase(Ellipsoid(1.0), AsymmetricSelfConsistent()),
    "sc.spheres" => () -> two_phase(Ellipsoid(1.0), SelfConsistent()),
    "differential.spheres" => () -> two_phase(Ellipsoid(1.0), DifferentialScheme()),
    "conduction.mt.spheroids" => conduction,
    "hill.ellipsoid.ortho" => () -> hill_tensor(Ellipsoid(1.0, 0.5, 0.2), ortho()),
    "mt.crack.iso" => () -> homogenize(cracked(C₀), MoriTanaka(), :C),
    "mt.crack.ortho" => () -> homogenize(cracked(ortho()), MoriTanaka(), :C),
]
println("load\t", LOAD)
for (name, f) in paths
    println(name, "\t", first_call(f))
end
"""

function parse_args(argv)
    o = Dict("runs" => 3, "out" => "")
    for a in argv
        k, v = split(a[3:end], '='; limit = 2)
        o[k] = k == "runs" ? parse(Int, v) : v
    end
    return o
end

# One fresh process: its measures, in the order printed.
function run_child(project)
    out = read(`$(Base.julia_cmd()) --startup-file=no --project=$project -e $CHILD`, String)
    pairs = [(String(first(p)), parse(Float64, last(p))) for p in split.(filter(!isempty, split(out, '\n')), '\t') if length(p) == 2]
    return pairs
end

median(v) = (s = sort(v); n = length(s); isodd(n) ? s[(n + 1) ÷ 2] : (s[n ÷ 2] + s[n ÷ 2 + 1]) / 2)

function main(argv)
    o = parse_args(argv)
    project = dirname(Base.active_project())
    runs = [run_child(project) for _ in 1:o["runs"]]
    names = first.(runs[1])
    result = Dict(name => median([last(r[i]) for r in runs]) for (i, name) in enumerate(names))
    for name in names
        println(rpad(name, 26), round(result[name]; digits = 3), " s")
    end
    isempty(o["out"]) || write(o["out"], JSON.json(Dict("runs" => o["runs"], "median_s" => result), 2))
    return nothing
end

main(ARGS)
