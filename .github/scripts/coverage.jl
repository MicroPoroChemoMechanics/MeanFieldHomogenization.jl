# Build `lcov.info` for Codecov.
#
# Why not `julia-actions/julia-processcoverage`?  That action calls
# CoverageTools' `amend_coverage_from_src!`, which marks as "executable" every
# line the *source parser* believes to be code — including lines Julia's
# instrumentation never emits a count for.  Methods generated inside `@eval`
# loops are the main offender, and both TensND and MeanFieldHomogenization use that style
# heavily.  Such lines can never be covered by any test, yet they sit in the
# denominator forever: TensND measures 1352/1418 = 95 % of the lines Julia
# actually instruments, but the amended ratio is 1352/2278 = 59 %.
#
# So we mix the two:
#   • a file that was loaded      → raw `.cov` counts, no amendment (faithful);
#   • a file never loaded at all  → amended, i.e. every executable line counted
#     as missed, so genuine gaps (`src/submanifold.jl`, `ext/…Ext.jl`) stay
#     visible at 0 % instead of silently vanishing from the report.
#
# Usage: `julia coverage.jl [output [folder...] [--exclude folder...]]`, by
# default `lcov.info` from `src` and `ext`. MFH Studio (`src/studio`) is
# reported by its own workflow, which runs its compute process: CI.yml excludes
# it, since its worker files, never loaded in-process, would land here amended
# and Codecov's union would then count the lines Julia never instruments as
# missed. Studio.yml excludes `src/studio/library`, the worked examples, which
# are scripts the studio opens, not code the package loads.

# CoverageTools is a dependency of Coverage
using Coverage
using Coverage.CoverageTools

const SPLIT = something(findfirst(==("--exclude"), ARGS), length(ARGS) + 1)
const OUTPUT = SPLIT > 1 ? ARGS[1] : "lcov.info"
const DIRS = SPLIT > 2 ? ARGS[2:(SPLIT - 1)] : ["src", "ext"]
const EXCLUDED = [normpath(d) * Base.Filesystem.path_separator for d in ARGS[(SPLIT + 1):end]]
# `.cov` files are also dropped next to the test sources; wipe them too so a
# local run leaves the working tree clean.
const CLEAN_DIRS = ["src", "ext", "test", "docs"]

# The Gridap backend is left out of the report. CI does not install Gridap: its
# Gmsh reader pulls GridapDistributed, PartitionedArrays (hence MPI) and Metis,
# which is a lot to install on every run for a second backend of the
# axisymmetric Fourier solve that Ferrite already provides. Its tests run where
# the stack is installed (`test/FiniteElements/test_gridap_backend.jl`), and
# counted as never run it would only hide the coverage of everything else.
const NOT_MEASURED = [normpath("ext/MeanFieldHomogenizationGridapExt")]

kept(fc) = !any(d -> startswith(normpath(fc.filename), d), vcat(EXCLUDED, NOT_MEASURED))
collect_dirs() = filter(kept, reduce(vcat, (process_folder(d) for d in DIRS if isdir(d)); init = FileCoverage[]))

"""
Raw counts for loaded files, amended (all-missed) counts for never-loaded ones.
"""
function merged_coverage()
    raw = withenv("DISABLE_AMEND_COVERAGE_FROM_SRC" => "yes") do
        collect_dirs()
    end
    amended = Dict(fc.filename => fc for fc in collect_dirs())
    out = FileCoverage[]
    for fc in raw
        # `process_cov` returns an all-`nothing` vector when no `.cov` file
        # exists, i.e. when nothing in the file was ever executed.
        push!(out, any(!isnothing, fc.coverage) ? fc : amended[fc.filename])
    end
    return out
end

function report(fcs)
    tot = 0
    cov = 0
    for fc in sort(fcs, by = x -> x.filename)
        lines = count(!isnothing, fc.coverage)
        lines == 0 && continue
        hit = count(x -> !isnothing(x) && x > 0, fc.coverage)
        tot += lines
        cov += hit
        println(rpad(fc.filename, 46), lpad(hit, 5), "/", lpad(lines, 5), "  ", round(100 * hit / lines, digits = 1), "%")
    end
    println("TOTAL ", cov, "/", tot, " = ", round(100 * cov / tot, digits = 2), "%")
    return (cov, tot)
end

fcs = merged_coverage()
covered, total = report(fcs)
# Without `.cov` files every source lands in the "never loaded" branch and the
# report is a silent 0 % — which would be uploaded and shown on the badge as if
# it were real.  Fail loudly instead.
covered == 0 && error(
    "No coverage data found: every file reports 0 %.  Did `julia-runtest` run " *
        "with `coverage: true`, and is this script running from the repo root?"
)
# Paths with forward slashes, so that a report written on Windows names the
# same files as the others when Codecov merges them.
LCOV.writefile(OUTPUT, [FileCoverage(replace(fc.filename, '\\' => '/'), fc.source, fc.coverage) for fc in fcs])
clean_folder.(filter(isdir, CLEAN_DIRS))
