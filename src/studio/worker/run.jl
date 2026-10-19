# Running a script: in a fresh module, so that one run leaves nothing behind
# for the next, with what it prints and what it warns kept for the page.

"""
    check_report() -> Vector{String}

The lines `mfhstudio(check = true)` prints: the versions the scripts run with.
"""
check_report() = [
    "Julia $VERSION, MeanFieldHomogenization $(pkgversion(MeanFieldHomogenization)), " *
        "TensND $(pkgversion(TensND)).",
]

"""
    run_in_module(source, name) -> NamedTuple

Run `source` in a fresh module, its output and its log messages written to one
text: `(; ok, error, output, mod)`. The file the script is read as lies in an
empty folder: the preamble of a studio script activates the first project it
finds above its own folder, which must not be one where the compute process
happens to have been started. A failure of the code is reported with its line;
an interrupt (the page's Stop) propagates, so that the caller knows the run was
stopped rather than failed.
"""
function run_in_module(source::AbstractString, name::Symbol)
    mod = Module(name)
    failure = Ref("")
    log = tempname()
    mktempdir() do dir
        open(log, "w") do io
            redirect_stdio(; stdout = io, stderr = io) do
                with_logger(ConsoleLogger(io)) do
                    try
                        Base.include_string(mod, source, joinpath(dir, "mfhstudio_script.jl"))
                    catch err
                        # `include_string` wraps the interrupt of a Stop in a LoadError
                        (err isa LoadError ? err.error : err) isa InterruptException && rethrow()
                        # the script's failure is the answer the page shows, with its line
                        failure[] = sprint(showerror, err)
                    end
                end
            end
        end
    end
    output = read(log, String)
    rm(log; force = true)
    return (; ok = isempty(failure[]), error = failure[], output, mod)
end

"The results a script publishes, as JSON can carry them: a non-finite number is `null`."
jsonable(x::AbstractDict) = Dict{String, Any}(string(k) => jsonable(v) for (k, v) in x)
jsonable(x::Union{AbstractVector, Tuple}) = Any[jsonable(v) for v in x]
jsonable(x::Bool) = x
jsonable(x::Real) = isfinite(x) ? Float64(x) : nothing
jsonable(x::Union{AbstractString, Symbol}) = String(x)
jsonable(::Nothing) = nothing
jsonable(x) = string(x)

"""
    run_script(source) -> Dict{String, Any}

Run a studio script and return whether it succeeded, what it printed, its
error, and the curves it published in `MFHSTUDIO_RESULTS`.
"""
function run_script(source::AbstractString)
    # no display in the compute process: a figure is drawn to nothing
    get!(ENV, "GKSwstype", "100")
    r = run_in_module(source, :MFHStudioRun)
    found = Base.invokelatest(isdefined, r.mod, :MFHSTUDIO_RESULTS)
    results = found ? jsonable(Base.invokelatest(getfield, r.mod, :MFHSTUDIO_RESULTS)) : nothing
    return Dict{String, Any}("ok" => r.ok, "stdout" => r.output, "error" => r.ok ? nothing : r.error, "results" => results)
end
