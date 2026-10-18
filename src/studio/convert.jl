# Opening an Echoes script converts it. The translator, tools/echoes2mfh, is a
# Python program: the studio runs it in a Python of its own when one is there
# (3.10 or later), and says what to do when none is. Python serves this
# conversion only; the studio itself needs none.

const ECHOES2MFH = normpath(STUDIO_ROOT, "..", "..", "tools", "echoes2mfh")

# Run by the Python found: translate the file named, and answer in JSON.
const CONVERT_SCRIPT = """
import ast, json, os, sys
sys.path.insert(0, sys.argv[1])
from echoes2mfh.emit import emit
from echoes2mfh.extract import extract
path = sys.argv[2]
with open(path, encoding="utf-8", errors="replace") as fh:
    source = fh.read()
try:
    ast.parse(source)
except SyntaxError as exc:
    print(json.dumps({"error": f"{os.path.basename(path)} does not parse as Python 3: "
                      f"{exc.msg} (line {exc.lineno}). If it is Python 2, run `2to3 -w` on it first."}))
    sys.exit(0)
script = extract(source, path)
print(json.dumps({
    "julia": emit(script, os.path.splitext(os.path.basename(path))[0]),
    "findings": [{"line": f.lineno, "severity": f.severity, "reason": f.reason,
                  "suggestion": f.suggestion} for f in script.findings],
    "blocking": script.finding_count("blocking"),
}))
"""

"The Pythons tried, in order: the one `MFHSTUDIO_PYTHON` names, then those on the path."
function python_candidates()
    named = get(ENV, "MFHSTUDIO_PYTHON", "")
    isempty(named) || return [split(named)]
    return Sys.iswindows() ? [["python"], ["py", "-3"], ["python3"]] : [["python3"], ["python"]]
end

"Can `argv` run the translator: Python 3.10 or later, the translator importable?"
function translator_runs(argv)
    probe = "import sys; sys.path.insert(0, sys.argv[1]); import echoes2mfh.emit, echoes2mfh.extract; " *
        "sys.exit(0 if sys.version_info >= (3, 10) else 3)"
    Sys.which(first(argv)) === nothing && return false
    return success(pipeline(ignorestatus(`$argv -I -c $probe $ECHOES2MFH`); stdout = devnull, stderr = devnull))
end

"The command of the first Python that runs the translator, or `nothing`."
find_translator() = (i = findfirst(translator_runs, python_candidates()); i === nothing ? nothing : python_candidates()[i])

"Where the translation of `path` goes unless the user says otherwise: beside it, as `.jl`."
suggest_output(path::AbstractString) = splitext(abspath(path))[1] * ".jl"

is_echoes_script(path::AbstractString) = lowercase(splitext(path)[2]) == ".py"

"""
    convert_echoes(path) -> Dict{String, Any}

The Julia translation of the Echoes script at `path`, with the findings of the
translator (`line`, `severity`, `reason`, `suggestion`) and the number of
blocking ones. An `ArgumentError` says what to do when no Python can run the
translator, or when the script is not Python 3.
"""
function convert_echoes(path::AbstractString)
    isfile(path) || throw(ArgumentError("no such file: $path"))
    argv = find_translator()
    argv === nothing && throw(
        ArgumentError(
            "opening an Echoes script needs Python 3.10 or later, to run the translator " *
                "tools/echoes2mfh, and none was found (set MFHSTUDIO_PYTHON to name one). " *
                "Translate it in a terminal instead, from $ECHOES2MFH: " *
                "python3 -m echoes2mfh translate $path -o $(suggest_output(path))"
        )
    )
    out = IOBuffer()
    err = IOBuffer()
    p = run(pipeline(ignorestatus(`$argv -I -c $CONVERT_SCRIPT $ECHOES2MFH $path`); stdout = out, stderr = err))
    success(p) || throw(ArgumentError("the translator failed: " * String(take!(err))))
    answer = JSON.parse(String(take!(out)))
    haskey(answer, "error") && throw(ArgumentError(answer["error"]))
    answer["summary"] = conversion_summary(length(answer["findings"]), answer["blocking"])
    return answer
end

"What the translation did, in a sentence."
conversion_summary(n::Integer, blocking::Integer) = n == 0 ? "Translated with no findings." :
    "Translated with $n finding$(n == 1 ? "" : "s") ($blocking blocking). Each is marked " *
    "UNTRANSLATED in the script and raises at run time, so a half-translated model cannot " *
    "quietly produce numbers."
