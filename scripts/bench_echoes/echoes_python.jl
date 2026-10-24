# Chooses the Python interpreter in which Echoes is installed, for the scripts of
# this directory and for the Echoes cross-checks of `scripts/` (15, 60).
#
# MUST be included BEFORE `using PythonCall` (and before anything that loads it,
# `MeanFieldHomogenization` and `TensND` included): PythonCall reads its settings
# once, when it is loaded.
#
# The interpreter is, in this order:
#   1. `ENV["ECHOES_PYTHON"]`, when set;
#   2. the `python3`, then the `python`, found on the PATH, which is the one of the
#      Python environment activated before starting Julia.
# It must import both `echoes` and `sympy` (`pip install sympy`: the symbolic layer
# of the stack needs it). CondaPkg is then told to manage nothing, so that the
# packages of that environment (the numpy version Echoes was built against, often
# numpy 1.x) are kept as they are. Settings already present in the environment
# (`JULIA_CONDAPKG_BACKEND`, `JULIA_PYTHONCALL_EXE`) are respected, and nothing is
# written outside this process.
let
    if haskey(ENV, "JULIA_PYTHONCALL_EXE") && haskey(ENV, "JULIA_CONDAPKG_BACKEND")
        # explicit settings: nothing to choose
    else
        if any(m -> m.name == "PythonCall", keys(Base.loaded_modules))
            @warn "PythonCall is already loaded: the Python chosen by echoes_python.jl comes too late. Include this file first, in a fresh session."
        end
        py = get(ENV, "ECHOES_PYTHON", "")
        if isempty(py)
            found = Sys.which("python3")
            found === nothing && (found = Sys.which("python"))
            py = found === nothing ? "" : found
        end
        ok = !isempty(py) && success(pipeline(`$py -c "import echoes, sympy"`; stdout = devnull, stderr = devnull))
        ok || error(
            "No Python interpreter that imports both `echoes` and `sympy` was found" *
                (isempty(py) ? "" : " (tried `$py`)") * ". Activate the Python environment " *
                "where Echoes is installed (it must also have sympy: `pip install sympy`) " *
                "before starting Julia, or set the environment variable ECHOES_PYTHON to that interpreter."
        )
        haskey(ENV, "JULIA_CONDAPKG_BACKEND") || (ENV["JULIA_CONDAPKG_BACKEND"] = "Null")
        haskey(ENV, "JULIA_PYTHONCALL_EXE") || (ENV["JULIA_PYTHONCALL_EXE"] = py)
    end
end
