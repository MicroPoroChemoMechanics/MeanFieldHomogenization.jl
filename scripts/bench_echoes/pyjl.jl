# Conversion of Python data (dicts, lists, tuples, numbers, strings) to plain Julia
# values, for the benchmark scripts that read the cases defined in the Python
# modules next to them.  PythonCall converts nothing implicitly: every Python
# result is a `Py` until it is asked to become a Julia value.
using PythonCall

"""
    pyjl(x)

Recursive conversion of a Python object to Julia: `None` → `nothing`, `bool`,
`int`, `float` and `str` → `Bool`, `Int`, `Float64` and `String`, `dict` →
`Dict{String,Any}`, `list` and `tuple` → `Vector{Any}`. Any other object (a
numpy array, an Echoes tensor) is returned unchanged, as a `Py`.
"""
function pyjl(x::Py)
    pyisnone(x) && return nothing
    pyisinstance(x, pybuiltins.bool) && return pyconvert(Bool, x)
    pyisinstance(x, pybuiltins.int) && return pyconvert(Int, x)
    pyisinstance(x, pybuiltins.float) && return pyconvert(Float64, x)
    pyisinstance(x, pybuiltins.str) && return pyconvert(String, x)
    pyisinstance(x, pybuiltins.dict) &&
        return Dict{String, Any}(pyconvert(String, k) => pyjl(x[k]) for k in x)
    (pyisinstance(x, pybuiltins.list) || pyisinstance(x, pybuiltins.tuple)) &&
        return Any[pyjl(v) for v in x]
    return x
end

"""
    pymat(x)

A Python 2-D array (or `None`, as `nothing`) as a Julia `Matrix{Float64}`.
"""
pymat(x::Py) = pyisnone(x) ? nothing : pyconvert(Matrix{Float64}, x)
