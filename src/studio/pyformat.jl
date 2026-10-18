# The text conventions of the scripts MFH Studio writes. The studio was first
# written in Python, and every script it saved carries numbers and an embedded
# model written Python's way: a float as Python's `repr` prints it (`1e-06`,
# `1000000.0`), the model as `json.dumps(…, indent = 1, sort_keys = True)`
# with every character beyond ASCII escaped. Writing them the same way keeps
# those scripts reading back unchanged, byte for byte.

"A JSON object as the page sends it and the model keeps it: keys in the order given."
const JDict = JSON.Object{String, Any}

jdict(pairs::Pair...) = (d = JDict(); foreach(p -> d[p.first] = p.second, pairs); d)

"""
    truthy(x) -> Bool

Python's truth value: `nothing`, `false`, a zero, an empty string or an empty
collection are false. The model reads optional fields this way (`x or default`).
"""
truthy(::Nothing) = false
truthy(x::Bool) = x
truthy(x::Number) = !iszero(x)
truthy(x::AbstractString) = !isempty(x)
truthy(x::Union{AbstractVector, AbstractDict, Tuple}) = !isempty(x)
truthy(x) = true

"`x` when it is truthy, else `default`: Python's `x or default`."
either(x, default) = truthy(x) ? x : default

"""
    shortest_digits(x) -> (digits, decpt)

The shortest decimal digits that read back as `x > 0`, and the position of the
decimal point: `x = 0.d₁d₂… × 10^decpt`. Julia prints the shortest
round-tripping digits, as Python does; only the layout differs.
"""
function shortest_digits(x::Float64)
    s = string(x)
    m, e = occursin('e', s) ? split(s, 'e') : (s, "0")
    whole, frac = split(m, '.')
    digits = whole * frac
    decpt = length(whole) + parse(Int, e)
    k = findfirst(!=('0'), digits)
    decpt -= k - 1
    return rstrip(digits[k:end], '0'), decpt
end

"""
    py_repr(x::Float64) -> String

Python's `repr` of a float: fixed notation for `1e-4 ≤ |x| < 1e16`, otherwise
`d.ddde±XX`; `inf`, `-inf`, `nan`.
"""
function py_repr(x::Float64)
    isnan(x) && return "nan"
    isinf(x) && return x > 0 ? "inf" : "-inf"
    iszero(x) && return signbit(x) ? "-0.0" : "0.0"
    sign = x < 0 ? "-" : ""
    digits, decpt = shortest_digits(abs(x))
    n = length(digits)
    if -4 < decpt <= 16
        decpt <= 0 && return sign * "0." * "0"^(-decpt) * digits
        decpt >= n && return sign * digits * "0"^(decpt - n) * ".0"
        return sign * digits[1:decpt] * "." * digits[(decpt + 1):end]
    end
    mantissa = n == 1 ? digits : digits[1:1] * "." * digits[2:end]
    exponent = decpt - 1
    return sign * mantissa * "e" * (exponent < 0 ? "-" : "+") * lpad(abs(exponent), 2, '0')
end

"""
    julia_number(x) -> String

A value as Julia code, keeping float-ness explicit: `true`, `3`, `1.0`,
`1.0e-6`; a string is code already and is written as it is.
"""
julia_number(x::Bool) = x ? "true" : "false"
julia_number(x::Integer) = string(x)
function julia_number(x::AbstractFloat)
    s = py_repr(Float64(x))
    if occursin('e', s)
        mantissa, exponent = split(s, 'e')
        occursin('.', mantissa) || (mantissa *= ".0")
        return mantissa * "e" * string(parse(Int, exponent))
    end
    return occursin('.', s) ? s : s * ".0"
end
julia_number(x::AbstractString) = String(x)
julia_number(::Nothing) = "None"
julia_number(x) = py_str(x)

"""
    julia_float(x) -> String

[`julia_number`](@ref), but never a bare integer: the sizes of a geometry form an
`NTuple{N, T}`, which `(1, 0.6)` is not.
"""
julia_float(x::Bool) = julia_number(x)
julia_float(x::Integer) = julia_number(Float64(x))
julia_float(x) = julia_number(x)

"The characters Python's `str.splitlines` breaks a line on."
const LINE_BREAKS = ('\n', '\r', '\v', '\f', '\x1c', '\x1d', '\x1e', '\u85', '\u2028', '\u2029')

"""
    py_splitlines(s) -> Vector{String}

The lines of `s` as Python's `str.splitlines` gives them: broken on every line
break (`\r\n` counting as one), with no empty last line for a trailing break.
"""
function py_splitlines(s::AbstractString)
    lines = String[]
    start = firstindex(s)
    i = start
    while i <= ncodeunits(s)
        c = s[i]
        if c in LINE_BREAKS
            push!(lines, s[start:prevind(s, i)])
            i = nextind(s, i)
            c == '\r' && i <= ncodeunits(s) && s[i] == '\n' && (i = nextind(s, i))
            start = i
        else
            i = nextind(s, i)
        end
    end
    start <= ncodeunits(s) && push!(lines, s[start:end])
    return lines
end

# ── Python's json.dumps(x, indent = 1, sort_keys = True) ─────────────────────

function py_json_string(io::IO, s::AbstractString)
    print(io, '"')
    for c in s
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\r'
            print(io, "\\r")
        elseif c == '\t'
            print(io, "\\t")
        elseif c == '\b'
            print(io, "\\b")
        elseif c == '\f'
            print(io, "\\f")
        elseif ' ' <= c <= '~'
            print(io, c)
        else
            u = UInt32(c)
            if u > 0xffff
                u -= 0x00010000
                print(io, "\\u", string(0xd800 + (u >> 10); base = 16, pad = 4), "\\u", string(0xdc00 + (u & 0x03ff); base = 16, pad = 4))
            else
                print(io, "\\u", string(u; base = 16, pad = 4))
            end
        end
    end
    print(io, '"')
    return nothing
end

py_json_value(io::IO, x::AbstractString, ::Int) = py_json_string(io, x)
py_json_value(io::IO, ::Nothing, ::Int) = print(io, "null")
py_json_value(io::IO, x::Bool, ::Int) = print(io, x ? "true" : "false")
py_json_value(io::IO, x::Integer, ::Int) = print(io, x)
function py_json_value(io::IO, x::AbstractFloat, ::Int)
    isnan(x) && return print(io, "NaN")
    isinf(x) && return print(io, x > 0 ? "Infinity" : "-Infinity")
    return print(io, py_repr(Float64(x)))
end
function py_json_value(io::IO, x::Union{AbstractVector, Tuple}, level::Int)
    isempty(x) && return print(io, "[]")
    print(io, "[")
    for (i, v) in enumerate(x)
        print(io, i == 1 ? "\n" : ",\n", " "^(level + 1))
        py_json_value(io, v, level + 1)
    end
    print(io, "\n", " "^level, "]")
    return nothing
end
function py_json_value(io::IO, x::AbstractDict, level::Int)
    isempty(x) && return print(io, "{}")
    print(io, "{")
    for (i, k) in enumerate(sort!(collect(String, keys(x))))
        print(io, i == 1 ? "\n" : ",\n", " "^(level + 1))
        py_json_string(io, k)
        print(io, ": ")
        py_json_value(io, x[k], level + 1)
    end
    print(io, "\n", " "^level, "}")
    return nothing
end

"""
    py_json(x) -> String

`x` as Python's `json.dumps(x, indent = 1, sort_keys = True)` writes it.
"""
py_json(x) = sprint(py_json_value, x, 0)

# ── Python's json.loads ─────────────────────────────────────────────────────

"The tokens beyond JSON that Python's `json` reads and writes, longest first."
const SPECIAL_FLOATS = ("-Infinity" => -Inf, "Infinity" => Inf, "NaN" => NaN)
const SPECIAL_MARK = "\0py:"

"""
    parse_python_json(text) -> Any

JSON as Python's `json.loads` reads it: `NaN`, `Infinity` and `-Infinity` are
floats, and every other number is read as JSON.jl reads it, an integer staying
an integer (JSON.jl's `allownan` would read every number as a float). The
special tokens outside the strings become marked strings for the strict
parser, then floats again.
"""
function parse_python_json(text::AbstractString)
    io = IOBuffer()
    i = firstindex(text)
    instring = false
    while i <= ncodeunits(text)
        c = text[i]
        if instring
            print(io, c)
            if c == '\\'
                i = nextind(text, i)
                i <= ncodeunits(text) && print(io, text[i])
            elseif c == '"'
                instring = false
            end
        elseif c == '"'
            instring = true
            print(io, c)
        else
            k = findfirst(p -> startswith(SubString(text, i), p.first), SPECIAL_FLOATS)
            if k === nothing
                print(io, c)
            else
                token = SPECIAL_FLOATS[k].first
                print(io, "\"\\u0000py:", token, '"')
                i += ncodeunits(token) - 1
            end
        end
        i = nextind(text, i)
    end
    return restore_special(JSON.parse(String(take!(io))))
end

function restore_special(x::AbstractString)
    startswith(x, SPECIAL_MARK) || return x
    token = x[(ncodeunits(SPECIAL_MARK) + 1):end]
    return last(SPECIAL_FLOATS[findfirst(p -> p.first == token, SPECIAL_FLOATS)])
end
restore_special(x::AbstractVector) = map!(restore_special, x, x)
function restore_special(x::AbstractDict)
    for (k, v) in x
        x[k] = restore_special(v)
    end
    return x
end
restore_special(x) = x
