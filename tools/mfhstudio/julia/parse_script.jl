# =============================================================================
#  parse_script.jl — read an existing Julia script back into the studio's IR.
#
#  Two levels, and the second one is the important one:
#
#  1. Scripts written by the studio carry an embedded `#= mfhstudio-model =#`
#     block holding the serialized model, so they round-trip exactly.
#
#  2. Everything else is parsed with `Meta.parseall` -- Julia's own parser, not
#     an approximation of it -- and matched against the vocabulary the emitter
#     produces. Whatever is not recognized becomes an **opaque block carrying
#     its source verbatim**, and is re-emitted byte for byte.
#
#  The second rule is what makes the interface safe to point at a script
#  somebody hand-tuned: the parts the studio does not understand are the parts
#  it must not touch. Guessing would be worse than refusing.
# =============================================================================

const MODEL_OPEN = "#= mfhstudio-model"
const MODEL_CLOSE = "=#"

"""
    embedded_model(source) -> Union{Nothing, String}

The serialized model a studio-written script carries, if any.
"""
function embedded_model(source::AbstractString)
    i = findfirst(MODEL_OPEN, source)
    i === nothing && return nothing
    rest = SubString(source, last(i) + 1)
    j = findfirst(MODEL_CLOSE, rest)
    j === nothing && return nothing
    body = strip(SubString(rest, 1, first(j) - 1))
    # the version tag sits on the first line
    nl = findfirst('\n', body)
    nl === nothing && return nothing
    return String(strip(SubString(body, nl + 1)))
end

# ── Source slicing ──────────────────────────────────────────────────────────
#
# Verbatim preservation needs exact line spans, so statements are located by
# the `LineNumberNode`s the parser interleaves and sliced out of the original
# text rather than re-printed from the AST (which would lose comments,
# alignment and any formatting the author chose).

struct Chunk
    first_line::Int
    last_line::Int
    ex::Any
end

function toplevel_chunks(source::AbstractString, filename::AbstractString)
    lines = split(source, '\n'; keepempty = true)
    parsed = Meta.parseall(source; filename = filename)
    chunks = Chunk[]
    pending_line = 1
    exprs = parsed isa Expr && parsed.head === :toplevel ? parsed.args : Any[parsed]
    for a in exprs
        if a isa LineNumberNode
            pending_line = a.line
            continue
        end
        push!(chunks, Chunk(pending_line, pending_line, a))
    end
    # A statement runs until the line before the next one starts. Trailing
    # blank lines and comments are *not* part of it though: a comment sitting
    # just above the next statement introduces that one, and swallowing it
    # here would make the statement's recorded text differ from the statement
    # itself. They are handed back as trivia instead, so nothing is lost.
    out = Chunk[]
    for (i, c) in enumerate(chunks)
        stop = i < length(chunks) ? chunks[i + 1].first_line - 1 : length(lines)
        stop = max(stop, c.first_line)
        while stop > c.first_line
            s = strip(lines[stop])
            (isempty(s) || startswith(s, "#")) || break
            stop -= 1
        end
        push!(out, Chunk(c.first_line, stop, c.ex))
    end
    return out, lines
end

slice(lines, a, b) = join(lines[max(a, 1):min(b, length(lines))], "\n")

# ── Recognition ─────────────────────────────────────────────────────────────

_iscall(ex, name) = ex isa Expr && ex.head === :call && ex.args[1] === name

function _kwargs(ex::Expr)
    out = Dict{String, String}()
    for a in ex.args
        if a isa Expr && a.head === :kw
            out[String(a.args[1])] = string(a.args[2])
        elseif a isa Expr && a.head === :parameters
            for p in a.args
                p isa Expr && p.head === :kw && (out[String(p.args[1])] = string(p.args[2]))
            end
        end
    end
    return out
end

_positional(ex::Expr) = [a for a in ex.args[2:end] if !(a isa Expr && a.head in (:kw, :parameters))]

"""
Recognize an RVE builder: a function whose body creates an `RVE` and fills it
with `add_phase!` calls (and, for scripts written before v0.8, `add_matrix!`).
This is the shape the emitter writes
and the shape the hand-written demos use.
"""
function recognize_builder(ex)
    (ex isa Expr && ex.head === :function) || return nothing
    sig = ex.args[1]
    (sig isa Expr && sig.head === :call) || return nothing
    fname = String(sig.args[1])
    args = String[string(a) for a in sig.args[2:end] if !(a isa Expr && a.head === :parameters)]

    matrix = nothing
    phases = Dict{String, Any}[]
    var = nothing
    found = false
    rve_opts = Dict{String, String}()

    for st in ex.args[2].args
        st isa LineNumberNode && continue
        # rve = RVE()
        if st isa Expr && st.head === :(=) && _iscall(st.args[2], :RVE)
            var = String(st.args[1])
            a = _positional(st.args[2])
            # Pre-0.8 scripts named the matrix here. Kept so the studio can
            # still open them; scripts written since carry no such name.
            isempty(a) || (matrix = replace(string(a[1]), ":" => ""))
            # `RVE(; T = ComplexF64)` — the element type is part of the model
            rve_opts = _kwargs(st.args[2])
            found = true
        elseif _iscall(st, :add_matrix!)
            # Legacy, pre-0.8: `add_matrix!(rve, geom, props)`. Read only —
            # the emitter never writes it any more.
            a = _positional(st)
            length(a) >= 3 || continue
            push!(
                phases, Dict{String, Any}(
                    "name" => matrix === nothing ? "MATRIX" : matrix,
                    "is_matrix" => true,
                    "geometry" => string(a[2]),
                    "properties" => string(a[3]),
                    "options" => _kwargs(st),
                )
            )
        elseif _iscall(st, :add_phase!)
            a = _positional(st)
            length(a) >= 4 || continue
            opts = _kwargs(st)
            push!(
                phases, Dict{String, Any}(
                    "name" => replace(string(a[2]), ":" => ""),
                    # `fraction = :rest` is what marks the phase taking up the
                    # volume complement — the successor of the matrix flag.
                    "is_matrix" => get(opts, "fraction", nothing) == ":rest",
                    "geometry" => string(a[3]),
                    "properties" => string(a[4]),
                    "options" => opts,
                )
            )
        end
    end

    found || return nothing
    return Dict{String, Any}(
        "kind" => "rve_builder",
        "function" => fname,
        "params" => args,
        "var" => var,
        "matrix" => matrix,
        "rve_options" => rve_opts,
        "phases" => phases,
    )
end

"""
Recognize a laminate builder: a function whose body creates a `Laminate` and
fills it with `add_layer!` calls.

Deliberately a separate recognizer rather than a branch of `recognize_builder`:
the two cells share nothing but the `function … end` wrapper, and a laminate has
no matrix, no per-member geometry and no orientation average to look for.

The frame keyword is kept as written — `normal = (0, 0, 1)` and
`euler_angles = (θ, ϕ, ψ)` are the two forms the emitter produces, and
MeanFieldHomogenization refuses both at once, so at most one is ever there.
"""
function recognize_laminate(ex)
    (ex isa Expr && ex.head === :function) || return nothing
    sig = ex.args[1]
    (sig isa Expr && sig.head === :call) || return nothing
    fname = String(sig.args[1])
    args = String[string(a) for a in sig.args[2:end] if !(a isa Expr && a.head === :parameters)]

    layers = Dict{String, Any}[]
    var = nothing
    found = false
    lam_opts = Dict{String, String}()

    for st in ex.args[2].args
        st isa LineNumberNode && continue
        if st isa Expr && st.head === :(=) && _iscall(st.args[2], :Laminate)
            var = String(st.args[1])
            lam_opts = _kwargs(st.args[2])
            found = true
        elseif _iscall(st, :add_layer!)
            a = _positional(st)
            length(a) >= 3 || continue
            push!(
                layers, Dict{String, Any}(
                    "name" => replace(string(a[2]), ":" => ""),
                    "properties" => string(a[3]),
                    "options" => _kwargs(st),
                )
            )
        end
    end

    found || return nothing
    return Dict{String, Any}(
        "kind" => "laminate_builder",
        "function" => fname,
        "params" => args,
        "var" => var,
        "laminate_options" => lam_opts,
        "layers" => layers,
    )
end

"""
Recognize a top-level `const NAME = expr` binding, which the emitter uses for
every scalar and tensor parameter.
"""
function recognize_const(ex)
    (ex isa Expr && ex.head === :const) || return nothing
    a = ex.args[1]
    (a isa Expr && a.head === :(=)) || return nothing
    lhs = a.args[1]
    lhs isa Symbol || return nothing
    return Dict{String, Any}(
        "kind" => "param",
        "name" => String(lhs),
        "value" => string(a.args[2]),
    )
end

"""
Recognize `X = homogenize(cell, scheme, :prop)` and the ALV variant.
"""
function recognize_homogenize(ex)
    (ex isa Expr && ex.head === :(=)) || return nothing
    rhs = ex.args[2]
    (rhs isa Expr && rhs.head === :call) || return nothing
    fn = rhs.args[1]
    fn in (:homogenize, :homogenize_alv) || return nothing
    a = _positional(rhs)
    return Dict{String, Any}(
        "kind" => "homogenize",
        "alv" => fn === :homogenize_alv,
        "target" => string(ex.args[1]),
        "cell" => length(a) >= 1 ? string(a[1]) : "",
        "scheme" => length(a) >= 2 ? string(a[2]) : "",
        "property" => length(a) >= 3 ? replace(string(a[3]), ":" => "") : "C",
        "options" => _kwargs(rhs),
    )
end

const RECOGNIZERS = [
    recognize_const, recognize_builder, recognize_laminate, recognize_homogenize,
]

"""
    parse_script(source; filename = "script.jl") -> Dict

The whole file as a list of nodes. Recognized constructs carry structured
fields; everything else is `{"kind": "opaque", "source": <verbatim>}`.

Concatenating every node's source in order reproduces the input exactly --
that property is what the round-trip test checks.
"""
function parse_script(source::AbstractString; filename::AbstractString = "script.jl")
    embedded = embedded_model(source)
    if embedded !== nothing
        return Dict{String, Any}("exact" => true, "model" => embedded, "nodes" => Any[])
    end

    local chunks, lines
    try
        chunks, lines = toplevel_chunks(source, filename)
    catch e
        return Dict{String, Any}(
            "exact" => false, "nodes" => Any[],
            "error" => "cannot parse: " * sprint(showerror, e),
        )
    end

    nodes = Any[]

    # The nodes must *tile* the file: every line belongs to exactly one node,
    # in order, with no gap and no overlap. Reconstruction is then a plain
    # join, and byte-exactness is a property of the tiling rather than
    # something the emitter has to be careful about.
    cursor = 1
    for c in chunks
        if c.first_line > cursor
            # Text between statements — shebang, header comment, blank lines.
            push!(
                nodes, Dict{String, Any}(
                    "kind" => "trivia",
                    "source" => slice(lines, cursor, c.first_line - 1),
                    "first_line" => cursor, "last_line" => c.first_line - 1,
                )
            )
        end
        node = nothing
        for r in RECOGNIZERS
            node = r(c.ex)
            node === nothing || break
        end
        node === nothing && (node = Dict{String, Any}("kind" => "opaque"))
        node["source"] = slice(lines, c.first_line, c.last_line)
        node["first_line"] = c.first_line
        node["last_line"] = c.last_line
        push!(nodes, node)
        cursor = c.last_line + 1
    end
    if cursor <= length(lines)
        push!(
            nodes, Dict{String, Any}(
                "kind" => "trivia",
                "source" => slice(lines, cursor, length(lines)),
                "first_line" => cursor, "last_line" => length(lines),
            )
        )
    end

    return Dict{String, Any}(
        "exact" => false,
        "nodes" => nodes,
        "recognized" => count(n -> !(n["kind"] in ("opaque", "trivia")), nodes),
        "opaque" => count(n -> n["kind"] == "opaque", nodes),
    )
end

"""
    reconstruct(parsed) -> String

The inverse of `parse_script` for the non-exact path: gluing every node's
source back together must reproduce the input byte for byte.
"""
reconstruct(parsed) = join([n["source"] for n in parsed["nodes"]], "\n")
