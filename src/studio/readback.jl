# Reading a script back into the model, at two levels. A script the studio
# wrote embeds its model and reopens exactly. Any other is parsed (parse.jl)
# and matched against the vocabulary the code generation writes; everything
# else is kept as an opaque block, verbatim. What the studio does not
# understand, it does not touch: a script tuned by hand can be opened, edited in
# the parts the forms model, and written back with the rest byte for byte.

"""
    py_float(text) -> Union{Nothing, Float64}

The number Python's `float` reads in `text`, or `nothing`: Julia's parser
besides refuses an underscore between digits, which Python accepts, and reads
a hexadecimal integer, which Python refuses.
"""
function py_float(text::AbstractString)
    t = strip(text)
    occursin(r"^[+-]?0[xXoObB]", t) && return nothing
    if occursin('_', t)
        occursin(r"(?<![0-9])_|_(?![0-9])", t) && return nothing
        t = replace(t, "_" => "")
    end
    return tryparse(Float64, t)
end

"A literal becomes a float; anything else stays the Julia expression it is."
number_or_text(text::AbstractString) = (t = String(strip(text)); something(py_float(t), t))

"The captured groups of a match, as strings."
groups(m::RegexMatch) = [String(something(c, "")) for c in m.captures]

const GEOM_PATTERNS = [
    ("spheroid", r"(*UCP)^Spheroid\(\s*([^;,)]+)", ["omega"]),
    ("ellipsoid", r"(*UCP)^Ellipsoid\(\s*([^;,)]+),\s*([^;,)]+),\s*([^;,)]+)", ["a", "b", "c"]),
    ("cylinder", r"(*UCP)^Cylinder\(\s*([^;,)]+),\s*([^;,)]+)", ["b", "c"]),
    ("penny_crack", r"(*UCP)^PennyCrack\(\s*([^;,)]+)", ["a"]),
    ("elliptic_crack", r"(*UCP)^EllipticCrack\(\s*([^;,)]+),\s*([^;,)]+)", ["a", "b"]),
    ("ribbon_crack", r"(*UCP)^RibbonCrack\(\s*([^;,)]+)", ["b"]),
]

"""
    parse_geometry(expr) -> Union{Nothing, Geometry}

The shape an expression builds, when it is one the forms model.
"""
function parse_geometry(expr::AbstractString)
    e = strip(expr)
    for (kind, pattern, names) in GEOM_PATTERNS
        m = match(pattern, e)
        m === nothing && continue
        geo = Geometry(; kind, args = jdict((n => number_or_text(v) for (n, v) in zip(names, groups(m)))...))
        ang = match(r"(*UCP)euler_angles\s*=\s*\(([^)]*)\)", e)
        ang === nothing || (geo.euler_angles = Any[number_or_text(x) for x in split(ang[1], ",") if !isempty(strip(x))])
        return geo
    end
    return nothing
end

const PROP_PATTERNS = [
    ("iso_stiffness", ["k", "mu"], r"(*UCP)^iso_stiffness\(\s*([^,)]+),\s*([^,)]+)\)"),
    ("iso_stiffness_E_nu", ["E", "nu"], r"(*UCP)^iso_stiffness_E_nu\(\s*([^,)]+),\s*([^,)]+)\)"),
    (
        "hoenig_stiffness", ["E1", "h", "nu1", "nu2", "gamma"],
        r"(*UCP)^hoenig_stiffness\(\s*([^,)]+),\s*([^,)]+),\s*([^,)]+),\s*([^,)]+),\s*([^,)]+)\)",
    ),
]

"A literal of a keyword argument as the model keeps it: a number or a flag as it is, else its code."
literal_value(v::Union{Number, Bool, AbstractString}) = v
literal_value(v) = string(v)

"""
    parse_seam(key, expr) -> Union{Nothing, Property}

The multiscale seam `Homogenized(builder(args...), Scheme(; options...))` as a
property whose cell is named by its builder (`"@builder:<name>"`, resolved
once every builder is read), with the scheme and its options; `nothing` for any
other expression.
"""
function parse_seam(key::AbstractString, expr::AbstractString)
    ex = Meta.parse(expr; raise = false)
    (is_call(ex, :Homogenized) && length(ex.args) == 3) || return nothing
    inner, scheme = ex.args[2], ex.args[3]
    (inner isa Expr && inner.head === :call && inner.args[1] isa Symbol) || return nothing
    (scheme isa Expr && scheme.head === :call && scheme.args[1] isa Symbol) || return nothing
    isempty(positional_args(scheme)) || return nothing
    options = jdict((k => literal_value(Meta.parse(v)) for (k, v) in sort!(collect(keyword_args(scheme))))...)
    p = Property(; key, source = "cell", scheme = String(scheme.args[1]), scheme_options = options)
    p.cell = "@builder:" * String(inner.args[1])
    return p
end

"""
    parse_property(key, expr) -> Property

A property as the forms model it: the multiscale seam (its inner cell resolved
once every builder is read), a tensor from its moduli, or the expression as
written.
"""
function parse_property(key::AbstractString, expr::AbstractString)
    e = String(strip(expr))
    seam = parse_seam(key, e)
    seam === nothing || return seam
    for (b, names, pattern) in PROP_PATTERNS
        mm = match(pattern, e)
        mm === nothing && continue
        return Property(; key, source = "builder", builder = b, args = jdict((n => number_or_text(v) for (n, v) in zip(names, groups(mm)))...))
    end
    return Property(; key, source = "expr", expr = e)
end

"""
    split_toplevel(body) -> Vector{String}

The pieces of `body` between commas at bracket depth zero, stripped, the empty
ones left out: a matrix `[1.0 0.0; 0.0 2.0]` is one piece.
"""
function split_toplevel(body::AbstractString)
    out = String[]
    depth = 0
    start = firstindex(body)
    for i in eachindex(body)
        ch = body[i]
        if ch in ('(', '[', '{')
            depth += 1
        elseif ch in (')', ']', '}')
            depth -= 1
        elseif ch == ',' && depth == 0
            push!(out, body[start:prevind(body, i)])
            start = nextind(body, i)
        end
    end
    push!(out, body[start:end])
    return [String(strip(p)) for p in out if !isempty(strip(p))]
end

"""
    parse_properties(expr) -> Vector{Property}

`Dict(:C => iso_stiffness(72.0, 32.0), :K => …)` as properties.
"""
function parse_properties(expr::AbstractString)
    e = strip(expr)
    startswith(e, "Dict") || return Property[]
    open = findfirst('(', e)
    close = findlast(')', e)
    from = open === nothing ? firstindex(e) : nextind(e, open)
    to = close === nothing ? prevind(e, lastindex(e)) : prevind(e, close)
    inner = from <= to ? e[from:to] : ""
    parts = String[]
    depth = 0
    start = firstindex(inner)
    for i in eachindex(inner)
        ch = inner[i]
        if ch in ('(', '[', '{')
            depth += 1
        elseif ch in (')', ']', '}')
            depth -= 1
        elseif ch == ',' && depth == 0
            push!(parts, inner[start:prevind(inner, i)])
            start = nextind(inner, i)
        end
    end
    push!(parts, inner[start:end])
    out = Property[]
    for part in parts
        arrow = findfirst("=>", part)
        arrow === nothing && continue
        key = strip(part[1:prevind(part, first(arrow))])
        startswith(key, ":") || continue
        push!(out, parse_property(key, part[(last(arrow) + 1):end]))
    end
    return out
end

const INTERFACE_PATTERN = r"(*UCP)^([A-Za-z_]\w*Interface)\s*\((.*)\)\s*$"s

"""
The scalar fields each interface takes, in constructor order: the read-back
names them in the form, the code generation only needs their order.
"""
const INTERFACE_FIELDS = Dict(
    "PerfectInterface" => String[],
    "SpringInterface" => ["kn", "kt"],
    "MembraneInterface" => ["k2D", "mu2D"],
    "KapitzaInterface" => ["h"],
    "SurfaceConductiveInterface" => ["ks"],
    "AnisotropicSpringInterface" => ["compliance"],
    "AnisotropicMembraneInterface" => ["stiffness"],
    "AnisotropicSurfaceConductiveInterface" => ["conductance"],
)

"`SpringInterface(1.0e-3, 2.0e-3)` as the fields of the form, or `nothing`."
function parse_interface(expr::AbstractString)
    m = match(INTERFACE_PATTERN, strip(expr))
    m === nothing && return nothing
    kind, body = String(m[1]), strip(m[2])
    names = get(INTERFACE_FIELDS, kind, nothing)
    names === nothing && return nothing
    parts = isempty(body) ? String[] : split_toplevel(body)
    length(parts) == length(names) || return nothing
    return jdict("kind" => kind, "args" => jdict((n => number_or_text(v) for (n, v) in zip(names, parts))...))
end

"The elements of a Julia tuple literal, as written."
function tuple_items(text::AbstractString)
    t = strip(text)
    (startswith(t, "(") && endswith(t, ")")) && (t = t[nextind(t, firstindex(t)):prevind(t, lastindex(t))])
    return split_toplevel(t)
end

const IDENTIFIER = r"(*UCP)^[A-Za-z_\u00C0-\uFFFF][\w\u00C0-\uFFFF!]*$"

"""
    unrepresentable(node) -> Union{Nothing, String}

Why a builder cannot be written back faithfully, or `nothing`: a signature with
types, defaults or keywords, a shape or properties the forms do not model.
"""
function unrepresentable(node::AbstractDict)
    for p in either(get(node, "params", nothing), String[])
        s = strip(string(p))
        occursin(IDENTIFIER, s) || return "the builder signature uses `$s`, which the studio does not model"
    end
    get(node, "kind", nothing) == "laminate_builder" && return unrepresentable_laminate(node)
    for ph in either(get(node, "phases", nothing), Any[])
        geom = field(ph, "geometry", "")
        parse_geometry(geom) === nothing && return "geometry `$geom` is not one the studio can rebuild"
        props = field(ph, "properties", "")
        (isempty(strip(props)) || startswith(strip(props), "Dict")) ||
            return "properties `$props` are not a Dict the studio can rebuild"
        for (opt, val) in either(get(ph, "options", nothing), Dict{String, String}())
            if opt == "symmetrize" && !(occursin("IsoSymmetrize", val) || occursin("TISymmetrize", val) || strip(val) == "nothing")
                return "symmetrize = `$val` is not one the studio can rebuild"
            end
        end
    end
    return nothing
end

function unrepresentable_laminate(node::AbstractDict)
    for k in keys(either(get(node, "laminate_options", nothing), Dict{String, String}()))
        # `basis = …` gives a symbolic frame, which the model would have to invent
        k in ("normal", "euler_angles") || return "`$k = …` on the Laminate is not one the studio models"
    end
    layers = either(get(node, "layers", nothing), Any[])
    isempty(layers) && return "the laminate has no layer"
    kinds = Set{String}()
    for lay in layers
        props = field(lay, "properties", "")
        (isempty(strip(props)) || startswith(strip(props), "Dict")) ||
            return "properties `$props` are not a Dict the studio can rebuild"
        lopts = either(get(lay, "options", nothing), Dict{String, String}())
        for k in keys(lopts)
            k in ("fraction", "thickness", "interface") || return "`$k = …` on add_layer! is not one the studio models"
        end
        if haskey(lopts, "fraction")
            push!(kinds, "fraction")
        elseif haskey(lopts, "thickness")
            push!(kinds, "thickness")
        else
            return "layer `$(py_str(get(lay, "name", nothing)))` gives neither a fraction nor a thickness"
        end
        itf = strip(either(get(lopts, "interface", nothing), ""))
        (!isempty(itf) && parse_interface(itf) === nothing) && return "interface `$itf` is not one the studio can rebuild"
    end
    # the package refuses this too: the source is kept rather than rewritten
    length(kinds) > 1 && return "the stack mixes fractions and absolute thicknesses"
    return nothing
end

"The name of a cell from the name of its builder: `build_foam` is `foam`."
cell_name(fname::AbstractString) = startswith(fname, "build_") ? fname[(ncodeunits("build_") + 1):end] : String(fname)

function cell_from_node(node::AbstractDict)
    get(node, "kind", nothing) == "laminate_builder" && return laminate_from_node(node)
    fname = field(node, "function", "build_rve")
    c = Cell(;
        name = cell_name(fname),
        matrix_name = either(get(node, "matrix", nothing), "MATRIX"),
        params = collect(String, either(get(node, "params", nothing), String[])),
        # the function keeps its own name: its callers refer to it
        builder_name = fname,
        rve_options = JDict(either(get(node, "rve_options", nothing), JDict())),
    )
    for p in field(node, "phases", Any[])
        ph = Phase(;
            name = field(p, "name", "PHASE"),
            amount_kind = truthy(get(p, "is_matrix", nothing)) ? "rest" : field(p, "amount_kind", "fraction"),
            geometry = something(parse_geometry(field(p, "geometry", "")), Geometry()),
            properties = parse_properties(field(p, "properties", "")),
        )
        opts = either(get(p, "options", nothing), Dict{String, String}())
        # `fraction = :rest` marks the phase taking up the rest, not an amount
        if haskey(opts, "fraction") && opts["fraction"] != ":rest"
            ph.amount_kind, ph.amount = "fraction", number_or_text(opts["fraction"])
        elseif haskey(opts, "density")
            ph.amount_kind, ph.amount = "density", number_or_text(opts["density"])
        end
        sym = get(opts, "symmetrize", "")
        occursin("IsoSymmetrize", sym) ? (ph.symmetrize = "iso") : occursin("TISymmetrize", sym) && (ph.symmetrize = "ti")
        push!(c.phases, ph)
    end
    return c
end

function laminate_from_node(node::AbstractDict)
    fname = field(node, "function", "build_lam")
    opts = either(get(node, "laminate_options", nothing), Dict{String, String}())
    c = Cell(; name = cell_name(fname), kind = "laminate", params = collect(String, either(get(node, "params", nothing), String[])), builder_name = fname)
    if haskey(opts, "euler_angles")
        c.frame_mode = "euler"
        c.euler_angles = Any[number_or_text(x) for x in tuple_items(opts["euler_angles"])]
    elseif haskey(opts, "normal")
        c.frame_mode = "normal"
        c.normal = Any[number_or_text(x) for x in tuple_items(opts["normal"])]
    else
        # no keyword is the canonical frame, which the code generation writes
        # for e₃: read back, it lands on the model the studio would create
        c.frame_mode = "euler"
        c.euler_angles = Any[]
        c.normal = Any[0.0, 0.0, 1.0]
    end
    for lay in field(node, "layers", Any[])
        lopts = either(get(lay, "options", nothing), Dict{String, String}())
        layer = Layer(; name = field(lay, "name", "A"), properties = parse_properties(field(lay, "properties", "")))
        if haskey(lopts, "thickness")
            layer.amount_kind, layer.amount = "thickness", number_or_text(lopts["thickness"])
        else
            layer.amount_kind, layer.amount = "fraction", number_or_text(lopts["fraction"])
        end
        perfect = jdict("kind" => "PerfectInterface", "args" => JDict())
        layer.interface = haskey(lopts, "interface") ? something(parse_interface(lopts["interface"]), perfect) : perfect
        push!(c.layers, layer)
    end
    return c
end

"The seams of `c` pointed at the cells of `by_builder` (by the name of their builder)."
function resolve_seams!(c::Cell, by_builder::AbstractDict)
    for mb in members(c), pr in mb.properties
        (pr.source == "cell" && pr.cell isa String && startswith(pr.cell, "@builder:")) || continue
        target = get(by_builder, pr.cell[(ncodeunits("@builder:") + 1):end], nothing)
        target === nothing || (pr.cell = target.id)
    end
    return c
end

"""
    fails_to_reproduce(node, original, others) -> Union{Nothing, String}

Would writing this cell back change the file? The cell is rendered as it would
be saved, its seams pointing at the builders `others` names, and compared with
the text it came from, indentation and blank lines aside: a construct the
studio parses but cannot reproduce is one it leaves alone. (`unrepresentable`
has refused first what the cell could not be built from.)
"""
function fails_to_reproduce(node::AbstractDict, original::AbstractString, others::AbstractDict)
    norm(text) = [String(strip(l)) for l in py_splitlines(text) if !isempty(strip(l))]
    c = cell_from_node(node)
    stubs = Dict{String, Cell}(f => cell_from_node(n) for (f, n) in others)
    stubs[builder(c)] = c
    resolve_seams!(c, stubs)
    m = StudioModel(; cells = vcat(c, [x for x in values(stubs) if x !== c]))
    norm(render_cell(m, c)) == norm(original) && return nothing
    return "writing it back would not reproduce the original text"
end

"""
    readable_builders(nodes) -> IdDict

Why each builder node is kept as written (`nothing` for one the studio
models). A cell is checked with the builders it reads through its seams, which
must be readable too: the check is repeated until no builder drops out, so that
a seam never points at a builder kept as written.
"""
function readable_builders(nodes)
    why = IdDict{Any, Union{Nothing, String}}(n => unrepresentable(n) for n in nodes)
    dropped = true
    while dropped
        accepted = Dict(String(n["function"]) => n for n in nodes if why[n] === nothing)
        dropped = false
        for n in values(accepted)
            reason = fails_to_reproduce(n, n["source"], accepted)
            reason === nothing && continue
            why[n] = reason
            dropped = true
        end
    end
    return why
end

"""
    model_from_script(source) -> (model, report)

The model a script describes, and a report on what was understood: the
embedded model of a script the studio wrote, else the cells and parameters
recognized, with everything else kept verbatim. Any text is read: what is not
understood, syntax errors included, is kept as written.
"""
function model_from_script(source::AbstractString)
    embedded = extract_embedded(source)
    if embedded !== nothing
        report = jdict(
            "exact" => true, "recognized" => nothing, "opaque" => 0,
            "note" => "reopened from the model the studio embedded in the file",
        )
        return model_from_dict(embedded), report
    end
    parsed = parse_script(source)
    m = StudioModel(; title = "opened")
    why = readable_builders([n for n in parsed["nodes"] if get(n, "kind", nothing) in ("rve_builder", "laminate_builder")])
    by_builder = Dict{String, Cell}()
    order = 0
    for node in parsed["nodes"]
        kind = get(node, "kind", nothing)
        src = field(node, "source", "")
        if kind == "trivia"
            # blank lines and comments: kept, unless they are only whitespace
            if !isempty(strip(src))
                push!(m.opaque, OpaqueBlock(; source = src, order))
                order += 1
            end
        elseif kind == "param"
            push!(m.params, Param(; name = node["name"], value = node["value"], origin = src))
            order += 1
        elseif kind in ("rve_builder", "laminate_builder")
            if why[node] === nothing
                c = cell_from_node(node)
                push!(m.cells, c)
                by_builder[builder(c)] = c
            else
                # regenerating it would change its signature and break its callers
                push!(m.opaque, OpaqueBlock(; source = src, order, note = "kept as written: $(why[node])"))
            end
            order += 1
        else
            # `homogenize`, and everything else, as written
            push!(m.opaque, OpaqueBlock(; source = src, order))
            order += 1
        end
    end
    # the seams, now that every cell is read: each points at a readable builder
    foreach(c -> resolve_seams!(c, by_builder), m.cells)
    isempty(m.cells) || (m.root_cell = last(m.cells).id)
    report = jdict(
        "exact" => false, "recognized" => parsed["recognized"], "opaque" => length(m.opaque),
        "cells" => [c.name for c in m.cells],
        "note" => "the parts shown as preserved were not recognized and will be written back unchanged",
    )
    return m, report
end
