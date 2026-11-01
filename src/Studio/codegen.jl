# Model → Julia. The conventions are those of the scripts the studio has always
# written, and of `tools/echoes2mfh`, so that a script written by the studio
# and one translated from Echoes read the same way. Three of them carry weight:
#
# * `iso_stiffness(k, μ)` takes physical moduli, where the raw `TensISO{3}(a, b)`
#   takes `(3k, 2μ)`: only the former is written.
# * The amount of the phase taking up the rest is derived (`1 − Σ f`): it is
#   written `fraction = :rest`, never as a number.
# * Solver options belong to the scheme instance, not to `homogenize`.
#
# The text is that of the Python studio byte for byte, so that a script it saved
# and the same model written now compare equal.

const IND = "    "
const RULE = "# " * "="^77
const MODEL_TAG = "mfhstudio-model v1"

rule(title::AbstractString) = "# ── $title " * "─"^max(3, 72 - length(title))

"The ZYZ angles, padded to three and emptied for the canonical frame."
function padded_angles(euler)
    a = Any[x for x in either(euler, Any[]) if x != "" && x !== nothing]
    a = vcat(a, Any[0.0, 0.0, 0.0])[1:3]
    return all(x -> x == 0, a) ? Any[] : a
end

"Is the normal the default stacking direction, e₃?"
function is_canonical_normal(n)
    vals = collect(Any, either(n, Any[]))
    (length(vals) != 3 || any(x -> x isa AbstractString, vals)) && return false
    return vals[1] == 0 && vals[2] == 0 && vals[3] > 0
end

"A number, or a Julia expression the user typed (`π / 4`), written as it is."
code_or_float(x) = x isa AbstractString ? String(x) : julia_float(x)

"The basis the constants are written in; the canonical one is spelled out."
function frame_expr(euler)
    a = padded_angles(euler)
    isempty(a) && return "CanonicalBasis{3, Float64}()"
    return "RotatedBasis(" * join((code_or_float(x) for x in a), ", ") * ")"
end

"The symmetry axis of a transversely isotropic tensor: the third vector of the frame."
function axis_expr(euler, axis = nothing)
    a = padded_angles(euler)
    isempty(a) || return "vecbasis($(frame_expr(a)))[:, 3]"
    axis isa AbstractVector && length(axis) == 3 && return tuple_expr(julia_float(x) for x in axis)
    return "(0.0, 0.0, 1.0)"
end

"`axis_expr` as a tuple, which `LayeredSpheroid` requires."
function axis_tuple_expr(euler, axis = nothing)
    e = axis_expr(euler, axis)
    return startswith(e, "(") ? e : "Tuple($e)"
end

"A Julia tuple of the non-empty `items`; only a 1-tuple takes the trailing comma."
function tuple_expr(items)
    parts = [String(p) for p in items if truthy(p)]
    isempty(parts) && return "()"
    length(parts) == 1 && return "($(only(parts)),)"
    return "(" * join(parts, ", ") * ")"
end

# The activation preamble, shared with echoes2mfh: a script runs from wherever
# it is saved.
const ACTIVATE = """import Pkg
let d = @__DIR__
    while true
        pt = joinpath(d, "Project.toml")
        if isfile(pt) && occursin("MeanFieldHomogenization", read(pt, String))
            Pkg.activate(d; io = devnull)
            break
        end
        parent = dirname(d)
        parent == d && break
        d = parent
    end
end"""

"The writer of one script: the model, whether to embed it, and the lines so far."
mutable struct CodeGen
    const m::StudioModel
    const embed::Bool
    out::Vector{String}
end

CodeGen(m::StudioModel; embed::Bool = true) = CodeGen(m, embed, String[])

w!(g::CodeGen, line::AbstractString = "") = (push!(g.out, line); nothing)
blank!(g::CodeGen) = ((isempty(g.out) || last(g.out) == "") || push!(g.out, ""); nothing)

"""
    generate(model; embed_model = true) -> String

The script of `model`: a header naming its problems, the packages, the
parameters, one builder per cell (inner scales first), the code kept verbatim,
the computation, and the model embedded so that the studio reopens it exactly.
"""
function generate(m::StudioModel; embed_model::Bool = true)
    g = CodeGen(m; embed = embed_model)
    header!(g, validate(m))
    preamble!(g)
    params!(g)
    builders!(g)
    opaque!(g)
    main!(g)
    embedded_model!(g)
    return rstrip(join(g.out, "\n")) * "\n"
end

function header!(g::CodeGen, problems::Vector{String})
    w!(g, RULE)
    w!(g, "#  $(g.m.title).jl")
    w!(g, "#")
    w!(g, "#  Built with MFH Studio. Editing this file by hand is fine: the")
    w!(g, "#  studio reads it back and preserves anything it does not")
    w!(g, "#  recognize.")
    if !isempty(g.m.description)
        w!(g, "#")
        foreach(line -> w!(g, "#  $line"), py_splitlines(g.m.description))
    end
    if !isempty(problems)
        w!(g, "#")
        w!(g, "#  UNRESOLVED PROBLEMS — this script will not run as-is:")
        foreach(p -> w!(g, "#    * $p"), problems)
    end
    w!(g, RULE)
    w!(g)
    return nothing
end

function preamble!(g::CodeGen)
    foreach(line -> w!(g, line), py_splitlines(ACTIVATE))
    blank!(g)
    w!(g, "using MeanFieldHomogenization")
    w!(g, "using TensND")
    w!(g, "using LinearAlgebra")
    w!(g, "using Printf")
    # only the computation that plots loads Plots: a sensitivity prints a table
    m = g.m
    plots = m.alv.enabled ? m.alv.plot :
        m.sens.enabled ? false : (m.sweep.enabled && m.sweep.mode != "single" && m.sweep.plot)
    if plots
        w!(g, "using Plots")
        w!(g, "gr()")
    end
    blank!(g)
    return nothing
end

function params!(g::CodeGen)
    isempty(g.m.params) && return nothing
    w!(g, rule("Parameters"))
    for p in g.m.params
        # a parameter read from a file keeps its text until it is edited
        if truthy(p.origin) && !p.edited
            foreach(line -> w!(g, line), py_splitlines(p.origin))
            continue
        end
        w!(g, "const $(p.name) = $(p.value)" * (isempty(p.comment) ? "" : "  # $(p.comment)"))
    end
    blank!(g)
    return nothing
end

function builders!(g::CodeGen)
    isempty(g.m.cells) && return nothing
    order = try
        topological_order(g.m)
    catch err
        # the cycle is named in the header; the cells follow in their order
        err isa ArgumentError || rethrow()
        g.m.cells
    end
    if uses_multiscale(g.m)
        w!(g, rule("Scales"))
        w!(g, "#")
        w!(g, "# Inner scales are defined first. A phase property holding a")
        w!(g, "# `Homogenized(cell, scheme)` is the seam: the outer scheme")
        w!(g, "# resolves the inner scale when it reads that property, and")
        w!(g, "# memoizes it for the duration of the call.")
        blank!(g)
    end
    foreach(c -> builder!(g, c), order)
    return nothing
end

function builder!(g::CodeGen, c::Cell)
    w!(g, "function $(builder(c))($(join(c.params, ", ")))")
    foreach(line -> w!(g, line), cell_body(g, c))
    w!(g, "end")
    blank!(g)
    return nothing
end

"The statements of a builder, without `function`/`end`: the 3-D view reads them too."
function cell_body(g::CodeGen, c::Cell)
    saved, g.out = g.out, String[]
    try
        is_laminate(c) ? laminate_body!(g, c) : rve_body!(g, c)
        return g.out
    finally
        g.out = saved
    end
end

"Keyword options as `k = v`, sorted by name."
options_text(opts) = ["$k = $(py_str(opts[k]))" for k in sort!(collect(String, keys(opts)))]

function rve_body!(g::CodeGen, c::Cell)
    opts = either(c.rve_options, JDict())
    tail = isempty(opts) ? "" : "; " * join(options_text(opts), ", ")
    # an RVE designates no matrix: the phase taking up the rest is written first
    w!(g, "$(IND)rve = RVE($tail)")
    rest = remainder(c)
    rest === nothing || emit_phase!(g, rest)
    foreach(ph -> emit_phase!(g, ph), inclusions(c))
    w!(g, "$(IND)return rve")
    return nothing
end

"An angle tuple as the page entered it: `(a,)` for one angle."
angle_tuple(vals, n) = n == 1 ? "($vals,)" : "($vals)"

function laminate_body!(g::CodeGen, c::Cell)
    # one way only to give the frame; the canonical one is written as nothing
    opts = String[]
    if c.frame_mode == "euler" && !isempty(padded_angles(c.euler_angles))
        vals = join((code_or_float(x) for x in c.euler_angles), ", ")
        push!(opts, "euler_angles = " * angle_tuple(vals, length(c.euler_angles)))
    elseif c.frame_mode != "euler" && !is_canonical_normal(c.normal)
        push!(opts, "normal = (" * join((code_or_float(x) for x in either(c.normal, Any[])), ", ") * ")")
    end
    append!(opts, options_text(either(c.rve_options, JDict())))
    w!(g, "$(IND)lam = Laminate(" * (isempty(opts) ? "" : "; " * join(opts, ", ")) * ")")
    for lay in c.layers
        kw = ["$(lay.amount_kind) = $(amount_text(lay.amount))"]
        itf = interface_expr(lay.interface)
        isempty(itf) || push!(kw, "interface = $itf")
        call!(g, "add_layer!(lam, :$(lay.name), $(properties_expr(g, lay.properties)); " * join(kw, ", ") * ")")
    end
    w!(g, "$(IND)return lam")
    return nothing
end

"An interface, or `\"\"` for the perfect one, which `add_layer!` assumes."
function interface_expr(itf)
    itf = either(itf, JDict())
    kind = field(itf, "kind", "PerfectInterface")
    kind == "PerfectInterface" && return ""
    args = join((code_or_number(v) for v in values(either(get(itf, "args", nothing), JDict()))), ", ")
    return isempty(args) ? "$(py_str(kind))()" : "$(py_str(kind))($args)"
end

code_or_number(x) = x isa AbstractString ? String(x) : julia_number(x)
amount_text(x) = x isa AbstractString ? String(x) : julia_float(x)

const SYMMETRIZE_CODE = Dict("iso" => "IsoSymmetrize()", "ti" => "TISymmetrize()")

function emit_phase!(g::CodeGen, ph::Phase)
    opts = String[]
    (truthy(ph.symmetrize) && ph.symmetrize != "none") && push!(opts, "symmetrize = " * SYMMETRIZE_CODE[ph.symmetrize])
    # `fraction = :rest` marks the phase taking up the volume the others leave
    pushfirst!(opts, ph.amount_kind == "rest" ? "fraction = :rest" : "$(ph.amount_kind) = $(amount_text(ph.amount))")
    call!(g, "add_phase!(rve, :$(ph.name), $(geometry_expr(g, ph)), $(properties_expr(g, ph.properties)); " * join(opts, ", ") * ")")
    return nothing
end

"""
    call!(g, line)

A call on one line, or, beyond 92 columns, broken at its own `;`: only a
separator at bracket depth zero belongs to the call, one deeper belongs to a
nested constructor. A call with no such separator stays long.
"""
function call!(g::CodeGen, line::AbstractString)
    full = IND * line
    length(full) <= 92 && return w!(g, full)
    open = findfirst('(', line)
    head, tail = open === nothing ? (line, "") : (line[1:prevind(line, open)], line[nextind(line, open):end])
    close = findlast(')', tail)
    body = close === nothing ? "" : tail[1:prevind(tail, close)]
    cut = toplevel_kw_split(body)
    cut === nothing && return w!(g, full)
    w!(g, "$IND$head(")
    w!(g, "$IND$IND$(body[1:prevind(body, cut)]);")
    w!(g, "$IND$IND$(body[nextind(body, cut, 2):end])")
    w!(g, "$IND)")
    return nothing
end

"The index of the `;` separating positional from keyword arguments, at depth zero."
function toplevel_kw_split(body::AbstractString)
    depth = 0
    for i in eachindex(body)
        ch = body[i]
        if ch in ('(', '[', '{')
            depth += 1
        elseif ch in (')', ']', '}')
            depth -= 1
        elseif ch == ';' && depth == 0
            return i
        end
    end
    return nothing
end

"A value of an argument dictionary: the typed expression, else the number or its default."
arg_text(a, key, default) = (v = get(a, key, default); v isa AbstractString ? String(v) : julia_float(v))

function geometry_expr(g::CodeGen, ph::Phase)
    geo = ph.geometry
    a = geo.args
    v(key, default = 0.0) = arg_text(a, key, default)
    ang = ""
    if truthy(geo.euler_angles)
        # a bare `0` beside `0.5` would make a `Tuple{Int, Float64}`
        ang = "; euler_angles = " * angle_tuple(join((code_or_float(x) for x in geo.euler_angles), ", "), length(geo.euler_angles))
    end
    k = geo.kind
    k == "spheroid" && return "Spheroid($(v("omega", 1.0))$ang)"
    k == "ellipsoid" && return "Ellipsoid($(v("a", 1.0)), $(v("b", 1.0)), $(v("c", 1.0))$ang)"
    k == "cylinder" && return "Cylinder($(v("b", 1.0)), $(v("c", 1.0))$ang)"
    k == "penny_crack" && return "PennyCrack($(v("a", 1.0))$ang)"
    k == "elliptic_crack" && return "EllipticCrack($(v("a", 1.0)), $(v("b", 0.5))$ang)"
    k == "ribbon_crack" && return "RibbonCrack($(v("b", 1.0))$ang)"
    k == "layered_sphere" && return layered_sphere_expr(g, geo)
    k == "layered_spheroid" && return layered_spheroid_expr(g, geo)
    return "Spheroid(1.0)"
end

function layered_sphere_expr(g::CodeGen, geo::Geometry)
    radii = join((arg_text(l, "radius", 1.0) for l in geo.layers), ", ")
    moduli = join((prop_expr(g, property_from_dict(l["property"])) for l in geo.layers), ", ")
    # the joined lists are split again on ", ", as the studio always wrote them
    radii_t = isempty(radii) ? "()" : tuple_expr(split(radii, ", "))
    mod_t = isempty(moduli) ? "()" : tuple_expr(split(moduli, ", "))
    ifaces = [get(l, "interface", nothing) for l in geo.layers]
    if any(i -> truthy(i) && field(i, "kind", "PerfectInterface") != "PerfectInterface", ifaces)
        parts = map(ifaces) do i
            i = either(i, jdict("kind" => "PerfectInterface", "args" => JDict()))
            kind = py_str(field(i, "kind", "PerfectInterface"))
            args = join((julia_number(x) for x in values(either(get(i, "args", nothing), JDict()))), ", ")
            isempty(args) ? "$kind()" : "$kind($args)"
        end
        it = length(parts) == 1 ? "($(only(parts)),)" : "(" * join(parts, ", ") * ")"
        return "LayeredSphere($radii_t, $mod_t; interfaces = $it)"
    end
    return "LayeredSphere($radii_t, $mod_t)"
end

"""
`layered_spheroid_from_fractions`: the fraction constructor solves for the
confocal radii itself, which radii typed layer by layer essentially never are.
"""
function layered_spheroid_expr(g::CodeGen, geo::Geometry)
    fractions = tuple_expr(arg_text(l, "fraction", 1.0) for l in geo.layers)
    moduli = tuple_expr(prop_expr(g, property_from_dict(l["property"])) for l in geo.layers)
    omega = field(geo.args, "omega", 0.5)
    radius = field(geo.args, "radius", 1.0)
    ns = field(geo.args, "Nseries", 5)
    # the solver honors the axis of a spheroid of revolution
    axis = isempty(padded_angles(geo.euler_angles)) ? "" : ", axis = $(axis_tuple_expr(geo.euler_angles))"
    return "layered_spheroid_from_fractions($(julia_float(omega)), $(julia_float(radius)), " *
        "$fractions, $moduli; Nseries = $(julia_number(ns))$axis)"
end

function properties_expr(g::CodeGen, props::Vector{Property})
    isempty(props) && return "Dict{Symbol, Any}()"
    return "Dict(" * join(("$(p.key) => $(prop_expr(g, p))" for p in props), ", ") * ")"
end

"""
    prop_expr(g, p) -> String

A property as Julia: the typed expression, the multiscale seam
`Homogenized(inner_builder(...), scheme)`, a viscoelastic law, or a tensor
built from its constants (physical moduli for an isotropic one).
"""
function prop_expr(g::CodeGen, p::Property)
    p.source == "expr" && return either(p.expr, "one(TensISO{3})")
    if p.source == "cell"
        inner = cell(g.m, p.cell)
        inner === nothing && return "#= missing inner cell =# one(TensISO{3})"
        call = "$(builder(inner))(" * join(inner.params, ", ") * ")"
        return "Homogenized($call, $(scheme_expr(either(p.scheme, "MoriTanaka"), either(p.scheme_options, JDict()))))"
    end
    (p.source == "visco" || truthy(p.visco)) && return visco_expr(either(p.visco, JDict()))
    a = p.args
    num(key, default = 0.0) = arg_text(a, key, default)
    b = p.builder
    b == "iso_stiffness" && return "iso_stiffness($(num("k", 1.0)), $(num("mu", 1.0)))"
    b == "iso_stiffness_E_nu" && return "iso_stiffness_E_nu($(num("E", 1.0)), $(num("nu", 0.2)))"
    # the axis is an argument of `hoenig_stiffness`, not a keyword
    b == "hoenig_stiffness" && return "hoenig_stiffness($(num("E1", 30.0)), $(num("h", 0.3)), " *
        "$(num("nu1", 0.2)), $(num("nu2", 0.25)), $(num("gamma", 0.5)), $(axis_expr(p.euler_angles)))"
    # one argument to TensISO{dim} is the second-order (conductivity) form
    b == "TensISO{3}" && return "TensISO{3}($(num("k", 1.0)))"
    b == "TensTI2" && return "TensTI{2}($(num("kt", 1.0)), $(num("ka", 1.0)), $(axis_expr(p.euler_angles, get(a, "axis", nothing))))"
    if b == "TensDiag2"
        # `Tens(A, basis)` holds A as the components in that basis
        diag = "[$(num("k1", 1.0)) 0.0 0.0; 0.0 $(num("k2", 1.0)) 0.0; 0.0 0.0 $(num("k3", 1.0))]"
        return isempty(padded_angles(p.euler_angles)) ? "Tens($diag)" : "Tens($diag, $(frame_expr(p.euler_angles)))"
    end
    if b == "TensOrtho"
        consts = join(
            (
                num(k, d) for (k, d) in (
                        ("C11", 120.0), ("C22", 90.0), ("C33", 70.0), ("C12", 40.0), ("C13", 35.0),
                        ("C23", 30.0), ("C44", 25.0), ("C55", 22.0), ("C66", 20.0),
                    )
            ), ", "
        )
        return "TensOrtho($consts, $(frame_expr(p.euler_angles)))"
    end
    b == "maxwell_iso" && return "maxwell_iso($(num("k", 10.0)), $(num("mu", 5.0)), $(num("eta_k", 1.0)), $(num("eta_mu", 1.0)))"
    b == "kelvin_iso" && return "kelvin_iso($(num("k0", 10.0)), $(num("mu0", 5.0)), " *
        "[$(num("k1", 20.0))], [$(num("mu1", 10.0))], [$(num("tau_k", 1.0))], [$(num("tau_mu", 1.0))])"
    b == "heaviside_law" && return "heaviside_law(iso_stiffness($(num("k", 10.0)), $(num("mu", 5.0))))"
    if b == "ViscoLaw"
        expr = py_str(either(get(a, "expr", nothing), "iso_stiffness(10.0, 5.0)"))
        mode = lstrip(strip(py_str(either(get(a, "mode", nothing), "creep"))), ':')
        mode = mode in ("creep", "relaxation") ? mode : "creep"
        return "ViscoLaw((t, t′) -> $expr, :$mode)"
    end
    return "iso_stiffness($(num("k", 1.0)), $(num("mu", 1.0)))"
end

"""
The `visco` field of a property, kept for the projects saved with it: one time
per law, taken as both the bulk and the shear time.
"""
function visco_expr(v)
    kind = field(v, "kind", "maxwell_iso")
    a = field(v, "args", JDict())
    num(key, default = 1.0) = julia_number(get(a, key, default))
    kind == "maxwell_iso" && return "maxwell_iso($(num("k", 10.0)), $(num("mu", 5.0)), $(num("tau")), $(num("tau")))"
    kind == "kelvin_iso" && return "kelvin_iso(Inf, Inf, [$(num("k", 10.0))], [$(num("mu", 5.0))], [$(num("tau"))], [$(num("tau"))])"
    kind == "heaviside" && return "heaviside_law(iso_stiffness($(num("k", 10.0)), $(num("mu", 5.0))))"
    return "ViscoLaw((t, t′) -> $(py_str(field(a, "expr", "1.0"))), :$(py_str(field(v, "mode", "creep"))))"
end

"A scheme with its solver options, which belong to the instance."
function scheme_expr(name, options)
    opts = [k => v for (k, v) in either(options, JDict()) if v !== nothing]
    isempty(opts) && return "$(py_str(name))()"
    sort!(opts; by = first)
    return "$(py_str(name))(; " * join(("$k = $(julia_number(v))" for (k, v) in opts), ", ") * ")"
end

function opaque!(g::CodeGen)
    isempty(g.m.opaque) && return nothing
    w!(g, rule("Preserved from the original script"))
    w!(g, "#")
    w!(g, "# MFH Studio did not recognize the code below, so it is kept")
    w!(g, "# exactly as it was rather than rewritten.")
    blank!(g)
    for blk in sort(g.m.opaque; by = b -> b.order, alg = Base.Sort.DEFAULT_STABLE)
        isempty(blk.note) || w!(g, "# $(blk.note)")
        foreach(line -> w!(g, line), py_splitlines(blk.source))
        blank!(g)
    end
    return nothing
end

function main!(g::CodeGen)
    r = root(g.m)
    r === nothing && return nothing
    w!(g, rule("Result"))
    m = g.m
    if m.alv.enabled
        alv_main!(g, r)
    elseif m.sens.enabled
        sens_main!(g, r)
    elseif !m.sweep.enabled || m.sweep.mode == "single"
        single_main!(g, r)
    else
        sweep_main!(g, r, m.sweep)
    end
    return nothing
end

const PROJECTION_CODE = Dict("iso" => "best_fit_iso", "ti" => "best_fit_ti", "ortho" => "best_fit_ortho")

function project(g::CodeGen, var::AbstractString)
    fn = get(PROJECTION_CODE, g.m.sweep.projection, nothing)
    return fn === nothing ? String(var) : "$fn($var)"
end

clamped(g::CodeGen, expr::AbstractString) = g.m.sweep.clamp_zero ? "max($expr, 0.0)" : String(expr)

"""
What each output reduces the effective tensor with, and the local it is bound
to: `k` and `μ` come out of one `k_mu`, a component and a trace out of one
`Array`, so asking for both must not compute the pair twice.
"""
const REDUCERS = Dict(
    "k" => ("km", "k_mu"), "mu" => ("km", "k_mu"), "E" => ("Enu", "E_nu"), "nu" => ("Enu", "E_nu"),
    "km" => ("KMC", "mandel"), "comp" => ("arr", "Array"), "trace3" => ("arr", "Array"),
)

"""
    bindings(outputs, var, suffix) -> Vector{Tuple{String, String, String}}

`(name, local, expression)` for every reduction used more than once, in the
order of first use; a reduction used once stays inline.
"""
function bindings(outputs, var::AbstractString, suffix::AbstractString = "")
    seen = Tuple{String, String, Int}[]
    for o in outputs
        entry = get(REDUCERS, field(o, "kind", "k"), nothing)
        entry === nothing && continue
        name, fn = entry
        i = findfirst(s -> s[1] == name, seen)
        i === nothing ? push!(seen, (name, fn, 1)) : (seen[i] = (name, fn, seen[i][3] + 1))
    end
    return [(name, name * suffix, "$fn($var)") for (name, fn, n) in seen if n > 1]
end

"`s` padded with spaces to `n` characters, as Python's `str.ljust` (not by display width)."
py_ljust(s::AbstractString, n::Integer) = s * " "^max(0, n - length(s))

"Python's `int` of an index the page sent."
py_int(x::Integer) = Int(x)
py_int(x::AbstractFloat) = trunc(Int, x)
py_int(x::AbstractString) = parse(Int, strip(x))

"""
    output_expr(o, var, binds) -> String

One plotted quantity. `k`, `μ`, `E` and `ν` exist for an isotropic tensor only;
Kelvin-Mandel components whatever the symmetry. `binds` maps a reduction to the
local it was bound to.
"""
function output_expr(o, var::AbstractString = "C", binds::AbstractDict = Dict{String, String}())
    red(kind) = (entry = REDUCERS[kind]; get(binds, entry[1], "$(entry[2])($var)"))
    kind = field(o, "kind", "k")
    i, j = py_int(field(o, "i", 1)), py_int(field(o, "j", 1))
    kind == "mu" && return "$(red("mu"))[2]"
    kind == "E" && return "$(red("E"))[1]"
    kind == "nu" && return "$(red("nu"))[2]"
    kind in ("km", "comp") && return "$(red(kind))[$i, $j]"
    kind == "trace3" && return "tr($(red("trace3"))) / 3"
    return "$(red("k"))[1]"
end

function output_label(o)
    kind = field(o, "kind", "k")
    i, j = py_int(field(o, "i", 1)), py_int(field(o, "j", 1))
    kind == "km" && return "KM$i$j"
    kind == "comp" && return "C$i$j"
    kind == "trace3" && return "tr/3"
    return py_str(kind)
end

scheme_list(sw::Sweep) = [
    (py_str(field(x, "name", "MoriTanaka")), scheme_expr(field(x, "name", "MoriTanaka"), either(get(x, "options", nothing), JDict())))
        for x in either(sw.schemes, Any[jdict("name" => "MoriTanaka", "options" => JDict())])
]

builder_call(c::Cell) = "$(builder(c))(" * join(c.params, ", ") * ")"

function single_main!(g::CodeGen, r::Cell)
    sw = g.m.sweep
    w!(g, "# One homogenization with the amounts entered in the model.")
    w!(g, "const cell = $(builder_call(r))")
    blank!(g)
    names = either(String[output_label(o) for o in sw.outputs], ["value"])
    for (scheme_name, scheme) in scheme_list(sw)
        var = "C_$scheme_name"
        w!(g, "$var = homogenize(cell, $scheme, $(sw.property))")
        proj = project(g, var)
        proj == var || w!(g, "$var = $proj")
        # top-level names: they carry the scheme's suffix
        binds = Dict{String, String}()
        for (name, bound, expr) in bindings(sw.outputs, var, "_$scheme_name")
            w!(g, "$bound = $expr")
            binds[name] = bound
        end
        # `@printf` takes its arguments separated by spaces, not commas
        vals = either(join((clamped(g, output_expr(o, var, binds)) for o in sw.outputs), " "), var)
        fmt = join(("$n = %.6f" for n in names), "  ")
        w!(g, "@printf \"$(py_ljust(scheme_name, 24))  $fmt\\n\" $vals")
    end
    blank!(g)
    return nothing
end

function sweep_main!(g::CodeGen, r::Cell, sw::Sweep)
    lens = lens_expr(sw.lens)
    names = either(String[output_label(o) for o in sw.outputs], ["value"])
    x = sw.variable
    w!(g, "const $(x)s = range($(julia_float(sw.start)), $(julia_float(sw.stop)); length = $(julia_number(sw.length)))")
    blank!(g)
    w!(g, "#")
    w!(g, "# `set_param` returns a *new* cell, leaving the original intact,")
    w!(g, "# so the sweep is a pure map rather than a mutation.")
    w!(g, "const base_cell = $(builder_call(r))")
    blank!(g)
    w!(g, "function evaluate(scheme, $x)")
    w!(g, "$(IND)cell = set_param(base_cell, $lens, $x)")
    w!(g, "$(IND)C = homogenize(cell, scheme, $(sw.property))")
    proj = project(g, "C")
    proj == "C" || w!(g, "$(IND)C = $proj")
    binds = Dict{String, String}()
    for (name, bound, expr) in bindings(sw.outputs, "C")
        w!(g, "$IND$bound = $expr")
        binds[name] = bound
    end
    outs = either(join((clamped(g, output_expr(o, "C", binds)) for o in sw.outputs), ", "), "C")
    w!(g, length(names) == 1 ? "$(IND)return ($outs,)" : "$(IND)return ($outs)")
    w!(g, "end")
    blank!(g)
    w!(g, "const SCHEMES = [")
    foreach(((name, scheme),) -> w!(g, "$IND(\"$name\", $scheme),"), scheme_list(sw))
    w!(g, "]")
    blank!(g)
    w!(g, "const RESULTS = Dict{String, Any}()")
    w!(g, "for (name, scheme) in SCHEMES")
    w!(g, "$(IND)rows = [evaluate(scheme, $x) for $x in $(x)s]")
    for (i, n) in enumerate(names)
        w!(g, "$(IND)RESULTS[\"\$(name) $n\"] = [r[$(i)] for r in rows]")
    end
    w!(g, "end")
    blank!(g)
    w!(g, "# Published for MFH Studio; harmless when the script runs alone.")
    w!(g, "const MFHSTUDIO_RESULTS = merge(Dict(\"x\" => collect($(x)s), \"xlabel\" => \"$x\"), RESULTS)")
    blank!(g)
    # sorted once: the table and the figure walk the same curves in the same order
    w!(g, "const CURVES = sort!(collect(RESULTS); by = first)")
    blank!(g)
    w!(g, "for (label, ys) in CURVES")
    w!(g, "$(IND)@printf \"%-28s  first = %.6f   last = %.6f\\n\" label first(ys) last(ys)")
    w!(g, "end")
    if sw.plot
        blank!(g)
        w!(g, "p = plot(; xlabel = \"$x\", ylabel = \"effective property\", legend = :best)")
        w!(g, "for (label, ys) in CURVES")
        w!(g, "$(IND)plot!(p, $(x)s, ys; label = label, lw = 2)")
        w!(g, "end")
        w!(g, "display(p)")
    end
    return nothing
end

"""
`derivative`, `gradient` or `jacobian` of the effective property, straight from
the lenses: the wrappers build the perturbed cell and differentiate through
`homogenize` themselves. A jacobian passes no `indexer`: it differentiates the
whole tensor, flattened.
"""
function sens_main!(g::CodeGen, r::Cell)
    s = g.m.sens
    c = something(cell(g.m, s.cell), r)
    lenses = either([lens_expr(l) for l in s.lenses], ["amount(:PHASE)"])
    w!(g, "const cell = $(builder_call(c))")
    w!(g, "const scheme = $(scheme_expr(s.scheme, either(s.scheme_options, JDict())))")
    blank!(g)
    if s.kind == "derivative"
        w!(g, "const param = " * first(lenses))
    else
        w!(g, "const params = [")
        foreach(l -> w!(g, "$IND$l,"), lenses)
        w!(g, "]")
    end
    blank!(g)
    indexer = ""
    if s.kind != "jacobian"
        # the projection sits inside the differentiated function
        fn = get(PROJECTION_CODE, s.projection, nothing)
        indexer = ", indexer = C -> $(output_expr(s.output, fn === nothing ? "C" : "$fn(C)"))"
    end
    arg = s.kind == "derivative" ? "param" : "params"
    w!(g, "# `get_param` reads the point the derivative is taken at, so the")
    w!(g, "# values in the model are the x₀ — nothing is entered twice.")
    w!(g, "const D = $(s.kind)(cell, scheme, $arg; output = $(s.property)$indexer)")
    blank!(g)
    label = output_label(s.output)
    if s.kind == "derivative"
        w!(g, "@printf \"d($label) / d(%s) = %.8g\\n\" \"$(lens_label(first(s.lenses)))\" D")
        return nothing
    end
    w!(g, "const LABELS = [")
    foreach(l -> w!(g, "$IND\"$(lens_label(l))\","), s.lenses)
    w!(g, "]")
    if s.kind == "gradient"
        w!(g, "@printf \"%-34s  d($label) / d(p)\\n\" \"parameter\"")
        w!(g, "for (name, g) in zip(LABELS, D)")
        w!(g, "$(IND)@printf \"%-34s  %.8g\\n\" name g")
        w!(g, "end")
        return nothing
    end
    w!(g, "# One row per component of the flattened effective tensor.")
    w!(g, "println(\"jacobian \", size(D, 1), \" x \", size(D, 2), \"  columns: \",")
    w!(g, "$(IND)join(LABELS, \", \"))")
    w!(g, "for i in axes(D, 1)")
    w!(g, "$(IND)row = @view D[i, :]")
    w!(g, "$(IND)any(!iszero, row) || continue")
    w!(g, "$(IND)@printf \"  [%3d]  %s\\n\" i join(")
    w!(g, "$IND$IND[@sprintf(\"%12.6g\", x) for x in row], \"\")")
    w!(g, "end")
    return nothing
end

"How a parameter is named in the printed table."
function lens_label(l::Lens)
    k = l.kind
    k == "amount" && return "amount($(l.phase))"
    k == "thickness" && return "thickness($(l.phase))"
    k == "interface_param" && return "interface $(py_str(l.index)).$(l.field_name)"
    k == "property" && return "$(l.phase)$(l.property)[$(py_str(l.index))]"
    k == "geometry" && return "$(l.phase).$(l.field_name)[$(py_str(l.index))]"
    k == "shape_param" && return "shape.$(l.field_name)[$(py_str(l.index))]"
    k == "nested" && return "$(l.member)$(l.property) → $(lens_label(lens_from_dict(either(l.inner, JDict()))))"
    return k
end

function alv_main!(g::CodeGen, r::Cell)
    alv = g.m.alv
    c = something(cell(g.m, alv.cell), r)
    i, j = py_int(alv.component[1]), py_int(alv.component[2])
    grid = "$(julia_float(alv.t_start)), $(julia_float(alv.t_stop)); length = $(julia_number(alv.length))"
    w!(g, alv.log_time ? "const times = vcat(0.0, 10 .^ range($grid))" : "const times = range($grid)")
    blank!(g)
    w!(g, "const cell = $(builder_call(c))")
    blank!(g)
    w!(g, "#")
    w!(g, "# `homogenize_alv` returns the effective relaxation operator as a")
    w!(g, "# 6n x 6n block matrix on the time grid. Its Volterra inverse is")
    w!(g, "# the creep operator, and the response to a unit stress step is")
    w!(g, "# the row sum of its ($i$j) blocks.")
    w!(g, "function uniaxial(R)")
    w!(g, "$(IND)J = volterra_inverse(R; block_size = 6)")
    w!(g, "$(IND)n = size(J, 1) ÷ 6")
    w!(g, "$(IND)return [sum(J[6 * (a - 1) + $i, 6 * (b - 1) + $j] for b in 1:n) for a in 1:n]")
    w!(g, "end")
    blank!(g)
    w!(g, "const RESULTS = Dict{String, Any}()")
    for (name, scheme) in scheme_list(g.m.sweep)
        w!(g, "RESULTS[\"$name\"] = uniaxial(homogenize_alv(cell, $scheme, $(alv.property); times = times))")
    end
    blank!(g)
    w!(g, "# Published for MFH Studio; harmless when the script runs alone.")
    w!(g, "const MFHSTUDIO_RESULTS = merge(Dict(\"x\" => collect(times), \"xlabel\" => \"t\"), RESULTS)")
    blank!(g)
    w!(g, "const CURVES = sort!(collect(RESULTS); by = first)")
    blank!(g)
    w!(g, "for (label, ys) in CURVES")
    w!(g, "$(IND)@printf \"%-24s  J(t₁) = %.6g   J(t_end) = %.6g\\n\" label first(ys) last(ys)")
    w!(g, "end")
    if alv.plot
        blank!(g)
        w!(g, "p = plot(; xlabel = \"t\", ylabel = \"uniaxial creep\", legend = :best" * (alv.log_time ? ", xscale = :log10)" : ")"))
        w!(g, "for (label, ys) in CURVES")
        w!(g, "$(IND)plot!(p, times, ys; label = label, lw = 2)")
        w!(g, "end")
        w!(g, "display(p)")
    end
    return nothing
end

function embedded_model!(g::CodeGen)
    g.embed || return nothing
    blank!(g)
    w!(g, "#= $MODEL_TAG")
    w!(g, "The studio reads this block to reopen the model exactly as it was.")
    w!(g, "Deleting it costs nothing but a best-effort re-reading of the code.")
    w!(g, py_json(to_dict(g.m)))
    w!(g, "=#")
    return nothing
end

"A parameter lens as the package writes it."
function lens_expr(l::Lens)
    k = l.kind
    k == "property" && return "property(:$(l.phase), $(l.property), $(py_str(l.index)))"
    k == "geometry" && return "geometry(:$(l.phase), :$(l.field_name), $(py_str(l.index)))"
    k == "shape_param" && return "shape_param(:$(l.field_name), $(py_str(l.index)))"
    k == "thickness" && return "thickness(:$(l.phase))"
    # the field names one scalar of the interface on top of layer `index`
    k == "interface_param" && return "interface_param($(py_str(l.index)), :$(l.field_name))"
    k == "nested" && return "nested(:$(l.member), $(l.property), $(lens_expr(lens_from_dict(either(l.inner, JDict())))))"
    return "amount(:$(l.phase))"
end

"""
    cell_expression(model, cell) -> String

The cell as one Julia expression, `let … end`, built from the statements of its
builder: the 3-D view draws what the script builds.
"""
function cell_expression(m::StudioModel, c::Cell)
    body = String[]
    for ln in cell_body(CodeGen(m; embed = false), c)
        isempty(strip(ln)) && continue
        stripped = lstrip(ln)
        if startswith(stripped, "return ")
            ln = ln[1:(ncodeunits(ln) - ncodeunits(stripped))] * stripped[(ncodeunits("return ") + 1):end]
        end
        push!(body, ln)
    end
    return "let\n" * join(body, "\n") * "\nend"
end

"""
    render_cell(model, cell) -> String

One builder as the script writes it: the read-back checks that a cell it
recognized would be written back unchanged.
"""
function render_cell(m::StudioModel, c::Cell)
    g = CodeGen(m; embed = false)
    builder!(g, c)
    return rstrip(join(g.out, "\n"))
end

"""
    extract_embedded(source) -> Union{Nothing, JDict}

The model a script written by the studio carries, or `nothing`.
"""
function extract_embedded(source::AbstractString)
    i = findfirst("#= $MODEL_TAG", source)
    i === nothing && return nothing
    j = findnext("\n=#", source, first(i))
    j === nothing && return nothing
    body = source[first(i):prevind(source, first(j))]
    brace = findfirst('{', body)
    brace === nothing && return nothing
    value = try
        parse_python_json(body[brace:end])
    catch err
        err isa ArgumentError || rethrow()
        return nothing
    end
    return value isa AbstractDict ? value : nothing
end
