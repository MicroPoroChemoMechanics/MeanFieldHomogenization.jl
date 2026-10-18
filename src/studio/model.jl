# The authoring model: what the page edits and what the code generation reads.
# It is a graph of cells, not a single RVE, because the package chains scales
# declaratively: a phase property may hold `Homogenized(inner_cell, scheme)`
# instead of a tensor, and the outer scheme resolves the inner scale when it
# reads the property. Read and written as JSON field for field as the Python
# studio did, so that the models embedded in the scripts it saved reopen
# unchanged.

# The fields typed `Any` hold a JSON value as the page sends it: a number, an
# integer kept apart from a float, or a string naming a parameter. The model is
# read once per request, never on a numerical path.

"A fresh cell id: eight hexadecimal digits."
new_id() = Random.randstring(Random.RandomDevice(), "0123456789abcdef", 8)

"""
The only schemes defined for a `Laminate`. `Laminated` is the exact periodic
solution; `Voigt` and `Reuss` apply because neither needs a matrix phase, and
they bracket it. Every other scheme needs a matrix and a reference medium.
"""
const LAMINATE_SCHEMES = ("Laminated", "Voigt", "Reuss")

"""
Schemes that act on a `ParticleAssembly`, a cell carrying the positions of the
inclusions, rather than on an `RVE`. The studio has no positional cell, so they
are left out of its catalog.
"""
const ASSEMBLY_SCHEMES = ("ClusterModel", "EquivalentInclusion")

"The outputs only an isotropic tensor has."
const ISOTROPIC_ONLY = ("k", "mu", "E", "nu")

"""
    py_str(x) -> String

`x` as a Python f-string interpolates it, for the messages and the code that
interpolate a value of the model as it stands: `None`, `True`, `1e-06`.
"""
py_str(x::AbstractString) = String(x)
py_str(::Nothing) = "None"
py_str(x::Bool) = x ? "True" : "False"
py_str(x::Integer) = string(x)
py_str(x::AbstractFloat) = py_repr(Float64(x))
py_str(x::AbstractVector) = "[" * join((py_item(v) for v in x), ", ") * "]"
py_str(x::AbstractDict) = "{" * join((py_item(k) * ": " * py_item(v) for (k, v) in x), ", ") * "}"
py_item(x::AbstractString) = "'" * replace(x, "\\" => "\\\\", "'" => "\\'") * "'"
py_item(x) = py_str(x)

"Is `x` a number to Python (a `bool` is one)?"
py_number(x) = x isa Union{Integer, AbstractFloat}

"The first element of `xs` satisfying `pred`, or `nothing`."
first_or_nothing(pred, xs) = (i = findfirst(pred, xs); i === nothing ? nothing : xs[i])

# ── Reading the fields of a JSON object ─────────────────────────────────────

"`d[key]` when `d` has it, else `default`: Python's `d.get(key, default)`."
field(d::AbstractDict, key::AbstractString, default) = haskey(d, key) ? d[key] : default

"""
    update_fields!(x, d) -> x

Set every field of `x` that `d` names, as Python's `Type(**{k: v for k in
fields})` does: what `d` leaves out keeps its default, what it gives is taken
as it is.
"""
function update_fields!(x::T, d::AbstractDict, skip = ()) where {T}
    for name in fieldnames(T)
        key = String(name)
        (key in skip || !haskey(d, key)) && continue
        setfield!(x, name, convert(fieldtype(T, name), d[key]))
    end
    return x
end

"Every field of `x`, as a JSON object."
function fields_dict(x::T) where {T}
    d = JDict()
    for name in fieldnames(T)
        d[String(name)] = getfield(x, name)
    end
    return d
end

# ── Values: a literal, a named parameter, or a nested scale ─────────────────

"""
    Property

One entry of a phase's property dictionary. `source` says where the value
comes from: `"builder"`, a tensor built from moduli (`iso_stiffness(k, μ)`);
`"expr"`, a Julia expression typed by the user; `"cell"`, the multiscale seam,
the effective property of another cell, written `Homogenized(cell, scheme)`.
`form` names the catalog entry that produced it (several share one builder);
`euler_angles` the frame its anisotropic constants are written in, which is
not the inclusion's.
"""
Base.@kwdef mutable struct Property
    key::String = ":C"
    source::String = "builder"
    builder::String = "iso_stiffness"
    form::String = "iso_kmu"
    args::JDict = jdict("k" => 10.0, "mu" => 5.0)
    expr::String = ""
    euler_angles::Vector{Any} = Any[]
    cell::Union{Nothing, String} = nothing
    scheme::Union{Nothing, String} = nothing
    scheme_options::JDict = JDict()
    visco::Union{Nothing, JDict} = nothing
end

property_from_dict(d::AbstractDict) = update_fields!(Property(), d)

"""
    Geometry

The shape of a phase: its `kind` (`spheroid`, `ellipsoid`, `cylinder`,
`penny_crack`, `elliptic_crack`, `ribbon_crack`, `layered_sphere`,
`layered_spheroid`), its sizes, its ZYZ angles, and the layers of a layered
inclusion (`radius` or `fraction`, `property`, `interface` each).
"""
Base.@kwdef mutable struct Geometry
    kind::String = "spheroid"
    args::JDict = jdict("omega" => 1.0)
    euler_angles::Vector{Any} = Any[]
    layers::Vector{Any} = Any[]
end

geometry_from_dict(d::AbstractDict) = update_fields!(Geometry(), d)

"""
    Phase

A phase of an RVE. `amount_kind` is `"fraction"`, `"density"` or `"rest"`, the
phase taking up the volume the others leave (its `amount` is then not read).
"""
Base.@kwdef mutable struct Phase
    name::String = "PHASE"
    geometry::Geometry = Geometry()
    properties::Vector{Property} = Property[]
    amount_kind::String = "fraction"
    amount::Any = 0.1
    symmetrize::String = "none"
end

function phase_from_dict(d::AbstractDict)
    return Phase(;
        name = field(d, "name", "PHASE"),
        # files written before v0.8 carry `is_matrix`, read as "rest"
        amount_kind = truthy(get(d, "is_matrix", nothing)) ? "rest" : field(d, "amount_kind", "fraction"),
        geometry = geometry_from_dict(field(d, "geometry", JDict())),
        properties = Property[property_from_dict(p) for p in field(d, "properties", Any[])],
        amount = field(d, "amount", 0.1),
        symmetrize = field(d, "symmetrize", "none"),
    )
end

"""
    Layer

A layer of a `Laminate`: a property dictionary, an amount (a `"fraction"` or a
`"thickness"`, the same for the whole stack), and the interface on top of it
(the last one closes the cell onto the first by periodicity).
"""
Base.@kwdef mutable struct Layer
    name::String = "A"
    properties::Vector{Property} = Property[]
    amount_kind::String = "fraction"
    amount::Any = 0.5
    interface::JDict = jdict("kind" => "PerfectInterface", "args" => JDict())
end

function layer_from_dict(d::AbstractDict)
    return Layer(;
        name = field(d, "name", "A"),
        properties = Property[property_from_dict(p) for p in field(d, "properties", Any[])],
        amount_kind = field(d, "amount_kind", "fraction"),
        amount = field(d, "amount", 0.5),
        interface = JDict(either(get(d, "interface", nothing), jdict("kind" => "PerfectInterface", "args" => JDict()))),
    )
end

"""
    tilted_normal(n) -> Bool

Does the normal `n` say anything other than e₃? A symbolic component counts as
tilted: nothing here can rule out that it turns the stack.
"""
function tilted_normal(n)
    vals = collect(Any, either(n, Any[]))
    isempty(vals) && return false
    (length(vals) != 3 || any(x -> x isa AbstractString, vals)) && return true
    return Float64[Float64(x) for x in vals] != [0.0, 0.0, 1.0]
end

"""
    Cell

One scale, an RVE (a matrix with inclusion phases) or a laminate (a periodic
stack of layers, solved exactly). The frame of a laminate is given by ZYZ
angles (`frame_mode = "euler"`) or by its normal. `builder_name` keeps the
name of the function a script read back used; new cells are `build_<name>`.
`ui` is where the page drew the cell in its graph of scales.
"""
Base.@kwdef mutable struct Cell
    id::String = new_id()
    name::String = "rve"
    kind::String = "rve"
    matrix_name::String = "MATRIX"
    phases::Vector{Phase} = Phase[]
    layers::Vector{Layer} = Layer[]
    frame_mode::String = "euler"
    normal::Vector{Any} = Any[0.0, 0.0, 1.0]
    euler_angles::Vector{Any} = Any[]
    params::Vector{String} = String[]
    builder_name::Union{Nothing, String} = nothing
    rve_options::JDict = JDict()
    ui::JDict = jdict("x" => 40, "y" => 40)
end

builder(c::Cell) = either(c.builder_name, "build_" * c.name)
is_laminate(c::Cell) = c.kind == "laminate"

"Is the stack normal to e₃, however its frame is given?"
function is_canonical_frame(c::Cell)
    c.frame_mode == "euler" && return !any(a -> a != "" && a !== nothing && a != 0, either(c.euler_angles, Any[]))
    return !tilted_normal(c.normal)
end

"The members carrying properties: the phases of an RVE, the layers of a laminate."
members(c::Cell) = is_laminate(c) ? c.layers : c.phases

"The phase taking up the volume the others leave, if the cell names one."
remainder(c::Cell) = first_or_nothing(p -> p.amount_kind == "rest", c.phases)
inclusions(c::Cell) = filter(p -> p.amount_kind != "rest", c.phases)

function cell_from_dict(d::AbstractDict)
    return Cell(;
        id = either(get(d, "id", nothing), new_id()),
        name = field(d, "name", "rve"),
        kind = field(d, "kind", "rve"),
        matrix_name = field(d, "matrix_name", "MATRIX"),
        phases = Phase[phase_from_dict(p) for p in field(d, "phases", Any[])],
        layers = Layer[layer_from_dict(l) for l in field(d, "layers", Any[])],
        # A model embedded before the frame had a mode carries only its normal;
        # reading it as "euler" would straighten a tilted stack.
        frame_mode = either(get(d, "frame_mode", nothing), tilted_normal(get(d, "normal", nothing)) ? "normal" : "euler"),
        normal = collect(Any, either(get(d, "normal", nothing), Any[0.0, 0.0, 1.0])),
        euler_angles = collect(Any, either(get(d, "euler_angles", nothing), Any[])),
        params = collect(String, field(d, "params", String[])),
        builder_name = get(d, "builder_name", nothing),
        rve_options = JDict(field(d, "rve_options", JDict())),
        ui = JDict(either(get(d, "ui", nothing), jdict("x" => 40, "y" => 40))),
    )
end

function to_dict(c::Cell)
    return jdict(
        "id" => c.id, "name" => c.name, "kind" => c.kind, "matrix_name" => c.matrix_name,
        "phases" => Any[to_dict(p) for p in c.phases], "layers" => Any[to_dict(l) for l in c.layers],
        "frame_mode" => c.frame_mode, "normal" => copy(c.normal), "euler_angles" => copy(c.euler_angles),
        "params" => copy(c.params), "builder_name" => c.builder_name,
        "rve_options" => copy(c.rve_options), "ui" => copy(c.ui),
    )
end

# ── Parameters, sweeps, outputs ─────────────────────────────────────────────

"""
    Param

A named constant, written `const name = value`. `origin` holds the text it had
in the file it was read from, written back as long as `edited` is false, so
that a constant laid out over several lines keeps its layout.
"""
Base.@kwdef mutable struct Param
    name::String = "k"
    value::String = "1.0"
    comment::String = ""
    origin::Union{Nothing, String} = nothing
    edited::Bool = false
end

param_from_dict(d::AbstractDict) = update_fields!(Param(), d)

"""
    Lens

A parameter of a sweep or a sensitivity, as the package's parameter lenses:
`amount`, `property`, `geometry`, `shape_param`, `thickness`,
`interface_param`, or `nested`, which reaches into an inner scale through
`inner`.
"""
Base.@kwdef mutable struct Lens
    kind::String = "amount"
    phase::String = ""
    property::String = ":C"
    field_name::String = "semi_axes"
    index::Any = 1
    member::String = ""
    inner::Union{Nothing, JDict} = nothing
end

lens_from_dict(d::AbstractDict) = update_fields!(Lens(), d)

"""
    Sweep

What to compute and what to plot: one homogenization with the amounts entered
(`mode = "single"`), or one lens swept over a range for several schemes.
`outputs` are specs (`k`, `mu`, `E`, `nu`, a Kelvin-Mandel component `km`, a
component `comp`, `trace3`); `clamp_zero` clips negative values for display.
"""
Base.@kwdef mutable struct Sweep
    enabled::Bool = false
    mode::String = "sweep"
    variable::String = "φ"
    start::Any = 0.0
    stop::Any = 1.0
    length::Any = 21
    lens::Lens = Lens()
    cell::Union{Nothing, String} = nothing
    schemes::Vector{Any} = Any[jdict("name" => "MoriTanaka", "options" => JDict())]
    property::String = ":C"
    projection::String = "none"
    outputs::Vector{Any} = Any[jdict("kind" => "k"), jdict("kind" => "mu")]
    clamp_zero::Bool = false
    plot::Bool = true
end

"An output spec: a bare kind, as older studios wrote it, or a spec already."
output_spec(o::AbstractString) = jdict("kind" => o)
output_spec(o::AbstractDict) = JDict(o)

needs_isotropy(s::Sweep) = any(o -> get(o, "kind", nothing) in ISOTROPIC_ONLY, s.outputs)

function to_dict(s::Sweep)
    d = fields_dict(s)
    d["lens"] = to_dict(s.lens)
    return d
end

function sweep_from_dict(d::AbstractDict)
    s = update_fields!(Sweep(), d, ("lens", "schemes", "outputs"))
    s.lens = lens_from_dict(field(d, "lens", JDict()))
    # Models written before schemes became a list, and before outputs were
    # specs, still open: migrated rather than lost.
    schemes = get(d, "schemes", nothing)
    truthy(schemes) || (schemes = Any[jdict("name" => either(get(d, "scheme", nothing), "MoriTanaka"), "options" => JDict(either(get(d, "scheme_options", nothing), JDict())))])
    s.schemes = Any[jdict("name" => field(x, "name", "MoriTanaka"), "options" => JDict(either(get(x, "options", nothing), JDict()))) for x in schemes]
    outs = Any[output_spec(o) for o in either(get(d, "outputs", nothing), Any[])]
    s.outputs = isempty(outs) ? Any[jdict("kind" => "k"), jdict("kind" => "mu")] : outs
    return s
end

"""
    Sens

Sensitivities of the effective property by automatic differentiation: a
`derivative` with respect to one lens, the `gradient` of one scalar with
respect to several, or the `jacobian` of the whole tensor.
"""
Base.@kwdef mutable struct Sens
    enabled::Bool = false
    kind::String = "derivative"
    cell::Union{Nothing, String} = nothing
    scheme::String = "MoriTanaka"
    scheme_options::JDict = JDict()
    property::String = ":C"
    lenses::Vector{Lens} = Lens[Lens()]
    output::JDict = jdict("kind" => "k")
    projection::String = "iso"
end

needs_isotropy(s::Sens) = get(s.output, "kind", nothing) in ISOTROPIC_ONLY && s.projection == "none"

function to_dict(s::Sens)
    d = fields_dict(s)
    d["lenses"] = Any[to_dict(l) for l in s.lenses]
    return d
end

function sens_from_dict(d::AbstractDict)
    s = update_fields!(Sens(), d, ("lenses", "output"))
    lenses = Lens[lens_from_dict(x) for x in either(get(d, "lenses", nothing), Any[])]
    s.lenses = isempty(lenses) ? Lens[Lens()] : lenses
    out = get(d, "output", nothing)
    s.output = output_spec(either(out, jdict("kind" => "k")))
    # a jacobian flattens the whole tensor; a derivative takes one parameter
    s.kind == "derivative" && (s.lenses = s.lenses[1:1])
    return s
end

"""
    Alv

Aging linear viscoelasticity: the time grid, the cell and scheme, and the
Kelvin-Mandel component of the creep operator to follow ((1, 1) is the
uniaxial response).
"""
Base.@kwdef mutable struct Alv
    enabled::Bool = false
    t_start::Any = 0.0
    t_stop::Any = 10.0
    length::Any = 41
    log_time::Bool = false
    cell::Union{Nothing, String} = nothing
    scheme::String = "MoriTanaka"
    property::String = ":C"
    component::Any = Any[1, 1]
    plot::Bool = true
end

function alv_from_dict(d::AbstractDict)
    a = update_fields!(Alv(), d)
    component = collect(Any, either(a.component, Any[1, 1]))
    a.component = either(component[1:min(2, length(component))], Any[1, 1])
    return a
end

"""
    OpaqueBlock

Source the studio did not recognize, kept verbatim and written back byte for
byte at its place (`order`), with a `note` saying why.
"""
Base.@kwdef mutable struct OpaqueBlock
    source::String = ""
    order::Any = 0
    note::String = ""
end

opaque_from_dict(d::AbstractDict) = update_fields!(OpaqueBlock(), d)

to_dict(x::Union{Property, Geometry, Param, Lens, Alv, OpaqueBlock}) = fields_dict(x)

function to_dict(p::Phase)
    return jdict(
        "name" => p.name, "geometry" => to_dict(p.geometry),
        "properties" => Any[to_dict(x) for x in p.properties],
        "amount_kind" => p.amount_kind, "amount" => p.amount, "symmetrize" => p.symmetrize,
    )
end

function to_dict(l::Layer)
    return jdict(
        "name" => l.name, "properties" => Any[to_dict(x) for x in l.properties],
        "amount_kind" => l.amount_kind, "amount" => l.amount, "interface" => l.interface,
    )
end

# ── The whole model ─────────────────────────────────────────────────────────

"""
    StudioModel

The whole model: its parameters, its cells, what to compute (a sweep, a
sensitivity or an aging viscoelastic run), the code kept verbatim, and the
cell carrying the final result (`root_cell`, else the last one).
"""
Base.@kwdef mutable struct StudioModel
    title::String = "model"
    description::String = ""
    params::Vector{Param} = Param[]
    cells::Vector{Cell} = Cell[]
    sweep::Sweep = Sweep()
    sens::Sens = Sens()
    alv::Alv = Alv()
    opaque::Vector{OpaqueBlock} = OpaqueBlock[]
    root_cell::Union{Nothing, String} = nothing
end

cell(::StudioModel, ::Nothing) = nothing
cell(m::StudioModel, id) = first_or_nothing(c -> c.id == id, m.cells)
cell_by_name(m::StudioModel, name::AbstractString) = first_or_nothing(c -> c.name == name, m.cells)

root(m::StudioModel) = something(cell(m, m.root_cell), isempty(m.cells) ? nothing : last(m.cells), Some(nothing))

"""
    dependencies(m, c) -> Vector{String}

The ids of the cells `c` reads through a `Homogenized` seam, over its members:
a layer carries the seam exactly as a phase does.
"""
dependencies(::StudioModel, c::Cell) =
    String[pr.cell for mb in members(c) for pr in mb.properties if pr.source == "cell" && truthy(pr.cell)]

"""
    topological_order(m) -> Vector{Cell}

The cells, each after the ones it reads. A cycle is refused with an
`ArgumentError` naming it: a scale cannot be built from itself.
"""
function topological_order(m::StudioModel)
    state = Dict{String, Symbol}()
    order = Cell[]
    function visit(id::String, trail::Vector{String})
        st = get(state, id, :new)
        st === :done && return nothing
        if st === :visiting
            names = [(c = cell(m, x); c === nothing ? x : c.name) for x in vcat(trail, id)]
            start = findfirst(==(last(names)), names)
            throw(ArgumentError("multiscale cycle: " * join(names[start:end], " → ")))
        end
        c = cell(m, id)
        c === nothing && return nothing
        state[id] = :visiting
        for dep in dependencies(m, c)
            visit(dep, vcat(trail, id))
        end
        state[id] = :done
        push!(order, c)
        return nothing
    end
    for c in m.cells
        visit(c.id, String[])
    end
    return order
end

uses_multiscale(m::StudioModel) = any(c -> !isempty(dependencies(m, c)), m.cells)

"Each value of `xs` appearing more than once, in the order of first appearance."
repeated(xs) = unique(filter(x -> count(==(x), xs) > 1, xs))

"""
    validate(m) -> Vector{String}

The problems worth blocking on, each a sentence; the script lists them in its
header.
"""
function validate(m::StudioModel)
    problems = String[]
    try
        topological_order(m)
    catch err
        err isa ArgumentError || rethrow()
        push!(problems, err.msg)
    end
    for n in repeated([c.name for c in m.cells])
        push!(problems, "two cells are both named `$n`")
    end
    for c in m.cells
        if is_laminate(c)
            append!(problems, laminate_problems(c))
        elseif remainder(c) === nothing
            push!(problems, "cell `$(c.name)` has no matrix phase")
        else
            # the matrix is the phase whose amount is the rest; two of them would
            # leave one out of the script without a word
            rest = [p.name for p in c.phases if p.amount_kind == "rest"]
            length(rest) > 1 && push!(problems, "cell `$(c.name)` has $(length(rest)) matrix phases ($(join(rest, ", "))); a cell has one")
        end
        what = is_laminate(c) ? "layers" : "phases"
        for n in repeated([mb.name for mb in members(c)])
            push!(problems, "cell `$(c.name)` has two $what named `$n`")
        end
    end
    append!(problems, scheme_problems(m))
    # `k_mu` and `E_nu` have methods for TensISO only; a laminate of several
    # layers is transversely isotropic even when every layer is isotropic
    anisotropic = any(ph -> ph.symmetrize == "none", (ph for c in m.cells for ph in c.phases)) ||
        any(c -> is_laminate(c) && length(c.layers) > 1, m.cells)
    if m.sweep.enabled && needs_isotropy(m.sweep) && m.sweep.projection == "none" && anisotropic
        push!(
            problems,
            "k, μ, E and ν are only defined for an isotropic result. This " *
                "model has phases with no orientation average, so the effective " *
                "tensor need not be isotropic: pick a reporting projection, or " *
                "plot Kelvin-Mandel components instead."
        )
    end
    m.sens.enabled && append!(problems, sens_problems(m))
    # an inner Homogenized cannot sit inside an aging viscoelastic chain: the
    # inner result would have to be re-expressed as a ViscoLaw
    if m.alv.enabled && uses_multiscale(m)
        push!(
            problems,
            "aging viscoelasticity cannot be combined with a nested scale: " *
                "MeanFieldHomogenization cannot re-express a homogenized inner result as a " *
                "ViscoLaw"
        )
    end
    return problems
end

"What the package refuses about a stack, said before the run."
function laminate_problems(c::Cell)
    problems = String[]
    isempty(c.layers) && return push!(problems, "laminate `$(c.name)` has no layer")
    kinds = unique(l.amount_kind for l in c.layers)
    if length(kinds) > 1
        push!(
            problems,
            "laminate `$(c.name)` mixes absolute thicknesses with volume " *
                "fractions. MeanFieldHomogenization refuses that: with an " *
                "imperfect interface the period carries the size effect, so a " *
                "half-specified stack is ambiguous."
        )
    elseif kinds == ["fraction"]
        # the package checks Σf ≈ 1 rather than rescaling silently
        vals = [l.amount for l in c.layers]
        if all(py_number, vals)
            total = sum(vals)
            abs(total - 1.0) > 1.0e-8 && push!(problems, "the fractions of laminate `$(c.name)` sum to $(Printf.@sprintf("%.6g", Float64(total))), not 1")
        end
    end
    for l in c.layers
        py_number(l.amount) && l.amount < 0 && push!(problems, "layer `$(l.name)` has a negative $(l.amount_kind)")
    end
    c.frame_mode == "normal" && all(x -> py_number(x) && x == 0, either(c.normal, Any[])) &&
        push!(problems, "laminate `$(c.name)` has a null normal")
    return problems
end

function sens_problems(m::StudioModel)
    s = m.sens
    problems = String[]
    isempty(s.lenses) && push!(problems, "the sensitivity needs at least one parameter")
    s.kind == "derivative" && length(s.lenses) != 1 &&
        push!(problems, "a derivative is with respect to one parameter; use a gradient for several")
    target = something(cell(m, s.cell), root(m), Some(nothing))
    if target !== nothing && is_laminate(target) && any(l -> l.kind == "amount", s.lenses)
        push!(
            problems,
            "a laminate has no phase amount: differentiate a layer " *
                "thickness instead (changing hᵢ also moves the period, " *
                "which is what carries the interface size effect)."
        )
    end
    i = findfirst(l -> l.kind in ("thickness", "interface_param") && (target === nothing || !is_laminate(target)), s.lenses)
    i === nothing || push!(problems, "the `$(s.lenses[i].kind)` parameter only exists on a laminate")
    if s.kind != "jacobian" && needs_isotropy(s)
        free = target !== nothing && (
            (is_laminate(target) && length(target.layers) > 1) || any(ph -> ph.symmetrize == "none", target.phases)
        )
        free && push!(
            problems,
            "k, μ, E and ν are only defined for an isotropic result. " *
                "Pick a reporting projection, or differentiate a " *
                "Kelvin-Mandel component instead."
        )
    end
    return problems
end

"Schemes asked of a cell that does not support them."
function scheme_problems(m::StudioModel)
    problems = String[]
    function check(id, name, place)
        c = something(cell(m, id), root(m), Some(nothing))
        (c === nothing || !is_laminate(c)) && return nothing
        name in LAMINATE_SCHEMES && return nothing
        push!(
            problems,
            "$place: `$(py_str(name))` needs a matrix phase, so it does not " *
                "apply to the laminate `$(c.name)`. A laminate takes " *
                join(("`$s`" for s in LAMINATE_SCHEMES), ", ") * "."
        )
        return nothing
    end
    if m.sweep.enabled
        for s in m.sweep.schemes
            check(m.sweep.cell, field(s, "name", "MoriTanaka"), "Sweep")
        end
    end
    if m.alv.enabled
        check(m.alv.cell, m.alv.scheme, "Viscoelastic")
        # `laminate_alv` builds the order-2 kernel in the canonical frame
        c = something(cell(m, m.alv.cell), root(m), Some(nothing))
        if c !== nothing && is_laminate(c) && m.alv.property != ":C" && !is_canonical_frame(c)
            push!(
                problems,
                "aging viscoelasticity of a laminate in transport (`:K`) " *
                    "requires the canonical frame: MeanFieldHomogenization " *
                    "builds that kernel with the normal along e₃."
            )
        end
    end
    m.sens.enabled && check(m.sens.cell, m.sens.scheme, "Sensitivity")
    # every property a `Homogenized` reads carries its own scheme
    for c in m.cells, mb in members(c), pr in mb.properties
        pr.source == "cell" && truthy(pr.cell) &&
            check(pr.cell, either(pr.scheme, "MoriTanaka"), "`$(c.name)`/`$(mb.name)`$(pr.key)")
    end
    return problems
end

function to_dict(m::StudioModel)
    return jdict(
        "title" => m.title, "description" => m.description,
        "params" => Any[to_dict(p) for p in m.params], "cells" => Any[to_dict(c) for c in m.cells],
        "sweep" => to_dict(m.sweep), "sens" => to_dict(m.sens), "alv" => to_dict(m.alv),
        "opaque" => Any[to_dict(o) for o in m.opaque], "root_cell" => m.root_cell,
    )
end

"""
    model_from_dict(d) -> StudioModel

The model a JSON object describes, as the page sends it or a script embeds it;
what it leaves out takes its default, and the forms of older studios are
migrated.
"""
function model_from_dict(d::AbstractDict)
    return StudioModel(;
        title = field(d, "title", "model"),
        description = field(d, "description", ""),
        params = Param[param_from_dict(p) for p in field(d, "params", Any[])],
        cells = Cell[cell_from_dict(c) for c in field(d, "cells", Any[])],
        opaque = OpaqueBlock[opaque_from_dict(o) for o in field(d, "opaque", Any[])],
        root_cell = get(d, "root_cell", nothing),
        sweep = sweep_from_dict(field(d, "sweep", JDict())),
        sens = sens_from_dict(field(d, "sens", JDict())),
        alv = alv_from_dict(field(d, "alv", JDict())),
    )
end

"The porous benchmark, the shortest useful model to open on."
function default_model()
    solid = Phase(;
        name = "SOLID", amount_kind = "rest",
        properties = [Property(; key = ":C", form = "iso_kmu", args = jdict("k" => 72.0, "mu" => 32.0))],
    )
    pore = Phase(;
        name = "PORE", amount_kind = "fraction", amount = 0.1,
        properties = [Property(; key = ":C", form = "void", args = jdict("k" => 1.0e-6, "mu" => 1.0e-6))],
    )
    c = Cell(; name = "rve", matrix_name = "SOLID", phases = [solid, pore])
    m = StudioModel(; title = "porous_benchmark", cells = [c], root_cell = c.id)
    m.sweep = Sweep(;
        enabled = true, mode = "sweep", variable = "φ", start = 0.0, stop = 0.9, length = 19,
        lens = Lens(; kind = "amount", phase = "PORE"), cell = c.id,
        projection = "iso",
    )
    return m
end

"The 30/70 bilayer: a stiff layer and a compliant one, transversely isotropic together."
default_layers() = [
    Layer(; name = "A", amount = 0.3, properties = [Property(; args = jdict("k" => 2.0, "mu" => 0.8))]),
    Layer(; name = "B", amount = 0.7, properties = [Property(; args = jdict("k" => 0.5, "mu" => 0.2))]),
]
