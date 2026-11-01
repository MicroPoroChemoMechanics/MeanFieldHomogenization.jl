# The 3-D view: the shape of a phase, or the stack of a laminate, as Plotly
# traces. The shape is built from the model by the package's own constructors,
# so that the picture shows what the package builds (a zero semi-axis becomes a
# crack, an infinite one a cylinder), and nothing typed in the page is ever run:
# a size or an angle is a number, or an arithmetic expression of numbers and of
# the model's parameters. The surface parametrizations are those that draw the
# figures of the documentation (scripts/common/docviz.jl).

# ── Numbers the page may type ───────────────────────────────────────────────

const ARITHMETIC = Dict{Symbol, Function}(
    :+ => +, :- => -, :* => *, :/ => /, :^ => ^, :sqrt => sqrt, :cbrt => cbrt,
    :sin => sin, :cos => cos, :tan => tan, :asin => asin, :acos => acos, :atan => atan,
    :exp => exp, :log => log, :deg2rad => deg2rad,
)
const NAMED_NUMBERS = Dict{Symbol, Float64}(:π => π, :pi => π, :ℯ => ℯ, :Inf => Inf)

"""
    numeric(m, x) -> Union{Nothing, Float64}

The value of a size or an angle of the model: a number, or the text of an
arithmetic expression of numbers, `π`, and the model's parameters (`const`
values, read the same way); `nothing` for anything else, which is not drawn.
"""
numeric(m::StudioModel, x; depth::Int = 0) = numeric_value(m, x, depth)
numeric_value(::StudioModel, x::Real, ::Int) = Float64(x)
function numeric_value(m::StudioModel, x::AbstractString, depth::Int)
    ex = Meta.parse(x; raise = false)
    return ex isa Expr && ex.head in (:error, :incomplete) ? nothing : numeric_value(m, ex, depth)
end
function numeric_value(m::StudioModel, x::Symbol, depth::Int)
    haskey(NAMED_NUMBERS, x) && return NAMED_NUMBERS[x]
    depth < 8 || return nothing                     # parameters defined in a cycle
    p = first_or_nothing(p -> p.name == String(x), m.params)
    return p === nothing ? nothing : numeric_value(m, p.value, depth + 1)
end
function numeric_value(m::StudioModel, x::Expr, depth::Int)
    (x.head === :call && x.args[1] isa Symbol && haskey(ARITHMETIC, x.args[1])) || return nothing
    vals = [numeric_value(m, a, depth) for a in x.args[2:end]]
    any(isnothing, vals) && return nothing
    return Float64(ARITHMETIC[x.args[1]](vals...))
end
numeric_value(::StudioModel, x, ::Int) = nothing

"The angles of a shape or a stack as numbers (an empty entry skipped), or `nothing`."
function numeric_angles(m::StudioModel, angles)
    vals = [numeric(m, a) for a in angles if a != ""]
    return any(isnothing, vals) ? nothing : Tuple(Float64.(vals))
end

# ── The shapes, built from the model ────────────────────────────────────────

"Keyword arguments for the angles: none for none."
angle_keywords(angles::Tuple) = isempty(angles) ? (;) : (; euler_angles = angles)

"""
    shape(m, phase) -> Union{Nothing, geometry}

The inclusion a phase describes, built by the package's constructors, or
`nothing` when one of its sizes or angles is not a number. The moduli of a
layered inclusion are left out of the picture: placeholders of the right order
stand for them.
"""
const SHAPE_SIZES = Dict(
    "spheroid" => ("omega",), "ellipsoid" => ("a", "b", "c"), "cylinder" => ("b", "c"),
    "penny_crack" => ("a",), "elliptic_crack" => ("a", "b"), "ribbon_crack" => ("b",),
)

function shape(m::StudioModel, ph::Phase)
    geo = ph.geometry
    k = geo.kind
    # a kind the forms do not know is written `Spheroid(1.0)`, and drawn so
    (haskey(SHAPE_SIZES, k) || k in ("layered_sphere", "layered_spheroid")) || return MeanFieldHomogenization.Spheroid(1.0)
    arg(key, default) = numeric(m, get(geo.args, key, default))
    angles = numeric_angles(m, geo.euler_angles)
    angles === nothing && return nothing
    kw = angle_keywords(angles)
    if k == "layered_sphere"
        radii = [numeric(m, get(l, "radius", 1.0)) for l in geo.layers]
        any(isnothing, radii) && return nothing
        isempty(radii) && return nothing
        moduli = Tuple(MeanFieldHomogenization.iso_stiffness(1.0, 1.0) for _ in radii)
        return MeanFieldHomogenization.LayeredSphere(Tuple(Float64.(radii)), moduli)
    elseif k == "layered_spheroid"
        fractions = [numeric(m, get(l, "fraction", 1.0)) for l in geo.layers]
        omega, radius, ns = arg("omega", 0.5), arg("radius", 1.0), arg("Nseries", 5)
        (any(isnothing, fractions) || isempty(fractions) || any(isnothing, (omega, radius, ns))) && return nothing
        # the order of the series is a count
        (isinteger(ns) && 1 <= ns <= 1000) || return nothing
        axis = isempty(angles) ? (0.0, 0.0, 1.0) :
            Tuple(MeanFieldHomogenization.TensND.vecbasis(MeanFieldHomogenization.TensND.RotatedBasis(vcat(collect(angles), zeros(3))[1:3]...))[:, 3])
        moduli = Tuple(MeanFieldHomogenization.TensND.TensISO{3}(1.0) for _ in fractions)
        return MeanFieldHomogenization.layered_spheroid_from_fractions(
            omega, radius, Tuple(Float64.(fractions)), moduli; Nseries = round(Int, ns), axis,
        )
    end
    defaults = Dict("omega" => 1.0, "a" => 1.0, "b" => k == "elliptic_crack" ? 0.5 : 1.0, "c" => 1.0)
    vals = [arg(s, defaults[s]) for s in SHAPE_SIZES[k]]
    any(isnothing, vals) && return nothing
    ctor = Dict(
        "spheroid" => MeanFieldHomogenization.Spheroid, "ellipsoid" => MeanFieldHomogenization.Ellipsoid,
        "cylinder" => MeanFieldHomogenization.Cylinder, "penny_crack" => MeanFieldHomogenization.PennyCrack,
        "elliptic_crack" => MeanFieldHomogenization.EllipticCrack, "ribbon_crack" => MeanFieldHomogenization.RibbonCrack,
    )[k]
    return ctor(vals...; kw...)
end

"""
    stack(m, cell) -> Union{Nothing, Laminate}

The stack of a laminate cell, its frame and its layers' amounts: what the
picture shows. Properties and interfaces do not change it and are left out,
so that a layer reading an inner scale is drawn too.
"""
function stack(m::StudioModel, c::Cell)
    if c.frame_mode == "euler"
        angles = numeric_angles(m, c.euler_angles)
        angles === nothing && return nothing
        lam = isempty(padded_angles(c.euler_angles)) ? MeanFieldHomogenization.Laminate() :
            MeanFieldHomogenization.Laminate(; euler_angles = angles)
    else
        n = [numeric(m, x) for x in either(c.normal, Any[])]
        any(isnothing, n) && return nothing
        lam = is_canonical_normal(c.normal) ? MeanFieldHomogenization.Laminate() :
            MeanFieldHomogenization.Laminate(; normal = Tuple(Float64.(n)))
    end
    for layer in c.layers
        amount = numeric(m, layer.amount)
        amount === nothing && return nothing
        kw = layer.amount_kind == "thickness" ? (; thickness = amount) : (; fraction = amount)
        MeanFieldHomogenization.add_layer!(lam, Symbol(layer.name), Dict{Symbol, Any}(); kw...)
    end
    return lam
end

# ── Surfaces (the parametrizations of scripts/common/docviz.jl) ─────────────

"Sample `f(u, v) -> (x, y, z)` on the grid `us × vs` into three matrices."
function param_surface(f, us, vs)
    X = [f(u, v)[1] for u in us, v in vs]
    Y = [f(u, v)[2] for u in us, v in vs]
    Z = [f(u, v)[3] for u in us, v in vs]
    return X, Y, Z
end

"The ellipsoid of semi-axes `(a, b, c)`, rotated by `R` (columns: the local axes)."
function ellipsoid_surface(a, b, c; R = Matrix{Float64}(I, 3, 3), nu::Integer = 33, nv::Integer = 17)
    f(u, v) = Tuple(R * [a * cos(v) * cos(u), b * cos(v) * sin(u), c * sin(v)])
    return param_surface(f, range(0, 2π; length = nu), range(-π / 2, π / 2; length = nv))
end

"The lateral surface of the elliptic cylinder of semi-axes `(b, c)` and length `L` along e₁."
function cylinder_surface(b, c, L; R = Matrix{Float64}(I, 3, 3), nu::Integer = 33, nv::Integer = 2)
    f(u, t) = Tuple(R * [L * t, b * cos(u), c * sin(u)])
    return param_surface(f, range(0, 2π; length = nu), range(-0.5, 0.5; length = max(nv, 2)))
end

"The flat elliptical disc of semi-axes `(a, b)` in the local (e₁, e₂) plane: a crack."
function disc_surface(a, b; R = Matrix{Float64}(I, 3, 3), nu::Integer = 49, nv::Integer = 5)
    f(u, r) = Tuple(R * [a * r * cos(u), b * r * sin(u), 0.0])
    return param_surface(f, range(0, 2π; length = nu), range(0, 1; length = nv))
end

"A blue-to-warm ramp, readable in the light and the dark theme."
function layer_colors(n::Integer)
    base = ["#1f4e9c", "#4a90d9", "#7ec8e3", "#f0ad4e", "#c0392b", "#8a6d3b"]
    return n ≤ length(base) ? base[1:n] : [base[mod1(k, length(base))] for k in 1:n]
end

# ── Traces ──────────────────────────────────────────────────────────────────

const FACE = "#4a90d9"
const CRACK = "#8a6d3b"
const GUIDE = "#c0392b"

grid_rows(M) = [collect(Float64, @view M[i, :]) for i in axes(M, 1)]

"A Plotly `surface` trace."
function surface(X, Y, Z; color = FACE, opacity = 0.55, name = "")
    d = Dict{String, Any}(
        "type" => "surface", "x" => grid_rows(X), "y" => grid_rows(Y), "z" => grid_rows(Z),
        "opacity" => opacity, "showscale" => false, "colorscale" => [[0, color], [1, color]],
        "contours" => Dict(a => Dict("highlight" => false) for a in ("x", "y", "z")),
        "hoverinfo" => "skip",
    )
    isempty(name) || (d["name"] = name; d["showlegend"] = true)
    return d
end

"A line through the columns of `P`: scenery, kept out of the legend."
function polyline(P::AbstractMatrix; color = GUIDE, width = 4, dash = "solid")
    return Dict{String, Any}(
        "type" => "scatter3d", "mode" => "lines",
        "x" => collect(Float64, @view P[1, :]), "y" => collect(Float64, @view P[2, :]), "z" => collect(Float64, @view P[3, :]),
        "line" => Dict("color" => color, "width" => width, "dash" => dash),
        "hoverinfo" => "skip", "showlegend" => false,
    )
end

function labels3d(P::AbstractMatrix, texts; color = GUIDE)
    return Dict{String, Any}(
        "type" => "scatter3d", "mode" => "text",
        "x" => collect(Float64, @view P[1, :]), "y" => collect(Float64, @view P[2, :]), "z" => collect(Float64, @view P[3, :]),
        "text" => collect(String, texts), "textfont" => Dict("color" => color, "size" => 13),
        "hoverinfo" => "skip", "showlegend" => false,
    )
end

"The axes of the frame `R`, each drawn to its length (an infinite one to 3) and named."
function axis_guides(R, lengths, names)
    out = Dict{String, Any}[]
    tips = Vector{Float64}[]
    for (i, (L, nm)) in enumerate(zip(lengths, names))
        isfinite(L) || (L = 3.0)
        e = R[:, i] .* L
        push!(out, polyline(hcat(zeros(3), e); color = GUIDE, width = 3))
        push!(tips, e .* 1.12)
    end
    push!(out, labels3d(reduce(hcat, tips), collect(String, names)))
    return out
end

"The frame of a shape or a stack: the columns are its axes in the global frame."
frame_matrix(x::Union{MeanFieldHomogenization.Ellipsoid, MeanFieldHomogenization.Cylinder, MeanFieldHomogenization.EllipticCrack, MeanFieldHomogenization.RibbonCrack, MeanFieldHomogenization.Laminate}) =
    Matrix{Float64}(MeanFieldHomogenization.TensND.vecbasis(x.basis))

# A layered spheroid stores its revolution axis, not a basis: a frame whose
# third vector is that axis draws it, the transverse plane being isotropic.
function frame_matrix(ls::MeanFieldHomogenization.LayeredSpheroid)
    e3 = normalize(Float64.(collect(ls.axis)))
    t = zeros(3)
    t[argmin(abs.(e3))] = 1.0       # the canonical vector least aligned with e3
    e1 = normalize(cross(t, e3))
    return hcat(e1, cross(e3, e1), e3)
end

"""
    traces(geometry; cutaway, guides) -> Vector{Dict}

The traces of a shape, of a layered inclusion (in a cut-away view, which
alone shows the inner layers) or of a laminate's stack.
"""
function traces end

function traces(ell::MeanFieldHomogenization.Ellipsoid{3}; guides::Bool = true, kw...)
    a, b, c = Float64.(ell.semi_axes)
    R = frame_matrix(ell)
    out = Dict{String, Any}[surface(ellipsoid_surface(a, b, c; R)...)]
    guides && append!(out, axis_guides(R, (a, b, c), ("a", "b", "c")))
    return out
end

function traces(cyl::MeanFieldHomogenization.Cylinder; guides::Bool = true, length_shown = 6.0, kw...)
    b, c = Float64.(cyl.semi_axes)
    R = frame_matrix(cyl)
    out = Dict{String, Any}[surface(cylinder_surface(b, c, length_shown; R)...)]
    guides && append!(out, axis_guides(R, (length_shown / 2, b, c), ("L → ∞", "b", "c")))
    return out
end

function crack_traces(a, b, R; guides::Bool = true, kw...)
    out = Dict{String, Any}[surface(disc_surface(a, b; R)...; color = CRACK, opacity = 0.9)]
    n = R[:, 3] .* max(a, b) .* 1.5
    push!(out, polyline(hcat(zeros(3), n); color = GUIDE, width = 5))
    push!(out, labels3d(reshape(n .* 1.12, 3, 1), ["n"]))
    guides && append!(out, axis_guides(R, (a, b, max(a, b) / 2), ("a", "b", "")))
    return out
end

traces(cr::MeanFieldHomogenization.EllipticCrack; kw...) = crack_traces(Float64(cr.a), Float64(cr.b), frame_matrix(cr); kw...)
traces(cr::MeanFieldHomogenization.RibbonCrack; length_shown = 6.0, kw...) =
    crack_traces(length_shown / 2, Float64(cr.b), frame_matrix(cr); kw...)

function traces(ls::MeanFieldHomogenization.LayeredSphere; cutaway::Bool = true, kw...)
    radii = Float64.(collect(ls.radii))
    n = length(radii)
    colors = layer_colors(n)
    out = Dict{String, Any}[]
    # a cut-away keeps half of each shell, so that the inner ones show
    ulim = cutaway ? 1.0π : 2.0π
    for (i, r) in enumerate(radii)
        r > 0 || continue
        f(u, v) = (r * cos(v) * cos(u), r * cos(v) * sin(u), r * sin(v))
        X, Y, Z = param_surface(f, range(0, ulim; length = 41), range(-π / 2, π / 2; length = 21))
        push!(out, surface(X, Y, Z; color = colors[i], opacity = i == n ? 0.55 : 0.9, name = "layer $i (r = $(round(r; digits = 3)))"))
    end
    return out
end

function traces(ls::MeanFieldHomogenization.LayeredSpheroid; cutaway::Bool = true, guides::Bool = false, kw...)
    out = Dict{String, Any}[]
    semi = Float64.(collect(MeanFieldHomogenization.outer_semiaxes(ls)))
    n = MeanFieldHomogenization.layer_count(ls)
    colors = layer_colors(n)
    ulim = cutaway ? 1.0π : 2.0π
    R = frame_matrix(ls)
    for i in 1:n
        c, aeq = Float64.(MeanFieldHomogenization.layer_semiaxes(ls, i))
        # written in the spheroid's frame (revolution axis along e₃), then rotated
        f(u, v) = Tuple(R * [aeq * cos(v) * cos(u), aeq * cos(v) * sin(u), c * sin(v)])
        X, Y, Z = param_surface(f, range(0, ulim; length = 41), range(-π / 2, π / 2; length = 21))
        push!(out, surface(X, Y, Z; color = colors[i], opacity = i == n ? 0.55 : 0.9, name = "layer $i"))
    end
    guides && append!(out, axis_guides(R, (semi[end], semi[end], semi[1]), ("ρt", "ρt", "ρa")))
    return out
end

"""
One layer of a stack: a slab of thickness `h`, its faces normal to the third
column of `R`, its lower face at `z0` along it, drawn as one closed `mesh3d`
(a stack of transparent surfaces reads as a muddle of planes).
"""
function slab(R, z0, h, half, ymax; color = FACE, opacity = 0.75, name = "")
    corners = [R * [x, y, z] for z in (z0, z0 + h) for (x, y) in ((-half, -half), (half, -half), (half, ymax), (-half, ymax))]
    P = reduce(hcat, corners)
    d = Dict{String, Any}(
        "type" => "mesh3d",
        "x" => collect(Float64, @view P[1, :]), "y" => collect(Float64, @view P[2, :]), "z" => collect(Float64, @view P[3, :]),
        # two triangles per face, bottom vertices 0-3, top 4-7, in the same order
        "i" => [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3],
        "j" => [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4],
        "k" => [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7],
        "color" => color, "opacity" => opacity, "flatshading" => true, "hoverinfo" => "skip",
    )
    isempty(name) || (d["name"] = name; d["showlegend"] = true)
    return d
end

"""
One period of the stack and its normal; the cut-away halves the stack along one
direction of its plane, exposing the cross-section where the layers show.
"""
function traces(lam::MeanFieldHomogenization.Laminate; cutaway::Bool = true, guides::Bool = true, kw...)
    names = MeanFieldHomogenization.layer_names(lam)
    isempty(names) && return Dict{String, Any}[]
    R = frame_matrix(lam)
    hs = Float64[Float64(MeanFieldHomogenization.layer_thickness(lam, nm)) for nm in names]
    L = sum(hs)
    L > 0 || return Dict{String, Any}[]
    # the layers are unbounded in their plane: the extent drawn follows the period
    half = 0.6 * L
    colors = layer_colors(length(names))
    out = Dict{String, Any}[]
    z = -L / 2
    for (i, (nm, h)) in enumerate(zip(names, hs))
        push!(out, slab(R, z, h, half, cutaway ? 0.0 : half; color = colors[i], opacity = 0.9, name = "$(nm) (f = $(round(h / L; digits = 3)))"))
        z += h
    end
    if guides
        nvec = R[:, 3] .* (L * 1.1)
        push!(out, polyline(hcat(zeros(3), nvec); color = GUIDE, width = 5))
        push!(out, labels3d(reshape(nvec .* 1.12, 3, 1), ["n"]))
    end
    return out
end

"""
    scene(geometry; kw...) -> Dict

A whole Plotly payload, `{data, layout}`.
"""
function scene(geom; kw...)
    data = traces(geom; kw...)
    return Dict{String, Any}(
        "data" => data,
        "layout" => Dict(
            "margin" => Dict("l" => 0, "r" => 0, "t" => 0, "b" => 0),
            "showlegend" => length(data) > 1,
            "legend" => Dict("x" => 0, "y" => 1),
            "scene" => Dict(
                "aspectmode" => "data",
                "xaxis" => Dict("title" => "x"), "yaxis" => Dict("title" => "y"), "zaxis" => Dict("title" => "z"),
            ),
            "paper_bgcolor" => "rgba(0,0,0,0)",
        ),
    )
end

"""
    view_of(m, cell_id, phase; cutaway, guides) -> Dict

The picture the page asks for: a phase's shape, or, for a laminate, its stack.
A shape whose sizes are not numbers, or that the package refuses, is not drawn,
and `note` says why.
"""
function view_of(m::StudioModel, cell_id, phase; kw...)
    empty(note) = Dict{String, Any}("data" => Any[], "layout" => Dict{String, Any}(), "note" => note)
    c = cell(m, cell_id)
    c === nothing && return empty("no such cell")
    geom = try
        is_laminate(c) ? stack(m, c) : (ph = first_or_nothing(p -> p.name == phase, c.phases); ph === nothing ? nothing : shape(m, ph))
    catch err
        # the package refuses the shape (a negative size, radii out of order):
        # its message is the note
        err isa Union{ArgumentError, DomainError} || rethrow()
        return empty(sprint(showerror, err))
    end
    geom === nothing && return empty("sizes that are not numbers cannot be drawn")
    return scene(geom; kw...)
end
