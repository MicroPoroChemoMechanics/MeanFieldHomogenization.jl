# =============================================================================
#  docviz.jl — shared 3-D figure helpers for the demo scripts and the docs.
#
#  The docs are rendered by VitePress, which bundles plotly.js itself
#  (`plotly.js-gl3d-dist-min`, declared in `docs/package.json`) and exposes a
#  single drawing entry point, `window.mfhPlotly`, from
#  `docs/src/.vitepress/theme/plotly-figures.ts`. A figure therefore emits
#  nothing but a container and a one-line call — see `_plotly_html` — and names
#  its colors by theme token (`FIG_*` below) so that it reads on the light and
#  the dark scheme alike.
#
#  Standalone `julia scripts/NN_*.jl` runs write the same HTML; opening one
#  outside the docs needs a `window.mfhPlotly` of its own, which is the price of
#  having a single emitter for both uses.
#
#  Include it with
#
#      include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
#
#  — the only form that works both in a standalone `julia scripts/NN_*.jl` run
#  and inside a Documenter `@example` block, where `@__DIR__` points at the
#  build directory rather than at the script.
#
#  Everything here emits a `Base.HTML`, which Documenter embeds verbatim as the
#  output of the block.  The plot data are inlined and then travel through Vite,
#  so keep an eye on their size: the knobs that matter are the mesh resolution
#  (`nu`, `nv`) and `DIGITS`.
# =============================================================================

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Random

# ── JSON/JS serialization ────────────────────────────────────────────────────
#
# Three decimals is ~0.1 % of a unit-sized inclusion — below what a reader can
# resolve on screen, and it roughly halves the inlined payload compared with
# the 17 significant digits `string(::Float64)` would emit.

const DIGITS = 3

_js(x::Real) = string(round(Float64(x); digits = DIGITS))
_jsvec(v) = "[" * join((_js(x) for x in v), ",") * "]"
_jsmat(M) = "[" * join(("[" * join((_js(M[i, j]) for j in axes(M, 2)), ",") * "]"
    for i in axes(M, 1)), ",") * "]"
_jsints(v) = "[" * join(v, ",") * "]"
_jsstrs(v) = "[" * join(("\"" * string(s) * "\"" for s in v), ",") * "]"
_jsbool(x::Bool) = x ? "true" : "false"

# ── Colors ───────────────────────────────────────────────────────────────────
#
# Figures are drawn on a transparent background, so whatever has to contrast
# with the page (text, guides, unit vectors, the haze of a cell) names a theme
# token instead of a color. The tokens are defined once per color scheme in
# `docs/src/.vitepress/theme/overrides.css`; `plotly-figures.ts`, next to it,
# substitutes their current values before drawing and again when the reader
# toggles the scheme. Filled surfaces keep plain colors that read on both
# backgrounds.

const FIG_INK = "var(--mfh-fig-ink)"
const FIG_GUIDE = "var(--mfh-fig-guide)"
const FIG_GRID = "var(--mfh-fig-grid)"
const FIG_VECTOR = "var(--mfh-fig-vector)"
const FIG_EDGE = "var(--mfh-fig-edge)"
const FIG_BODY = "var(--mfh-fig-body)"

# The part of a layout that puts a figure on the page rather than on a white
# card: transparent paper, text and modebar in the theme's ink.
const _THEMED_LAYOUT = """paper_bgcolor:"rgba(0,0,0,0)", plot_bgcolor:"rgba(0,0,0,0)",
    font:{color:"$FIG_INK"},
    modebar:{bgcolor:"rgba(0,0,0,0)",color:"$FIG_GUIDE",activecolor:"$FIG_INK"},"""
# ── Labels ───────────────────────────────────────────────────────────────────
#
# A label is anchored at a point of the scene but drawn in the SVG layer above
# it, as a Plotly scene annotation. A WebGL text trace would be cheaper, but it
# knows no underline, and the vectors are underlined in this documentation
# (Notation page): `<u><i>n</i></u>` only renders as an annotation. It also
# stays upright and the same size however the scene is turned.

"""
    SceneLabel

A text anchored at a point of a 3-D scene, built by [`scene_label`](@ref).
[`plotly_scene`](@ref) routes it to the layout, so labels and trace strings can
be mixed in the vector it receives.
"""
struct SceneLabel
    pos::NTuple{3, Float64}
    text::String
    color::String
    size::Int
    xanchor::String
    yanchor::String
    xshift::Int
    yshift::Int
end

"What a scene is built from: Plotly trace strings and labels."
const SceneItem = Union{String, SceneLabel}

"""
    scene_label(p, text; color = FIG_INK, size = 17, xanchor = "center",
                yanchor = "middle", xshift = 0, yshift = 0) -> SceneLabel

`text` takes Plotly's markup: `<i>a</i>` for a scalar, `<u><i>n</i></u>` for a
vector, `<sub>`/`<sup>`. `xshift`/`yshift` move it by that many pixels on
screen, which keeps it clear of the point it names whichever way the scene is
rotated.
"""
function scene_label(
        p, text::AbstractString;
        color::AbstractString = FIG_INK, size::Integer = 17,
        xanchor::AbstractString = "center", yanchor::AbstractString = "middle",
        xshift::Integer = 0, yshift::Integer = 0,
    )
    x, y, z = Float64.(Tuple(p))
    return SceneLabel((x, y, z), text, color, size, xanchor, yanchor, xshift, yshift)
end

# A serif family for the labels, which are symbols of the equations and should
# look like them rather than like the sans-serif title.
const LABEL_FONT = "STIX Two Text, Cambria, Times New Roman, serif"

_annotation(l::SceneLabel) =
    """{x:$(_js(l.pos[1])),y:$(_js(l.pos[2])),z:$(_js(l.pos[3])),text:"$(l.text)",
    showarrow:false,font:{family:"$LABEL_FONT",color:"$(l.color)",size:$(l.size)},
    xanchor:"$(l.xanchor)",yanchor:"$(l.yanchor)",xshift:$(l.xshift),yshift:$(l.yshift)}"""

# ── The scene emitter ────────────────────────────────────────────────────────

"""
    plotly_scene(items; uid, title = "", height = 480, aspectmode = "data",
                 axes_visible = false, xlabel = "x", ylabel = "y", zlabel = "z",
                 showlegend = false, camera = nothing, orthographic = false,
                 bounds = nothing, zoom = 1, sliders = "")

Wrap a vector of ready-made Plotly trace strings, possibly mixed with
[`SceneLabel`](@ref)s, into an interactive 3-D figure. `uid` must be unique
**within a page**: it is the id of the `<div>` the plot is drawn into.

The axes are hidden unless `axes_visible = true`: a picture of a shape, a cell
or a field has no use for coordinates, and they are kept only where they carry a
quantity (a stress, a friction angle). `camera` is the eye position
`(x, y, z)`, `orthographic` drops the perspective, and `sliders` is a ready-made
Plotly slider object (see [`flattening_figure`](@ref)).

`bounds`, three `(min, max)` pairs from [`scene_bounds`](@ref), fixes the axis
ranges, and `zoom` then enlarges the picture by that factor. Without
perspective, moving the eye does not change the size of the picture: Plotly
zooms by scaling the aspect ratios, and does so here from the same geometric
mean of the ranges as its own `aspectmode = "data"`, so that `zoom = 1` is the
default size and the proportions stay exact.
"""
function plotly_scene(
        items::AbstractVector;
        uid::AbstractString,
        title::AbstractString = "",
        height::Integer = 480,
        aspectmode::AbstractString = "data",
        axes_visible::Bool = false,
        xlabel::AbstractString = "x",
        ylabel::AbstractString = "y",
        zlabel::AbstractString = "z",
        showlegend::Bool = false,
        camera = nothing,
        orthographic::Bool = false,
        bounds = nothing,
        zoom::Real = 1,
        sliders::AbstractString = "",
    )
    all(x -> x isa AbstractString || x isa SceneLabel, items) ||
        throw(ArgumentError("a scene is built from trace strings and SceneLabels"))
    traces = String[x for x in items if x isa AbstractString]
    labels = SceneLabel[x for x in items if x isa SceneLabel]
    ## Left-aligned: Plotly puts its modebar at the top right, and a centered
    ## title long enough to be useful runs straight into it.
    ttl = isempty(title) ? "" :
        "title:{text:\"$title\",x:0.02,xanchor:\"left\",font:{size:13}},"
    eye = camera === nothing ? "" :
        "eye:{x:$(_js(camera[1])),y:$(_js(camera[2])),z:$(_js(camera[3]))},"
    proj = orthographic ? "projection:{type:\"orthographic\"}," : ""
    cam = isempty(eye) && isempty(proj) ? "" : "camera:{$eye$proj up:{x:0,y:0,z:1}},"
    ## Without axes there is nothing to read in a hover box but meaningless
    ## coordinates, so hovering is switched off with them.
    hover = axes_visible ? "\"closest\"" : "false"
    rng(k) = bounds === nothing ? "" : "range:[$(_js(bounds[k][1])),$(_js(bounds[k][2]))],"
    axis(t, k) = "{title:{text:\"$t\"},$(rng(k))visible:$(_jsbool(axes_visible)),color:\"$FIG_INK\"," *
        "gridcolor:\"$FIG_GRID\",zerolinecolor:\"$FIG_GUIDE\",showspikes:false}"
    aspect = if bounds === nothing
        "aspectmode:\"$aspectmode\""
    else
        Δ = [Float64(b[2] - b[1]) for b in bounds]
        r = zoom .* Δ ./ cbrt(prod(Δ))
        "aspectmode:\"manual\",aspectratio:{x:$(_js(r[1])),y:$(_js(r[2])),z:$(_js(r[3]))}"
    end
    sl = isempty(sliders) ? "" : ", sliders:[$sliders]"
    return _plotly_html(
        uid, height,
        join(traces, ","),
        """{height:$height, margin:{l:0,r:0,t:$(isempty(title) ? 10 : 46),b:$(isempty(sliders) ? 0 : 64)},
            $(_THEMED_LAYOUT) uirevision:"$uid",
            $ttl showlegend:$(showlegend),
            scene:{$aspect, bgcolor:"rgba(0,0,0,0)", hovermode:$hover,
                   $cam
                   xaxis:$(axis(xlabel, 1)), yaxis:$(axis(ylabel, 2)), zaxis:$(axis(zlabel, 3)),
                   annotations:[$(join(map(_annotation, labels), ","))]}$sl}"""
    )
end


"""
    plotly_surface(x, y, Z; title, zlabel, uid, colorscale = "RdBu",
                   reversescale = true, colorbar_title = "", height = 520)

A single interactive surface `Z[j, i]` over the grid `(x[i], y[j])` — the
2½-D counterpart of [`plotly_scene`](@ref), used by the percolation maps of
the cement-paste diffusion chapter.
"""
function plotly_surface(
        x, y, Z;
        title::AbstractString, zlabel::AbstractString, uid::AbstractString,
        colorscale::AbstractString = "RdBu", reversescale::Bool = true,
        colorbar_title::AbstractString = "", height::Integer = 520,
        xlabel::AbstractString = "x", ylabel::AbstractString = "y",
    )
    cb = isempty(colorbar_title) ? "" : ",colorbar:{title:\"$colorbar_title\"}"
    trace = """{type:"surface", x:$(_jsvec(x)), y:$(_jsvec(y)), z:$(_jsmat(Z)),
        colorscale:"$colorscale", reversescale:$(reversescale)$cb}"""
    ## The axes of a response surface carry its variables, so they stay; only
    ## their colors follow the theme.
    axis(t) = "{title:{text:\"$t\"},color:\"$FIG_INK\",gridcolor:\"$FIG_GRID\",zerolinecolor:\"$FIG_GUIDE\"}"
    return _plotly_html(
        uid, height, trace,
        """{title:{text:"$title"}, height:$height, margin:{l:0,r:0,t:40,b:0},
            $(_THEMED_LAYOUT) uirevision:"$uid",
            scene:{bgcolor:"rgba(0,0,0,0)",
                   xaxis:$(axis(xlabel)), yaxis:$(axis(ylabel)), zaxis:$(axis(zlabel))}}"""
    )
end

# The shared container + drawing call.  Every emitter above funnels here so that
# the contract with the VitePress theme is written exactly once.
#
# The writer wraps `text/html` output containing a `<script>` in
# `<ClientOnly v-exec-scripts v-html=…>`; because `v-html` assigns `innerHTML`,
# which never runs scripts, the directive re-creates each one on mount — and
# again on every client-side navigation, so a figure reached through the sidebar
# draws just like one reached by a full page load.
function _plotly_html(uid, height, data, layout)
    return Base.HTML(
        """
        <div id="$uid" style="width:100%;height:$(height)px;"></div>
        <script>
        window.mfhPlotly("$uid", [$data], $layout);
        </script>
        """
    )
end

"""
    scene_bounds(parts...; pad = 0.04) -> NTuple{3, Tuple{Float64, Float64}}

Axis ranges enclosing every point of `parts`, each a `3 × n` matrix (`NaN`
columns ignored), widened by `pad` times the largest extent. A degenerate
extent (a flat picture) is opened to the same margin.
"""
function scene_bounds(parts::AbstractMatrix...; pad::Real = 0.04)
    lo = fill(Inf, 3)
    hi = fill(-Inf, 3)
    for P in parts, j in axes(P, 2), i in 1:3
        x = Float64(P[i, j])
        isnan(x) && continue
        lo[i] = min(lo[i], x)
        hi[i] = max(hi[i], x)
    end
    δ = pad * maximum(hi .- lo)
    return ntuple(i -> (lo[i] - δ, hi[i] + δ), 3)
end

# ── Sections of a surface, cut with the mouse ────────────────────────────────

"""
    CutFamily(name, symbol, labels, planes, curves, sections, labels2d;
              active = 1, equal = true)

One family of cutting planes for [`section_figure`](@ref). For the `k`-th
plane, `labels[k]` names it on the slider (after `symbol = `), `planes[k]` is
the patch drawn in space (`(X, Y, Z)`, see [`plane_patch`](@ref)), `curves[k]`
the section in space (`3 × n`) and `sections[k]` the same curve in coordinates
of the plane (`2 × n`), whose axes `labels2d` names. `NaN` columns break a
curve. `equal` keeps the same scale on both axes of the panel, which only
makes sense when they carry the same quantity.
"""
struct CutFamily
    name::String
    symbol::String
    labels::Vector{String}
    planes::Vector{NTuple{3, Matrix{Float64}}}
    curves::Vector{Matrix{Float64}}
    sections::Vector{Matrix{Float64}}
    labels2d::NTuple{2, String}
    active::Int
    equal::Bool
end

CutFamily(name, symbol, labels, planes, curves, sections, labels2d; active::Integer = 1, equal::Bool = true) =
    CutFamily(
    String(name), String(symbol), String.(labels), planes, curves, sections,
    (String(labels2d[1]), String(labels2d[2])), Int(active), equal
)

"""
    plane_patch(o, u, w, hu, hw) -> (X, Y, Z)

The rectangle of center `o` and half-sides `hu`, `hw` along the unit vectors
`u`, `w`, as the `2 × 2` matrices of a surface trace.
"""
function plane_patch(o, u, w, hu, hw)
    P = [o .+ s * hu .* u .+ t * hw .* w for s in (-1, 1), t in (-1, 1)]
    return getindex.(P, 1), getindex.(P, 2), getindex.(P, 3)
end

"""
    section_figure(items, families; uid, title = "", height = 620,
                   xlabel = "x", ylabel = "y", zlabel = "z", camera = nothing,
                   aspectmode = "data") -> Base.HTML

A surface in space (`items`, as for [`plotly_scene`](@ref)), with a panel on its
right showing it cut by a plane. Buttons choose the family of planes, or none,
and a slider moves the plane; the section is drawn on the surface and, in the
panel, in coordinates of the plane, which is where a two-dimensional criterion
is read. Where a family asks for it, the panel keeps equal scales on its two
axes by shrinking the side that has room to spare, rather than its range, so
that no section is clipped.
"""
function section_figure(
        items::AbstractVector, families::AbstractVector{CutFamily};
        uid::AbstractString, title::AbstractString = "", height::Integer = 620,
        xlabel::AbstractString = "x", ylabel::AbstractString = "y", zlabel::AbstractString = "z",
        camera = nothing, aspectmode::AbstractString = "data",
    )
    traces = String[x for x in items if x isa AbstractString]
    labels = SceneLabel[x for x in items if x isa SceneLabel]
    n0 = length(traces)
    idx = _jsints(n0:(n0 + 2))
    ## The three traces of the cut, hidden until a family is chosen: the plane,
    ## the section on the surface, the section in the panel.
    push!(
        traces, """{type:"surface",x:[[0,0],[0,0]],y:[[0,0],[0,0]],z:[[0,0],[0,0]],visible:false,
        opacity:0.3,showscale:false,colorscale:[[0,"$FIG_GUIDE"],[1,"$FIG_GUIDE"]],
        lighting:{ambient:1,diffuse:0,specular:0,roughness:1,fresnel:0},hoverinfo:"skip"}"""
    )
    push!(
        traces, """{type:"scatter3d",mode:"lines",x:[0],y:[0],z:[0],visible:false,
        line:{width:7,color:"$FIG_EDGE"},hoverinfo:"skip"}"""
    )
    push!(
        traces, """{type:"scatter",mode:"lines",x:[0],y:[0],visible:false,xaxis:"x",yaxis:"y",
        line:{width:3,color:"$FIG_EDGE"},hovertemplate:"(%{x:.3f}, %{y:.3f})<extra></extra>"}"""
    )
    function cut_args(f, k)
        X, Y, Z = f.planes[k]
        C, S = f.curves[k], f.sections[k]
        return replace(
            "{x:[$(_jsmat(X)),$(_jsvec(@view C[1, :])),$(_jsvec(@view S[1, :]))]," *
                "y:[$(_jsmat(Y)),$(_jsvec(@view C[2, :])),$(_jsvec(@view S[2, :]))]," *
                "z:[$(_jsmat(Z)),$(_jsvec(@view C[3, :])),null]}",
            "NaN" => "null"
        )
    end
    nf = length(families)
    shown(i) = join(("\"sliders[$(j - 1)].visible\":$(_jsbool(j == i))" for j in 1:nf), ",")
    sliders = map(enumerate(families)) do (i, f)
        steps = join(
            (
                "{label:\"$(f.labels[k])\",method:\"restyle\",args:[$(cut_args(f, k)),$idx]}"
                    for k in eachindex(f.labels)
            ), ","
        )
        """{visible:false,active:$(f.active - 1),x:0,len:0.62,pad:{t:6,b:6},
        currentvalue:{prefix:"$(f.symbol) = ",font:{color:"$FIG_INK",size:14}},
        font:{color:"$FIG_INK"},bgcolor:"$FIG_GRID",activebgcolor:"$FIG_GUIDE",
        bordercolor:"$FIG_GUIDE",tickcolor:"$FIG_GUIDE",steps:[$steps]}"""
    end
    off = """{label:"No cut",method:"update",args:[{visible:[false,false,false]},
        {$(shown(0)),"xaxis.visible":false,"yaxis.visible":false},$idx]}"""
    ons = map(enumerate(families)) do (i, f)
        a = cut_args(f, f.active)
        """{label:"$(f.name)",method:"update",args:[$(a[1:(end - 1)]),visible:[true,true,true]},
        {$(shown(i)),"sliders[$(i - 1)].active":$(f.active - 1),
         "xaxis.visible":true,"yaxis.visible":true,
         "xaxis.title.text":"$(f.labels2d[1])","yaxis.title.text":"$(f.labels2d[2])",
         "yaxis.scaleanchor":$(f.equal ? "\"x\"" : "null")},$idx]}"""
    end
    ## Plotly paints the pressed button near white whatever the theme, so the
    ## buttons keep a light face and dark text in both schemes.
    menus = """{type:"buttons",direction:"right",showactive:true,active:0,
        x:0,xanchor:"left",y:1,yanchor:"top",pad:{t:0,r:0,l:0,b:0},
        font:{color:"#24292f",size:12},bgcolor:"#e6e9ed",bordercolor:"#9da7b3",
        buttons:[$off,$(join(ons, ","))]}"""
    ttl = isempty(title) ? "" :
        "title:{text:\"$title\",x:0.02,xanchor:\"left\",font:{size:13}},"
    eye = camera === nothing ? "" :
        "camera:{eye:{x:$(_js(camera[1])),y:$(_js(camera[2])),z:$(_js(camera[3]))},up:{x:0,y:0,z:1}},"
    axis3(t) = "{title:{text:\"$t\"},color:\"$FIG_INK\",gridcolor:\"$FIG_GRID\",zerolinecolor:\"$FIG_GUIDE\",showspikes:false}"
    axis2(extra) = "{$(extra)visible:false,title:{text:\"\"},color:\"$FIG_INK\",gridcolor:\"$FIG_GRID\",zerolinecolor:\"$FIG_GUIDE\"}"
    return _plotly_html(
        uid, height, join(traces, ","),
        """{height:$height, margin:{l:0,r:10,t:$(isempty(title) ? 10 : 46),b:64},
        $(_THEMED_LAYOUT) uirevision:"$uid", $ttl showlegend:false, hovermode:"closest",
        scene:{domain:{x:[0,0.62],y:[0,0.9]}, aspectmode:"$aspectmode", bgcolor:"rgba(0,0,0,0)",
               hovermode:"closest", $eye
               xaxis:$(axis3(xlabel)), yaxis:$(axis3(ylabel)), zaxis:$(axis3(zlabel)),
               annotations:[$(join(map(_annotation, labels), ","))]},
        xaxis:$(axis2("domain:[0.74,1],anchor:\"y\",constrain:\"domain\",")),
        yaxis:$(axis2("domain:[0.1,0.85],anchor:\"x\",scaleanchor:\"x\",scaleratio:1,constrain:\"domain\",")),
        updatemenus:[$menus], sliders:[$(join(sliders, ","))]}"""
    )
end

# ── Trace constructors ───────────────────────────────────────────────────────

"""
    surface_trace(X, Y, Z; color = "#4a90d9", opacity = 1.0, name = "",
                  flat = false, fade = nothing)

A one-color parametric surface.  Plotly has no flat-color surface, so the
color is passed as a degenerate two-stop colorscale.

`flat = true` turns the shading off, for a planar region that should read as
one tint from every angle (a crack face). `fade = (W, w)` makes the surface
transparent where the matrix `W` (sampled like `X`) exceeds `w`, opaque at
`opacity` below: the way an unbounded body is drawn as a finite piece that
dies out rather than one that ends.
"""
function surface_trace(
        X, Y, Z;
        color::AbstractString = "#4a90d9", opacity::Real = 1.0,
        name::AbstractString = "", flat::Bool = false, fade = nothing,
    )
    nm = isempty(name) ? "" : ",name:\"$name\",showlegend:true"
    lit = flat ? ",lighting:{ambient:1,diffuse:0,specular:0,roughness:1,fresnel:0}" : ""
    op = if fade === nothing
        "opacity:$(_js(opacity))"
    else
        W, w = fade
        ## The surface color is |W| rescaled to [0, 1]; the opacity scale is
        ## flat up to w, then falls to zero at the largest |W|.
        Wmax = maximum(abs, W)
        "surfacecolor:$(_jsmat(abs.(W) ./ Wmax)),cmin:0,cmax:1," *
            "opacityscale:[[0,$(_js(opacity))],[$(_js(w / Wmax)),$(_js(opacity))],[1,0]]"
    end
    return """{type:"surface",x:$(_jsmat(X)),y:$(_jsmat(Y)),z:$(_jsmat(Z)),
        $op,showscale:false,
        colorscale:[[0,"$color"],[1,"$color"]]$lit,
        contours:{x:{highlight:false},y:{highlight:false},z:{highlight:false}}$nm}"""
end

"""
    mesh_trace(V, F; color = "#4a90d9", opacity = 1.0, name = "")

A triangular mesh from a `3 × n` vertex matrix `V` and a `3 × m` **0-based**
face matrix `F`.  This is the form to use when many inclusions share one
figure: merging them into a single trace keeps the browser responsive, where
one trace per inclusion does not.
"""
function mesh_trace(
        V::AbstractMatrix, F::AbstractMatrix;
        color::AbstractString = "#4a90d9", opacity::Real = 1.0,
        name::AbstractString = "", flatshading::Bool = false,
    )
    nm = isempty(name) ? "" : ",name:\"$name\",showlegend:true"
    return """{type:"mesh3d",x:$(_jsvec(@view V[1, :])),y:$(_jsvec(@view V[2, :])),
        z:$(_jsvec(@view V[3, :])),
        i:$(_jsints(@view F[1, :])),j:$(_jsints(@view F[2, :])),k:$(_jsints(@view F[3, :])),
        color:"$color",opacity:$(_js(opacity)),flatshading:$(flatshading)$nm}"""
end

"""
    line_trace(pts; color = "#c0392b", width = 4, dash = "solid", name = "")

A 3-D polyline through the columns of a `3 × n` matrix (or a vector of
triples) — used for principal-axis guides and section outlines.  A column of
`NaN` breaks the line, so one trace can carry several separate pieces.
"""
function line_trace(
        pts;
        color::AbstractString = "#c0392b", width::Real = 4,
        dash::AbstractString = "solid", name::AbstractString = "",
    )
    P = _as_matrix(pts)
    nm = isempty(name) ? "" : ",name:\"$name\",showlegend:true"
    return replace(
        """{type:"scatter3d",mode:"lines",x:$(_jsvec(@view P[1, :])),
        y:$(_jsvec(@view P[2, :])),z:$(_jsvec(@view P[3, :])),
        line:{width:$width,color:"$color",dash:"$dash"},showlegend:false$nm}""",
        "NaN" => "null"
    )
end

"""
    marker_trace(pts; color = FIG_EDGE, size = 5)

Dots at the columns of a `3 × n` matrix — the vertices a dimension is read
between.
"""
function marker_trace(pts; color::AbstractString = FIG_EDGE, size::Real = 5)
    P = _as_matrix(pts)
    return """{type:"scatter3d",mode:"markers",x:$(_jsvec(@view P[1, :])),
        y:$(_jsvec(@view P[2, :])),z:$(_jsvec(@view P[3, :])),
        marker:{size:$size,color:"$color"},showlegend:false}"""
end

"""
    arrow_trace(origin, direction; color = FIG_VECTOR, size = 0.2, label = "",
                shaft = true) -> Vector{SceneItem}

An arrow from `origin` to `origin + direction`, marking a direction (a unit
vector of a frame, a crack normal). `size` is the length of the head, which is
a cone meshed explicitly so that it keeps that length whatever the scene; a
Plotly `cone` trace rescales itself from the data instead. `label`, if given, is
set beyond the tip.
"""
function arrow_trace(
        origin, direction;
        color::AbstractString = FIG_VECTOR, size::Real = 0.2,
        label::AbstractString = "", shaft::Bool = true, width::Real = 6,
    )
    o = collect(Float64, origin)
    d = collect(Float64, direction)
    L = norm(d)
    u = d ./ L
    tip = o .+ d
    h = min(Float64(size), 0.6 * L)
    items = SceneItem[]
    shaft && push!(items, line_trace(hcat(o, tip .- 0.9h .* u); color = color, width = width))
    V, F = _cone_mesh(tip, u, h, 0.38h)
    push!(items, mesh_trace(V, F; color = color, flatshading = true))
    ## The camera keeps z up, so the label of a vertical arrow (a normal, most
    ## often) goes just above or below its tip; any other one, a little beyond.
    if !isempty(label)
        up = abs(u[3]) > 0.7
        yanchor = up ? (u[3] > 0 ? "bottom" : "top") : "middle"
        push!(items, scene_label(tip .+ (up ? 0.04L : 0.22L) .* u, label; color = color, yanchor = yanchor))
    end
    return items
end

# Apex at `tip`, axis `-u`, height `h`, base radius `r`, closed by its base.
function _cone_mesh(tip, u, h, r; n::Integer = 18)
    a = abs(u[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
    e1 = normalize(a .- (a ⋅ u) .* u)
    e2 = u × e1
    c = tip .- h .* u
    V = Matrix{Float64}(undef, 3, n + 2)
    V[:, 1] = tip
    V[:, 2] = c
    for k in 1:n
        θ = 2π * (k - 1) / n
        V[:, k + 2] = c .+ r .* (cos(θ) .* e1 .+ sin(θ) .* e2)
    end
    F = Matrix{Int}(undef, 3, 2n)
    for k in 1:n
        i, j = k + 1, mod1(k + 1, n) + 1          # 0-based ring indices
        F[:, k] = [0, i, j]
        F[:, n + k] = [1, j, i]
    end
    return V, F
end

_as_matrix(P::AbstractMatrix) = P
_as_matrix(P::AbstractVector{<:Real}) = reshape(collect(Float64, P), :, 1)   # one point
_as_matrix(P) = reduce(hcat, (collect(Float64, p) for p in P))

# ── Parametric surfaces ──────────────────────────────────────────────────────

"""
    param_surface(f, us, vs) -> (X, Y, Z)

Sample `f(u, v) -> (x, y, z)` on the grid `us × vs` into three matrices ready
for [`surface_trace`](@ref).
"""
function param_surface(f, us, vs)
    X = [f(u, v)[1] for u in us, v in vs]
    Y = [f(u, v)[2] for u in us, v in vs]
    Z = [f(u, v)[3] for u in us, v in vs]
    return X, Y, Z
end

"""
    ellipsoid_surface(a, b, c; R = I, center = (0, 0, 0), nu = 33, nv = 17)

Sampled surface of the ellipsoid of semi-axes `(a, b, c)`, rotated by `R`
(columns = local axes in the global frame) and translated to `center`.
"""
function ellipsoid_surface(
        a, b, c;
        R = Matrix{Float64}(I, 3, 3), center = (0.0, 0.0, 0.0),
        nu::Integer = 33, nv::Integer = 17,
    )
    o = collect(Float64, center)
    f = function (u, v)
        p = (a * cos(v) * cos(u), b * cos(v) * sin(u), c * sin(v))
        return Tuple(o .+ R * collect(Float64, p))
    end
    return param_surface(f, range(0, 2π; length = nu), range(-π / 2, π / 2; length = nv))
end

"""
    supersphere_surface(p, a, b, c; kwargs...)

Surface of the supersphere `|x/a|^p + |y/b|^p + |z/c|^p = 1`.  A morphology
with no Hill tensor of its own — the standard illustration of what the
[custom-inclusion contract](@ref man-custom-inclusions) is for.
"""
function supersphere_surface(
        p, a, b, c;
        R = Matrix{Float64}(I, 3, 3), center = (0.0, 0.0, 0.0),
        nu::Integer = 65, nv::Integer = 33,
    )
    o = collect(Float64, center)
    sgnpow(x, q) = sign(x) * abs(x)^q
    f = function (u, v)
        q = (
            a * sgnpow(cos(v) * cos(u), 2 / p),
            b * sgnpow(cos(v) * sin(u), 2 / p),
            c * sgnpow(sin(v), 2 / p),
        )
        return Tuple(o .+ R * collect(Float64, q))
    end
    return param_surface(f, range(0, 2π; length = nu), range(-π / 2, π / 2; length = nv))
end

"""
    cylinder_surface(b, c, L; kwargs...)

Lateral surface of the elliptic cylinder of transverse semi-axes `(b, c)` and
length `L` along its axis (`e₁` in the local frame, matching the
`Ellipsoid(Inf, b, c) → Cylinder` redirection).
"""
function cylinder_surface(
        b, c, L;
        R = Matrix{Float64}(I, 3, 3), center = (0.0, 0.0, 0.0),
        nu::Integer = 33, nv::Integer = 2,
    )
    o = collect(Float64, center)
    f = function (u, t)
        q = (L * t, b * cos(u), c * sin(u))
        return Tuple(o .+ R * collect(Float64, q))
    end
    return param_surface(f, range(0, 2π; length = nu), range(-0.5, 0.5; length = max(nv, 2)))
end

"""
    disc_surface(a, b; kwargs...)

Flat elliptical disc in the local `(e₁, e₂)` plane — the `c → 0` limit an
[`EllipticCrack`](@ref) stands for.
"""
function disc_surface(
        a, b;
        R = Matrix{Float64}(I, 3, 3), center = (0.0, 0.0, 0.0),
        nu::Integer = 49, nv::Integer = 5,
    )
    o = collect(Float64, center)
    f = function (u, r)
        q = (a * r * cos(u), b * r * sin(u), 0.0)
        return Tuple(o .+ R * collect(Float64, q))
    end
    return param_surface(f, range(0, 2π; length = nu), range(0, 1; length = nv))
end

# ── Triangular meshes (for merged, many-inclusion figures) ───────────────────

"""
    ellipsoid_mesh(a, b, c; R, center, nu = 17, nv = 9) -> (V, F)

Vertices (`3 × n`) and 0-based triangles (`3 × m`) of an ellipsoid.  The
default resolution is deliberately coarse: these meshes are meant to be
merged by the dozen inside one RVE figure.
"""
function ellipsoid_mesh(
        a, b, c;
        R = Matrix{Float64}(I, 3, 3), center = (0.0, 0.0, 0.0),
        nu::Integer = 17, nv::Integer = 9,
    )
    o = collect(Float64, center)
    us = range(0, 2π; length = nu)      # last point duplicates the first
    vs = range(-π / 2, π / 2; length = nv)
    V = Matrix{Float64}(undef, 3, (nu - 1) * nv)
    n = 0
    for j in 1:nv, i in 1:(nu - 1)
        u, v = us[i], vs[j]
        p = (a * cos(v) * cos(u), b * cos(v) * sin(u), c * sin(v))
        V[:, n += 1] = o .+ R * collect(Float64, p)
    end
    # Quad strips between consecutive parallels, split into two triangles.
    m = nu - 1
    F = Matrix{Int}(undef, 3, 2 * m * (nv - 1))
    t = 0
    idx(i, j) = (j - 1) * m + mod(i - 1, m) + 1      # 1-based, wrapping in u
    for j in 1:(nv - 1), i in 1:m
        v00, v10 = idx(i, j), idx(i + 1, j)
        v01, v11 = idx(i, j + 1), idx(i + 1, j + 1)
        F[:, t += 1] = [v00 - 1, v10 - 1, v11 - 1]
        F[:, t += 1] = [v00 - 1, v11 - 1, v01 - 1]
    end
    return V, F
end

"""
    merge_meshes(meshes) -> (V, F)

Concatenate `(V, F)` pairs, offsetting the face indices — the whole point
being to end up with **one** `mesh3d` trace for a whole population.
"""
function merge_meshes(meshes)
    Vs, Fs, offset = Matrix{Float64}[], Matrix{Int}[], 0
    for (V, F) in meshes
        push!(Vs, V)
        push!(Fs, F .+ offset)
        offset += size(V, 2)
    end
    return reduce(hcat, Vs), reduce(hcat, Fs)
end

# ── Random microstructures ───────────────────────────────────────────────────

"""
    random_direction(rng) -> NTuple{3, Float64}

Uniform sampling on the unit sphere (inverse-transform on `cos θ` — sampling
the two angles uniformly would crowd the poles).
"""
function random_direction(rng::AbstractRNG)
    z = 2 * rand(rng) - 1
    r = sqrt(max(0.0, 1 - z^2))
    φ = 2π * rand(rng)
    return (r * cos(φ), r * sin(φ), z)
end

"""
    rotation_from_axis(n) -> Matrix{Float64}

An orthonormal frame whose **third** column is the unit vector `n` — i.e. the
rotation that takes a shape defined about `e₃` to one about `n`.
"""
function rotation_from_axis(n)
    e3 = normalize(collect(Float64, n))
    a = abs(e3[1]) < 0.9 ? [1.0, 0.0, 0.0] : [0.0, 1.0, 0.0]
    e1 = normalize(a - (a ⋅ e3) * e3)
    return hcat(e1, e3 × e1, e3)
end

"""
    rve_traces(; n = 60, R = 1.0, semi_axes = (0.05, 0.05, 0.05),
               orientation = :random, seed = 1949, color = "#1f4e9c",
               opacity = 1.0, cell_color = FIG_INK, cell_opacity = 0.12,
               nu = 13, nv = 7, max_tries = 4000)

A realization of a matrix/inclusion RVE: `n` non-overlapping inclusions of
semi-axes `semi_axes`, dropped inside a translucent sphere of radius `R` by
random sequential addition, all merged into a single `mesh3d` trace.

`orientation` is `:random` (isotropic distribution), `:aligned` (all axes on
`e₃`) or any 3-vector giving a common axis.  Flatten the third semi-axis to
get a crack population — `semi_axes = (0.09, 0.09, 0.004)` reads as a disc.

`seed` fixes the realization, so the same call always draws the same picture.
"""
function rve_traces(;
        n::Integer = 60, R::Real = 1.0,
        semi_axes = (0.05, 0.05, 0.05),
        orientation = :random,
        seed::Integer = 1949,
        color::AbstractString = "#1f4e9c", opacity::Real = 1.0,
        cell_color::AbstractString = FIG_INK, cell_opacity::Real = 0.12,
        nu::Integer = 13, nv::Integer = 7,
        max_tries::Integer = 4000,
    )
    rng = MersenneTwister(seed)
    a, b, c = float.(semi_axes)
    rmax = max(a, b, c)
    centers = NTuple{3, Float64}[]
    meshes = Tuple{Matrix{Float64}, Matrix{Int}}[]
    tries = 0
    while length(centers) < n && tries < max_tries
        tries += 1
        # Uniform in the ball: cube-root of a uniform on the radius.
        d = random_direction(rng)
        ρ = (R - rmax) * cbrt(rand(rng))
        ctr = ρ .* d
        # Rejection on center distance — cheap, and enough to keep the picture
        # readable (a real packing algorithm is not the point here).
        any(√sum((ctr .- p) .^ 2) ≤ 2.2 * rmax for p in centers) && continue
        push!(centers, ctr)
        Rot = orientation === :random ? rotation_from_axis(random_direction(rng)) :
            orientation === :aligned ? Matrix{Float64}(I, 3, 3) :
            rotation_from_axis(orientation)
        push!(meshes, ellipsoid_mesh(a, b, c; R = Rot, center = ctr, nu = nu, nv = nv))
    end
    V, F = merge_meshes(meshes)
    Xc, Yc, Zc = ellipsoid_surface(R, R, R; nu = 41, nv = 21)
    return [
        surface_trace(Xc, Yc, Zc; color = cell_color, opacity = cell_opacity),
        mesh_trace(V, F; color = color, opacity = opacity),
    ]
end

# ── Adapters: draw the MeanFieldHomogenization geometry objects themselves ──────────────

"""
    frame_matrix(basis) -> Matrix{Float64}

The 3×3 matrix whose columns are the local axes of a TensND basis in the
canonical frame.
"""
function frame_matrix(basis::TensND.AbstractBasis)
    E = TensND.vecbasis(basis, :cov)
    return Float64[E[i, j] for i in 1:size(E, 1), j in 1:size(E, 2)]
end

"""
    shape_traces(x; kwargs...) -> Vector{SceneItem}

Draw a `MeanFieldHomogenization` inclusion object.  Implemented for [`Ellipsoid`](@ref),
[`Cylinder`](@ref), [`EllipticCrack`](@ref), [`RibbonCrack`](@ref),
[`LayeredSphere`](@ref) and [`LayeredSpheroid`](@ref), so a figure and the
computation it illustrates are driven by the *same* object.
"""
function shape_traces end

function shape_traces(
        ell::MeanFieldHomogenization.Ellipsoid{3};
        color::AbstractString = "#4a90d9", opacity::Real = 0.55,
        axes_guides::Bool = true, labels = ("<i>a</i>", "<i>b</i>", "<i>c</i>"),
        nu::Integer = 41, nv::Integer = 21,
    )
    a, b, c = Float64.(ell.semi_axes)
    R = frame_matrix(ell.basis)
    X, Y, Z = ellipsoid_surface(a, b, c; R = R, nu = nu, nv = nv)
    traces = SceneItem[surface_trace(X, Y, Z; color = color, opacity = opacity)]
    axes_guides && append!(traces, _axis_guides(R, (a, b, c), labels))
    return traces
end

function shape_traces(
        cyl::MeanFieldHomogenization.Cylinder;
        color::AbstractString = "#4a90d9", opacity::Real = 0.55,
        length_shown::Real = 6.0, labels = ("<i>L</i> → ∞", "<i>b</i>", "<i>c</i>"),
        axes_guides::Bool = true,
    )
    b, c = Float64.(cyl.semi_axes)
    R = frame_matrix(cyl.basis)
    X, Y, Z = cylinder_surface(b, c, length_shown; R = R)
    traces = SceneItem[surface_trace(X, Y, Z; color = color, opacity = opacity)]
    if axes_guides
        append!(traces, _axis_guides(R, (length_shown / 2, b, c), labels))
    end
    return traces
end

function shape_traces(
        cr::MeanFieldHomogenization.EllipticCrack;
        color::AbstractString = "#8a6d3b", opacity::Real = 0.9,
        normal::Bool = true, axes_guides::Bool = true,
        labels = ("<i>a</i>", "<i>b</i>", "<u><i>n</i></u>"),
    )
    a, b = Float64(cr.a), Float64(cr.b)
    R = frame_matrix(cr.basis)
    X, Y, Z = disc_surface(a, b; R = R)
    traces = SceneItem[surface_trace(X, Y, Z; color = color, opacity = opacity)]
    if axes_guides
        append!(traces, _axis_guides(R, (a, b, 0.0), (labels[1], labels[2], "")))
    end
    if normal
        append!(
            traces,
            arrow_trace(
                (0.0, 0.0, 0.0), R[:, 3] .* (0.8 * a);
                size = 0.18 * a, label = labels[3]
            )
        )
    end
    return traces
end

function shape_traces(
        cr::MeanFieldHomogenization.RibbonCrack;
        color::AbstractString = "#8a6d3b", opacity::Real = 0.9,
        length_shown::Real = 6.0, normal::Bool = true,
    )
    b = Float64(cr.b)
    R = frame_matrix(cr.basis)
    # A ribbon is the a → ∞ limit of an elliptic crack: draw a long strip.
    f = (u, t) -> Tuple(R * [length_shown * (u / 2π - 0.5), b * (2t - 1), 0.0])
    X, Y, Z = param_surface(f, range(0, 2π; length = 33), range(0, 1; length = 5))
    traces = SceneItem[surface_trace(X, Y, Z; color = color, opacity = opacity)]
    if normal
        append!(
            traces,
            arrow_trace(
                (0.0, 0.0, 0.0), R[:, 3] .* (2b);
                size = 0.36 * b, label = "<u><i>n</i></u>"
            )
        )
    end
    return traces
end

# Azimuth at which a cutaway starts. Plotly's default camera looks down on the
# +x +y octant, i.e. azimuth 45°, so a cutaway must *remove* the sector around
# 45° for the opening to face the reader. Drawing `[0, 1.5π]` removes
# `[270°, 360°]`, which faces away: every layered figure then showed nothing but
# the outermost shell, a uniform blob. Starting at 90° removes `[0°, 90°]`,
# centered on the camera, and the whole stack of shells is on show.
const CUT_START = π / 2

function shape_traces(
        ls::MeanFieldHomogenization.LayeredSphere;
        colors = nothing, cutaway::Bool = true,
        opacity::Real = 1.0, nu::Integer = 41, nv::Integer = 21,
    )
    radii = Float64.(ls.radii)
    cols = colors === nothing ? _layer_colors(length(radii)) : colors
    # Every shell cut open over the same quarter of its azimuth, so the inner
    # ones stay visible — the flat picture of the recurrence. Opaque on purpose:
    # six translucent nested surfaces blend into one muddy color.
    u0 = cutaway ? CUT_START : 0.0
    umax = cutaway ? 1.5π : 2π
    traces = SceneItem[]
    for (k, r) in enumerate(radii)
        f = (u, v) -> (r * cos(v) * cos(u), r * cos(v) * sin(u), r * sin(v))
        X, Y, Z = param_surface(f, range(u0, u0 + umax; length = nu), range(-π / 2, π / 2; length = nv))
        push!(traces, surface_trace(X, Y, Z; color = cols[mod1(k, length(cols))], opacity = opacity))
    end
    return traces
end

function shape_traces(
        sp::MeanFieldHomogenization.LayeredSpheroid;
        colors = nothing, cutaway::Bool = true,
        opacity::Real = 1.0, nu::Integer = 41, nv::Integer = 21,
    )
    N = length(sp.q)
    cols = colors === nothing ? _layer_colors(N) : colors
    R = rotation_from_axis(sp.axis)
    u0 = cutaway ? CUT_START : 0.0   # see CUT_START above
    umax = cutaway ? 1.5π : 2π
    traces = SceneItem[]
    for k in 1:N
        ax, disk = Float64.(MeanFieldHomogenization.layer_semiaxes(sp, k))
        f = function (u, v)
            p = (disk * cos(v) * cos(u), disk * cos(v) * sin(u), ax * sin(v))
            return Tuple(R * collect(Float64, p))
        end
        X, Y, Z = param_surface(f, range(u0, u0 + umax; length = nu), range(-π / 2, π / 2; length = nv))
        push!(traces, surface_trace(X, Y, Z; color = cols[mod1(k, length(cols))], opacity = opacity))
    end
    return traces
end

"""
    box_mesh(xspan, yspan, zspan) -> (V, F)

The eight vertices and twelve triangles of an axis-aligned box.  Used to give
a laminate layer an actual *volume*: two horizontal faces read as a pair of
floating planes, not as a layer of matter.
"""
function box_mesh(xspan, yspan, zspan)
    x0, x1 = xspan
    y0, y1 = yspan
    z0, z1 = zspan
    V = Matrix{Float64}(undef, 3, 8)
    n = 0
    for z in (z0, z1), y in (y0, y1), x in (x0, x1)
        V[:, n += 1] = [x, y, z]
    end
    # 0-based indices, as `mesh_trace` expects: two triangles per face.
    F = reshape(
        [0, 1, 3, 0, 3, 2,      # z = z0
            4, 5, 7, 4, 7, 6,      # z = z1
            0, 1, 5, 0, 5, 4,      # y = y0
            2, 3, 7, 2, 7, 6,      # y = y1
            0, 2, 6, 0, 6, 4,      # x = x0
            1, 3, 7, 1, 7, 5],     # x = x1
        3, 12
    )
    return V, F
end

"""
    laminate_traces(thicknesses; kwargs...)

An exploded stack of layers with its normal — the periodic multilayer cell of
[`theory/laminate.md`](@ref th-laminate-pinv).

Each layer is drawn as a **solid box**.  It used to be two horizontal faces,
which reads as a stack of planes rather than of matter, and hides the very
thing the figure is about: the layer thicknesses, which `thickness_labels`
names ``h_1, h_2, …`` along the edge that faces right in the default view.
"""
function laminate_traces(
        thicknesses;
        width::Real = 2.0, colors = nothing, opacity::Real = 0.95,
        normal::Bool = true, gap::Real = 0.0, thickness_labels::Bool = true,
    )
    t = Float64.(collect(thicknesses))
    cols = colors === nothing ? _layer_colors(length(t)) : colors
    traces = SceneItem[]
    w = width / 2
    z = -sum(t) / 2
    for (k, tk) in enumerate(t)
        z0, z1 = z, z + tk
        V, F = box_mesh((-w, w), (-w, w), (z0, z1))
        push!(
            traces,
            mesh_trace(V, F; color = cols[mod1(k, length(cols))], opacity = opacity, flatshading = true)
        )
        ## The camera looks from (+x, +y, +z), so the vertical edge at
        ## (-w, w) is the one on the right of the picture.
        thickness_labels && push!(
            traces, scene_label(
                (-w, w, (z0 + z1) / 2), "<i>h</i><sub>$k</sub>";
                xanchor = "left", xshift = 8
            )
        )
        z = z1 + gap
    end
    if normal
        append!(traces, arrow_trace((0.0, 0.0, z), (0.0, 0.0, 0.6); size = 0.15, label = "<u><i>n</i></u>"))
    end
    return traces
end

# Dashed guides along the three local axes, with their labels at the tips.
function _axis_guides(R, lengths, labels)
    traces = SceneItem[]
    for m in 1:3
        L = Float64(lengths[m])
        L > 0 || continue
        push!(traces, line_trace(hcat(zeros(3), R[:, m] .* L); color = FIG_EDGE, width = 3, dash = "dash"))
        isempty(labels[m]) || push!(traces, scene_label(R[:, m] .* (1.12 * L), labels[m]))
    end
    return traces
end

# Blue-to-warm ramp, readable in both the light and the dark Documenter theme.
function _layer_colors(n::Integer)
    base = ["#1f4e9c", "#4a90d9", "#7ec8e3", "#f0ad4e", "#c0392b", "#8a6d3b"]
    return n ≤ length(base) ? base[1:n] : [base[mod1(k, length(base))] for k in 1:n]
end

# ── Crack sketches ───────────────────────────────────────────────────────────
#
# The figures of the geometry section of `theory/cod_tensors.md`: the elliptic
# and the ribbon crack with their frame, then the flat ellipsoid and the flat
# elliptic cylinder of which they are the limits. Each is drawn from the crack
# object itself (its semi-axes and its frame), the crack in FIG_EDGE, the body
# that degenerates onto it in FIG_BODY, the unit vectors in FIG_VECTOR.

# Seen from (+ℓ, +m) and somewhat above, without perspective: the view of the
# usual hand-drawn sketch, in which ℓ runs to the lower left and m to the lower
# right.
const SKETCH_EYE = (1.0, 1.0, 0.8)

# How much larger than Plotly's default size a sketch is drawn: the default
# leaves room for any rotation, which a crack lying flat does not need. A tilted
# one keeps the default, being seen face on from the default eye.
const SKETCH_ZOOM = 1.6
_sketch_zoom(R, zoom) = zoom === nothing ? (abs(R[3, 3]) > 0.99 ? SKETCH_ZOOM : 1.0) : Float64(zoom)

const _VEC_LABELS = ("<u><i>ℓ</i></u>", "<u><i>m</i></u>", "<u><i>n</i></u>")

# Length of the unit vectors drawn in a crack of in-plane half-sizes a ≥ b: long
# enough to read, short enough to stay inside the crack.
_unit_length(a, b) = min(a / 3, 0.7 * b)

# A dashed axis from -L to L along u, interrupted over [0, v] where the unit
# vector is drawn: a guide drawn on top of an arrow hides it.
function _axis_line(u, L, v)
    gap = v < L ? hcat(fill(NaN, 3), v .* u, L .* u) : zeros(3, 0)
    return hcat(-L .* u, zeros(3), gap)
end

# Points of the ellipse of half-axes p along u and q along w, centered at o.
_ellipse(o, u, p, w, q; n::Integer = 97) =
    reduce(hcat, (o .+ p * cos(t) .* u .+ q * sin(t) .* w for t in range(0, 2π; length = n)))

# The unit vectors of the frame, labeled. `shifts` says, for each of the three,
# where its label sits relative to its tip on screen.
function _frame_arrows(R, v)
    items = SceneItem[]
    shifts = ((xanchor = "center", yanchor = "bottom", xshift = 0, yshift = 6),
        (xanchor = "center", yanchor = "bottom", xshift = 0, yshift = 6),
        (xanchor = "right", yanchor = "bottom", xshift = -6, yshift = 0))
    for k in 1:3
        append!(items, arrow_trace(zeros(3), v .* R[:, k]; size = 0.22 * v))
        push!(items, scene_label(v .* R[:, k], _VEC_LABELS[k]; color = FIG_VECTOR, shifts[k]...))
    end
    return items
end

# The crack face of an ellipse: a flat tint and its outline.
function _elliptic_face(a, b, R)
    X, Y, Z = disc_surface(a, b; R = R, nu = 73, nv = 3)
    return SceneItem[
        surface_trace(X, Y, Z; color = FIG_EDGE, opacity = 0.2, flat = true),
        line_trace(_ellipse(zeros(3), R[:, 1], a, R[:, 2], b); color = FIG_EDGE, width = 5),
    ]
end

# The face of a ribbon over |s| ≤ L along ℓ, dying out beyond `solid`: outline
# solid up to there, dotted after.
function _ribbon_face(b, R, L, solid)
    ℓ, m = R[:, 1], R[:, 2]
    ss = range(-L, L; length = 61)
    ts = range(-b, b; length = 3)
    S = [s for s in ss, t in ts]
    f = (s, t) -> Tuple(s .* ℓ .+ t .* m)
    X, Y, Z = param_surface(f, ss, ts)
    edge(s0, s1) = hcat(s0 .* ℓ .+ b .* m, s1 .* ℓ .+ b .* m, fill(NaN, 3),
        s0 .* ℓ .- b .* m, s1 .* ℓ .- b .* m, fill(NaN, 3))
    return SceneItem[
        surface_trace(X, Y, Z; color = FIG_EDGE, opacity = 0.2, flat = true, fade = (S, solid)),
        line_trace(edge(-solid, solid); color = FIG_EDGE, width = 5),
        line_trace(hcat(edge(solid, L), edge(-L, -solid)); color = FIG_EDGE, width = 4, dash = "dot"),
    ]
end

"""
    crack_figure(crack; uid, title = "", height = 380, zoom = nothing) -> Base.HTML

The crack alone, in its plane: its face, its axes (dashed), the vertices that
carry its semi-axes, and the unit vectors ``(\\underline{\\ell},
\\underline{m}, \\underline{n})`` of its frame. Implemented for an
[`EllipticCrack`](@ref) and a [`RibbonCrack`](@ref); a ribbon is drawn over six
half-widths and fades out along ``\\underline{\\ell}``. `zoom` defaults to a
close view of a crack lying flat and to Plotly's default size otherwise.
"""
function crack_figure end

function crack_figure(
        cr::MeanFieldHomogenization.EllipticCrack;
        uid::AbstractString, title::AbstractString = "", height::Integer = 380,
        zoom = nothing,
    )
    a, b = Float64(cr.a), Float64(cr.b)
    R = frame_matrix(cr.basis)
    ℓ, m = R[:, 1], R[:, 2]
    v = _unit_length(a, b)
    items = _elliptic_face(a, b, R)
    push!(items, line_trace(hcat(_axis_line(ℓ, a, v), fill(NaN, 3), _axis_line(m, b, v)); color = FIG_EDGE, width = 3, dash = "dash"))
    push!(items, marker_trace(hcat(a .* ℓ, b .* m)))
    append!(items, _frame_arrows(R, v))
    push!(items, scene_label(a .* ℓ, "<i>a</i>"; xanchor = "right", xshift = -8))
    push!(items, scene_label(b .* m, "<i>b</i> = <i>ηa</i>"; xanchor = "left", xshift = 8))
    push!(items, scene_label(0.5a .* ℓ .- 0.55b .* m, "ℰ"; color = FIG_EDGE, size = 22))
    bounds = scene_bounds(_ellipse(zeros(3), ℓ, a, m, b), 1.1v .* R)
    return plotly_scene(items; uid, title, height, camera = SKETCH_EYE, orthographic = true, bounds, zoom = _sketch_zoom(R, zoom))
end

function crack_figure(
        cr::MeanFieldHomogenization.RibbonCrack;
        uid::AbstractString, title::AbstractString = "", height::Integer = 380,
        zoom = nothing,
    )
    b = Float64(cr.b)
    R = frame_matrix(cr.basis)
    ℓ, m = R[:, 1], R[:, 2]
    L, solid = 3b, 1.6b
    v = _unit_length(L, b)
    items = _ribbon_face(b, R, L, solid)
    push!(items, line_trace(hcat(_axis_line(ℓ, solid, v), fill(NaN, 3), _axis_line(m, b, v)); color = FIG_EDGE, width = 3, dash = "dash"))
    push!(items, marker_trace(b .* m))
    append!(items, _frame_arrows(R, v))
    push!(items, scene_label(b .* m, "<i>b</i>"; xanchor = "left", xshift = 8))
    push!(items, scene_label(solid .* ℓ, "<i>a</i> → ∞"; xanchor = "right", yanchor = "top", xshift = -6, yshift = -4))
    push!(items, scene_label(-1.1b .* ℓ .- 0.55b .* m, "ℛ"; color = FIG_EDGE, size = 22))
    bounds = scene_bounds(hcat(L .* ℓ .+ b .* m, L .* ℓ .- b .* m, -L .* ℓ .+ b .* m, -L .* ℓ .- b .* m), 1.1v .* R)
    return plotly_scene(items; uid, title, height, camera = SKETCH_EYE, orthographic = true, bounds, zoom = _sketch_zoom(R, zoom))
end

"""
    flattening_figure(crack; uid, ratios = (0.4, 0.25, 0.12, 0.05, 0.02),
                      active = 2, title = "", height = 460, zoom = nothing) -> Base.HTML

The body a crack is the flat limit of, around the crack: the ellipsoid of
semi-axes ``(a, b, c = \\omega a)`` for an [`EllipticCrack`](@ref), the elliptic
cylinder of axis ``\\underline{\\ell}`` and half-axes ``(b, c = \\rho b)`` for a
[`RibbonCrack`](@ref). A slider runs the flatness through `ratios`, starting
from `ratios[active]`, so that the body is seen collapsing onto the crack.
"""
function flattening_figure end

function flattening_figure(
        cr::MeanFieldHomogenization.EllipticCrack;
        uid::AbstractString, ratios = (0.4, 0.25, 0.12, 0.05, 0.02), active::Integer = 2,
        title::AbstractString = "", height::Integer = 460, zoom = nothing,
    )
    a, b = Float64(cr.a), Float64(cr.b)
    R = frame_matrix(cr.basis)
    ℓ, m, n = R[:, 1], R[:, 2], R[:, 3]
    v = _unit_length(a, b)
    ## The traces that depend on c come first, in this order, so that the slider
    ## can address them by index.
    body(c) = ellipsoid_surface(a, b, c; R = R, nu = 41, nv = 19)
    moving(c) = (
        body(c),
        _ellipse(zeros(3), ℓ, a, n, c),
        _ellipse(zeros(3), m, b, n, c),
        hcat(-c .* n, c .* n),
        reshape(c .* n, 3, 1),
    )
    return _flattening(
        moving, a .* Float64.(ratios), active, ratios, "ω", "<i>c</i> = <i>ωa</i>", n;
        uid, title, height, extent = 1.1v .* R, zoom = _sketch_zoom(R, zoom),
        static = vcat(
            _elliptic_face(a, b, R),
            [
                line_trace(hcat(_axis_line(ℓ, a, v), fill(NaN, 3), _axis_line(m, b, v)); color = FIG_EDGE, width = 3, dash = "dash"),
                marker_trace(hcat(a .* ℓ, b .* m)),
            ],
            _frame_arrows(R, v),
            [
                scene_label(a .* ℓ, "<i>a</i>"; xanchor = "right", xshift = -8),
                scene_label(b .* m, "<i>b</i> = <i>ηa</i>"; xanchor = "left", xshift = 8),
            ],
        ),
    )
end

function flattening_figure(
        cr::MeanFieldHomogenization.RibbonCrack;
        uid::AbstractString, ratios = (0.4, 0.25, 0.12, 0.05, 0.02), active::Integer = 2,
        title::AbstractString = "", height::Integer = 460, zoom = nothing,
    )
    b = Float64(cr.b)
    R = frame_matrix(cr.basis)
    ℓ, m, n = R[:, 1], R[:, 2], R[:, 3]
    L, solid = 3b, 1.6b
    v = _unit_length(L, b)
    ss = range(-L, L; length = 31)
    us = range(0, 2π; length = 33)
    S = [s for s in ss, u in us]
    body(c) = param_surface((s, u) -> Tuple(s .* ℓ .+ b * cos(u) .* m .+ c * sin(u) .* n), ss, us)
    moving(c) = (
        body(c),
        _ellipse(zeros(3), m, b, n, c),
        hcat(_ellipse(solid .* ℓ, m, b, n, c), fill(NaN, 3), _ellipse(-solid .* ℓ, m, b, n, c)),
        hcat(-c .* n, c .* n),
        reshape(c .* n, 3, 1),
    )
    return _flattening(
        moving, b .* Float64.(ratios), active, ratios, "ρ", "<i>c</i> = <i>ρb</i>", n;
        uid, title, height, fade = (S, solid), extent = 1.1v .* R, zoom = _sketch_zoom(R, zoom),
        static = vcat(
            _ribbon_face(b, R, L, solid),
            [
                line_trace(hcat(_axis_line(ℓ, solid, v), fill(NaN, 3), _axis_line(m, b, v)); color = FIG_EDGE, width = 3, dash = "dash"),
                marker_trace(b .* m),
            ],
            _frame_arrows(R, v),
            [
                scene_label(b .* m, "<i>b</i>"; xanchor = "left", xshift = 8),
                scene_label(solid .* ℓ, "<i>a</i> → ∞"; xanchor = "right", yanchor = "top", xshift = -6, yshift = -4),
            ],
        ),
    )
end

# The figure shared by the two bodies. `moving(c)` returns, for the third
# half-axis c, the body surface (X, Y, Z), its two section outlines and the
# guide and vertex of c, as 3 × n matrices; `cs` are the values of c the slider
# visits, named `symbol = ratios[k]` on it.
function _flattening(
        moving, cs, active, ratios, symbol, clabel, n;
        uid, title, height, static, fade = nothing, extent = zeros(3, 0), zoom = SKETCH_ZOOM,
    )
    nmov = 5
    function geometry(c)
        (X, Y, Z), sec1, sec2, guide, vertex = moving(c)
        return (X, Y, Z), (sec1, sec2, guide, vertex)
    end
    (X, Y, Z), lines = geometry(cs[active])
    items = SceneItem[
        surface_trace(X, Y, Z; color = FIG_BODY, opacity = 0.22, fade = fade),
        line_trace(lines[1]; color = FIG_BODY, width = 3),
        line_trace(lines[2]; color = FIG_BODY, width = 3),
        line_trace(lines[3]; color = FIG_BODY, width = 3, dash = "dash"),
        marker_trace(lines[4]; color = FIG_BODY),
    ]
    append!(items, static)
    ## The label of c is the first annotation, so the slider moves
    ## `scene.annotations[0]`.
    label(c) = scene_label(c .* n, clabel; color = FIG_BODY, xanchor = "left", yanchor = "bottom", xshift = 6, yshift = 2)
    pushfirst!(items, label(cs[active]))
    steps = map(eachindex(cs)) do k
        (Xk, Yk, Zk), lk = geometry(cs[k])
        xs = "[" * join(vcat(_jsmat(Xk), [_jsvec(@view l[1, :]) for l in lk]), ",") * "]"
        ys = "[" * join(vcat(_jsmat(Yk), [_jsvec(@view l[2, :]) for l in lk]), ",") * "]"
        zs = "[" * join(vcat(_jsmat(Zk), [_jsvec(@view l[3, :]) for l in lk]), ",") * "]"
        p = cs[k] .* n
        replace(
            """{label:"$(ratios[k])",method:"update",args:[{x:$xs,y:$ys,z:$zs},
            {"scene.annotations[0].x":$(_js(p[1])),"scene.annotations[0].y":$(_js(p[2])),
             "scene.annotations[0].z":$(_js(p[3]))},$(_jsints(0:(nmov - 1)))]}""",
            "NaN" => "null"
        )
    end
    sliders = """{active:$(active - 1),x:0.1,len:0.8,pad:{t:6,b:6},
        currentvalue:{prefix:"$symbol = ",font:{color:"$FIG_INK",size:14}},
        font:{color:"$FIG_INK"},bgcolor:"$FIG_GRID",activebgcolor:"$FIG_GUIDE",
        bordercolor:"$FIG_GUIDE",tickcolor:"$FIG_GUIDE",
        steps:[$(join(steps, ","))]}"""
    ## The ranges hold the thickest body of the slider, and stay put as it moves.
    (Xm, Ym, Zm), _ = geometry(maximum(cs))
    bounds = scene_bounds(permutedims(hcat(vec(Xm), vec(Ym), vec(Zm))), extent)
    return plotly_scene(items; uid, title, height, camera = SKETCH_EYE, orthographic = true, bounds, zoom, sliders)
end
