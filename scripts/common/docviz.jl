# =============================================================================
#  docviz.jl — shared 3-D figure helpers for the demo scripts and the docs.
#
#  The docs are rendered by VitePress, which bundles plotly.js itself
#  (`plotly.js-gl3d-dist-min`, declared in `docs/package.json`) and exposes a
#  single drawing entry point, `window.mfhPlotly`, from
#  `docs/src/.vitepress/theme/index.ts`. A figure therefore emits nothing but a
#  container and a one-line call — see `_plotly_html`.
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

# ── The scene emitter ────────────────────────────────────────────────────────

"""
    plotly_scene(traces; uid, title = "", height = 480, aspectmode = "data",
                 xlabel = "x", ylabel = "y", zlabel = "z", showlegend = false,
                 axes_visible = true)

Wrap a vector of ready-made Plotly trace strings into an interactive 3-D
figure.  `uid` must be unique **within a page** — it is the id of the `<div>`
the plot is drawn into.
"""
function plotly_scene(
        traces::AbstractVector{<:AbstractString};
        uid::AbstractString,
        title::AbstractString = "",
        height::Integer = 480,
        aspectmode::AbstractString = "data",
        xlabel::AbstractString = "x",
        ylabel::AbstractString = "y",
        zlabel::AbstractString = "z",
        showlegend::Bool = false,
        axes_visible::Bool = true,
    )
    vis = axes_visible ? "true" : "false"
    ## Left-aligned: Plotly puts its modebar at the top right, and a centered
    ## title long enough to be useful runs straight into it.
    ttl = isempty(title) ? "" :
        "title:{text:\"$title\",x:0.02,xanchor:\"left\",font:{size:13}},"
    return _plotly_html(
        uid, height,
        join(traces, ","),
        """{height:$height, margin:{l:0,r:0,t:$(isempty(title) ? 10 : 46),b:0},
            $ttl showlegend:$(showlegend),
            scene:{aspectmode:"$aspectmode",
                   xaxis:{title:"$xlabel",visible:$vis},
                   yaxis:{title:"$ylabel",visible:$vis},
                   zaxis:{title:"$zlabel",visible:$vis}}}"""
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
    return _plotly_html(
        uid, height, trace,
        """{title:{text:"$title"}, height:$height, margin:{l:0,r:0,t:40,b:0},
            scene:{xaxis:{title:"$xlabel"}, yaxis:{title:"$ylabel"},
                   zaxis:{title:"$zlabel"}}}"""
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

# ── Trace constructors ───────────────────────────────────────────────────────

"""
    surface_trace(X, Y, Z; color = "#4a90d9", opacity = 1.0, name = "")

A one-color parametric surface.  Plotly has no flat-color surface, so the
color is passed as a degenerate two-stop colorscale.
"""
function surface_trace(
        X, Y, Z;
        color::AbstractString = "#4a90d9", opacity::Real = 1.0,
        name::AbstractString = "",
    )
    nm = isempty(name) ? "" : ",name:\"$name\",showlegend:true"
    return """{type:"surface",x:$(_jsmat(X)),y:$(_jsmat(Y)),z:$(_jsmat(Z)),
        opacity:$(_js(opacity)),showscale:false,
        colorscale:[[0,"$color"],[1,"$color"]],
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
triples) — used for principal-axis guides and section outlines.
"""
function line_trace(
        pts;
        color::AbstractString = "#c0392b", width::Real = 4,
        dash::AbstractString = "solid", name::AbstractString = "",
    )
    P = _as_matrix(pts)
    nm = isempty(name) ? "" : ",name:\"$name\",showlegend:true"
    return """{type:"scatter3d",mode:"lines",x:$(_jsvec(@view P[1, :])),
        y:$(_jsvec(@view P[2, :])),z:$(_jsvec(@view P[3, :])),
        line:{width:$width,color:"$color",dash:"$dash"},showlegend:false$nm}"""
end

"""
    text_trace(pts, labels; color = "#111", size = 15)

Floating labels at given points.  The Asymptote originals of these figures
carried their labels commented out; annotating is most of what a reader
actually needs from a shape picture.
"""
function text_trace(
        pts, labels;
        color::AbstractString = "#111111", size::Integer = 15,
    )
    P = _as_matrix(pts)
    return """{type:"scatter3d",mode:"text",x:$(_jsvec(@view P[1, :])),
        y:$(_jsvec(@view P[2, :])),z:$(_jsvec(@view P[3, :])),
        text:$(_jsstrs(labels)),textfont:{size:$size,color:"$color"},
        showlegend:false}"""
end

"""
    arrow_trace(origin, direction; color = "#1565c0", size = 0.35, label = "")

A cone marking a direction (a crack normal, a revolution axis), optionally
labeled at its tip.
"""
function arrow_trace(
        origin, direction;
        color::AbstractString = "#1565c0", size::Real = 0.35,
        label::AbstractString = "", shaft::Bool = true,
    )
    o = collect(Float64, origin)
    d = collect(Float64, direction)
    tip = o .+ d
    traces = String[]
    shaft && push!(traces, line_trace(hcat(o, tip); color = color, width = 5))
    push!(
        traces, """{type:"cone",x:$(_jsvec([tip[1]])),y:$(_jsvec([tip[2]])),
        z:$(_jsvec([tip[3]])),u:$(_jsvec([d[1] * size])),v:$(_jsvec([d[2] * size])),
        w:$(_jsvec([d[3] * size])),anchor:"tail",showscale:false,
        colorscale:[[0,"$color"],[1,"$color"]]}"""
    )
    isempty(label) || push!(traces, text_trace(reshape(tip .* 1.15, 3, 1), [label]))
    return traces
end

_as_matrix(P::AbstractMatrix) = P
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
               opacity = 1.0, cell_color = "#000000", cell_opacity = 0.12,
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
        cell_color::AbstractString = "#000000", cell_opacity::Real = 0.12,
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
    shape_traces(x; kwargs...) -> Vector{String}

Draw a `MeanFieldHomogenization` inclusion object.  Implemented for [`Ellipsoid`](@ref),
[`Cylinder`](@ref), [`EllipticCrack`](@ref), [`RibbonCrack`](@ref),
[`LayeredSphere`](@ref) and [`LayeredSpheroid`](@ref), so a figure and the
computation it illustrates are driven by the *same* object.
"""
function shape_traces end

function shape_traces(
        ell::MeanFieldHomogenization.Ellipsoid{3};
        color::AbstractString = "#4a90d9", opacity::Real = 0.55,
        axes_guides::Bool = true, labels = ("a", "b", "c"),
        nu::Integer = 41, nv::Integer = 21,
    )
    a, b, c = Float64.(ell.semi_axes)
    R = frame_matrix(ell.basis)
    X, Y, Z = ellipsoid_surface(a, b, c; R = R, nu = nu, nv = nv)
    traces = [surface_trace(X, Y, Z; color = color, opacity = opacity)]
    axes_guides && append!(traces, _axis_guides(R, (a, b, c), labels))
    return traces
end

function shape_traces(
        cyl::MeanFieldHomogenization.Cylinder;
        color::AbstractString = "#4a90d9", opacity::Real = 0.55,
        length_shown::Real = 6.0, labels = ("L → ∞", "b", "c"),
        axes_guides::Bool = true,
    )
    b, c = Float64.(cyl.semi_axes)
    R = frame_matrix(cyl.basis)
    X, Y, Z = cylinder_surface(b, c, length_shown; R = R)
    traces = [surface_trace(X, Y, Z; color = color, opacity = opacity)]
    if axes_guides
        append!(traces, _axis_guides(R, (length_shown / 2, b, c), labels))
    end
    return traces
end

function shape_traces(
        cr::MeanFieldHomogenization.EllipticCrack;
        color::AbstractString = "#8a6d3b", opacity::Real = 0.9,
        normal::Bool = true, axes_guides::Bool = true,
        labels = ("a", "b", "n̂"),
    )
    a, b = Float64(cr.a), Float64(cr.b)
    R = frame_matrix(cr.basis)
    X, Y, Z = disc_surface(a, b; R = R)
    traces = [surface_trace(X, Y, Z; color = color, opacity = opacity)]
    if axes_guides
        append!(traces, _axis_guides(R, (a, b, 0.0), (labels[1], labels[2], "")))
    end
    if normal
        append!(
            traces,
            arrow_trace(
                (0.0, 0.0, 0.0), R[:, 3] .* (0.8 * a);
                color = "#1565c0", size = 0.3 * a, label = labels[3]
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
    traces = [surface_trace(X, Y, Z; color = color, opacity = opacity)]
    if normal
        append!(
            traces,
            arrow_trace(
                (0.0, 0.0, 0.0), R[:, 3] .* (2b);
                color = "#1565c0", size = 0.6 * b, label = "n̂"
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
    traces = String[]
    # Every shell cut open over the same quarter of its azimuth, so the inner
    # ones stay visible — the flat picture of the recurrence. Opaque on purpose:
    # six translucent nested surfaces blend into one muddy color.
    u0 = cutaway ? CUT_START : 0.0
    umax = cutaway ? 1.5π : 2π
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
    traces = String[]
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
thing the figure is about: the layer thicknesses.
"""
function laminate_traces(
        thicknesses;
        width::Real = 2.0, colors = nothing, opacity::Real = 0.95,
        normal::Bool = true, gap::Real = 0.0,
    )
    t = Float64.(collect(thicknesses))
    cols = colors === nothing ? _layer_colors(length(t)) : colors
    traces = String[]
    z = -sum(t) / 2
    for (k, tk) in enumerate(t)
        z0, z1 = z, z + tk
        V, F = box_mesh((-width / 2, width / 2), (-width / 2, width / 2), (z0, z1))
        push!(
            traces,
            mesh_trace(V, F; color = cols[mod1(k, length(cols))], opacity = opacity)
        )
        z = z1 + gap
    end
    if normal
        append!(
            traces,
            arrow_trace(
                (0.0, 0.0, z), (0.0, 0.0, 0.6);
                color = "#1565c0", size = 0.25, label = "n̂"
            )
        )
    end
    return traces
end

# Dashed guides along the three local axes, with their labels at the tips.
function _axis_guides(R, lengths, labels)
    traces = String[]
    pts, txt = Vector{Float64}[], String[]
    for m in 1:3
        L = Float64(lengths[m])
        L > 0 || continue
        push!(traces, line_trace(hcat(zeros(3), R[:, m] .* L); color = "#c0392b", width = 3, dash = "dash"))
        isempty(labels[m]) && continue
        push!(pts, R[:, m] .* (1.12 * L))
        push!(txt, labels[m])
    end
    isempty(pts) || push!(traces, text_trace(reduce(hcat, pts), txt))
    return traces
end

# Blue-to-warm ramp, readable in both the light and the dark Documenter theme.
function _layer_colors(n::Integer)
    base = ["#1f4e9c", "#4a90d9", "#7ec8e3", "#f0ad4e", "#c0392b", "#8a6d3b"]
    return n ≤ length(base) ? base[1:n] : [base[mod1(k, length(base))] for k in 1:n]
end
