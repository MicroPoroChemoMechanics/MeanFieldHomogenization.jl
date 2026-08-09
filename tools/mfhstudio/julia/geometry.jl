# =============================================================================
#  geometry.jl — 3-D traces for the web view.
#
#  The parametrizations come from `scripts/common/docviz.jl`, the same code
#  that draws the figures in the documentation, so a shape looks the same in
#  the interface as it does in the manual.
#
#  What is *not* reused is docviz's trace builders: they emit JavaScript object
#  literals (unquoted keys), which are not JSON and would have to be injected
#  into the page as executable script. The numeric layer -- `ellipsoid_surface`,
#  `cylinder_surface`, `disc_surface`, `param_surface`, `_layer_colors` --
#  returns plain arrays, so real JSON is built here instead.
# =============================================================================

using MeanFieldHomogenization
using LinearAlgebra

const DOCVIZ = joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl")
isfile(DOCVIZ) || error("mfhstudio: cannot find $DOCVIZ")
include(DOCVIZ)

const FACE = "#4a90d9"
const CRACK = "#8a6d3b"
const GUIDE = "#c0392b"

_grid(M) = [collect(Float64, @view M[i, :]) for i in axes(M, 1)]

"""
    surface(X, Y, Z; color, opacity, name) -> Dict

A Plotly `surface` trace as a JSON-ready dictionary.
"""
function surface(X, Y, Z; color = FACE, opacity = 0.55, name = "")
    d = Dict{String, Any}(
        "type" => "surface",
        "x" => _grid(X), "y" => _grid(Y), "z" => _grid(Z),
        "opacity" => opacity,
        "showscale" => false,
        "colorscale" => [[0, color], [1, color]],
        "contours" => Dict(
            "x" => Dict("highlight" => false),
            "y" => Dict("highlight" => false),
            "z" => Dict("highlight" => false),
        ),
        "hoverinfo" => "skip",
    )
    isempty(name) || (d["name"] = name; d["showlegend"] = true)
    return d
end

function polyline(pts; color = GUIDE, width = 4, dash = "solid", name = "")
    P = pts isa AbstractMatrix ? pts : reduce(hcat, pts)
    d = Dict{String, Any}(
        "type" => "scatter3d", "mode" => "lines",
        "x" => collect(Float64, @view P[1, :]),
        "y" => collect(Float64, @view P[2, :]),
        "z" => collect(Float64, @view P[3, :]),
        "line" => Dict("color" => color, "width" => width, "dash" => dash),
        "hoverinfo" => "skip",
        # Guides are scenery, not data: naming them fills the legend with
        # "trace 1", "trace 2", … and hides the layers that do matter.
        "showlegend" => false,
    )
    isempty(name) || (d["name"] = name; d["showlegend"] = true)
    d
end

function labels3d(pts, texts; color = GUIDE)
    P = pts isa AbstractMatrix ? pts : reduce(hcat, pts)
    return Dict{String, Any}(
        "type" => "scatter3d", "mode" => "text",
        "x" => collect(Float64, @view P[1, :]),
        "y" => collect(Float64, @view P[2, :]),
        "z" => collect(Float64, @view P[3, :]),
        "text" => collect(String, texts),
        "textfont" => Dict("color" => color, "size" => 13),
        "hoverinfo" => "skip", "showlegend" => false,
    )
end

# ── Axis guides ─────────────────────────────────────────────────────────────

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

# `vecbasis`, not `get_array`: the latter is for tensors and throws on a basis.
# It used to be called here inside a `try`, so every rotated inclusion was drawn
# unrotated — the picture disagreed with the script and nothing said so.
_rot(b) = Matrix{Float64}(I, 3, 3)  # canonical fallback
function _rot(ell::MeanFieldHomogenization.Ellipsoid)
    try
        return Matrix{Float64}(MeanFieldHomogenization.TensND.vecbasis(ell.basis))
    catch
        return Matrix{Float64}(I, 3, 3)
    end
end

# A `LayeredSpheroid` stores its orientation as the unit revolution axis, not
# as a basis, so there is no `.basis` to read here — and `inclusion_basis`
# returns the canonical one whatever the axis. Build a frame whose third
# vector IS that axis; the transverse plane of a body of revolution is
# isotropic, so any orthonormal complement draws the same surface.
function _rot(ls::MeanFieldHomogenization.LayeredSpheroid)
    e3 = Float64.(collect(ls.axis))
    nrm = norm(e3)
    nrm ≈ 0 && return Matrix{Float64}(I, 3, 3)
    e3 ./= nrm
    # Cross with the canonical vector least aligned with e3, so the complement
    # never degenerates (a fixed choice fails when e3 is that very vector).
    t = zeros(3)
    t[argmin(abs.(e3))] = 1.0
    e1 = normalize(cross(t, e3))
    e2 = cross(e3, e1)
    return hcat(e1, e2, e3)
end

# ── Per-geometry traces ─────────────────────────────────────────────────────

traces(x; kw...) = Dict{String, Any}[]

function traces(ell::MeanFieldHomogenization.Ellipsoid{3}; guides::Bool = true, kw...)
    a, b, c = Float64.(ell.semi_axes)
    R = _rot(ell)
    X, Y, Z = ellipsoid_surface(a, b, c; R = R)
    out = Dict{String, Any}[surface(X, Y, Z)]
    guides && append!(out, axis_guides(R, (a, b, c), ("a", "b", "c")))
    return out
end

function traces(cyl::MeanFieldHomogenization.Cylinder; guides::Bool = true, length_shown = 6.0, kw...)
    b, c = Float64.(cyl.semi_axes)
    R = try
        Matrix{Float64}(MeanFieldHomogenization.TensND.vecbasis(cyl.basis))
    catch
        Matrix{Float64}(I, 3, 3)
    end
    X, Y, Z = cylinder_surface(b, c, length_shown; R = R)
    out = Dict{String, Any}[surface(X, Y, Z)]
    guides && append!(out, axis_guides(R, (length_shown / 2, b, c), ("L → ∞", "b", "c")))
    return out
end

function _crack_traces(a, b, R; guides::Bool = true, normal::Bool = true, kw...)
    X, Y, Z = disc_surface(a, b; R = R)
    out = Dict{String, Any}[surface(X, Y, Z; color = CRACK, opacity = 0.9)]
    if normal
        n = R[:, 3] .* max(a, b) .* 1.5
        push!(out, polyline(hcat(zeros(3), n); color = GUIDE, width = 5))
        push!(out, labels3d(reshape(n .* 1.12, 3, 1), ["n"]))
    end
    guides && append!(out, axis_guides(R, (a, b, max(a, b) / 2), ("a", "b", "")))
    return out
end

function traces(cr::MeanFieldHomogenization.EllipticCrack; kw...)
    R = try
        Matrix{Float64}(MeanFieldHomogenization.TensND.vecbasis(cr.basis))
    catch
        Matrix{Float64}(I, 3, 3)
    end
    return _crack_traces(Float64(cr.a), Float64(cr.b), R; kw...)
end

function traces(cr::MeanFieldHomogenization.RibbonCrack; length_shown = 6.0, kw...)
    R = try
        Matrix{Float64}(MeanFieldHomogenization.TensND.vecbasis(cr.basis))
    catch
        Matrix{Float64}(I, 3, 3)
    end
    return _crack_traces(length_shown / 2, Float64(cr.b), R; kw...)
end

"""
    traces(ls::LayeredSphere; cutaway = true)

Concentric shells. The cut-away view is what makes a layered inclusion
readable at all -- without it only the outermost shell is visible.
"""
function traces(ls::MeanFieldHomogenization.LayeredSphere; cutaway::Bool = true, guides::Bool = false, kw...)
    radii = Float64.(collect(ls.radii))
    n = length(radii)
    colors = _layer_colors(n)
    out = Dict{String, Any}[]
    # A cut-away keeps half of each shell, so the inner ones stay visible.
    ulim = cutaway ? 1.0π : 2.0π
    for (i, r) in enumerate(radii)
        r > 0 || continue
        f = (u, v) -> (r * cos(v) * cos(u), r * cos(v) * sin(u), r * sin(v))
        X, Y, Z = param_surface(f, range(0, ulim; length = 41), range(-π / 2, π / 2; length = 21))
        push!(
            out,
            surface(
                X, Y, Z;
                color = colors[i], opacity = i == n ? 0.55 : 0.9,
                name = "layer $i (r = $(round(r; digits = 3)))",
            ),
        )
    end
    return out
end

function traces(ls::MeanFieldHomogenization.LayeredSpheroid; cutaway::Bool = true, guides::Bool = false, kw...)
    out = Dict{String, Any}[]
    semi = Float64.(collect(MeanFieldHomogenization.outer_semiaxes(ls)))
    n = MeanFieldHomogenization.layer_count(ls)
    colors = _layer_colors(n)
    ulim = cutaway ? 1.0π : 2.0π
    R = _rot(ls)
    for i in 1:n
        c, aeq = try
            Float64.(MeanFieldHomogenization.layer_semiaxes(ls, i))
        catch
            (semi[1] * i / n, semi[end] * i / n)
        end
        # The parametrization is written in the spheroid's own frame (revolution
        # axis along local ê₃) and rotated into the global one, because
        # `param_surface` samples a point map and takes no `R` of its own.
        f = function (u, v)
            p = R * [aeq * cos(v) * cos(u), aeq * cos(v) * sin(u), c * sin(v)]
            return (p[1], p[2], p[3])
        end
        X, Y, Z = param_surface(f, range(0, ulim; length = 41), range(-π / 2, π / 2; length = 21))
        push!(out, surface(X, Y, Z; color = colors[i], opacity = i == n ? 0.55 : 0.9, name = "layer $i"))
    end
    guides && append!(
        out, axis_guides(R, (semi[end], semi[end], semi[1]), ("ρt", "ρt", "ρa"))
    )
    return out
end

# ── Laminate ────────────────────────────────────────────────────────────────

"""
    _slab(R, z0, h, half; color, opacity, name) -> Dict

One layer of a stack: a rectangular slab of thickness `h` whose faces are
normal to the third column of `R`, its lower face at height `z0` along that
axis and its in-plane extent `[-half, half] × [-half, ymax]`.

Drawn as a single closed `mesh3d` rather than six `surface` patches, because a
stack of transparent surfaces reads as a muddle of overlapping planes while a
mesh keeps its edges.
"""
function _slab(R, z0, h, half, ymax; color = FACE, opacity = 0.75, name = "")
    # Eight corners in the laminate's own frame, rotated into the global one.
    corners = Vector{Float64}[]
    for z in (z0, z0 + h),
            (x, y) in ((-half, -half), (half, -half), (half, ymax), (-half, ymax))
        push!(corners, R * [x, y, z])
    end
    P = reduce(hcat, corners)
    # Two triangles per face, the bottom four vertices being 0-3 and the top 4-7
    # in the same rotational order.
    i = [0, 0, 4, 4, 0, 0, 1, 1, 2, 2, 3, 3]
    j = [1, 2, 5, 6, 1, 5, 2, 6, 3, 7, 0, 4]
    k = [2, 3, 6, 7, 5, 4, 6, 5, 7, 6, 4, 7]
    d = Dict{String, Any}(
        "type" => "mesh3d",
        "x" => collect(Float64, @view P[1, :]),
        "y" => collect(Float64, @view P[2, :]),
        "z" => collect(Float64, @view P[3, :]),
        "i" => i, "j" => j, "k" => k,
        "color" => color, "opacity" => opacity,
        "flatshading" => true, "hoverinfo" => "skip",
    )
    isempty(name) || (d["name"] = name; d["showlegend"] = true)
    return d
end

# The stacking direction is the third axis of the stored basis, and `vecbasis`
# gives the whole frame — so `(ℓ, m)` come from the same place the kernel takes
# them, and a tilted stack is drawn tilted rather than merely labelled so.
function _rot(lam::MeanFieldHomogenization.Laminate)
    try
        return Matrix{Float64}(MeanFieldHomogenization.TensND.vecbasis(lam.basis))
    catch
        return Matrix{Float64}(I, 3, 3)
    end
end

"""
    traces(lam::Laminate; cutaway = true, guides = true)

The periodic cell, drawn as one period of the stack plus the normal.

`cutaway` cuts the stack in half along one in-plane direction, exposing the
cross-section — which is where a laminate is legible, the layers being
unbounded in their own plane and identical from above.
"""
function traces(
        lam::MeanFieldHomogenization.Laminate;
        cutaway::Bool = true, guides::Bool = true, kw...
    )
    names = MeanFieldHomogenization.layer_names(lam)
    n = length(names)
    n == 0 && return Dict{String, Any}[]

    R = _rot(lam)
    hs = Float64[Float64(MeanFieldHomogenization.layer_thickness(lam, nm)) for nm in names]
    L = sum(hs)
    L > 0 || return Dict{String, Any}[]
    # The lateral extent is arbitrary — the layers are unbounded in their plane
    # — so it is tied to the period, which keeps the picture in proportion
    # whatever the thicknesses are entered in.
    half = 0.6 * L
    ymax = cutaway ? 0.0 : half
    colors = _layer_colors(n)

    out = Dict{String, Any}[]
    z = -L / 2
    for (idx, nm) in enumerate(names)
        h = hs[idx]
        f = h / L
        push!(
            out,
            _slab(
                R, z, h, half, ymax; color = colors[idx], opacity = 0.9,
                name = "$(nm) (f = $(round(f; digits = 3)))",
            ),
        )
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
    scene(geom; kw...) -> Dict

A complete Plotly payload: `{data, layout}`, ready for `Plotly.newPlot`.
"""
function scene(geom; kw...)
    data = traces(geom; kw...)
    return Dict(
        "data" => data,
        "layout" => Dict(
            "margin" => Dict("l" => 0, "r" => 0, "t" => 0, "b" => 0),
            "showlegend" => length(data) > 1,
            "legend" => Dict("x" => 0, "y" => 1),
            "scene" => Dict(
                "aspectmode" => "data",
                "xaxis" => Dict("title" => "x"),
                "yaxis" => Dict("title" => "y"),
                "zaxis" => Dict("title" => "z"),
            ),
            "paper_bgcolor" => "rgba(0,0,0,0)",
        ),
    )
end
