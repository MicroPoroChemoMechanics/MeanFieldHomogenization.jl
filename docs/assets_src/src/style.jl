# =============================================================================
#  style.jl — the shared look of every drawn figure of the documentation: one
#  palette, one panel, one way of writing a label. The MPCM talk draws its own
#  figures with these same files.
#
#  Drawing: Luxor. Labels: Typst, through Typstry — mathematics in New Computer
#  Modern Math with the notation of the documentation (underlined vectors,
#  bold italic second-order tensors, blackboard fourth-order tensors), words in
#  DejaVu Sans. Every label is compiled to SVG by Typst and placed as vector
#  paths, so the output SVG is self-contained.
#
#  Colors are named by ROLE (matrix, inclusion, crack, coating…), never used as
#  raw hex codes in the figure files, so that one change here recolors every
#  figure consistently.
# =============================================================================

using Luxor
using Typstry
using StableRNGs

# ── Palette ───────────────────────────────────────────────────────────────────
# The MFH logo family (Julia colors), with tints for large areas. Every figure
# sits on its own white panel: an SVG served as an image follows the operating
# system's color scheme, not the site's toggle, so it must not rely on the page
# background.

const INK = "#24292f"          # text and outlines
const MUTED = "#57606a"        # secondary text, guides
const GUIDE = "#8c959f"        # dashed guides, zoom cones
const PANEL = "#ffffff"
const PANEL_EDGE = "#d0d7de"

const BLUE = "#4063D8"
const GREEN = "#389826"
const PURPLE = "#9558B2"
const RED = "#CB3C33"
const ORANGE = "#E07B39"
const TEAL = "#1F9E89"
const GOLD = "#D4A017"
const SLATE = "#4A5568"

"Mix `hex` with white: `t = 0` keeps the color, `t = 1` gives white."
function tint(hex::AbstractString, t::Real)
    c = parse(Luxor.Colors.RGB, hex)
    f(x) = x + (1 - x) * t
    return Luxor.Colors.RGB(f(c.r), f(c.g), f(c.b))
end

# Roles.
const MATRIX = tint(BLUE, 0.88)        # a matrix, an unbounded reference medium
const MATRIX_EDGE = MUTED
const EFFECTIVE = tint(PURPLE, 0.75)   # a homogenized (effective) medium
const INCLUSIONS = (BLUE, ORANGE, GREEN, PURPLE, TEAL, GOLD)
const CRACK = RED
const COATING = tint(ORANGE, 0.3)      # coatings, interfacial transition zones
const VECTOR = "#1a5fb4"               # arrows that carry a quantity
const HIGHLIGHT = ORANGE               # zoom spots

# Materials of the application figures: binders in blue and slate, from the
# lightest (cement paste) to the darkest (bitumen); mineral grains sand-colored;
# empty pores white.
const PASTE = MATRIX
const AGGREGATE = tint(GOLD, 0.45)
const PORE = PANEL
const PLATELET = tint(BLUE, 0.55)
const INTERLAYER = tint(ORANGE, 0.7)

# ── Sheet ─────────────────────────────────────────────────────────────────────

"Where the sheets go: `docs/src/assets/`, under the relative path each one names."
const OUT = Ref(normpath(joinpath(@__DIR__, "..", "..", "src", "assets")))

"""
    sheet(f, relpath, w, h; panel = true)

Draw `f()` into `OUT[]/relpath.svg`, a `w × h` drawing with the origin at the
top-left corner, on a white rounded panel.
"""
function sheet(f, relpath::AbstractString, w, h; panel::Bool = true)
    path = joinpath(OUT[], relpath * ".svg")
    mkpath(dirname(path))
    empty!(_GLYPHS)
    empty!(_PLACED)
    Drawing(w, h, path)
    origin()
    translate(-w / 2, -h / 2)
    if panel
        setcolor(PANEL)
        box(Point(w / 2, h / 2), w - 2, h - 2, 12, :fill)
        setcolor(PANEL_EDGE)
        setline(1.2)
        box(Point(w / 2, h / 2), w - 2, h - 2, 12, :stroke)
    end
    setline(1.6)
    setcolor(INK)
    f()
    finish()
    place_labels(path)
    compact_svg(path)
    println("  ", rpad(relpath * ".svg", 44), w, "×", h, "  ", round(Int, filesize(path) / 1024), " kB")
    return path
end

_short(x::AbstractString) = (v = round(parse(Float64, x); digits = 2); v == 0 ? "0" : rstrip(rstrip(string(v), '0'), '.'))

"""
    compact_svg(path)

Rewrite Cairo's SVG output without its redundancy: every number rounded to the
hundredth of a drawing unit (Typst glyph outlines come out with six decimals),
and the attributes that only restate SVG defaults dropped. Nothing visible
changes; the files shrink by about half.
"""
function compact_svg(path::AbstractString)
    s = read(path, String)
    s = replace(s, r"-?\d+\.\d{3,}" => _short)
    for a in (
            " fill-opacity=\"1\"", " stroke-opacity=\"1\"", " stroke-linecap=\"butt\"",
            " stroke-linejoin=\"miter\"", " fill-rule=\"nonzero\"",
        )
        s = replace(s, a => "")
    end
    write(path, s)
    return nothing
end

# ── Labels (Typst) ────────────────────────────────────────────────────────────

const TEXT_FONT = "DejaVu Sans"
const REF_PT = 20.0          # labels are compiled at this size, then scaled

"A compiled label: its size as Luxor measures it, its body, its glyph definitions."
struct Compiled
    width::Float64
    height::Float64
    body::String
    defs::Vector{Pair{String, String}}
end

const _LABELS = Dict{Tuple{String, String, String}, Compiled}()
const _GLYPHS = Dict{String, String}()                       # id => definition, for one sheet
const _PLACED = Tuple{String, Float64, Float64, Float64}[]  # (body, x, y, scale), for one sheet

_hex(c::AbstractString) = c
_hex(c) = "#" * Luxor.Colors.hex(Luxor.Colors.RGB(c))

const _DEF = r"<(symbol|clipPath|linearGradient|radialGradient|pattern)\b[^>]*\bid=\"([^\"]+)\".*?</\1>"s

function _compile(src::AbstractString, color, font::AbstractString)
    key = (String(src), _hex(color), String(font))
    return get!(_LABELS, key) do
        # The Typst compiler shipped with Typstry, called directly: a page that
        # shrinks to its content, no margin, no background.
        doc = """
        #set page(width: auto, height: auto, margin: 0pt, fill: none)
        #set text(size: $(REF_PT)pt, fill: rgb("$(_hex(color))"), font: "$(font)")
        #show math.equation: set text(font: "New Computer Modern Math")
        $(src)
        """
        io, err = IOBuffer(), IOBuffer()
        try
            Typstry.run_typst(
                c -> pipeline(c; stdin = IOBuffer(doc), stdout = io, stderr = err),
                TypstCommand(["compile", "--format", "svg", "-", "-"]),
            )
        catch
            error("Typst rejected the label `$(src)`:\n", String(take!(err)))
        end
        svg = String(take!(io))
        # Luxor's measure of the label sets the layout, as it always has; the
        # label itself is Typst's own SVG, whose glyphs are symbols named by a
        # hash of their outline, so that a sheet defines each glyph once.
        img = readsvg(svg)
        defs = [m[2] => m.match for m in eachmatch(_DEF, svg)]
        inner = svg[(findfirst('>', svg) + 1):(findlast("</svg>", svg)[1] - 1)]
        body = replace(inner, r"<defs\b.*?</defs>"s => "")
        Compiled(img.width, img.height, body, defs)
    end
end

"""
    label(src, pos; size = 18, color = INK, halign = :center, valign = :middle, bg = nothing)

Typeset the Typst source `src` (words, or mathematics between `\$…\$`) at
`pos`, with a font size of `size` drawing units, on a plate of color `bg` if
given. Returns `(width, height)`. The label is written into the sheet when it is
finished, above the drawing.
"""
function label(
        src::AbstractString, pos::Point;
        size::Real = 18, color = INK, halign::Symbol = :center, valign::Symbol = :middle,
        font::AbstractString = TEXT_FONT, bg = nothing,
    )
    lab = _compile(src, color, font)
    k = size / REF_PT
    w, h = lab.width * k, lab.height * k
    x = halign === :left ? pos.x : halign === :right ? pos.x - w : pos.x - w / 2
    y = valign === :top ? pos.y : valign === :bottom ? pos.y - h : pos.y - h / 2
    if bg !== nothing   # a backing plate, for a label over a busy drawing
        @layer (setcolor(bg); box(Point(x + w / 2, y + h / 2), w + 8, h + 6, 4, :fill))
    end
    # the user point (x, y) in the sheet's own coordinates
    M = getmatrix()
    push!(_PLACED, (lab.body, M[1] * x + M[3] * y + M[5], M[2] * x + M[4] * y + M[6], k))
    foreach(((id, d),) -> get!(_GLYPHS, id, d), lab.defs)
    return (w, h)
end

"Write the labels of a finished sheet into its SVG: the glyphs once, then each label."
function place_labels(path::AbstractString)
    isempty(_PLACED) && return nothing
    s = read(path, String)
    root = findfirst("<svg", s)[1]
    occursin("xmlns:xlink", s[root:findnext('>', s, root)]) ||
        (s = replace(s, "<svg " => "<svg xmlns:xlink=\"http://www.w3.org/1999/xlink\" "; count = 1))
    io = IOBuffer()
    print(io, "<defs>")
    foreach(id -> print(io, _GLYPHS[id]), sort!(collect(keys(_GLYPHS))))
    print(io, "</defs>")
    for (body, x, y, k) in _PLACED
        print(io, "<g transform=\"matrix(", k, " 0 0 ", k, " ", x, " ", y, ")\">", body, "</g>")
    end
    i = findlast("</svg>", s)[1]
    write(path, s[1:(i - 1)] * String(take!(io)) * s[i:end])
    return nothing
end

"Mathematics: `m(\"bb(C)^I\")` is the Typst source `\$bb(C)^I\$`."
m(s::AbstractString) = "\$" * s * "\$"

# ── Primitives ────────────────────────────────────────────────────────────────

"An ellipse of semi-axes `(a, b)` centered at `c`, its `a` axis at angle `θ`."
function ell(c::Point, a, b, θ = 0.0; fill = nothing, stroke = INK, lw = 1.4, dash = nothing)
    @layer begin
        translate(c)
        rotate(θ)
        if fill !== nothing
            setcolor(fill)
            ellipse(O, 2a, 2b, :fill)
        end
        if stroke !== nothing
            setcolor(stroke)
            setline(lw)
            dash === nothing ? setdash("solid") : setdash(dash)
            ellipse(O, 2a, 2b, :stroke)
            setdash("solid")
        end
    end
    return nothing
end

disk(c::Point, r; kw...) = ell(c, r, r; kw...)

"A straight arrow from `a` to `b`."
function arrow_ab(a::Point, b::Point; color = INK, lw = 1.6, head = 9)
    @layer begin
        setcolor(color)
        setline(lw)
        arrow(a, b; arrowheadlength = head, arrowheadangle = π / 7, linewidth = lw)
    end
    return nothing
end

"""
    unbounded_box(c, w, h; fill = MATRIX)

The infinite medium of an auxiliary problem: a rounded box with a dashed edge
and `∞` in its upper right corner.
"""
function unbounded_box(c::Point, w, h; fill = MATRIX, label_tensor = nothing)
    setcolor(fill)
    box(c, w, h, 10, :fill)
    @layer begin
        setcolor(MATRIX_EDGE)
        setline(1.6)
        setdash([7, 5])
        box(c, w, h, 10, :stroke)
    end
    label(m("infinity"), c + Point(w / 2 - 18, -h / 2 + 16); size = 18)
    label_tensor === nothing ||
        label(m(label_tensor), c + Point(w / 2 - 9, h / 2 - 17); size = 20, halign = :right)
    return nothing
end

"""
    remote_loading(corner, src; dir = (1, -1))

The arrow of a remote boundary condition leaving a box's corner, with its label.
"""
function remote_loading(corner::Point, src::AbstractString; len = 34, size = 18)
    tip = corner + Point(len, -len)
    arrow_ab(corner, tip; color = INK)
    label(m(src), tip + Point(-10, -14); size = size, halign = :right, valign = :bottom)
    return nothing
end

"The two tangents common to the circles `(c1, r1)` and `(c2, r2)` that leave both on the same side."
function external_tangents(c1::Point, r1, c2::Point, r2)
    v = c2 - c1
    d = hypot(v.x, v.y)
    u = v / d
    w = Point(-u.y, u.x)
    φ = acos((r1 - r2) / d)
    return map((1, -1)) do s
        n = cos(φ) * u + s * sin(φ) * w
        (c1 + r1 * n, c2 + r2 * n)
    end
end

"The two points where the tangents from `p` touch the circle `(c, r)`."
function tangent_points(p::Point, c::Point, r)
    v = p - c
    d = hypot(v.x, v.y)
    u = v / d
    w = Point(-u.y, u.x)
    β = acos(r / d)
    return (c + r * (cos(β) * u + sin(β) * w), c + r * (cos(β) * u - sin(β) * w))
end

function _cone_line(a::Point, b::Point)
    @layer begin
        setcolor(HIGHLIGHT)
        setline(1.6)
        setdash([6, 4])
        line(a, b, :stroke)
    end
    return nothing
end

"""
    zoom_cone(spot, rs, target, rt)

The dashed lines that carry a spot of radius `rs` to the circle of radius `rt`
it is magnified into: the two common tangents, drawn over both panels so that
they reach the spot.
"""
function zoom_cone(spot::Point, rs, target::Point, rt)
    foreach(((a, b),) -> _cone_line(a, b), external_tangents(spot, rs, target, rt))
    return nothing
end

"The same, to a rectangle: from the spot to the two corners `top` and `bottom`."
function zoom_cone(spot::Point, rs, top::Point, bottom::Point)
    p, q = tangent_points(top, spot, rs)
    _cone_line(p.y < q.y ? p : q, top)
    p, q = tangent_points(bottom, spot, rs)
    _cone_line(p.y > q.y ? p : q, bottom)
    return nothing
end

spot(c::Point, r) = disk(c, r; fill = nothing, stroke = HIGHLIGHT, lw = 2.4)

"""
    pack(rng, n, inside, radius; gap = 3, tries = 20000) -> Vector{Tuple{Point, Float64}}

Up to `n` non-overlapping disks, their radii drawn by `radius(rng)`, whose
centers satisfy `inside(p, r)`.
"""
function pack(rng, n, inside, radius; gap = 3, tries = 20000, avoid = Tuple{Point, Float64}[])
    out = Tuple{Point, Float64}[]
    lo, hi = -2000.0, 2000.0
    for _ in 1:tries
        length(out) >= n && break
        r = radius(rng)
        p = inside(rng, r)
        p === nothing && continue
        any(distance(p, q) < r + s + gap for (q, s) in out) && continue
        any(distance(p, q) < r + s + gap for (q, s) in avoid) && continue
        push!(out, (p, r))
    end
    return out
end

"A uniform point of the disk of center `c` and radius `R - r` (`nothing` if none)."
in_disk(c::Point, R) = (rng, r) -> begin
    R - r <= 0 && return nothing
    ρ = (R - r) * sqrt(rand(rng))
    θ = 2π * rand(rng)
    c + Point(ρ * cos(θ), ρ * sin(θ))
end

"A uniform point of the ellipse of center `c` and semi-axes `(A, B)`, shrunk by `r`."
in_ellipse(c::Point, A, B) = (rng, r) -> begin
    (A - r <= 0 || B - r <= 0) && return nothing
    ρ = sqrt(rand(rng))
    θ = 2π * rand(rng)
    c + Point((A - r) * ρ * cos(θ), (B - r) * ρ * sin(θ))
end

"A uniform point of the box of center `c` and size `w × h`, shrunk by `r`."
in_box(c::Point, w, h) = (rng, r) -> c + Point((w / 2 - r) * (2rand(rng) - 1), (h / 2 - r) * (2rand(rng) - 1))

# ── Voronoi cells (half-plane clipping) ───────────────────────────────────────

function _clip(poly::Vector{Point}, a::Point, n::Point, d)
    # keep the part of `poly` where n⋅x ≤ d
    out = Point[]
    m = length(poly)
    for i in 1:m
        p, q = poly[i], poly[mod1(i + 1, m)]
        sp, sq = n.x * p.x + n.y * p.y - d, n.x * q.x + n.y * q.y - d
        sp <= 0 && push!(out, p)
        if sp * sq < 0
            t = sp / (sp - sq)
            push!(out, p + t * (q - p))
        end
    end
    return out
end

"""
    voronoi(seeds, frame) -> Vector{Vector{Point}}

The Voronoi cell of every seed, clipped to the convex polygon `frame`.
"""
function voronoi(seeds::Vector{Point}, frame::Vector{Point})
    cells = Vector{Vector{Point}}()
    for (i, s) in enumerate(seeds)
        cell = copy(frame)
        for (j, t) in enumerate(seeds)
            i == j && continue
            n = t - s
            mid = (s + t) / 2
            cell = _clip(cell, mid, n, n.x * mid.x + n.y * mid.y)
            isempty(cell) && break
        end
        push!(cells, cell)
    end
    return cells
end

"Interpolate between two colors: `t = 0` gives `a`, `t = 1` gives `b`."
function mix(a, b, t::Real)
    ca, cb = parse(Luxor.Colors.RGB, _hex(a)), parse(Luxor.Colors.RGB, _hex(b))
    f(x, y) = (1 - t) * x + t * y
    return Luxor.Colors.RGB(f(ca.r, cb.r), f(ca.g, cb.g), f(ca.b, cb.b))
end

"The centroid of a simple polygon."
function centroid(poly::Vector{Point})
    A, cx, cy = 0.0, 0.0, 0.0
    n = length(poly)
    for i in 1:n
        p, q = poly[i], poly[mod1(i + 1, n)]
        c = p.x * q.y - q.x * p.y
        A += c
        cx += (p.x + q.x) * c
        cy += (p.y + q.y) * c
    end
    return Point(cx / (3A), cy / (3A))
end

"""
    lloyd(seeds, frame; iterations = 2) -> (seeds, cells)

Voronoi cells relaxed by `iterations` Lloyd steps (each seed moved to the
centroid of its cell), for grains of comparable size.
"""
function lloyd(seeds::Vector{Point}, frame::Vector{Point}; iterations::Int = 2)
    cells = voronoi(seeds, frame)
    for _ in 1:iterations
        seeds = [centroid(c) for c in cells]
        cells = voronoi(seeds, frame)
    end
    return seeds, cells
end

# ── Microstructure: the inclusion families of the scheme drawings ─────────────
# After the Echoes book: ellipsoids, needles and coated spheres, one color each,
# drawn identically in every figure.

abstract type Family end
struct Ellipsoidal <: Family end
struct Needle <: Family end
struct Coated <: Family end

const THREE_FAMILIES = (Ellipsoidal(), Needle(), Coated())

"Radius of the disk that holds an inclusion of the family at scale 1."
reach(::Ellipsoidal) = 14.0
reach(::Needle) = 17.0
reach(::Coated) = 10.0

"Draw one inclusion centered at `p`, its long axis at angle `θ`, at scale `s`."
draw_inclusion(::Ellipsoidal, p::Point, θ, s) = ell(p, 14s, 9s, θ; fill = BLUE, stroke = nothing)
draw_inclusion(::Needle, p::Point, θ, s) = ell(p, 17s, 4.5s, θ; fill = ORANGE, stroke = nothing)
function draw_inclusion(::Coated, p::Point, θ, s)
    disk(p, 10s; fill = GREEN, stroke = nothing)
    disk(p, 5.5s; fill = GOLD, stroke = nothing)
    return nothing
end

"""
    scatter(rng, kinds, inside; s = 1, gap = 6, avoid = []) -> Vector

One inclusion of each family of `kinds`, at a point drawn by `inside(rng, r)`
and a random angle, none overlapping another nor a disk of `avoid`. Returns
`(family, center, angle)` triples; a family that finds no room is left out.
"""
function scatter(
        rng, kinds, inside; s::Real = 1.0, gap::Real = 6.0, tries::Int = 5000,
        avoid = Tuple{Point, Float64}[]
    )
    placed = Tuple{Point, Float64}[]
    out = Tuple{Family, Point, Float64}[]
    for k in kinds
        r = reach(k) * s
        for _ in 1:tries
            p = inside(rng, r)
            p === nothing && continue
            all(distance(p, q) >= r + t + gap for (q, t) in placed) || continue
            all(distance(p, q) >= r + t + gap for (q, t) in avoid) || continue
            push!(placed, (p, r))
            push!(out, (k, p, π * rand(rng)))
            break
        end
    end
    return out
end

draw_all(items; s::Real = 1.0) = foreach(((k, p, θ),) -> draw_inclusion(k, p, θ, s), items)

"A bounded RVE: a rounded box with a solid edge."
function rve_box(c::Point, w, h; fill = MATRIX)
    setcolor(fill)
    box(c, w, h, 10, :fill)
    @layer (setcolor(INK); setline(1.6); box(c, w, h, 10, :stroke))
    return nothing
end

"The auxiliary problem: an unbounded medium, its tensor, its remote loading."
function aux_box(
        c::Point, w, h; fill = MATRIX, tensor = "bb(C)_0",
        load = "underline(u) = bold(E)^0 dot underline(x)", size = 16
    )
    unbounded_box(c, w, h; fill = fill, label_tensor = tensor)
    remote_loading(c + Point(w / 2, -h / 2), load; size = size)
    return nothing
end

"The arrow that reads «this RVE gives these auxiliary problems»."
then_arrow(a::Point, b::Point) = arrow_ab(a, b; color = MUTED, lw = 2.2, head = 12)

"Words typeset verbatim: no Typst markup is interpreted (`_`, `*`, `<`… stay as they are)."
txt(s::AbstractString) = "#\"" * replace(s, "\\" => "\\\\", "\"" => "\\\"") * "\""

"""
    settle(rng, radii, c; spread, iterations = 300, pull = 0.02, overlap = 0)

Disks of the given radii, packed densely around `c`: scattered over a square of
half-side `spread`, then pulled toward `c` and pushed apart wherever two of them
overlap by more than `overlap`, until they press on one another. The result
has the contacts a random sequential packing lacks.
"""
function settle(rng, radii, c::Point; spread, iterations::Int = 300, pull = 0.02, overlap = 0.0)
    ps = [c + Point(spread * (2rand(rng) - 1), spread * (2rand(rng) - 1)) for _ in radii]
    n = length(ps)
    for _ in 1:iterations
        for i in 1:n
            ps[i] += pull * (c - ps[i])
        end
        for _ in 1:3, i in 1:n, j in (i + 1):n
            v = ps[j] - ps[i]
            d = max(hypot(v.x, v.y), 1.0e-6)
            m = radii[i] + radii[j] - overlap
            if d < m
                push = ((m - d) / 2 / d) * v
                ps[i] -= push
                ps[j] += push
            end
        end
    end
    return ps
end
