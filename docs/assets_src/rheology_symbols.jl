# =============================================================================
#  rheology_symbols.jl — the drawing primitives for rheological networks.
#
#  Springs, dashpots and springpots, plus the two combinators (series and
#  parallel) that every model in the catalog is built from.  Luxor gives vector
#  output, so the SVGs stay crisp at any zoom and weigh a few kilobytes.
#
#  Everything is laid out on a **left-to-right axis**: an element occupies the
#  segment `[x1, x2]` at height `y`, and returns nothing.  Composition is done
#  by the caller, which knows the geometry it wants — that is simpler than a
#  layout engine and produces better-looking diagrams, because the spacing of a
#  three-branch Prony chain is not the spacing of a Maxwell unit.
#
#  Colors follow the package documentation's palette: elastic elements blue,
#  viscous elements orange, fractional (springpot) elements purple.
# =============================================================================

using Luxor
using LaTeXStrings
using MathTeXEngine   # activates Luxor's LaTeX text extension

const COL_SPRING = "#2E6DA4"     # elastic
const COL_DASHPOT = "#D9822B"    # viscous
const COL_POT = "#7B4EA3"        # fractional
const COL_WIRE = "#333333"
const COL_LABEL = "#1A1A1A"

const LW = 2.0                   # line width of the elements
const LW_WIRE = 1.6

# ── Layout constants ────────────────────────────────────────────────────────
#
#  Hand-tuned offsets are what produce overlapping labels, so the generator
#  uses these instead and never places anything by eye.

"Vertical distance between two branches of a parallel stack."
const PITCH = 58

"Offset from an element's axis to its label, placed **above** it."
const LABEL_UP = 26

"Offset from the lowest element of a diagram to its caption."
const CAPTION_DROP = 46


"Straight lead wire from `a` to `b`."
function wire(a::Point, b::Point)
    setcolor(COL_WIRE)
    setline(LW_WIRE)
    line(a, b, :stroke)
    return nothing
end

"""
    spring(x1, x2, y; coils = 6, h = 11, label = "", color = COL_SPRING)

A helical spring drawn as a zigzag between `(x1, y)` and `(x2, y)`, with short
straight leads at each end so it can be wired to its neighbors.

`label` is placed **below** the element; the caller places anything above.
"""
function spring(
        x1, x2, y; coils::Union{Nothing, Int} = nothing, h = 11, label = "",
        color = COL_SPRING, lead = 0.16, label_above::Bool = true
    )
    L = x2 - x1
    l = lead * L
    xa, xb = x1 + l, x2 - l
    ## Coil *density* rather than a fixed count: a spring spanning a whole
    ## parallel frame must not come out as four enormous zigzags.
    coils = coils === nothing ? clamp(round(Int, (xb - xa) / 26), 4, 12) : coils
    wire(Point(x1, y), Point(xa, y))
    wire(Point(xb, y), Point(x2, y))

    setcolor(color)
    setline(LW)
    n = 2coils
    step = (xb - xa) / n
    pts = [Point(xa, y)]
    for i in 1:n
        # A zigzag: alternate above and below the axis, with the first and last
        # half-steps flat so the coil meets the lead squarely.
        yy = y + (isodd(i) ? -h : h)
        push!(pts, Point(xa + (i - 0.5) * step, yy))
    end
    push!(pts, Point(xb, y))
    poly(pts, :stroke)
    _label(label, (x1 + x2) / 2, label_above ? y - LABEL_UP : y + LABEL_UP)
    return nothing
end

"""
    dashpot(x1, x2, y; h = 11, label = "", color = COL_DASHPOT)

A dashpot: an open cylinder with a piston on a rod. The cylinder opens to the
**right**, which is the usual convention and makes a series chain read left to
right.
"""
function dashpot(
        x1, x2, y; h = 11, label = "", color = COL_DASHPOT, lead = 0.14,
        label_above::Bool = true
    )
    L = x2 - x1
    l = lead * L
    xa, xb = x1 + l, x2 - l
    body = xb - xa
    ## The cylinder is closed on the **right** and open on the left; the rod
    ## enters from the left and carries the piston plate.  Drawn the other way
    ## round, the rod would cross the open end, which is what makes a dashpot
    ## look like a flat box.
    xcyl0 = xa + 0.2body             # open end (left)
    xpiston = xa + 0.5body           # piston plate, inside the cylinder
    xclosed = xb                      # closed end (right)

    ## A light fill for the bore: it is what makes the symbol read as a
    ## dashpot at a glance rather than as a box with a bar in it.
    setcolor(color)
    setopacity(0.13)
    rect(xcyl0, y - h, xclosed - xcyl0, 2h, :fill)
    setopacity(1.0)

    wire(Point(x1, y), Point(xpiston, y))     # rod
    wire(Point(xclosed, y), Point(x2, y))

    setcolor(color)
    setline(LW)
    ## cylinder: top, closed right end, bottom — deliberately open on the left
    poly(
        [
            Point(xcyl0, y - h), Point(xclosed, y - h),
            Point(xclosed, y + h), Point(xcyl0, y + h),
        ], :stroke
    )
    ## piston plate, filling the bore
    setline(LW + 1.4)
    line(Point(xpiston, y - h + 1.5), Point(xpiston, y + h - 1.5), :stroke)
    _label(label, (x1 + x2) / 2, label_above ? y - LABEL_UP : y + LABEL_UP)
    return nothing
end

"""
    springpot(x1, x2, y; order = "α", label = "", color = COL_POT)

The **parabolic element**, also called a springpot or a Scott-Blair element —
three names for one object, and the reason the drawing carries a parabola.

Its creep function is a power law, `J(t) ∝ t^α/Γ(1+α)`: a *parabolic branch*
when `α = 1/2`, which is where the asphalt literature's name comes from. The
element interpolates continuously between a spring (`α = 0`, the arc flattens
onto the axis) and a dashpot (`α = 1`, it straightens into a line), and that is
what the arc in the box shows.

`order` is drawn in the upper-left of the box — the exponent, `k` or `h` in the
bituminous models, `α` elsewhere — and `label` below the element.
"""
function springpot(
        x1, x2, y; h = 11, label = "", order = "α", color = COL_POT,
        lead = 0.2, label_above::Bool = true
    )
    L = x2 - x1
    l = lead * L
    xa, xb = x1 + l, x2 - l
    wire(Point(x1, y), Point(xa, y))
    wire(Point(xb, y), Point(x2, y))

    setcolor(color)
    setopacity(0.1)
    rect(xa, y - h, xb - xa, 2h, :fill)
    setopacity(1.0)
    setline(LW)
    rect(xa, y - h, xb - xa, 2h, :stroke)

    ## The parabola itself, rising from the bottom-left corner: the shape the
    ## element is named after.  A straight diagonal would say "somewhere
    ## between a spring and a dashpot"; the arc says *how*.
    w = xb - xa
    setline(LW - 0.4)
    pts = [Point(xa + u * w, y + h - 2h * sqrt(u)) for u in range(0, 1; length = 40)]
    poly(pts, :stroke)

    setcolor(COL_LABEL)
    fontsize(11)
    text(order, Point(xa + 0.22w, y - 0.4h), halign = :center)
    _label(label, (x1 + x2) / 2, label_above ? y - LABEL_UP : y + LABEL_UP)
    return nothing
end

"""
    parallel_frame(x1, x2, ys; y = 0)

The two vertical bars that put the branches at heights `ys` in parallel between
`x1` and `x2`, plus the leads that join them to the outside.
"""
function parallel_frame(x1, x2, ys; y = 0, lead = 16)
    setcolor(COL_WIRE)
    setline(LW_WIRE)
    ytop, ybot = minimum(ys), maximum(ys)
    line(Point(x1, ytop), Point(x1, ybot), :stroke)
    line(Point(x2, ytop), Point(x2, ybot), :stroke)
    wire(Point(x1 - lead, y), Point(x1, y))
    wire(Point(x2, y), Point(x2 + lead, y))
    return nothing
end

"Terminal stub with a filled node, marking where the network is loaded."
function terminal(x, y; left::Bool = true, len = 18)
    wire(Point(x, y), Point(left ? x - len : x + len, y))
    setcolor(COL_WIRE)
    circle(Point(left ? x - len : x + len, y), 2.6, :fill)
    return nothing
end

"""
    _label(s, x, y; size = 13)

Place an element label, centered on `(x, y)`.

Labels are `LaTeXString`s so that `E_\\infty`, `\\tau_1`, `V_b` and `E_{00}`
typeset properly — Luxor renders them through `MathTeXEngine`. That matters
here beyond looks: the drawings have to carry *the same symbols as the formulas
and the constructor signatures on the page*, and half of those cannot be
written in Unicode at all (there is no subscript `b`).

`MathTeXEngine` does not center a LaTeX string on its own, so the horizontal
offset is computed from `latextextsize`.
"""
function _label(s, x, y; size = 13)
    (s isa AbstractString && isempty(s)) && return nothing
    setcolor(COL_LABEL)
    fontsize(size)
    _tex_centered(s, x, y)
    return nothing
end

"Draw `s` centered on `(x, y)`, whether it is a LaTeXString or a plain String."
function _tex_centered(s::LaTeXString, x, y)
    w, _ = latextextsize(s)
    text(s, Point(x - w / 2, y))
    return nothing
end

_tex_centered(s::AbstractString, x, y) = (text(s, Point(x, y), halign = :center); nothing)

"Caption centered under a whole diagram."
function caption(s, x, y; size = 13, color = COL_LABEL)
    setcolor(color)
    fontsize(size)
    _tex_centered(s, x, y)
    return nothing
end

"Title above a diagram, in the package's link blue."
function title_at(s, x, y; size = 14)
    setcolor("#1F4E79")
    fontsize(size)
    _tex_centered(s, x, y)
    return nothing
end
