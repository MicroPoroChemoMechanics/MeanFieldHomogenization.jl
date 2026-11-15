# =============================================================================
#  theory.jl — sketches that accompany a derivation rather than a morphology.
#  Docs: theory/ (residues and their derivative).
# =============================================================================

"A pole of order 1 (a disk) or of order 2 (a disk in a ring), at `p`."
function pole_marker(p::Point, order::Int; color = BLUE, r = 4.0)
    if order == 1
        disk(p, r; fill = color, stroke = nothing)
    else
        disk(p, r; fill = color, stroke = nothing)
        disk(p, 2r; fill = nothing, stroke = color, lw = 1.8)
    end
    return nothing
end

"A closed circle of center `c` and radius `r`, run counterclockwise (as the plane is read), with its arrow head at the top."
function contour_circle(c::Point, r; color = VECTOR)
    @layer begin
        setcolor(color)
        setline(1.8)
        circle(c, r, :stroke)
        # at the top of the circle the counterclockwise direction points left
        tip = c + Point(-3, -r)
        d, n = Point(-1, 0), Point(0, 1)
        poly([tip + 4 * d, tip - 5 * d + 4.5 * n, tip - 5 * d - 4.5 * n], :fill; close = true)
    end
    return nothing
end

"The complex plane of one panel, its poles drawn at the given order."
function pole_panel(x0, title, order::Int)
    axis_y = 150.0
    mirror(p) = Point(p.x, 2axis_y - p.y)
    group = [Point(x0 + 140, 78), Point(x0 + 158, 90), Point(x0 + 142, 102)]
    gc, gr = Point(x0 + 147, 90), 34.0
    lone, lr = Point(x0 + 238, 108), 16.0
    label(title, Point(x0 + 150, 18); size = 14)
    @layer (setcolor(GUIDE); setline(1.4); line(Point(x0 + 8, axis_y), Point(x0 + 292, axis_y), :stroke))
    label(m("RR"), Point(x0 + 284, axis_y + 12); size = 13, color = MUTED, halign = :right)
    # conjugates, faded
    pc = order == 1 ? BLUE : PURPLE
    foreach(p -> pole_marker(mirror(p), order; color = tint(pc, 0.7)), [group; lone])
    contour_circle(gc, gr)
    contour_circle(lone, lr)
    foreach(p -> pole_marker(p, order; color = pc), [group; lone])
    label(m("Gamma"), gc + Point(gr * 0.78 + 6, -gr * 0.78 - 4); size = 15, color = VECTOR)
    return nothing
end

"The poles of the residue of a value and of its derivative: the same, one order higher."
function theory_residue_poles(; path = "theory/residue_derivative_poles", panel::Bool = true)
    return sheet(path, 640, 290; panel = panel) do
        pole_panel(10, "value: residues of " * m("N slash Q_0"), 1)
        pole_panel(330, "derivative: residues of " * m("-N delta Q slash Q_0^2"), 2)
        @layer (setcolor(PANEL_EDGE); setline(1.2); line(Point(320, 12), Point(320, 250), :stroke))
        pole_marker(Point(210, 272), 1)
        label(txt("pole of order 1"), Point(222, 272); size = 12, color = MUTED, halign = :left)
        pole_marker(Point(390, 272), 2; color = PURPLE)
        label(txt("pole of order 2"), Point(406, 272); size = 12, color = MUTED, halign = :left)
    end
end

theory_figures() = (theory_residue_poles(),)
