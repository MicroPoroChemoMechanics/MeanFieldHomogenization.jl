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

"""
The Gaussian approximation of a planar crack: the nodes of a square grid over a
contour with a contact island, the nodes kept, those whose cell is cut by an
edge, and the footprint of one Gaussian (left); the edge rule on a line of
nodes (right). Docs: theory/gaussian_cracks.md.
"""
function theory_gaussian_crack(; path = "theory/gaussian_crack_grid", panel::Bool = true)
    return sheet(path, 720, 330; panel = panel) do
        # ── the grid over the contour ──
        c = Point(180, 165)
        a, b, θ = 135.0, 95.0, -0.25            # the contour: an ellipse
        island, ri = c + Point(38, -12), 30.0    # and a bonded island
        inside(p) = begin
            q = p - c
            x, y = q.x * cos(θ) + q.y * sin(θ), -q.x * sin(θ) + q.y * cos(θ)
            (x / a)^2 + (y / b)^2 ≤ 1 && distance(p, island) > ri
        end
        ell(c, a, b, θ; fill = tint(RED, 0.86), stroke = CRACK, lw = 2.0)
        disk(island, ri; fill = tint(SLATE, 0.8), stroke = CRACK, lw = 2.0)
        h = 22.0
        # covered fraction of a cell, on 5 × 5 points
        fraction(p) = count(inside(p + Point((i - 3) * h / 5, (j - 3) * h / 5)) for i in 1:5, j in 1:5) / 25
        for i in -7:7, j in -5:5
            p = c + Point(i * h, j * h)
            f = fraction(p)
            if f == 1
                disk(p, 3.4; fill = BLUE, stroke = nothing)
            elseif f > 0
                disk(p, 3.4; fill = ORANGE, stroke = nothing)
            else
                disk(p, 1.5; fill = GUIDE, stroke = nothing)
            end
        end
        # the footprint of one Gaussian, of radius h √H with H = 2
        g = c + Point(-3h, 1h)
        disk(g, h * sqrt(2); fill = nothing, stroke = VECTOR, lw = 1.4, dash = [4, 3])
        label(m("h sqrt(H)"), g + Point(0, h * sqrt(2) + 14); size = 15, color = VECTOR, bg = tint(RED, 0.86))
        # the step
        s0 = c + Point(-7h, -5h - 16)
        arrow_ab(s0, s0 + Point(h, 0); color = INK, lw = 1.2, head = 5)
        arrow_ab(s0 + Point(h, 0), s0; color = INK, lw = 1.2, head = 5)
        label(m("h"), s0 + Point(h / 2, -12); size = 15)
        label(m("cal(S)"), c + Point(-70, -50); size = 18, color = CRACK)
        label(txt("bonded"), island; size = 11, color = INK)
        # legend
        ly = 312.0
        disk(Point(40, ly), 3.4; fill = BLUE, stroke = nothing)
        label(txt("node inside"), Point(50, ly); size = 12, color = MUTED, halign = :left)
        disk(Point(150, ly), 3.4; fill = ORANGE, stroke = nothing)
        label(txt("cell cut by an edge"), Point(160, ly); size = 12, color = MUTED, halign = :left)
        disk(Point(300, ly), 1.5; fill = GUIDE, stroke = nothing)
        label(txt("outside"), Point(310, ly); size = 12, color = MUTED, halign = :left)

        @layer (setcolor(PANEL_EDGE); setline(1.2); line(Point(375, 14), Point(375, 316), :stroke))

        # ── the edge rule on a line of nodes ──
        x0, y, H1 = 420.0, 190.0, 52.0
        edge = x0 + 3.62 * H1                    # the edge of the region, inside the fourth cell
        @layer begin
            setcolor(tint(RED, 0.86))
            box(Point(x0 - 14, y - 9), Point(edge, y + 9), :fill)
            setcolor(CRACK)
            setline(2.0)
            line(Point(x0 - 14, y - 9), Point(edge, y - 9), :stroke)
            line(Point(x0 - 14, y + 9), Point(edge, y + 9), :stroke)
            line(Point(edge, y - 9), Point(edge, y + 9), :stroke)
        end
        for k in 0:5
            xc = x0 + k * H1
            @layer (setcolor(GUIDE); setline(1.0); setdash([3, 3]); line(Point(xc, y - 30), Point(xc, y + 30), :stroke))
        end
        for k in 0:4
            xn = x0 + (k + 0.5) * H1
            disk(Point(xn, y), 4.0; fill = k < 3 ? BLUE : k == 3 ? ORANGE : GUIDE, stroke = nothing)
        end
        # the covered fraction of the cut cell
        @layer begin
            setcolor(INK)
            setline(1.1)
            line(Point(x0 + 3H1, y + 44), Point(edge, y + 44), :stroke)
            line(Point(x0 + 3H1, y + 39), Point(x0 + 3H1, y + 49), :stroke)
            line(Point(edge, y + 39), Point(edge, y + 49), :stroke)
        end
        label(m("theta h"), Point((x0 + 3H1 + edge) / 2, y + 58); size = 15)
        # the spring that holds the node of the cut cell
        xs = x0 + 3.5 * H1
        @layer begin
            setcolor(ORANGE)
            setline(1.8)
            zig = [Point(xs + (isodd(k) ? 7 : -7) * (0 < k < 8), y - 12 - 6k) for k in 0:8]
            poly(zig, :stroke)
            line(Point(xs - 12, y - 60), Point(xs + 12, y - 60), :stroke)
        end
        label(m("gamma (1 - theta) / theta"), Point(xs + 16, y - 40); size = 15, color = ORANGE, halign = :left)
        label(txt("crack"), Point(x0 + 1.2 * H1, y - 22); size = 13, color = CRACK)
        label(txt("a spring on the node of the cut cell"), Point(550, 62); size = 13, color = MUTED)
        label(txt("coverage counted on the region eroded by ") * m("c_0 h"), Point(550, 290); size = 13, color = MUTED)
    end
end

theory_figures() = (theory_residue_poles(), theory_gaussian_crack())
