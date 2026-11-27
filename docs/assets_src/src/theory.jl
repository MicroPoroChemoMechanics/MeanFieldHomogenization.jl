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

"Image of the circle of radius `ρ` by the map of coefficients `ms`, as points around `c` at the scale `s`."
function mapped_circle(c::Point, s, ms, ρ; n = 361)
    ω(ζ) = ζ + sum(m * ζ^(-k) for (k, m) in ms)
    return [(z = ω(ρ * cis(θ)); c + Point(s * real(z), -s * imag(z))) for θ in range(0, 2π; length = n)]
end

"Image of the ray of angle `θ`, from the circle to the radius `ρ`."
function mapped_ray(c::Point, s, ms, θ, ρ; n = 40)
    ω(ζ) = ζ + sum(m * ζ^(-k) for (k, m) in ms)
    return [(z = ω(r * cis(θ)); c + Point(s * real(z), -s * imag(z))) for r in range(1, ρ; length = n)]
end

"One plane of the map: the matrix, the hole of contour the image of the unit circle, and the images of circles and rays."
function map_plane(c::Point, s, ms; ρmax = 1.9)
    @layer begin
        setcolor(MATRIX)
        poly(mapped_circle(c, s, ms, ρmax), :fill; close = true)
        setcolor(PORE)
        poly(mapped_circle(c, s, ms, 1.0), :fill; close = true)
        setcolor(GUIDE)
        setline(0.8)
        foreach(ρ -> poly(mapped_circle(c, s, ms, ρ), :stroke; close = true), (1.3, 1.6))
        foreach(θ -> poly(mapped_ray(c, s, ms, θ, ρmax), :stroke), range(0, 2π; length = 13)[1:12])
        setcolor(INK)
        setline(2.0)
        poly(mapped_circle(c, s, ms, 1.0), :stroke; close = true)
    end
    return nothing
end

"The conformal map of the outside of the unit disk onto the outside of a hole, and four holes it gives."
function theory_conformal_map(; path = "theory/conformal_map", panel::Bool = true)
    square = [(3, 1 / 6), (7, 1 / 56), (11, 1 / 176)]
    return sheet(path, 720, 400; panel = panel) do
        map_plane(Point(170, 135), 62, [(1, 0.0)])
        map_plane(Point(540, 135), 62, square)
        label(m("zeta"), Point(170, 135); size = 17)
        label(m("abs(zeta) = 1"), Point(170, 62); size = 13, bg = MATRIX)
        label(txt("hole"), Point(540, 135); size = 12, color = MUTED)
        label(txt("matrix"), Point(540, 30); size = 12, color = MUTED, bg = MATRIX)
        label(m("z = x_2 + upright(i) x_3"), Point(650, 250); size = 13, color = MUTED)
        arrow_ab(Point(300, 135), Point(410, 135); color = VECTOR, lw = 1.8)
        label(m("z = omega(zeta)"), Point(355, 118); size = 15, color = VECTOR)
        @layer (setcolor(PANEL_EDGE); setline(1.2); line(Point(20, 268), Point(700, 268), :stroke))
        holes = (
            ([(1, 0.4)], m("m_1 = 2 slash 5")),
            ([(2, 1 / 4)], m("m_2 = 1 slash 4")),
            (square, txt("square, 3 terms")),
            ([(4, 1 / 10), (9, 1 / 75)], txt("pentagon, 2 terms")),
        )
        for (i, (ms, name)) in enumerate(holes)
            c = Point(90 + 180 * (i - 1), 325)
            @layer begin
                setcolor(tint(ORANGE, 0.75))
                poly(mapped_circle(c, 30, ms, 1.0), :fill; close = true)
                setcolor(INK)
                setline(1.6)
                poly(mapped_circle(c, 30, ms, 1.0), :stroke; close = true)
            end
            label(name, c + Point(0, 58); size = 12, color = MUTED)
        end
    end
end

"A block of matrix whose front face, of side `a` and center `c`, carries the sections `holes`, extruded by the vector `depth`."
function extruded_block(c::Point, a, depth::Point, holes)
    h = a / 2
    front = [c + Point(-h, -h), c + Point(h, -h), c + Point(h, h), c + Point(-h, h)]
    @layer begin
        setline(1.4)
        # the top and the side, then the front face over them
        for face in ([front[1], front[2], front[2] + depth, front[1] + depth], [front[2], front[3], front[3] + depth, front[2] + depth])
            setcolor(tint(BLUE, 0.78))
            poly(face, :fill; close = true)
            setcolor(INK)
            poly(face, :stroke; close = true)
        end
        setcolor(MATRIX)
        poly(front, :fill; close = true)
        setcolor(INK)
        poly(front, :stroke; close = true)
        for (offset, s, ms) in holes
            setcolor(PORE)
            poly(mapped_circle(c + offset, s, ms, 1.0), :fill; close = true)
            setcolor(INK)
            poly(mapped_circle(c + offset, s, ms, 1.0), :stroke; close = true)
        end
    end
    return nothing
end

"""
    theory_plane_problem(; path = "theory/plane_problem")

The two states of a plane problem and the section they share: a long prism, in
plane strain, a thin plate, in plane stress, and the plane of the section with
its holes.
"""
function theory_plane_problem(; path = "theory/plane_problem", panel::Bool = true)
    holes = ((Point(-24, -20), 15, [(1, 0.0)]), (Point(22, 22), 16, [(2, 0.25)]), (Point(26, -30), 9, [(1, 0.45)]))
    return sheet(path, 720, 330; panel = panel) do
        # plane strain: a long prism
        extruded_block(Point(115, 185), 120, Point(70, -50), holes)
        arrow_ab(Point(175, 125), Point(231, 85); color = VECTOR, lw = 1.8)
        label(m("underline(e)_1"), Point(262, 84); size = 15, color = VECTOR)
        label(txt("plane strain"), Point(130, 278); size = 14)
        label(m("epsilon_(1 1) = epsilon_(1 2) = epsilon_(1 3) = 0"), Point(130, 302); size = 13, color = MUTED)
        # the section
        c = Point(380, 185)
        extruded_block(c, 120, Point(0, 0), holes)
        arrow_ab(c + Point(-60, 60), c + Point(-10, 60); color = VECTOR, lw = 1.8)
        arrow_ab(c + Point(-60, 60), c + Point(-60, 10); color = VECTOR, lw = 1.8)
        label(m("underline(e)_2"), c + Point(-8, 74); size = 15, color = VECTOR)
        label(m("underline(e)_3"), c + Point(-42, 18); size = 15, color = VECTOR)
        label(txt("the section"), Point(380, 278); size = 14)
        label(m("k, mu quad \"or\" quad k^*, mu"), Point(380, 302); size = 13, color = MUTED)
        # plane stress: a thin plate
        extruded_block(Point(600, 185), 120, Point(13, -9), holes)
        label(txt("plane stress"), Point(605, 278); size = 14)
        label(m("sigma_(1 1) = sigma_(1 2) = sigma_(1 3) = 0"), Point(605, 302); size = 13, color = MUTED)
        # both lead to the same problem of the plane
        arrow_ab(Point(262, 185), Point(306, 185); color = GUIDE, lw = 1.6)
        arrow_ab(Point(528, 185), Point(454, 185); color = GUIDE, lw = 1.6)
    end
end

"""
    theory_parameter_classes(; path = "theory/parameter_classes")

The symmetry classes a coefficient of a property lens may belong to, nested
around an isotropic tensor ℂ: isotropic, transversely isotropic about n,
orthotropic in a frame that has n as an axis, anisotropic. A dual number put on
one coefficient of a class (here ℓ₅) moves ℂ inside that class, out of the
smaller ones, the other coefficients staying fixed.
"""
function theory_parameter_classes(; path = "theory/parameter_classes", panel::Bool = true)
    return sheet(path, 720, 330; panel = panel) do
        c = Point(235, 180)
        classes = (
            (430, 270, tint(BLUE, 0.93), m("\"ANISO\" quad M_(i j)"), txt("Kelvin-Mandel components in a basis")),
            (330, 200, tint(BLUE, 0.84), m("\"ORTHO\" quad C_(1 1 1 1), dots, C_(1 2 1 2)"), txt("in a frame")),
            (230, 134, tint(BLUE, 0.72), m("\"TI\"(underline(n)) quad ell_1, dots, ell_6"), nothing),
            (120, 62, tint(PURPLE, 0.7), m("\"ISO\" quad k, mu"), nothing),
        )
        for (w, h, col, _, _) in classes
            @layer (setcolor(col); box(c, w, h, 14, :fill); setcolor(MUTED); setline(1.0); box(c, w, h, 14, :stroke))
        end
        for (k, (w, h, _, name, note)) in enumerate(classes)
            k == 4 && continue
            label(name, Point(c.x - w / 2 + 12, c.y - h / 2 + 15); size = 14, halign = :left)
            note === nothing || label(note, Point(c.x + w / 2 - 12, c.y - h / 2 + 15); size = 11, color = MUTED, halign = :right)
        end
        label(classes[4][4], c + Point(0, -17); size = 14)
        # the tensor, and its move along one coefficient of TI
        p = c + Point(-22, 8)
        @layer (setcolor(INK); circle(p, 3.5, :fill))
        label(m("bb(C)"), p + Point(-14, 0); size = 15)
        q = c + Point(72, 40)
        arrow_ab(p + Point(4, 2), q; color = HIGHLIGHT, lw = 2.0)
        label(m("ell_5 + epsilon"), q + Point(-28, 16); size = 14, color = HIGHLIGHT, halign = :left)
        # the rule, on the right
        x = 478
        label(txt("a coefficient of a class"), Point(x, 110); size = 14, halign = :left)
        label(txt("that contains the tensor:"), Point(x, 132); size = 14, halign = :left)
        label(m("ell_5 arrow.r.bar ell_5 + epsilon"), Point(x, 168); size = 15, halign = :left, color = HIGHLIGHT)
        label(txt("the others fixed;"), Point(x, 204); size = 14, halign = :left)
        label(m("epsilon") * txt(" a dual number gives"), Point(x, 226); size = 14, halign = :left)
        label(m("partial bb(C)^(\"hom\") \\/ partial ell_5"), Point(x, 254); size = 15, halign = :left)
    end
end

"""
The radial problem (theory/radial_problems.md). Left: a section of a sphere or
of a cylinder whose layers are uniform, transversely isotropic about the radius
(the arcs are its surfaces of isotropy) or graded, with the local frame at a
point. Right: the same layers along the radius, and what carries the state Y
across each of them, from the regular solutions at the center to the matrix.
"""
function theory_radial_problem(; path = "theory/radial_problem", panel::Bool = true)
    return sheet(path, 800, 380; panel = panel) do
        core, sheets, inner, outer = tint(BLUE, 0.55), tint(ORANGE, 0.45), tint(GREEN, 0.25), tint(GREEN, 0.85)
        graded(t) = mix(inner, outer, t)
        # ── the section ──
        c = Point(185, 195)
        unbounded_box(c, 330, 320; label_tensor = "bb(C)_0")
        r₁, r₂, R = 42.0, 84.0, 126.0
        for k in 12:-1:1
            disk(c, r₂ + (R - r₂) * k / 12; fill = graded((k - 0.5) / 12), stroke = nothing)
        end
        disk(c, r₂; fill = sheets, stroke = nothing)
        @layer begin
            setcolor(tint(ORANGE, 0.05))
            setline(0.9)
            for r in (r₁ + 7):7:(r₂ - 4)
                circle(c, r, :stroke)
            end
        end
        disk(c, r₁; fill = core, stroke = nothing)
        foreach(r -> disk(c, r; fill = nothing, stroke = INK, lw = 1.4), (r₁, r₂, R))
        # the local frame at a point of the second layer
        θ = -0.62
        er, eθ = Point(cos(θ), sin(θ)), Point(sin(θ), -cos(θ))
        x = c + 63 * er
        @layer (setcolor(GUIDE); setline(1.0); setdash([3, 4]); line(c, x, :stroke))
        arrow_ab(x, x + 40 * er; color = VECTOR, lw = 2.0)
        arrow_ab(x, x + 36 * eθ; color = VECTOR, lw = 2.0)
        @layer (setcolor(INK); circle(x, 2.8, :fill))
        label(m("underline(e)_r"), x + 40 * er + Point(13, 6); size = 15, color = VECTOR)
        label(m("underline(e)_theta"), x + 36 * eθ + Point(-4, -11); size = 15, color = VECTOR)
        # ── the chain along the radius ──
        x₀, x₁, x₂, x₃, x₄ = 400.0, 480.0, 575.0, 670.0, 770.0
        top, base = 205.0, 250.0
        band(xa, xb, fill) = @layer (setcolor(fill); box(Point((xa + xb) / 2, (top + base) / 2), xb - xa, base - top, :fill))
        band(x₀, x₁, core)
        band(x₁, x₂, sheets)
        for k in 1:12
            band(x₂ + (x₃ - x₂) * (k - 1) / 12, x₂ + (x₃ - x₂) * k / 12 + 0.6, graded((k - 0.5) / 12))
        end
        band(x₃, x₄, MATRIX)
        @layer begin
            setcolor(INK)
            setline(1.4)
            foreach(xk -> line(Point(xk, top - 8), Point(xk, base + 6), :stroke), (x₁, x₂, x₃))
            line(Point(x₀, top), Point(x₀, base + 6), :stroke)
        end
        arrow_ab(Point(x₀, base), Point(x₄ + 12, base); color = INK, lw = 1.4)
        mid = (top + base) / 2
        label(txt("uniform"), Point((x₀ + x₁) / 2, mid); size = 11)
        label(m("\"TI\"(underline(e)_r)"), Point((x₁ + x₂) / 2, mid); size = 13)
        label(txt("graded"), Point((x₂ + x₃) / 2, mid); size = 11)
        label(txt("matrix"), Point((x₃ + x₄) / 2, mid); size = 11)
        label(m("r"), Point(x₄ + 10, base + 15); size = 15)
        for (name, xk) in (("0", x₀), ("r_1", x₁), ("r_2", x₂), ("R", x₃))
            label(m(name), Point(xk, base + 18); size = 15)
        end
        for (name, xk) in (("bold(J)_1", x₁), ("bold(J)_2", x₂), ("bold(J)_3", x₃))
            label(m(name), Point(xk, top - 20); size = 15, color = HIGHLIGHT)
        end
        label(m("Y = (\"displacement\", thick r times \"traction\")"), Point((x₀ + x₄) / 2, 62); size = 15)
        label(txt("regular at the center"), Point((x₀ + x₁) / 2 + 4, 108); size = 11, color = MUTED)
        label(m("1/2 (bold(1) + op(\"sign\") bold(X))"), Point((x₀ + x₁) / 2 + 4, 132); size = 13)
        label(m("rho^(bold(M))"), Point((x₁ + x₂) / 2 + 18, 150); size = 17)
        @layer (setcolor(GUIDE); setline(1.0); line(Point((x₁ + x₂) / 2 + 18, 164), Point((x₁ + x₂) / 2 + 18, top - 4), :stroke))
        label(m("r thin (dif Y)/(dif r) = bold(M)(r) thin Y"), Point((x₂ + x₃) / 2 + 22, 120); size = 13)
        @layer (setcolor(GUIDE); setline(1.0); line(Point((x₂ + x₃) / 2, 138), Point((x₂ + x₃) / 2, top - 4), :stroke))
        @layer (setcolor(GUIDE); setline(1.0); line(Point((x₀ + x₁) / 2, 146), Point((x₀ + x₁) / 2, top - 4), :stroke))
        label(txt("uniform field"), Point((x₃ + x₄) / 2 + 2, 152); size = 11, color = MUTED)
        label(txt("+ decaying modes"), Point((x₃ + x₄) / 2 + 2, 168); size = 11, color = MUTED)
        then_arrow(Point(x₀ + 10, 312), Point(x₄ - 10, 312))
        label(txt("the state is carried outward, then matched to the matrix"), Point((x₀ + x₄) / 2, 334); size = 12, color = MUTED)
    end
end

# ── Appendices on the plane (theory/holomorphic_functions.md, complex_potentials.md) ──

"The point of the sheet at the complex number `z` of a plane drawn around `c` at the scale `s`."
plane_point(c::Point, s, z) = c + Point(s * real(z), -s * imag(z))

"The direction on the sheet of the complex number `e` (the sheet's y axis points down)."
plane_direction(e) = Point(real(e), -imag(e))

"""
The polar grid of the outside of the unit disk (left) and its image by the map
of an ellipse, ζ + m/ζ (right): circles become confocal ellipses, rays confocal
hyperbolas, and the two families still cross at right angles. One circle and
one ray are picked out, with the frame (e_ρ, e_θ) where they cross.
Docs: theory/holomorphic_functions.md.
"""
function theory_holomorphic_map(; path = "theory/holomorphic_map", panel::Bool = true)
    m₁ = 0.35                                    # `m` is the label function of style.jl
    disk_map, ellipse_map = [(1, 0.0)], [(1, m₁)]
    ρmax, s = 2.2, 52.0
    ρ₀, θ₀ = 1.5, 50 * π / 180
    return sheet(path, 720, 340; panel = panel) do
        for (c, ms) in ((Point(175, 165), disk_map), (Point(545, 165), ellipse_map))
            @layer begin
                setcolor(MATRIX)
                poly(mapped_circle(c, s, ms, ρmax), :fill; close = true)
                setcolor(PORE)
                poly(mapped_circle(c, s, ms, 1.0), :fill; close = true)
                setcolor(GUIDE)
                setline(0.8)
                foreach(ρ -> poly(mapped_circle(c, s, ms, ρ), :stroke; close = true), (1.25, 1.5, 1.8))
                foreach(θ -> poly(mapped_ray(c, s, ms, θ, ρmax), :stroke), range(0, 2π; length = 25)[1:24])
                # the circle ρ₀ and the ray θ₀
                setcolor(HIGHLIGHT)
                setline(2.0)
                poly(mapped_circle(c, s, ms, ρ₀), :stroke; close = true)
                poly(mapped_ray(c, s, ms, θ₀, ρmax), :stroke)
                setcolor(INK)
                setline(2.0)
                poly(mapped_circle(c, s, ms, 1.0), :stroke; close = true)
            end
            # the frame where they cross: e_ρ along ζ ω′(ζ), e_θ a quarter turn further
            ζ₀ = ρ₀ * cis(θ₀)
            ω(ζ) = ζ + sum(mk * ζ^(-k) for (k, mk) in ms)
            dω(ζ) = 1 - sum(k * mk * ζ^(-k - 1) for (k, mk) in ms)
            e = ζ₀ * dω(ζ₀) / abs(ζ₀ * dω(ζ₀))
            p = plane_point(c, s, ω(ζ₀))
            eρ, eθ = plane_direction(e), plane_direction(im * e)
            @layer begin
                setcolor(INK)
                setline(1.2)
                poly([p + 9 * eρ, p + 9 * (eρ + eθ), p + 9 * eθ], :stroke)
            end
            arrow_ab(p, p + 38 * eρ; color = VECTOR, lw = 2.0)
            arrow_ab(p, p + 38 * eθ; color = VECTOR, lw = 2.0)
            label(m("underline(e)_rho"), p + 52 * eρ; size = 15, color = VECTOR, bg = MATRIX)
            label(m("underline(e)_theta"), p + 52 * eθ; size = 15, color = VECTOR, bg = MATRIX)
        end
        # the foci of the confocal family, at ± 2 √m
        for x in (-1, 1)
            disk(Point(545 + x * s * 2 * sqrt(m₁), 165), 3.0; fill = INK, stroke = nothing)
        end
        label(m("plus.minus 2 sqrt(m)"), Point(545, 182); size = 13, color = MUTED)
        arrow_ab(Point(300, 165), Point(412, 165); color = VECTOR, lw = 1.8)
        label(m("omega(zeta) = zeta + m slash zeta"), Point(356, 146); size = 14, color = VECTOR)
        label(m("zeta = rho thin e^(upright(i) theta)"), Point(175, 28); size = 15)
        label(m("z = x_1 + upright(i) x_2"), Point(545, 28); size = 15)
        label(txt("circles ") * m("rho") * txt(" and rays ") * m("theta"), Point(175, 312); size = 13, color = MUTED)
        label(txt("confocal ellipses and hyperbolas"), Point(545, 312); size = 13, color = MUTED)
    end
end

"""
An ellipse of the physical plane and its affine images in the planes of
z₁ = x₁ + μ₁x₂ and z₂ = x₁ + μ₂x₂ (bottom), each the image of the unit circle
(top) by a map ασ + β/σ; the point σ of the circle goes to the three points
that correspond to one another, which is why the ellipse is solved in an
anisotropic matrix. The roots μ₁, μ₂ are illustrative values of positive
imaginary part. Docs: theory/complex_potentials.md.
"""
function theory_affine_ellipses(; path = "theory/affine_ellipses", panel::Bool = true)
    m₁, θ₀, s = 0.4, 1.0, 48.0                   # `m` is the label function of style.jl
    ξ, η = complex(1 + m₁), im * (1 - m₁)
    # the point x₁ + μ x₂ of the contour at θ: μ = i is the physical plane
    image(μ, θ) = (x₁ = real(ξ * cis(-θ)); x₂ = real(η * cis(-θ)); x₁ + μ * x₂)
    planes = (
        (Point(130, 255), im, m("x_1 + upright(i) x_2"), m("sigma + m slash sigma")),
        (Point(380, 255), 0.4 + 1.6im, m("z_1 = x_1 + mu_1 x_2"), m("alpha_1 sigma + beta_1 slash sigma")),
        (Point(630, 255), -0.3 + 0.6im, m("z_2 = x_1 + mu_2 x_2"), m("alpha_2 sigma + beta_2 slash sigma")),
    )
    return sheet(path, 760, 365; panel = panel) do
        # the unit circle and the point σ
        c₀, r₀ = Point(380, 80), 42.0
        @layer (setcolor(GUIDE); setline(0.8); line(c₀ - Point(58, 0), c₀ + Point(58, 0), :stroke))
        disk(c₀, r₀; fill = nothing, stroke = INK, lw = 2.0)
        q = plane_point(c₀, r₀, cis(θ₀))
        @layer (setcolor(GUIDE); setline(1.0); setdash([3, 3]); line(c₀, q, :stroke))
        disk(q, 4.5; fill = HIGHLIGHT, stroke = nothing)
        label(m("sigma = e^(upright(i) theta)"), q + Point(14, -10); size = 15, halign = :left)
        label(m("abs(zeta) = 1"), c₀ + Point(-r₀ - 14, -24); size = 13, color = MUTED, halign = :right)
        for (c, μ, name, map) in planes
            @layer begin
                setcolor(MATRIX)
                box(c, 220, 150, 10, :fill)
                setcolor(GUIDE)
                setline(0.8)
                line(c - Point(100, 0), c + Point(100, 0), :stroke)
            end
            contour = [plane_point(c, s, image(μ, θ)) for θ in range(0, 2π; length = 241)]
            @layer begin
                setcolor(PORE)
                poly(contour, :fill; close = true)
                setcolor(INK)
                setline(2.0)
                poly(contour, :stroke; close = true)
            end
            p = plane_point(c, s, image(μ, θ₀))
            @layer (setcolor(GUIDE); setline(1.0); setdash([3, 3]); line(c, p, :stroke))
            disk(p, 4.5; fill = HIGHLIGHT, stroke = nothing)
            label(name, c + Point(0, 92); size = 15)
            # the map from the circle to this plane
            d = (c + Point(0, -82)) - c₀
            u = d / hypot(d.x, d.y)
            a, b = c₀ + (r₀ + 8) * u, c + Point(0, -80)
            arrow_ab(a, b; color = VECTOR, lw = 1.6)
            mid = (a + b) / 2
            side = c.x < c₀.x ? Point(-12, -6) : c.x > c₀.x ? Point(12, -6) : Point(12, 0)
            label(map, mid + side; size = 13, color = VECTOR, halign = c.x < c₀.x ? :right : :left, bg = PANEL)
        end
    end
end

"""
The contour of a hole and its sign conventions (left): the contour run
counterclockwise, the hole on its left; at a point, the tangent t and the
normal n pointing into the matrix; the remote stress, a tension p along d at
the angle α. Right, magnified: the free facet of the contour and the hoop
stress σ_tt on the facets normal to t. Docs: theory/complex_potentials.md.
"""
function theory_contour_stress(; path = "theory/contour_stress", panel::Bool = true)
    c, a, b = Point(205, 165), 95.0, 50.0
    contour_at(θ) = c + Point(a * cos(θ), -b * sin(θ))
    θ₀, α = 40 * π / 180, 125 * π / 180
    return sheet(path, 720, 330; panel = panel) do
        rve_box(c, 340, 260)
        hole = [contour_at(θ) for θ in range(0, 2π; length = 241)]
        @layer (setcolor(PORE); poly(hole, :fill; close = true); setcolor(INK); setline(2.0); poly(hole, :stroke; close = true))
        # the direction of travel, at the top of the contour
        top = contour_at(π / 2)
        @layer (setcolor(INK); poly([top + Point(-6, 0), top + Point(5, -5), top + Point(5, 5)], :fill; close = true))
        # the remote tension along d, at the angle α
        d = Point(cos(α), -sin(α))
        across = Point(sin(α), cos(α))
        @layer (setcolor(GUIDE); setline(1.0); setdash([4, 3]); line(c, c + 118 * d, :stroke); line(c, c + Point(118, 0), :stroke))
        @layer (setcolor(MUTED); setline(1.2); arc(c, 34, -α, 0, :stroke))
        label(m("alpha"), c + 20 * Point(cos(α / 2), -sin(α / 2)); size = 14, color = MUTED)
        for sgn in (-1, 1), k in -1:1
            base = c + sgn * 122 * d + 20k * across
            arrow_ab(base, base + sgn * 26 * d; color = INK, lw = 1.6)
        end
        label(m("bold(Sigma) = p thin underline(d) times.circle underline(d)"), Point(205, 314); size = 14)
        # tangent and normal at the point
        p = contour_at(θ₀)
        t = Point(-a * sin(θ₀), -b * cos(θ₀))
        t = t / hypot(t.x, t.y)
        n = Point(-t.y, t.x)                     # a quarter turn clockwise on the sheet: out of the hole
        arrow_ab(p, p + 34 * t; color = VECTOR, lw = 2.0)
        arrow_ab(p, p + 34 * n; color = VECTOR, lw = 2.0)
        label(m("underline(t)"), p + 46 * t; size = 15, color = VECTOR)
        label(m("underline(n)"), p + 46 * n; size = 15, color = VECTOR, bg = MATRIX)
        label(txt("hole"), c + Point(-30, 18); size = 12, color = MUTED)
        # magnified: the free facet and the hoop stress
        rs, Z, R = 16.0, Point(555, 165), 112.0
        k = R / rs
        spot(p, rs)
        zoom_cone(p, rs, Z, R)
        @layer begin
            circle(Z, R, :clip)
            setcolor(MATRIX)
            circle(Z, R, :fill)
            setcolor(PORE)
            poly([Z + k * (q - p) for q in hole], :fill; close = true)
            setcolor(INK)
            setline(2.2)
            poly([Z + k * (q - p) for q in hole], :stroke; close = true)
            # an element of matrix on the contour, its sides along t and n
            h, w = 26.0, 40.0
            corners = [Z + 4n - h * t, Z + 4n + h * t, Z + (4 + w) * n + h * t, Z + (4 + w) * n - h * t]
            setcolor(tint(BLUE, 0.68))
            poly(corners, :fill; close = true)
            setcolor(INK)
            setline(1.4)
            poly(corners, :stroke; close = true)
        end
        disk(Z, R; fill = nothing, stroke = HIGHLIGHT, lw = 2.4)
        for sgn in (-1, 1)
            mid = Z + (4 + 40 / 2) * n + sgn * 26 * t
            arrow_ab(mid, mid + sgn * 28 * t; color = CRACK, lw = 2.0)
        end
        label(m("sigma_(t t)"), Z + 22 * n + 70 * t; size = 15, color = CRACK, bg = MATRIX)
        label(txt("free: ") * m("sigma_(n n) = sigma_(n t) = 0"), Z + Point(0, R + 22); size = 13, color = MUTED)
        label(m("sigma_(t t) = underline(t) dot bold(sigma) dot underline(t)"), Point(555, 28); size = 14)
        label(m("(n_1 + upright(i) n_2) dif s = - upright(i) dif z"), Point(555, 312); size = 13, color = MUTED)
    end
end

theory_figures() = (
    theory_residue_poles(), theory_gaussian_crack(), theory_conformal_map(), theory_plane_problem(),
    theory_parameter_classes(), theory_radial_problem(),
    theory_holomorphic_map(), theory_affine_ellipses(), theory_contour_stress(),
)
