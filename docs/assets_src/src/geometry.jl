# =============================================================================
#  geometry.jl — the auxiliary problems of the theory pages:
#  Eshelby's inclusion and inhomogeneity, the n-layer sphere, the n-layer
#  confocal spheroid, the decompositions of the proof of Levin's theorem.
#  Docs: theory/eshelby_problem.md, theory/layered_sphere.md,
#        theory/layered_spheroid.md, theory/levin.md.
# =============================================================================

"Eshelby's inclusion problem: a uniform polarization inside the ellipsoid, no remote loading."
function eshelby_inclusion()
    return sheet("geometry/eshelby_inclusion", 340, 400) do
        c = Point(160, 220)
        unbounded_box(c, 250, 300; label_tensor = "bb(C)")
        ell(c + Point(0, 10), 58, 30, -0.35; fill = tint(BLUE, 0.6), stroke = BLUE, lw = 2.0, dash = [5, 4])
        label(m("bold(tau)"), c + Point(0, 10); size = 24)
        label(m("cal(E)"), c + Point(52, -34); size = 20, color = BLUE)
        remote_loading(c + Point(125, -150), "underline(u) = underline(0)")
    end
end

"Eshelby's inhomogeneity problem: a different stiffness inside, a remote uniform strain outside."
function eshelby_inhomogeneity()
    return sheet("geometry/eshelby_inhomogeneity", 340, 400) do
        c = Point(160, 220)
        unbounded_box(c, 250, 300; label_tensor = "bb(C)")
        ell(c + Point(0, 10), 58, 30, -0.35; fill = BLUE, stroke = nothing)
        label(m("bb(C)^I"), c + Point(0, 10); size = 24, color = "#ffffff")
        remote_loading(c + Point(125, -150), "underline(u) = bold(E) dot underline(x)")
    end
end

"The n-layer sphere in an unbounded matrix under a remote uniform strain."
function layered_sphere()
    return sheet("geometry/layered_sphere", 640, 420) do
        c = Point(250, 225)
        unbounded_box(Point(320, 225), 600, 370; label_tensor = "bb(C)_0")
        # radii r_N, r_(N-1), r_2, r_1; the dashed circle stands for the layers left out
        radii = (150, 112, 80, 46)
        fills = (tint(BLUE, 0.15), tint(BLUE, 0.4), tint(BLUE, 0.62), tint(BLUE, 0.82))
        for (k, (r, f)) in enumerate(zip(radii, fills))
            disk(c, r; fill = f, stroke = nothing)
            disk(c, r; fill = nothing, stroke = INK, lw = 1.4, dash = k == 2 ? [5, 4] : nothing)
        end
        label(m("1"), c + Point(0, -23); size = 20)
        label(m("2"), c + Point(0, -63); size = 20)
        label(m("dots.c"), c + Point(0, -96); size = 20)
        label(m("N"), c + Point(0, -131); size = 20, color = "#ffffff")
        # the radius line and the radii of the interfaces
        @layer begin
            setcolor(INK)
            setline(1.2)
            line(c, c + Point(150, 0), :stroke)
            circle(c, 3, :fill)
        end
        for (name, r) in (("r_1", 46), ("r_2", 80), ("r_N", 150))
            @layer (setcolor(INK); circle(c + Point(r, 0), 2.6, :fill))
            label(m(name), c + Point(r - 4, 17); size = 16, halign = :right, color = r == 150 ? "#ffffff" : INK)
        end
        # remote loading
        for dy in (-60, 0, 60)
            arrow_ab(Point(430, 225 + dy * 0.2), Point(520, 225 + dy); color = VECTOR)
        end
        label(m("underline(u) = bold(E) dot underline(x)"), Point(485, 135); size = 18)
        label(m("abs(underline(x)) -> infinity"), Point(485, 318); size = 16, color = MUTED)
    end
end

"""
The n-layer cylinder: its section (left), with the radii and the interfaces,
and a cut along its axis (right), the layers running to infinity along e₃.
"""
function layered_cylinder()
    return sheet("geometry/layered_cylinder", 760, 420) do
        unbounded_box(Point(380, 225), 740, 370; label_tensor = "bb(C)_0")
        fills = (tint(BLUE, 0.15), tint(BLUE, 0.4), tint(BLUE, 0.62), tint(BLUE, 0.82))
        radii = (130, 98, 70, 40)
        # ── the section, the axis pointing out of the page ──
        c = Point(175, 225)
        for (k, (r, f)) in enumerate(zip(radii, fills))
            disk(c, r; fill = f, stroke = nothing)
            disk(c, r; fill = nothing, stroke = INK, lw = 1.4, dash = k == 2 ? [5, 4] : nothing)
        end
        label(m("1"), c + Point(0, -20); size = 18)
        label(m("2"), c + Point(0, -55); size = 18)
        label(m("dots.c"), c + Point(0, -84); size = 18)
        label(m("N"), c + Point(0, -114); size = 18, color = "#ffffff")
        @layer begin
            setcolor(INK)
            setline(1.2)
            line(c, c + Point(130, 0), :stroke)
            circle(c, 7, :stroke)            # the axis, out of the page
            circle(c, 2.2, :fill)
        end
        label(m("underline(e)_3"), c + Point(-14, 16); size = 15, halign = :right)
        for (name, r) in (("r_1", 40), ("r_2", 70), ("r_N", 130))
            @layer (setcolor(INK); circle(c + Point(r, 0), 2.6, :fill))
            label(m(name), c + Point(r - 4, 15); size = 15, halign = :right, color = r == 130 ? "#ffffff" : INK)
        end
        label(m("cal(I)_1"), c + Point(37, -37); size = 14)
        label(m("cal(I)_N"), c + Point(106, 104); size = 14)
        # ── the cut along the axis ──
        x0, x1, yc = 400.0, 650.0, 225.0
        for (k, (r, f)) in enumerate(zip(radii, fills))
            h = r * 0.62
            setcolor(f)
            box(Point((x0 + x1) / 2, yc), x1 - x0, 2h, :fill)
            @layer begin
                setcolor(INK)
                setline(1.3)
                k == 2 && setdash([5, 4])
                line(Point(x0, yc - h), Point(x1, yc - h), :stroke)
                line(Point(x0, yc + h), Point(x1, yc + h), :stroke)
            end
        end
        # the layers run on: dashed ends on both sides
        @layer begin
            setcolor(GUIDE)
            setline(1.1)
            setdash([3, 4])
            for r in radii, s in (-1, 1), (xa, xb) in ((x0 - 34, x0), (x1, x1 + 34))
                line(Point(xa, yc + s * r * 0.62), Point(xb, yc + s * r * 0.62), :stroke)
            end
        end
        @layer begin
            setcolor(INK)
            setline(1.0)
            setdash([8, 3, 2, 3])
            line(Point(x0 - 40, yc), Point(x1 + 40, yc), :stroke)
        end
        arrow_ab(Point(x1 + 40, yc), Point(x1 + 70, yc); color = INK)
        label(m("underline(e)_3"), Point(x1 + 58, yc + 16); size = 15)
        label(m("(k, l, m, n, p)_j"), Point((x0 + x1) / 2, yc - 102); size = 16)
        # remote loading
        label(m("underline(u) = bold(E) dot underline(x)"), Point(525, 360); size = 17)
        label(m("abs(underline(x) - (underline(x) dot underline(e)_3) underline(e)_3) -> infinity"), Point(525, 386); size = 14, color = MUTED)
    end
end

"The n-layer confocal spheroid: domains Ω₁…Ω_{N+1}, interfaces 𝓘₁…𝓘_N, common foci."
function layered_ellipsoid()
    return sheet("geometry/layered_ellipsoid", 520, 460) do
        c = Point(250, 235)
        unbounded_box(Point(260, 232), 492, 436; label_tensor = nothing)
        f = 120.0                            # half focal distance c, common to every layer
        axial = (205.0, 178.0, 152.0, 132.0) # axial semi-axes ρ_a; ρ_t = √(ρ_a² − c²)
        fills = (tint(BLUE, 0.15), tint(BLUE, 0.4), tint(BLUE, 0.62), tint(BLUE, 0.82))
        for (k, (ρa, fl)) in enumerate(zip(axial, fills))
            ρt = sqrt(ρa^2 - f^2)
            ell(c, ρt, ρa; fill = fl, stroke = nothing)
            ell(c, ρt, ρa; fill = nothing, stroke = INK, lw = 1.3, dash = k == 2 ? [5, 4] : nothing)
        end
        @layer begin
            setcolor(GUIDE)
            setline(1.0)
            setdash([6, 4])
            line(Point(c.x, 22), Point(c.x, 448), :stroke)
            line(Point(22, c.y), Point(498, c.y), :stroke)
        end
        # the common foci and the half focal distance c
        @layer (setcolor(RED); circle(c + Point(0, -f), 4, :fill); circle(c + Point(0, f), 4, :fill))
        @layer begin
            setcolor(RED)
            setline(1.3)
            x = c.x - 14
            line(Point(x, c.y), Point(x, c.y - f), :stroke)
            line(Point(x - 4, c.y), Point(x + 4, c.y), :stroke)
            line(Point(x - 4, c.y - f), Point(x + 4, c.y - f), :stroke)
        end
        label(m("c"), Point(c.x - 26, c.y - f / 2); size = 18, color = RED)
        # domains below the axis, interfaces above it
        ρts = reverse([sqrt(ρa^2 - f^2) for ρa in axial])   # inner to outer
        xs = vcat(0.0, ρts)
        names = ("Omega_1", "Omega_2", "dots.c", "Omega_N")
        for k in 1:4
            label(
                m(names[k]), c + Point(k == 1 ? 26 : (xs[k] + xs[k + 1]) / 2, 18); size = 15,
                color = k == 4 ? "#ffffff" : INK
            )
        end
        label(m("Omega_(N+1)"), c + Point(ρts[end] + 40, 18); size = 15)
        for (name, ρt) in zip(("cal(I)_1", "cal(I)_2", "", "cal(I)_N"), ρts)
            isempty(name) && continue
            label(m(name), c + Point(ρt + 3, -15); size = 15, halign = :left)
        end
    end
end

# ── Levin's theorem: a problem with eigen loads as the sum of two ────────────

const LEVIN_W = 210.0      # the volume element of each panel
const LEVIN_H = 150.0
const LEVIN_X = (135.0, 465.0, 795.0)
const LEVIN_Y = 200.0

"""
    levin_sum(path, titles, conditions, formulas, conclusion)

The complete problem, `=`, the problem without eigen load under the boundary
condition, `+`, the eigen loads alone under a vanishing boundary condition: one
volume element drawn three times with the same microstructure, its boundary
condition above, its local law, concentration and averages below, and Hill's
lemma on the crossed pairs at the bottom.
"""
function levin_sum(path, titles, conditions, formulas, conclusion)
    return sheet(path, 930, 560) do
        W, H = LEVIN_W, LEVIN_H
        # One microstructure, the same in the three panels.
        disks = pack(StableRNG(20261016), 7, in_box(Point(0, 0), W - 26, H - 26), rng -> 11 + 7rand(rng); gap = 10)
        for k in 1:3
            x = LEVIN_X[k]
            c = Point(x, LEVIN_Y)
            label(titles[k], Point(x, 52); size = 17)
            label(conditions[k], Point(x, 92); size = 16)
            rve_box(c, W, H)
            for (p, r) in disks
                disk(c + p, r; fill = INCLUSIONS[2], stroke = nothing)
            end
            for (i, f) in enumerate(formulas[k])
                label(f, Point(x, LEVIN_Y + H / 2 + 34 + 36 * (i - 1)); size = 16)
            end
        end
        label(m("="), Point((LEVIN_X[1] + LEVIN_X[2]) / 2, LEVIN_Y); size = 34)
        label(m("+"), Point((LEVIN_X[2] + LEVIN_X[3]) / 2, LEVIN_Y); size = 34)
        @layer (setcolor(PANEL_EDGE); setline(1.2); line(Point(60, 440), Point(870, 440), :stroke))
        label(txt("Hill's lemma on the crossed pairs of (a) and (b)"), Point(465, 470); size = 15, color = MUTED)
        label(conclusion, Point(465, 512); size = 17)
    end
end

"""
Levin's theorem, prestresses: under a macroscopic strain, the problem with
prestresses is the sum of (a) no prestress under the same strain and (b) the
prestresses alone under a vanishing macroscopic strain.
"""
levin_prestress() = levin_sum(
    "geometry/levin_prestress",
    (txt("prestressed phases"), txt("(a) no prestress"), txt("(b) prestresses alone")),
    (
        m("underline(u) = bold(E) dot underline(x) \"on\" partial Omega"),
        m("underline(u)^a = bold(E) dot underline(x) \"on\" partial Omega"),
        m("underline(u)^b = underline(0) \"on\" partial Omega"),
    ),
    (
        (
            m("bold(sigma) = bb(C)_r : bold(epsilon) + bold(sigma)^ast_r"),
            m("angle.l bold(epsilon) angle.r = bold(E)"),
            m("angle.l bold(sigma) angle.r = bb(C)^\"hom\" : bold(E) + bold(Sigma)^ast"),
        ),
        (
            m("bold(sigma)^a = bb(C)_r : bold(epsilon)^a"),
            m("bold(epsilon)^a = bb(A) : bold(E)"),
            m("angle.l bold(sigma)^a angle.r = bb(C)^\"hom\" : bold(E)"),
        ),
        (
            m("bold(sigma)^b = bb(C)_r : bold(epsilon)^b + bold(sigma)^ast_r"),
            m("angle.l bold(epsilon)^b angle.r = bold(0)"),
            m("angle.l bold(sigma)^b angle.r = bold(Sigma)^ast"),
        ),
    ),
    m(
        "angle.l bold(sigma)^b : bold(epsilon)^a angle.r = bold(Sigma)^ast : bold(E), quad " *
            "angle.l bold(sigma)^a : bold(epsilon)^b angle.r = 0 quad => quad " *
            "bold(Sigma)^ast = angle.l bold(sigma)^ast : bb(A) angle.r = sum_r f_r bold(sigma)^ast_r : bb(A)_r"
    ),
)

"""
Levin's theorem, free strains: under a macroscopic stress, the problem with
eigenstrains is the sum of (a) no eigenstrain under the same stress and (b) the
eigenstrains alone under a vanishing macroscopic stress.
"""
levin_free_strain() = levin_sum(
    "geometry/levin_free_strain",
    (txt("phases with eigenstrains"), txt("(a) no eigenstrain"), txt("(b) eigenstrains alone")),
    (
        m("bold(sigma) dot underline(n) = bold(Sigma) dot underline(n) \"on\" partial Omega"),
        m("bold(sigma)^a dot underline(n) = bold(Sigma) dot underline(n) \"on\" partial Omega"),
        m("bold(sigma)^b dot underline(n) = underline(0) \"on\" partial Omega"),
    ),
    (
        (
            m("bold(sigma) = bb(C)_r : (bold(epsilon) - bold(epsilon)^ast_r)"),
            m("angle.l bold(sigma) angle.r = bold(Sigma)"),
            m("angle.l bold(epsilon) angle.r = (bb(C)^\"hom\")^(-1) : bold(Sigma) + bold(E)^ast"),
        ),
        (
            m("bold(sigma)^a = bb(C)_r : bold(epsilon)^a"),
            m("bold(sigma)^a = bb(B) : bold(Sigma)"),
            m("angle.l bold(epsilon)^a angle.r = (bb(C)^\"hom\")^(-1) : bold(Sigma)"),
        ),
        (
            m("bold(sigma)^b = bb(C)_r : (bold(epsilon)^b - bold(epsilon)^ast_r)"),
            m("angle.l bold(sigma)^b angle.r = bold(0)"),
            m("angle.l bold(epsilon)^b angle.r = bold(E)^ast"),
        ),
    ),
    m(
        "angle.l bold(sigma)^a : bold(epsilon)^b angle.r = bold(Sigma) : bold(E)^ast, quad " *
            "angle.l bold(sigma)^b : bold(epsilon)^a angle.r = 0 quad => quad " *
            "bold(E)^ast = angle.l bold(epsilon)^ast : bb(B) angle.r = sum_r f_r bold(epsilon)^ast_r : bb(B)_r"
    ),
)

"""
The annular crack: a penny of radius a bonded on the disc of radius c, seen from
above (left) and cut along a diameter (right); the inner front grows from r = a
to r = c. Docs: theory/rice_crack_contributions.md.
"""
function annular_crack()
    return sheet("geometry/annular_crack", 640, 330) do
        c = Point(170, 170)
        A, C, ρ = 125.0, 50.0, 88.0
        disk(c, A; fill = tint(RED, 0.72), stroke = CRACK, lw = 2.0)
        disk(c, C; fill = tint(SLATE, 0.8), stroke = CRACK, lw = 2.0)
        disk(c, ρ; fill = nothing, stroke = CRACK, lw = 1.4, dash = [5, 4])
        label(txt("bonded"), c + Point(0, -24); size = 13, color = INK)
        # radii
        for (r, θ, name) in ((A, -0.55, "a"), (C, 2.2, "c"))
            q = c + Point(r * cos(θ), r * sin(θ))
            @layer (setcolor(INK); setline(1.2); line(c, q, :stroke); circle(q, 2.4, :fill))
            label(m(name), c + Point(0.6r * cos(θ), 0.6r * sin(θ)) + Point(10, -12); size = 18)
        end
        # the front at radius ρ, its outer normal ν = −e_r, the growth
        f = c + Point(ρ * cos(0.9), ρ * sin(0.9))
        arrow_ab(f, f - Point(30 * cos(0.9), 30 * sin(0.9)); color = VECTOR, lw = 2.0)
        label(m("underline(nu)"), f - Point(30 * cos(0.9), 30 * sin(0.9)) + Point(-12, 4); size = 18, color = VECTOR)
        label(m("rho"), c + Point(ρ * cos(1.6), ρ * sin(1.6)) + Point(-14, 6); size = 18, color = CRACK)
        # the section along a diameter
        y = 170.0
        x0 = 470.0
        s = 0.95
        @layer begin
            setcolor(tint(SLATE, 0.55))
            setline(5)
            line(Point(x0 - s * C, y), Point(x0 + s * C, y), :stroke)
            setcolor(CRACK)
            setline(5)
            line(Point(x0 - s * A, y), Point(x0 - s * C, y), :stroke)
            line(Point(x0 + s * C, y), Point(x0 + s * A, y), :stroke)
        end
        arrow_ab(Point(x0, y), Point(x0, y - 70); color = VECTOR, lw = 2.0)
        label(m("underline(n)"), Point(x0 + 14, y - 66); size = 18, color = VECTOR, halign = :left)
        for (xx, name) in ((x0 + s * A, "a"), (x0 + s * C, "c"))
            label(m(name), Point(xx, y + 20); size = 17)
        end
        label(m("r"), Point(x0 + s * A + 18, y); size = 17, color = MUTED)
        label(txt("growth of the inner front: from r = a to r = c"), Point(320, 312); size = 14, color = MUTED)
    end
end

"""
The circular arc crack of radius R and half-angle α, its frame (m, n) at the
middle of the arc and the frame (n, ν) of a tip; the axis ℓ points out of the
page. Docs: theory/rice_crack_contributions.md.
"""
function arc_crack()
    return sheet("geometry/arc_crack", 460, 360) do
        O = Point(230, 250)
        R = 150.0
        α = π / 3
        disk(O, R; fill = nothing, stroke = GUIDE, lw = 1.2, dash = [5, 4])
        # the arc: R (sin θ m + cos θ n), m to the right, n up
        pt(θ, r = R) = O + Point(r * sin(θ), -r * cos(θ))
        @layer begin
            setcolor(CRACK)
            setline(5)
            pts = [pt(θ) for θ in range(-α, α; length = 60)]
            poly(pts, :stroke)
        end
        @layer (setcolor(INK); circle(O, 3, :fill))
        @layer (setcolor(INK); setline(1.0); setdash([4, 3]); line(O, pt(0), :stroke); line(O, pt(α), :stroke))
        label(m("R"), O + Point(R * sin(α) / 2 + 10, -R * cos(α) / 2 + 8); size = 18)
        @layer (setcolor(INK); setline(1.0); arc(O, 40, -π / 2, -π / 2 + α, :stroke))
        label(m("alpha"), O + Point(22, -52); size = 18)
        # the frame at the middle of the arc
        mid = pt(0)
        arrow_ab(mid, mid + Point(55, 0); color = VECTOR, lw = 2.0)
        arrow_ab(mid, mid + Point(0, -55); color = VECTOR, lw = 2.0)
        label(m("underline(m)"), mid + Point(62, 0); size = 18, color = VECTOR, halign = :left)
        label(m("underline(n)"), mid + Point(10, -58); size = 18, color = VECTOR, halign = :left)
        @layer (setcolor(VECTOR); setline(1.6); circle(mid, 7, :stroke); circle(mid, 2, :fill))
        label(m("underline(ell)"), mid + Point(-14, 14); size = 18, color = VECTOR, halign = :right)
        # the frame of the tip θ = α
        tip = pt(α)
        nt = Point(sin(α), -cos(α))
        νt = Point(cos(α), sin(α))
        arrow_ab(tip, tip + 48 * nt; color = VECTOR, lw = 2.0)
        arrow_ab(tip, tip + 48 * νt; color = VECTOR, lw = 2.0)
        label(m("underline(n)"), tip + 60 * nt; size = 18, color = VECTOR)
        label(m("underline(nu)"), tip + 60 * νt; size = 18, color = VECTOR)
    end
end

geometry_figures() = (
    eshelby_inclusion(), eshelby_inhomogeneity(), layered_sphere(), layered_cylinder(), layered_ellipsoid(),
    levin_prestress(), levin_free_strain(), annular_crack(), arc_crack(),
)
