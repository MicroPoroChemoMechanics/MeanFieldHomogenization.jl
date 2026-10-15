# =============================================================================
#  geometry.jl — the auxiliary problems of the theory pages:
#  Eshelby's inclusion and inhomogeneity, the n-layer sphere, the n-layer
#  confocal spheroid.
#  Docs: theory/eshelby_problem.md, theory/layered_sphere.md,
#        theory/layered_spheroid.md.
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
            label(m(names[k]), c + Point(k == 1 ? 26 : (xs[k] + xs[k + 1]) / 2, 18); size = 15,
                color = k == 4 ? "#ffffff" : INK)
        end
        label(m("Omega_(N+1)"), c + Point(ρts[end] + 40, 18); size = 15)
        for (name, ρt) in zip(("cal(I)_1", "cal(I)_2", "", "cal(I)_N"), ρts)
            isempty(name) && continue
            label(m(name), c + Point(ρt + 3, -15); size = 15, halign = :left)
        end
    end
end

geometry_figures() = (eshelby_inclusion(), eshelby_inhomogeneity(), layered_sphere(), layered_ellipsoid())
