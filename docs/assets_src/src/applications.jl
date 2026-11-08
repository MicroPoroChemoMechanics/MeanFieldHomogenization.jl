# =============================================================================
#  applications.jl — the morphologies of the tutorial and application pages:
#  mortar with interfacial transition zones, the three-scale bituminous mix,
#  the lamellar clay (two scales and one interfoliar layer), the eccentric
#  core–shell aggregate of the recycled-concrete page, and the nested scales
#  of the hydrating cement paste, and the Brinkman inhomogeneity problem of
#  the Stokes-to-Darcy upscaling.
#  Docs: tutorials/transport.md, applications/bituminous.md,
#        applications/lamellar_clay.md, applications/recycled_aggregate.md,
#        applications/hydrating_blended_paste.md,
#        scripts/71_brinkman_permeability.jl.
# =============================================================================

"A leader line from `target` to a label written at `at`, left-aligned."
function leader(target::Point, at::Point, src::AbstractString; size = 17, color = INK)
    @layer begin
        setcolor(MUTED)
        setline(1.2)
        line(target, at - Point(8, 0), :stroke)
        circle(target, 2.6, :fill)
    end
    label(src, at; size = size, color = color, halign = :left)
    return nothing
end

"Content clipped to a disk, then the disk's outline."
function disk_panel(f, c::Point, R; fill, edge = INK)
    disk(c, R; fill = fill, stroke = nothing)
    @layer begin
        circle(c, R - 1, :clip)
        f()
    end
    disk(c, R; fill = nothing, stroke = edge, lw = 1.8)
    return nothing
end

# ── Mortar with interfacial transition zones ──────────────────────────────────

"Mortar: cement paste, aggregates, each in an ITZ shell."
function veritz()
    return sheet("veritz", 780, 300) do
        rng = StableRNG(31)
        c, A, B, e = Point(270, 150), 250.0, 132.0, 5.0     # e: ITZ thickness
        ell(c, A, B; fill = PASTE, stroke = INK, lw = 1.6)
        grains = pack(rng, 22, in_ellipse(c, A - 8, B - 8), rng -> 17 + 20 * rand(rng); gap = 7)
        for (p, r) in grains
            disk(p, r; fill = COATING, stroke = nothing)
            disk(p, r - e; fill = AGGREGATE, stroke = nothing)
        end
        # leaders: to the right-most grain, its ITZ, and the paste beside it
        p, r = grains[argmax([q.x + s for (q, s) in grains])]
        x = 560
        leader(p + Point(0, 0), Point(x, 205), txt("aggregate"))
        leader(p + Point((r - e / 2) * cos(-0.6), (r - e / 2) * sin(-0.6)), Point(x, 150), txt("ITZ"))
        free = c + Point(0.7A, -0.45B)
        for _ in 1:500    # a point of the paste, upper right, clear of every grain
            inside = ((free.x - c.x) / A)^2 + ((free.y - c.y) / B)^2 < 0.8
            inside && all(distance(free, q) > s + 6 for (q, s) in grains) && break
            free = c + Point(A * (0.5 + 0.3rand(rng)), -B * (0.2 + 0.45rand(rng)))
        end
        leader(free, Point(x, 95), txt("cement paste"))
    end
end

# ── The three-scale bituminous mix ────────────────────────────────────────────

const BITUMEN = INK
const MASTIC = SLATE
const MORTAR = tint(SLATE, 0.55)

"""
    contact_packing(rng, radii, c; e) -> centers

Coated grains (core radius `r`, film `e`) that touch some of their neighbors
and leave mortar elsewhere: packed until jammed, the packing opened by a
dilation, then each grain moved onto its nearest neighbor until their films
merge, without overlapping any other.
"""
function contact_packing(rng, radii, c::Point; e, dilation = 1.16)
    n = length(radii)
    ps = settle(rng, radii .+ e, c; spread = 130)
    ps = [c + dilation * (p - c) for p in ps]
    near(i, j) = radii[i] + radii[j] + e                 # films merged over half their thickness
    gap(i, j) = distance(ps[i], ps[j]) - near(i, j)
    for i in sortperm(rand(rng, n))
        j = argmin(k -> k == i ? Inf : gap(i, k), 1:n)
        rand(rng) < 0.25 && continue                     # some grains stay free
        u = ps[j] - ps[i]
        ps[i] += (gap(i, j) / distance(ps[i], ps[j])) * u
        for _ in 1:20                                    # then clear every other grain
            clash = findfirst(k -> k != i && k != j && gap(i, k) < 0, 1:n)
            clash === nothing && break
            v = ps[i] - ps[clash]
            ps[i] += (-gap(i, clash) / distance(ps[i], ps[clash])) * v
        end
    end
    return ps
end

const MIX_WORDS = (
    titles = ("grain contact", "full mix", "mortar", "mastic"),
    schemes = ("", "self-consistent", "Mori–Tanaka", "Mori–Tanaka"),
    contents = ("", "coated coarse aggregates, voids", "sand in mastic", "fillers in bitumen"),
    film = "mastic film",
    contact = "grain-to-grain contact",
)

"""
Grain contact ← full mix (SC) → mortar (MT) → mastic (MT), after someCBM2022.
The coarse aggregates are coated by a mastic film and press on one another:
where two of them touch, the films merge and thin down to the contact, which is
what the left panel magnifies.
"""
function ver_multi_mix(; path = "ver_multi_mix", words = MIX_WORDS)
    return sheet(path, 1000, 350) do
        rng = StableRNG(41)
        y, R, e = 172.0, 105.0, 3.5          # e: thickness of the mastic film, drawn
        cs = (Point(335, y), Point(600, y), Point(865, y))
        # full mix: coated coarse aggregates packed until they touch
        radii = sort([11 + 17 * rand(rng) for _ in 1:44]; rev = true)
        centers = contact_packing(rng, radii, cs[1]; e = e)
        grains = collect(zip(centers, radii))
        voids = pack(
            rng, 10, in_disk(cs[1], R - 6), rng -> 2.5 + 2.5 * rand(rng); gap = 1.5,
            avoid = [(p, r + e) for (p, r) in grains]
        )
        # the contact magnified: a pair in contact, well inside, on the left side
        contacts = [
            (i, j) for i in eachindex(grains) for j in (i + 1):lastindex(grains)
                if distance(centers[i], centers[j]) < radii[i] + radii[j] + e + 0.5
        ]
        at(i, j) = centers[i] + ((radii[i] + e / 2) / distance(centers[i], centers[j])) * (centers[j] - centers[i])
        inner = [(i, j) for (i, j) in contacts if distance(at(i, j), cs[1]) < R - 30]
        i, j = inner[argmin([at(i, j).x for (i, j) in inner])]
        sp1 = (at(i, j), 10.0)
        # the mortar magnified: the point of the right half farthest from every film
        clearance(p) = minimum(distance(p, q) - r - e for (q, r) in grains)
        cands = [cs[1] + Point(x, z) for x in 0:3:(R - 25) for z in -(R - 25):3:(R - 25) if hypot(x, z) < R - 25]
        pm = cands[argmax(clearance.(cands))]
        spm = (pm, clamp(clearance(pm) + 3, 7.0, 12.0))
        # mortar: sand in the mastic; mastic: fillers in the bitumen
        sand = pack(rng, 46, in_disk(cs[2], R + 8), rng -> 6 + 8 * rand(rng); gap = 3)
        sp2 = (cs[2] + Point(-30, -40), 12.0)
        sand = [(p, r) for (p, r) in sand if distance(p, sp2[1]) > r + sp2[2] + 3]
        fillers = pack(rng, 140, in_disk(cs[3], R + 6), rng -> 2.5 + 4 * rand(rng); gap = 2.2)
        disk_panel(cs[1], R; fill = MORTAR) do
            for (p, r) in grains
                disk(p, r + e; fill = MASTIC, stroke = nothing)
            end
            for (p, r) in grains
                disk(p, r; fill = AGGREGATE, stroke = nothing)
            end
            foreach(((p, r),) -> disk(p, r; fill = PORE, stroke = nothing), voids)
        end
        disk_panel(cs[2], R; fill = MASTIC) do
            foreach(((p, r),) -> disk(p, r; fill = AGGREGATE, stroke = nothing), sand)
        end
        disk_panel(cs[3], R; fill = BITUMEN) do
            foreach(((p, r),) -> disk(p, r; fill = AGGREGATE, stroke = nothing), fillers)
        end
        # grain contact: two coarse aggregates, their films merged down to the contact
        rc = Point(110, y)
        w, h, Rg, ef = 170.0, 150.0, 260.0, 9.0
        @layer begin
            box(rc, w, h, :clip)
            setcolor(MORTAR)
            box(rc, w, h, :fill)
            for s in (-1, 1)
                disk(rc + Point(0, s * (Rg + 1.5)), Rg + ef; fill = MASTIC, stroke = nothing)
            end
            for s in (-1, 1)
                disk(rc + Point(0, s * (Rg + 1.5)), Rg; fill = AGGREGATE, stroke = nothing)
            end
        end
        @layer (setcolor(INK); setline(1.4); box(rc, w, h, :stroke))
        film = rc + Point(58, sqrt(Rg^2 - 58^2) - Rg - 1.5 + ef / 2)
        @layer begin
            setcolor(INK)
            setline(1.1)
            line(film, Point(film.x, y - h / 2 - 12), :stroke)
            line(rc, Point(rc.x, y + h / 2 + 12), :stroke)
            circle(film, 2.4, :fill)
            circle(rc, 2.4, :fill)
        end
        label(txt(words.film), Point(film.x, y - h / 2 - 14); size = 13, valign = :bottom)
        label(txt(words.contact), Point(rc.x, y + h / 2 + 15); size = 13, valign = :top)
        # the zooms, over the panels, from each orange spot
        zoom_cone(sp1[1], sp1[2], rc + Point(w / 2, -h / 2), rc + Point(w / 2, h / 2))
        zoom_cone(spm[1], spm[2], cs[2], R)
        zoom_cone(sp2[1], sp2[2], cs[3], R)
        spot(sp1...)
        spot(spm...)
        spot(sp2...)
        titles = map(txt, words.titles)
        schemes = map(w -> isempty(w) ? "" : txt(w), words.schemes)
        contents = map(w -> isempty(w) ? "" : txt(w), words.contents)
        xs = (rc.x, cs[1].x, cs[2].x, cs[3].x)
        for k in 1:4
            label(titles[k], Point(xs[k], 36); size = 19, color = VECTOR)
            k == 1 && continue
            label(schemes[k], Point(xs[k], y + R + 26); size = 16, color = HIGHLIGHT)
            label(contents[k], Point(xs[k], y + R + 48); size = 14, color = MUTED)
        end
    end
end

# ── Lamellar clay ─────────────────────────────────────────────────────────────

"A lamellar particle: platelets and interfoliar spaces, the stack of normal n at angle θ."
function particle(c::Point, θ; L = 72, n = 4, tp = 7.0, tf = 3.0)
    H = n * tp + (n - 1) * tf
    @layer begin
        translate(c)
        rotate(θ)
        setcolor(INTERLAYER)
        box(O, L, H, :fill)
        y = -H / 2
        for _ in 1:n
            setcolor(PLATELET)
            box(Point(0, y + tp / 2), L, tp, :fill)
            y += tp + tf
        end
        setcolor(tint(BLUE, 0.2))
        setline(1.0)
        box(O, L, H, :stroke)
    end
    return H
end

"① the REV, a porous polycrystal of particles; ② one particle, a periodic laminate."
function lamellar_scales()
    return sheet("lamellar/lamellar_scales", 790, 340) do
        rng = StableRNG(53)
        c, side = Point(160, 172), 240.0
        rve_box(c, side, side; fill = PANEL)
        # particles of every orientation (bounded by their half-diagonal), then macropores
        parts = pack(rng, 8, in_box(c, side - 10, side - 10), rng -> 40.0; gap = 6)
        pores = pack(rng, 7, in_box(c, side - 14, side - 14), rng -> 7 + 7 * rand(rng); gap = 6, avoid = parts)
        θs = [π * rand(rng) for _ in parts]
        for ((p, _), θ) in zip(parts, θs)
            particle(p, θ)
        end
        for (p, r) in pores
            disk(p, r; fill = tint(GUIDE, 0.75), stroke = GUIDE, lw = 1.0)
        end
        # the particle zoomed on
        k = argmax([p.x - abs(p.y - c.y) for (p, _) in parts])
        pz = parts[k][1]
        disk(pz, 42; fill = nothing, stroke = HIGHLIGHT, lw = 2.2, dash = [5, 4])
        # ② the laminate
        x1, x2, top = 440.0, 580.0, 92.0
        tp, tf, nl = 26.0, 11.0, 5
        zoom_cone(pz, 42, Point(x1, top), Point(x1, top + nl * tp + (nl - 1) * tf))
        y = top
        ys = Float64[]
        for i in 1:nl
            setcolor(PLATELET)
            box(Point((x1 + x2) / 2, y + tp / 2), x2 - x1, tp, :fill)
            push!(ys, y)
            y += tp
            if i < nl
                setcolor(INTERLAYER)
                box(Point((x1 + x2) / 2, y + tf / 2), x2 - x1, tf, :fill)
                y += tf
            end
        end
        @layer (setcolor(INK); setline(1.2); box(Point((x1 + x2) / 2, (top + y) / 2), x2 - x1, y - top, :stroke))
        # h: the interfoliar thickness, between the first two platelets
        yh1, yh2 = ys[1] + tp, ys[2]
        @layer begin
            setcolor(INK)
            setline(1.1)
            for yy in (yh1, yh2)
                line(Point(x1 - 22, yy), Point(x1 - 4, yy), :stroke)
            end
        end
        arrow_ab(Point(x1 - 13, yh1 - 18), Point(x1 - 13, yh1); lw = 1.2, head = 6)
        arrow_ab(Point(x1 - 13, yh2 + 18), Point(x1 - 13, yh2); lw = 1.2, head = 6)
        label(m("h"), Point(x1 - 32, (yh1 + yh2) / 2); size = 17)
        arrow_ab(Point(x2 + 22, top + 60), Point(x2 + 22, top + 10); color = VECTOR, lw = 2.0)
        label(m("underline(n)"), Point(x2 + 36, top + 20); size = 18, color = VECTOR, halign = :left)
        leader(Point(x2 - 30, ys[3] + tp / 2), Point(x2 + 34, ys[3] + tp / 2 + 8), txt("platelets, ") * m("1 - f"); size = 14)
        leader(Point(x2 - 30, ys[4] - tf / 2), Point(x2 + 34, ys[4] - tf / 2 + 40), txt("interfoliar space, ") * m("f"); size = 14)
        # titles and legend
        label(txt("① REV — a porous polycrystal"), Point(c.x, 34); size = 15, color = VECTOR)
        label(txt("② one particle — a periodic two-layer laminate"), Point(580, 34); size = 15, color = VECTOR)
        disk(Point(52, 322), 6; fill = tint(GUIDE, 0.75), stroke = GUIDE, lw = 1.0)
        label(txt("macropores, ") * m("phi"), Point(64, 322); size = 13, halign = :left)
        setcolor(PLATELET)
        box(Point(190, 322), 18, 10, :fill)
        label(txt("lamellar particles, ") * m("1 - phi"), Point(204, 322); size = 13, halign = :left)
    end
end

"A spring drawn vertically between heights `y1 < y2` at abscissa `x`."
function vspring(x, y1, y2; coils = 5, half = 9, color = INK)
    L = y2 - y1
    ya, yb = y1 + 0.15L, y2 - 0.15L
    n = 2coils
    st = (yb - ya) / n
    pts = vcat(Point(x, y1), Point(x, ya), [Point(x + (isodd(i) ? -half : half), ya + (i - 0.5) * st) for i in 1:n], Point(x, yb), Point(x, y2))
    @layer (setcolor(color); setline(1.8); poly(pts, :stroke))
    return nothing
end

"One interfoliar layer: free sliding, a normal repulsion p + πᵍ, a normal stiffness Π."
function interfoliar_layer()
    return sheet("lamellar/interfoliar_layer", 600, 350) do
        x1, x2 = 90.0, 470.0
        xm = (x1 + x2) / 2
        y1, y2, y3, y4 = 70.0, 125.0, 205.0, 260.0
        setcolor(PLATELET)
        box(Point(xm, (y1 + y2) / 2), x2 - x1, y2 - y1, :fill)
        box(Point(xm, (y3 + y4) / 2), x2 - x1, y4 - y3, :fill)
        setcolor(INTERLAYER)
        box(Point(xm, (y2 + y3) / 2), x2 - x1, y3 - y2, :fill)
        @layer begin
            setcolor(INK)
            setline(1.2)
            box(Point(xm, (y1 + y2) / 2), x2 - x1, y2 - y1, :stroke)
            box(Point(xm, (y3 + y4) / 2), x2 - x1, y4 - y3, :stroke)
            setcolor(HIGHLIGHT)
            setdash([5, 4])
            line(Point(x1, y2), Point(x2, y2), :stroke)
            line(Point(x1, y3), Point(x2, y3), :stroke)
        end
        label(txt("solid platelet — elastic, incompressible"), Point(xm, (y1 + y2) / 2); size = 14)
        label(txt("solid platelet — elastic, incompressible"), Point(xm, (y3 + y4) / 2); size = 14)
        label(txt("interfoliar space — fluid + ions"), Point(xm, y2 + 20); size = 14, color = ORANGE)
        # the normal stiffness Π and the repulsion p + πᵍ
        vspring(112, y2, y3; color = HIGHLIGHT)
        label(m("Pi"), Point(134, (y2 + y3) / 2 + 14); size = 19, color = HIGHLIGHT)
        arrow_ab(Point(435, (y2 + y3) / 2), Point(435, y2 + 2); color = HIGHLIGHT, lw = 1.8, head = 8)
        arrow_ab(Point(435, (y2 + y3) / 2), Point(435, y3 - 2); color = HIGHLIGHT, lw = 1.8, head = 8)
        label(m("p + pi^g"), Point(425, (y2 + y3) / 2 + 14); size = 17, color = HIGHLIGHT, halign = :right)
        # h, n, free sliding
        @layer (setcolor(INK); setline(1.1); line(Point(x1 - 30, y2), Point(x1 - 6, y2), :stroke); line(Point(x1 - 30, y3), Point(x1 - 6, y3), :stroke))
        arrow_ab(Point(x1 - 18, (y2 + y3) / 2), Point(x1 - 18, y2); lw = 1.2, head = 7)
        arrow_ab(Point(x1 - 18, (y2 + y3) / 2), Point(x1 - 18, y3); lw = 1.2, head = 7)
        label(m("h"), Point(x1 - 40, (y2 + y3) / 2); size = 18)
        arrow_ab(Point(x2 + 40, (y2 + y3) / 2 + 26), Point(x2 + 40, (y2 + y3) / 2 - 26); color = VECTOR, lw = 2.0)
        label(m("underline(n)"), Point(x2 + 56, (y2 + y3) / 2); size = 18, color = VECTOR)
        arrow_ab(Point(x1 + 60, y1 - 22), Point(x1 + 200, y1 - 22); color = VECTOR, lw = 1.6)
        label(txt("free sliding"), Point(x1 + 210, y1 - 22); size = 13, color = MUTED, halign = :left)
        arrow_ab(Point(x2 - 60, y4 + 22), Point(x2 - 200, y4 + 22); color = VECTOR, lw = 1.6)
        label(txt("free sliding"), Point(x2 - 210, y4 + 22); size = 13, color = MUTED, halign = :right)
        label(
            txt("linearized around ") * m("h_o") * txt(", the layer has a normal stiffness ") * m("Pi") * txt(" and no other stiffness at all"),
            Point(300, 326); size = 13, color = MUTED
        )
    end
end

# ── The eccentric core–shell aggregate ────────────────────────────────────────

"The recycled-concrete aggregate in meridian section, for α = 0, 0.6 and 1."
function axi_schematic()
    w = 0.5                      # old aggregate / whole inclusion, the value of the page
    ratio = round(cbrt(w); digits = 3)
    return sheet("fe/axi_schematic", 960, 400) do
        label(
            m("a_c = a w^(1\\/3)") * txt(", ") * m("d = alpha (a - a_c)") * txt("      (here ") * m("w = $w") *
                txt(", so ") * m("a_c \\/ a = $ratio") * txt(")"), Point(480, 30); size = 16
        )
        titles = (m("alpha = 0") * txt(" — concentric"), m("alpha = 0.6"), m("alpha = 1") * txt(" — tangency, by definition"))
        for (k, α) in enumerate((0.0, 0.6, 1.0))
            o = Point(165 + 315 * (k - 1), 228)
            a = 100.0
            ac = a * cbrt(w)
            d = α * (a - ac)
            label(titles[k], Point(o.x, 70); size = 16)
            disk(o, 1.45a; fill = nothing, stroke = GUIDE, lw = 1.0, dash = [2, 3])
            disk(o, a; fill = tint(BLUE, 0.65), stroke = BLUE, lw = 1.6)
            core = o + Point(0, -d)
            disk(core, ac; fill = AGGREGATE, stroke = GOLD, lw = 1.6)
            # the symmetry axis
            @layer begin
                setcolor(INK)
                setline(0.9)
                setdash([8, 3, 2, 3])
                line(o + Point(0, 1.4a), o + Point(0, -1.38a), :stroke)
            end
            label(m("z"), o + Point(10, -1.38a); size = 15, halign = :left)
            # a_c, from the core center
            @layer (setcolor(INK); circle(core, 2.6, :fill))
            tip = core + ac * Point(cos(-π / 4), sin(-π / 4))
            @layer (setcolor(INK); setline(1.1); line(core, tip, :stroke))
            label(m("a_c"), tip + Point(6, -6); size = 15, halign = :left)
            # a, at mid-height of the lower shell, never across the core radius
            za = 0.55a
            xa = sqrt(a^2 - za^2)
            @layer begin
                setcolor(INK)
                setline(1.1)
                line(o + Point(0, za), o + Point(xa, za), :stroke)
                line(o + Point(xa, za - 5), o + Point(xa, za + 5), :stroke)
            end
            label(m("a"), o + Point(xa / 2, za + 13); size = 15)
            # d, on the left of the shell
            if d > 0
                xd = o.x - a - 12
                @layer begin
                    setcolor(GREEN)
                    setline(1.4)
                    line(Point(xd, o.y), Point(xd, o.y - d), :stroke)
                    line(Point(xd - 5, o.y), Point(xd + 5, o.y), :stroke)
                    line(Point(xd - 5, o.y - d), Point(xd + 5, o.y - d), :stroke)
                    setline(0.8)
                    setdash([2, 3])
                    line(Point(xd, o.y), o, :stroke)
                    line(Point(xd, o.y - d), core, :stroke)
                end
                label(m("d"), Point(xd - 12, o.y - d / 2); size = 15, color = GREEN)
            end
            label(txt("core ") * m("bb(C)_1"), core + Point(-ac / 2, ac / 3); size = 14)
            ring = o + 0.9a * Point(cos(2.3), sin(2.3))    # a point of the shell, below the core
            @layer (setcolor(BLUE); setline(1.0); line(ring, o + Point(-0.86a, 1.17a), :stroke); circle(ring, 2.4, :fill))
            label(txt("shell ") * m("bb(C)_2"), o + Point(-0.92a, 1.27a); size = 14, color = BLUE)
            label(txt("matrix ") * m("bb(C)_0"), o + Point(1.18a, 1.22a); size = 14, color = MUTED)
        end
    end
end

# ── The nested scales of a hydrating cement paste ─────────────────────────────

# One color per hydrate; they are those of the MPCM talk's plots, so that its
# chemistry and micromechanics slides name each phase alike.
const HYDRATE = (csh = "#2F2A85", ch = "#1EAFD0", aft = "#EB601D", afm = "#FBBA00")
const FOAM = tint(HYDRATE.csh, 0.82)
const SOLUTION = tint(HYDRATE.ch, 0.72)
const ANHYDROUS = SLATE

const PASTE_WORDS = (
    titles = ("Cement paste", "Hydrate foam", "Hydrates"),
    schemes = ("Mori–Tanaka", "self-consistent", "self-consistent"),
    contents = (
        txt("anhydrous grains and fillers in the foam"),
        txt("needles (") * m("omega = 7") * txt("), pore solution, void"),
        txt("C-S-H, portlandite, AFm, ettringite…"),
    ),
)

"""
    cement_paste_scales(; path, words = PASTE_WORDS)

The three inner scales of the four-scale paste model of Lavergne et al. (2018):
the paste (Mori–Tanaka: anhydrous grains and fillers in the hydrate foam), the
foam (self-consistent: hydrate needles of aspect ratio 7, pore solution, void)
and the hydrates (self-consistent, spherical phases). `words` carries the
titles, scheme names and contents, so that the talk draws the same figure in
French.
"""
function cement_paste_scales(; path = "applications/cement_paste_scales", words = PASTE_WORDS)
    return sheet(path, 1200, 440) do
        rng = StableRNG(2026)
        R, y = 145.0, 200.0
        cs = (Point(200, y), Point(600, y), Point(1000, y))
        spots = ((cs[1] + Point(50, -60), 26.0), (cs[2] + Point(40, -35), 26.0))
        # 1. the paste: anhydrous grains and filler in the hydrate foam
        clear = [(spots[1][1], spots[1][2] + 4)]
        grains = pack(rng, 16, in_disk(cs[1], R + 20), rng -> 12 + 22 * rand(rng); gap = 3, avoid = clear)
        fillers = pack(rng, 5, in_disk(cs[1], R + 20), rng -> 8 + 6 * rand(rng); gap = 3, avoid = vcat(grains, clear))
        disk_panel(cs[1], R; fill = FOAM) do
            foreach(((p, r),) -> disk(p, r; fill = ANHYDROUS, stroke = nothing), grains)
            foreach(((p, r),) -> disk(p, r; fill = AGGREGATE, stroke = nothing), fillers)
        end
        # 2. the hydrate foam: needles of aspect ratio 7 in the pore solution, some void
        voids = pack(rng, 14, in_disk(cs[2], R), rng -> 5 + 4 * rand(rng); gap = 2)
        disk_panel(cs[2], R; fill = SOLUTION) do
            foreach(((p, r),) -> disk(p, r; fill = PORE, stroke = nothing), voids)
            for _ in 1:260
                ρ, θ = (R + 10) * sqrt(rand(rng)), 2π * rand(rng)
                ell(cs[2] + Point(ρ * cos(θ), ρ * sin(θ)), 15.4, 2.2, π * rand(rng); fill = HYDRATE.csh, stroke = nothing)
            end
        end
        # 3. the hydrates: spherical phases, C-S-H dominant
        weights = ((HYDRATE.csh, 0.42), (HYDRATE.ch, 0.22), (HYDRATE.afm, 0.18), (HYDRATE.aft, 0.18))
        balls = pack(rng, 150, in_disk(cs[3], R + 15), rng -> 7 + 10 * rand(rng); gap = 1.2)
        disk_panel(cs[3], R; fill = tint(HYDRATE.csh, 0.12)) do
            for (p, r) in balls
                u, acc, col = rand(rng), 0.0, HYDRATE.csh
                for (c, w) in weights
                    acc += w
                    if u <= acc
                        col = c
                        break
                    end
                end
                disk(p, r; fill = col, stroke = PANEL, lw = 0.8)
            end
        end
        # the zooms, over the panels, from each orange spot
        zoom_cone(spots[1]..., cs[2], R)
        zoom_cone(spots[2]..., cs[3], R)
        spot(spots[1]...)
        spot(spots[2]...)
        for (c, t, s, d) in zip(cs, words.titles, words.schemes, words.contents)
            label(txt(t), Point(c.x, 30); size = 24, color = VECTOR)
            label(txt(s), Point(c.x, y + R + 30); size = 20, color = HIGHLIGHT)
            label(d, Point(c.x, y + R + 58); size = 17, color = MUTED)
        end
    end
end

# ── A solidifying composite sphere (aging viscoelasticity) ────────────────────

const SET_LAYER = tint(BLUE, 0.3)

const SOLID_WORDS = (
    time = "time",
    set = "set: creeps from its setting time on",
    unset = "not yet set: still a pore",
)

"""
    solidifying_sphere(; path, words = SOLID_WORDS)

The composite sphere of the aging-creep page, after Sanahuja (2013): a pore
wrapped in shells that set one after the other, from the outside in, each
creeping only from its own setting time; drawn at three ages. The setting times
are those of the page: fraction set f(t) = f∞ tᵅ/(1 + tᵅ), α = 4, the shell of
midpoint fraction fₖ setting at tₖ = (fₖ/(f∞ − fₖ))^(1/α), the earliest
outermost. Eight shells are drawn where the page computes with a hundred.
"""
function solidifying_sphere(; path = "applications/solidifying_sphere", words = SOLID_WORDS)
    N, α, finf, fp = 8, 4.0, 0.3, 0.1
    q = [(k + 0.5) / N for k in 0:(N - 1)]
    t_set = [(x / (1 - x))^(1 / α) for x in q]
    ρ = (cumsum(vcat(fp, fill(finf / N, N))) ./ (fp + finf)) .^ (1 / 3)   # pore, then shells outwards
    ages = (0.75, 1.0, 1.6)              # two, four and seven shells of eight set
    return sheet(path, 760, 340) do
        R, y = 92.0, 152.0
        xs = (140.0, 380.0, 620.0)
        arrow_ab(Point(40, 24), Point(690, 24); color = MUTED, lw = 1.6, head = 10)
        label(txt(words.time), Point(698, 24); size = 15, color = MUTED, halign = :left)
        for (j, (x, t)) in enumerate(zip(xs, ages))
            c = Point(x, y)
            rve_box(c, 220, 220)
            for k in N:-1:1              # shell k spans ρ[k]…ρ[k+1]; the outermost, k = N, sets first
                set = t >= t_set[N - k + 1]
                disk(c, R * ρ[k + 1]; fill = set ? SET_LAYER : PORE, stroke = set ? tint(BLUE, 0.05) : GUIDE, lw = set ? 0.5 : 0.8)
            end
            disk(c, R * ρ[1]; fill = PORE, stroke = GUIDE, lw = 0.8)
            label(m("t_$j"), Point(x, y + 130); size = 19)
        end
        # the legend
        y2 = 318
        disk(Point(150, y2), 8; fill = SET_LAYER, stroke = nothing)
        label(txt(words.set), Point(166, y2); size = 14, halign = :left)
        disk(Point(450, y2), 8; fill = PORE, stroke = GUIDE, lw = 0.8)
        label(txt(words.unset), Point(466, y2); size = 14, halign = :left)
    end
end

# ── Stokes-to-Darcy permeability: the Brinkman inhomogeneity problem ─────────

const SOLID = tint(SLATE, 0.35)      # an impervious grain
const FLUID = PANEL                  # clear fluid

"""
The three steps of a Stokes-to-Darcy estimate: the pore-scale cell (Stokes flow
in the pores, body force in the solid), one Brinkman inhomogeneity problem
(an inclusion of size R in a porous reference, its boundary layer of thickness
√k₀), and the patterns available to represent the regions.
"""
function brinkman_problems()
    return sheet("applications/brinkman_problems", 1100, 420) do
        top = 34.0
        # ── (a) the pore-scale cell
        ca = Point(170, 215)
        label(txt("pore scale: Stokes flow"), Point(ca.x, top); size = 16)
        rve_box(ca, 260, 260; fill = FLUID)
        rng = StableRNG(7)
        @layer begin
            box(ca, 256, 256, 9, :clip)
            for (p, r) in pack(
                    rng, 26, (rng, r) -> ca + Point((236 - 2r) * (rand(rng) - 0.5), (236 - 2r) * (rand(rng) - 0.5)),
                    rng -> 14 + 16 * rand(rng); gap = 5
                )
                disk(p, r; fill = SOLID, stroke = INK, lw = 1.0)
            end
        end
        label(m("Omega_f"), ca + Point(-108, 114); size = 17, bg = PANEL)
        label(m("angle.l underline(f) angle.r = nabla P"), Point(ca.x, 392); size = 16)
        # ── arrow
        then_arrow(Point(312, 215), Point(368, 215))
        # ── (b) the Brinkman inhomogeneity problem
        cb = Point(560, 215)
        label(txt("one Brinkman inhomogeneity problem"), Point(cb.x, top); size = 16)
        unbounded_box(cb, 360, 300; fill = MATRIX, label_tensor = "k_0")
        R, ℓ = 62.0, 26.0
        disk(cb, R + ℓ; fill = tint(BLUE, 0.7), stroke = BLUE, lw = 1.2, dash = [5, 4])
        disk(cb, R; fill = tint(ORANGE, 0.55), stroke = INK, lw = 1.6)
        label(m("k_1"), cb + Point(-14, -10); size = 20)
        @layer (setcolor(INK); setline(1.2); line(cb, cb + Point(R * cos(0.6), R * sin(0.6)), :stroke); circle(cb, 2.6, :fill))
        label(m("R"), cb + Point(0.5R * cos(0.6) + 10, 0.5R * sin(0.6) - 6); size = 17)
        @layer (setcolor(BLUE); setline(1.2); line(cb + Point(-R, 0), cb + Point(-R - ℓ, 0), :stroke))
        label(m("sqrt(k_0)"), cb + Point(-R - ℓ - 6, -18); size = 15, color = BLUE, halign = :right)
        for dy in (-100, -50, 0, 50, 100)
            arrow_ab(Point(cb.x - 168, cb.y + dy), Point(cb.x - 128, cb.y + dy); color = VECTOR)
        end
        label(m("underline(V)_0"), Point(cb.x - 148, cb.y - 126); size = 17, color = VECTOR)
        label(m("alpha_0 = R \\/ sqrt(k_0)"), Point(cb.x + 92, cb.y + 118); size = 16)
        label(m("-nabla p + mu Delta underline(v) - mu k^(-1) underline(v) = underline(0)"), Point(cb.x, 392); size = 15)
        # ── arrow
        then_arrow(Point(752, 215), Point(800, 215))
        # ── (c) the patterns
        x0 = 950.0
        label(txt("patterns of the regions"), Point(x0, top); size = 16)
        ys = (88.0, 150.0, 212.0, 274.0, 336.0)
        names = ("solid sphere", "fluid sphere", "cylinder", "layer", "coated sphere")
        for (y, nm) in zip(ys, names)
            label(txt(nm), Point(x0 + 10, y); size = 14, halign = :left)
        end
        xs = x0 - 40
        disk(Point(xs, ys[1]), 20; fill = SOLID, stroke = INK, lw = 1.4)
        disk(Point(xs, ys[2]), 20; fill = FLUID, stroke = INK, lw = 1.4)
        @layer begin
            setcolor(tint(ORANGE, 0.55))
            box(Point(xs, ys[3]), 76, 26, 0, :fill)
            setcolor(INK); setline(1.4)
            line(Point(xs - 38, ys[3] - 13), Point(xs + 38, ys[3] - 13), :stroke)
            line(Point(xs - 38, ys[3] + 13), Point(xs + 38, ys[3] + 13), :stroke)
            ell(Point(xs - 38, ys[3]), 5, 13; fill = nothing, stroke = INK, lw = 1.2)
            ell(Point(xs + 38, ys[3]), 5, 13; fill = tint(ORANGE, 0.4), stroke = INK, lw = 1.2)
        end
        @layer begin
            setcolor(tint(ORANGE, 0.55))
            box(Point(xs, ys[4]), 80, 18, 0, :fill)
            setcolor(INK); setline(1.4)
            line(Point(xs - 40, ys[4] - 9), Point(xs + 40, ys[4] - 9), :stroke)
            line(Point(xs - 40, ys[4] + 9), Point(xs + 40, ys[4] + 9), :stroke)
        end
        disk(Point(xs, ys[5]), 22; fill = FLUID, stroke = INK, lw = 1.4)
        disk(Point(xs, ys[5]), 12; fill = SOLID, stroke = INK, lw = 1.2)
    end
end

# ── The elastic limit of concrete: ITZ–aggregate interaction ──────────────────

const SEPARATION = ORANGE       # ITZ–aggregate separation, the interface criterion
const ITZ_FAILURE = PURPLE      # ITZ failure, the volume criterion

"A unit vector at angle `θ` from the upward vertical, clockwise, in drawing coordinates."
updir(θ) = Point(sin(θ), -cos(θ))

"""
The layered piece of interface the zooms show: aggregate below, the ITZ layer,
paste above, in the box of center `c` and size `w × h`; returns the heights of
the interface and of the outer face of the ITZ.
"""
function interface_piece(c::Point, w, h; t = 0.22h)
    y_int = c.y + 0.12h                              # aggregate | ITZ
    y_out = y_int - t                                # ITZ | paste
    @layer begin
        box(c, w, h, 10, :clip)
        setcolor(PASTE); box(c, w, h, 0, :fill)
        setcolor(COATING); box(Point(c.x, (y_int + y_out) / 2), w, t, 0, :fill)
        setcolor(AGGREGATE); box(Point(c.x, (y_int + c.y + h / 2) / 2), w, c.y + h / 2 - y_int, 0, :fill)
        setcolor(INK); setline(1.6)
        line(Point(c.x - w / 2, y_int), Point(c.x + w / 2, y_int), :stroke)
        setcolor(MUTED); setline(1.0); setdash([5, 4])
        line(Point(c.x - w / 2, y_out), Point(c.x + w / 2, y_out), :stroke)
    end
    @layer (setcolor(INK); setline(1.6); box(c, w, h, 10, :stroke))
    return y_int, y_out
end

"""
Two scales: concrete, where the ITZ is the surface of the aggregates; one
aggregate, its poles, equator and spherical frame; a piece of the interface,
where the ITZ is a layer bonded to the aggregate across the plane of normal n.
"""
function itz_scales()
    return sheet("applications/itz_scales", 1060, 430) do
        top = 32.0
        # ── (a) concrete
        ca = Point(170, 215)
        label(txt("concrete: the ITZ is a surface"), Point(ca.x, top); size = 16)
        rve_box(ca, 250, 250; fill = PASTE)
        rng = StableRNG(5)
        grains = pack(rng, 18, in_box(ca, 236, 236), rng -> 16 + 20 * rand(rng); gap = 7)
        for (p, r) in grains
            disk(p, r + 2.5; fill = COATING, stroke = nothing)
            disk(p, r; fill = AGGREGATE, stroke = INK, lw = 0.8)
        end
        corner = ca + Point(125, 125)                     # the load, at the lower right corner
        arrow_ab(corner, corner + Point(30, 30); color = INK)
        label(m("bold(Sigma)"), corner + Point(36, 22); size = 20, halign = :left)
        p0, r0 = grains[argmin([distance(q, ca + Point(60, 20)) for (q, _) in grains])]
        spot(p0, r0 + 7)
        # ── (b) one aggregate
        cb, R = Point(520, 222), 112.0
        label(txt("one aggregate"), Point(cb.x, top); size = 16)
        zoom_cone(p0, r0 + 7, cb, R + 6)
        disk(cb, R + 7; fill = COATING, stroke = nothing)
        disk(cb, R; fill = AGGREGATE, stroke = INK, lw = 1.6)
        @layer begin
            setcolor(GUIDE); setline(1.2); setdash([6, 5])
            line(cb + Point(0, -R - 46), cb + Point(0, R + 34), :stroke)
            line(cb + Point(-R - 14, 0), cb + Point(R + 14, 0), :stroke)
        end
        arrow_ab(cb + Point(0, -R - 20), cb + Point(0, -R - 54); color = INK)
        label(m("underline(e)_z"), cb + Point(16, -R - 50); size = 17, halign = :left)
        label(txt("pole"), cb + Point(-14, -R - 18); size = 13, color = MUTED, halign = :right)
        label(txt("equator"), cb + Point(-R - 18, -12); size = 13, color = MUTED, halign = :right)
        θ = 0.30π
        x = cb + R * updir(θ)
        @layer begin
            setcolor(INK); setline(1.0)
            line(cb, x, :stroke)
            arc2r(cb, cb + 34 * updir(0.0), cb + 34 * updir(θ), :stroke)
        end
        label(m("theta"), cb + 46 * updir(θ / 2); size = 16)
        arrow_ab(x, x + 46 * updir(θ); color = VECTOR, lw = 2)
        arrow_ab(x, x + 40 * updir(θ + π / 2); color = VECTOR, lw = 2)
        label(m("underline(n) = underline(e)_r"), x + 46 * updir(θ) + Point(-10, -14); size = 16, color = VECTOR, halign = :right, bg = PANEL)
        label(m("underline(e)_theta"), x + 40 * updir(θ + π / 2) + Point(-12, 4); size = 16, color = VECTOR, halign = :right)
        label(m("bold(sigma)_\"agg\" = bb(B)_\"agg\" : bold(Sigma)"), cb + Point(0, 34); size = 16)
        label(txt("uniform"), cb + Point(0, 60); size = 13, color = MUTED)
        spot(x, 13)
        # ── (c) a piece of the interface
        cc, w, h = Point(870, 222), 260.0, 230.0
        label(txt("a few micrometers: the ITZ is a layer"), Point(cc.x, top); size = 16)
        zoom_cone(x, 13, cc + Point(-w / 2, -h / 2), cc + Point(-w / 2, h / 2))
        y_int, y_out = interface_piece(cc, w, h)
        label(txt("aggregate"), Point(cc.x + w / 2 - 12, cc.y + h / 2 - 18); size = 14, halign = :right)
        label(txt("ITZ"), Point(cc.x + w / 2 - 12, (y_int + y_out) / 2); size = 14, halign = :right)
        label(txt("paste"), Point(cc.x + w / 2 - 12, cc.y - h / 2 + 18); size = 14, halign = :right)
        xn = cc.x - 70
        arrow_ab(Point(xn, y_int), Point(xn, y_int - 40); color = VECTOR, lw = 2)
        label(m("underline(n)"), Point(xn - 8, y_int - 34); size = 16, color = VECTOR, halign = :right)
        label(m("bold(sigma)_\"ITZ\" dot underline(n) = bold(sigma)_\"agg\" dot underline(n)"), Point(cc.x, cc.y + h / 2 + 30); size = 15)
        label(m("bold(epsilon)_\"ITZ\" = bold(epsilon)_\"agg\" + underline(a) limits(times.circle)^s underline(n)"), Point(cc.x, cc.y + h / 2 + 62); size = 15)
    end
end

"""
The two ways a crack starts next to an aggregate: the bond opens along the
interface (separation), or the ITZ breaks inside its thickness (ITZ failure).
"""
function itz_failure_modes()
    return sheet("applications/itz_failure_modes", 1000, 400) do
        w, h = 400.0, 230.0
        for (k, c) in enumerate((Point(255, 190), Point(745, 190)))
            y_int, y_out = interface_piece(c, w, h; t = 0.3h)
            k == 1 && for (txt_, y) in (("paste", c.y - h / 2 + 18), ("ITZ", (y_int + y_out) / 2), ("aggregate", c.y + h / 2 - 18))
                label(txt(txt_), Point(c.x - w / 2 + 12, y); size = 13, color = MUTED, halign = :left)
            end
            if k == 1
                label(txt("separation: a failure of the interface"), Point(c.x, 40); size = 17, color = SEPARATION)
                # the bond opens along the aggregate surface
                @layer begin
                    setcolor(PANEL)
                    poly([Point(c.x - 120, y_int), Point(c.x, y_int - 6), Point(c.x + 120, y_int), Point(c.x, y_int + 6)], :fill; close = true)
                    setcolor(SEPARATION); setline(2.4)
                    poly([Point(c.x - 120, y_int), Point(c.x, y_int - 6), Point(c.x + 120, y_int), Point(c.x, y_int + 6)], :stroke; close = true)
                end
                for dx in (-60, 0, 60)
                    arrow_ab(Point(c.x + dx, y_int - 10), Point(c.x + dx, y_int - 46); color = VECTOR, lw = 2)
                    arrow_ab(Point(c.x + dx, y_int + 10), Point(c.x + dx, y_int + 46); color = VECTOR, lw = 2)
                end
                label(m("T_r = underline(n) dot bold(sigma)_\"agg\" dot underline(n)"), Point(c.x, c.y + h / 2 + 28); size = 17)
                label(m("max_(underline(n)) T_r = T^\"ult\""), Point(c.x, c.y + h / 2 + 58); size = 17, color = SEPARATION)
            else
                label(txt("ITZ failure: a failure of the volume"), Point(c.x, 40); size = 17, color = ITZ_FAILURE)
                # a crack inside the layer, normal to the largest principal stress
                φ = 0.35π                                   # angle of e_I from the vertical
                ym = (y_int + y_out) / 2
                u = updir(φ + π / 2)                        # along the crack
                a, b = Point(c.x, ym) - 26 * u, Point(c.x, ym) + 26 * u
                @layer begin
                    setcolor(ITZ_FAILURE); setline(3.2)
                    line(a, b, :stroke)
                end
                e = updir(φ)
                arrow_ab(Point(c.x, ym) + 10 * e, Point(c.x, ym) + 52 * e; color = VECTOR, lw = 2)
                arrow_ab(Point(c.x, ym) - 10 * e, Point(c.x, ym) - 52 * e; color = VECTOR, lw = 2)
                label(m("underline(e)_\"I\""), Point(c.x, ym) + 66 * e; size = 16, color = VECTOR)
                label(m("sigma_\"I\" (bold(sigma)_\"ITZ\"), quad bold(sigma)_\"ITZ\" = bb(M)(underline(n)) : bold(Sigma)"), Point(c.x, c.y + h / 2 + 28); size = 17)
                label(m("max_(underline(n)) sigma_\"I\" = sigma^\"ult\"_\"ITZ\""), Point(c.x, c.y + h / 2 + 58); size = 17, color = ITZ_FAILURE)
            end
        end
    end
end

"""
Which mechanism starts the first crack, for each load, as the ratio of the two
strengths grows; the boundaries are the ratios at which both give the same
elastic limit.
"""
function itz_regimes()
    return sheet("applications/itz_regimes", 1000, 300) do
        x0, x1 = 250.0, 950.0
        lmin, lmax = log10(0.01), log10(5.0)
        X(ρ) = x0 + (x1 - x0) * (log10(ρ) - lmin) / (lmax - lmin)
        rows = (
            ("uniaxial tension", 0.98, 70.0),
            ("uniaxial compression", 0.05, 120.0),
            ("equibiaxial compression", 0.10, 170.0),
        )
        for (name, ρb, y) in rows
            label(txt(name), Point(x0 - 16, y); size = 15, halign = :right)
            @layer begin
                setcolor(tint(SEPARATION, 0.35)); box(Point((x0 + X(ρb)) / 2, y), X(ρb) - x0, 26, 4, :fill)
                setcolor(tint(ITZ_FAILURE, 0.35)); box(Point((X(ρb) + x1) / 2, y), x1 - X(ρb), 26, 4, :fill)
            end
        end
        for (_, ρb, y) in rows
            label(txt("separation"), Point((x0 + X(ρb)) / 2, y); size = 14)
            label(txt("ITZ failure"), Point((X(ρb) + x1) / 2, y); size = 14)
        end
        # the axis
        ya = 215.0
        @layer begin
            setcolor(INK); setline(1.4)
            line(Point(x0, ya), Point(x1, ya), :stroke)
        end
        arrow_ab(Point(x1 - 2, ya), Point(x1 + 22, ya); color = INK)
        for (ρ, s) in ((0.01, "0.01"), (0.05, "0.05"), (0.1, "0.1"), (0.5, "0.5"), (0.98, "0.98"), (5.0, "5"))
            @layer (setcolor(INK); setline(1.2); line(Point(X(ρ), ya - 5), Point(X(ρ), ya + 5), :stroke))
            label(m(s), Point(X(ρ), ya + 20); size = 14)
        end
        for ρ in (0.05, 0.10, 0.98)
            @layer (setcolor(GUIDE); setline(1.0); setdash([5, 4]); line(Point(X(ρ), 48), Point(X(ρ), ya), :stroke))
        end
        label(m("rho = T^\"ult\" \\/ sigma^\"ult\"_\"ITZ\""), Point(x0 - 16, ya); size = 16, halign = :right)
        # where the measured ratios of elastic limits are matched
        yb = 262.0
        @layer begin
            setcolor(GREEN); setline(2.4)
            line(Point(X(0.52), yb), Point(x1, yb), :stroke)
            line(Point(X(0.52), yb - 6), Point(X(0.52), yb + 6), :stroke)
        end
        label(txt("both measured ratios matched"), Point(X(0.52) - 10, yb); size = 14, color = GREEN, halign = :right)
    end
end

application_figures() = (
    veritz(), ver_multi_mix(), lamellar_scales(), interfoliar_layer(), axi_schematic(),
    cement_paste_scales(), solidifying_sphere(), brinkman_problems(),
    itz_scales(), itz_failure_modes(), itz_regimes(),
)
