# =============================================================================
#  schemes.jl — the homogenization schemes.
#
#  Two kinds of drawing:
#  - the morphologies the theory and manual pages already showed (RVE
#    decomposition, Mori–Tanaka RVE, self-consistent tessellation, Maxwell,
#    Ponte Castañeda–Willis), redrawn: schemes/rve_*.svg;
#  - one card per exported scheme, schemes/scheme_*.svg: the same RVE on the
#    left, read by the scheme into the auxiliary problem(s) on the right — the
#    reference medium (matrix ℂ₀ or effective ℂ^hom) and the remote loading
#    are what changes from one card to the next.
#  Docs: theory/homogenization.md, manual/schemes.md,
#        theory/differential_scheme.md, theory/cluster_model.md, theory/eim.md.
# =============================================================================

const SEED_RVE = 2026        # the RVE shared by every card
const CARD_W, CARD_H = 600, 290
const RVE_C, RVE_SIDE = Point(125, 180), 200

# Phase colors of the grains (self-consistent tessellation) and of the layers.
const GRAIN_TINTS = (tint(BLUE, 0.35), tint(ORANGE, 0.3), tint(GREEN, 0.35), tint(TEAL, 0.35), tint(GOLD, 0.3))

"The RVE of every card: a matrix carrying the three families."
function card_rve(c::Point = RVE_C, side = RVE_SIDE; s = 0.95)
    rng = StableRNG(SEED_RVE)
    rve_box(c, side, side)
    kinds = (Ellipsoidal(), Needle(), Coated(), Ellipsoidal(), Needle(), Coated(), Ellipsoidal(), Needle(), Coated())
    @layer begin
        box(c, side - 4, side - 4, 9, :clip)
        draw_all(
            scatter(
                rng, kinds, in_box(c, side - 22, side - 22); s = s, gap = 9,
                avoid = [(c + Point(side / 2 - 24, side / 2 - 20), 20.0)]
            ); s = s
        )
    end
    label(m("bb(C)_0"), c + Point(side / 2 - 9, side / 2 - 17); size = 18, halign = :right)
    return nothing
end

# ── The morphologies ──────────────────────────────────────────────────────────

"An RVE under u = E⋅x is a sum of single-inclusion problems in ℂ₀ under u = E⁰⋅x."
function rve_decomposition()
    return sheet("schemes/rve_decomposition", 900, 290) do
        rng = StableRNG(7)
        c, A, B = Point(175, 180), 150.0, 98.0
        ell(c, A, B; fill = MATRIX, stroke = INK, lw = 1.6)
        kinds = (Ellipsoidal(), Needle(), Coated(), Ellipsoidal(), Needle(), Coated(), Ellipsoidal(), Needle(), Coated(), Ellipsoidal())
        draw_all(scatter(rng, kinds, in_ellipse(c, A - 10, B - 10); gap = 10))
        remote_loading(c + Point(A * cos(π / 4), -B * sin(π / 4)), "underline(u) = bold(E) dot underline(x)")
        label(m("equiv"), Point(355, 180); size = 30)
        for (k, (x, fam)) in enumerate(zip((445, 615, 785), THREE_FAMILIES))
            aux_box(Point(x, 190), 130, 170)
            draw_inclusion(fam, Point(x, 195), -0.5, 1.3)
            k < 3 && label(m("+"), Point(x + 85, 190); size = 28)
        end
    end
end

"The matrix–inclusion morphology Mori–Tanaka reads."
function rve_mori_tanaka()
    return sheet("schemes/rve_mori_tanaka", 400, 260) do
        rng = StableRNG(11)
        c, A, B = Point(200, 130), 180.0, 112.0
        ell(c, A, B; fill = MATRIX, stroke = INK, lw = 1.6)
        kinds = (Ellipsoidal(), Needle(), Coated(), Ellipsoidal(), Needle(), Coated(), Ellipsoidal(), Needle(), Coated(), Ellipsoidal(), Needle(), Coated())
        draw_all(scatter(rng, kinds, in_ellipse(c, A - 12, B - 12); gap = 12))
        label(m("bb(C)_0"), c + Point(A * 0.6, B * 0.6); size = 20)
    end
end

"""
    grain_colors(rng, cells, ncolors) -> Vector{Int}

A color per cell, never that of a neighbor (two cells sharing an edge), and
each time the least used one among those allowed: no two touching grains alike,
and no color running across the drawing.
"""
function grain_colors(rng, cells, ncolors)
    n = length(cells)
    shared(a, b) = count(p -> any(q -> distance(p, q) < 0.5, b), a)
    neighbors = [[j for j in 1:n if j != i && shared(cells[i], cells[j]) >= 2] for i in 1:n]
    color = zeros(Int, n)
    used = zeros(Int, ncolors)
    for i in sortperm(rand(rng, n))
        allowed = setdiff(1:ncolors, color[neighbors[i]])
        isempty(allowed) && (allowed = collect(1:ncolors))
        least = minimum(used[allowed])
        k = rand(rng, filter(c -> used[c] == least, allowed))
        color[i] = k
        used[k] += 1
    end
    return color
end

"Grains of several phases, none surrounding the others."
function tessellation(c::Point, w, h; n = 54, seed = 3, edge = 1.6)
    rng = StableRNG(seed)
    frame = [c + Point(-w / 2, -h / 2), c + Point(w / 2, -h / 2), c + Point(w / 2, h / 2), c + Point(-w / 2, h / 2)]
    seeds = [c + Point(w * (rand(rng) - 0.5), h * (rand(rng) - 0.5)) for _ in 1:n]
    cells = voronoi(seeds, frame)     # unrelaxed: grains of unequal sizes, as in a real section
    colors = grain_colors(rng, cells, length(GRAIN_TINTS))
    @layer begin
        box(c, w, h, 10, :clip)
        for (cell, k) in zip(cells, colors)
            setcolor(GRAIN_TINTS[k])
            poly(cell, :fill; close = true)
            setcolor(PANEL)
            setline(edge)
            poly(cell, :stroke; close = true)
        end
    end
    @layer (setcolor(INK); setline(1.6); box(c, w, h, 10, :stroke))
    return nothing
end

rve_self_consistent() = sheet(() -> tessellation(Point(180, 170), 330, 310), "schemes/rve_self_consistent", 360, 340)

"Inclusions gathered in a cluster ≡ one equivalent inclusion Ω."
function rve_maxwell()
    return sheet("schemes/rve_maxwell", 680, 320) do
        rng = StableRNG(5)
        for x in (160, 490)
            unbounded_box(Point(x, 190), 280, 230; label_tensor = "bb(C)_0")
            remote_loading(Point(x + 140, 75), "underline(u) = bold(E) dot underline(x)")
        end
        c = Point(150, 195)
        kinds = (Ellipsoidal(), Needle(), Ellipsoidal(), Needle(), Ellipsoidal(), Needle(), Coated(), Needle())
        draw_all(scatter(rng, kinds, in_ellipse(c, 78, 58); gap = 7))
        ell(c, 88, 66; fill = nothing, stroke = MUTED, lw = 1.8, dash = [6, 5])
        label(m("equiv"), Point(325, 190); size = 30)
        ell(Point(480, 195), 88, 66; fill = tint(PURPLE, 0.45), stroke = nothing)
        label(m("Omega"), Point(480, 195); size = 26, color = "#ffffff")
    end
end

"""
    apollonian(rng, c, side; rmax, rmin, gap, step) -> Vector{Tuple{Point, Float64}}

Disks paving the square of center `c` as densely as they can, after Hashin's
composite-sphere assemblage: each new disk goes where the most room is left
(one of the roomiest spots, at random), as large as that room allows up to
`rmax`, so that smaller and smaller disks fill the interstices down to `rmin`.
"""
function apollonian(rng, c::Point, side; rmax = 46.0, rmin = 4.0, gap = 2.0, step = 2.0)
    h = side / 2
    cands = [c + Point(x, y) for x in (-h):step:h for y in (-h):step:h]
    room = [min(h - abs(p.x - c.x), h - abs(p.y - c.y)) - gap for p in cands]
    disks = Tuple{Point, Float64}[]
    while true
        best = maximum(room)
        best < rmin && break
        pool = findall(>=(0.85 * best), room)
        k = pool[rand(rng, 1:length(pool))]
        r = min(room[k], rmax * (0.5 + 0.5 * rand(rng)))
        p = cands[k]
        push!(disks, (p, r))
        for i in eachindex(cands)
            room[i] = min(room[i], distance(cands[i], p) - r - gap)
        end
    end
    return disks
end

"""
Ponte Castañeda–Willis: each flat inclusion inside its own spherical safety
domain, the domains paving the space as in Hashin's assemblage.
"""
function rve_pcw()
    return sheet("schemes/rve_pcw", 340, 340) do
        rng = StableRNG(9)
        c, side = Point(170, 170), 316.0
        rve_box(c, side, side)
        for (p, r) in apollonian(rng, c, side - 10; rmax = 52.0, rmin = 3.5, gap = 1.5)
            disk(p, r; fill = nothing, stroke = MUTED, lw = r < 10 ? 0.8 : 1.2, dash = r < 10 ? nothing : [4, 3])
            ell(p, 0.62r, 0.1r; fill = BLUE, stroke = nothing)
        end
    end
end

# ── One card per scheme ───────────────────────────────────────────────────────

"A card: the shared RVE, the arrow, then `right()`."
function card(right, name::AbstractString; w = CARD_W, h = CARD_H)
    return sheet("schemes/scheme_" * name, w, h) do
        card_rve()
        then_arrow(Point(240, 180), Point(284, 180))
        right()
    end
end

const AUX1 = (Point(418, 185), 230, 180)      # one auxiliary problem
const AUX2 = ((Point(377, 185), 146, 180), (Point(537, 185), 146, 180))

function aux_single(; fill = MATRIX, tensor = "bb(C)_0", load = "underline(u) = bold(E)^0 dot underline(x)", fam = Ellipsoidal())
    c, w, h = AUX1
    aux_box(c, w, h; fill = fill, tensor = tensor, load = load)
    draw_inclusion(fam, c + Point(-10, 5), -0.45, 1.8)
    return nothing
end

"Voigt: a uniform strain — exact for bars in parallel."
function bars(c::Point, parallel::Bool)
    w, h = 150.0, 130.0
    for (k, t) in enumerate(GRAIN_TINTS[1:3])
        setcolor(t)
        parallel ? box(c + Point((k - 2) * w / 3, 0), w / 3, h, :fill) :
            box(c + Point(0, (k - 2) * h / 3), w, h / 3, :fill)
    end
    @layer (setcolor(INK); setline(1.2); box(c, w, h, :stroke))
    for s in (-1, 1)
        @layer (setcolor(SLATE); box(c + Point(0, s * (h / 2 + 6)), w + 24, 12, 3, :fill))
        arrow_ab(c + Point(0, s * (h / 2 + 12)), c + Point(0, s * (h / 2 + 42)); color = VECTOR, lw = 2.0)
    end
    return nothing
end

scheme_voigt() = card("voigt"; w = 620) do
    bars(Point(390, 180), true)
    label(m("bold(epsilon)(underline(x)) = bold(E)"), Point(538, 180); size = 19, halign = :left)
end

scheme_reuss() = card("reuss"; w = 620) do
    bars(Point(390, 180), false)
    label(m("bold(sigma)(underline(x)) = bold(Sigma)"), Point(538, 180); size = 19, halign = :left)
end

"Laminated: the exact solution of a stack of layers of normal n."
function scheme_laminated()
    return sheet("schemes/scheme_laminated", 320, 290) do
        c, side, θ = Point(160, 148), 250.0, -0.42
        thick = (34, 22, 30, 18, 40, 30, 30, 22, 34, 26, 30)
        tones = (tint(BLUE, 0.55), tint(ORANGE, 0.5))
        et, ey = Point(cos(θ), sin(θ)), Point(-sin(θ), cos(θ))   # along and across the layers
        ys = cumsum(thick) .- thick ./ 2 .- sum(thick) / 2        # layer centers, across
        @layer begin
            box(c, side, side, 10, :clip)
            for (k, (t, y)) in enumerate(zip(thick, ys))
                @layer begin
                    translate(c + y * ey)
                    rotate(θ)
                    setcolor(tones[2 - isodd(k)])
                    box(O, 2side, t, :fill)
                end
            end
        end
        @layer (setcolor(INK); setline(1.6); box(c, side, side, 10, :stroke))
        n = -1 * ey                                               # the normal, pointing up
        o = c + ys[6] * ey + 40et
        arrow_ab(o, o + 78n; color = VECTOR, lw = 2.2, head = 11)
        label(m("underline(n)"), o + 78n + 18et; size = 20, color = VECTOR)
        label(m("bb(C)_1"), c + ys[5] * ey - 62et; size = 17)
        label(m("bb(C)_2"), c + ys[6] * ey - 62et; size = 17)
    end
end

scheme_dilute() = card(() -> aux_single(load = "underline(u) = bold(E) dot underline(x)"), "dilute")
scheme_dilute_dual() = card(() -> aux_single(load = "bold(sigma) dot underline(n) = bold(Sigma) dot underline(n)"), "dilute_dual")
scheme_mori_tanaka() = card(() -> aux_single(load = "underline(u) = angle.l bold(epsilon) angle.r_0 dot underline(x)"), "mori_tanaka")

"Self-consistent: every phase, the matrix included, embedded in the effective medium."
scheme_self_consistent() = card("self_consistent"; w = 660) do
    (c1, w1, h1), (c2, w2, h2) = AUX2
    aux_box(c1, w1, h1; fill = EFFECTIVE, tensor = "bb(C)^\"hom\"")
    disk(c1 + Point(-8, 5), 30; fill = MATRIX, stroke = MATRIX_EDGE, lw = 1.4)
    label(m("bb(C)_0"), c1 + Point(-8, 5); size = 16)
    aux_box(c2, w2, h2; fill = EFFECTIVE, tensor = "bb(C)^\"hom\"")
    draw_inclusion(Ellipsoidal(), c2 + Point(-8, 5), -0.45, 1.6)
end

"Asymmetric self-consistent: the inclusions see the effective medium, the matrix keeps its role."
scheme_asymmetric_self_consistent() = card(() -> aux_single(fill = EFFECTIVE, tensor = "bb(C)^\"hom\""), "asymmetric_self_consistent")

"Differential: inclusions added by small amounts to the current effective medium."
function scheme_differential()
    return sheet("schemes/scheme_differential", 700, 270) do
        rng = StableRNG(13)
        xs = (85, 265, 445, 625)
        tones = (MATRIX, mix(MATRIX, EFFECTIVE, 0.4), mix(MATRIX, EFFECTIVE, 0.55), mix(EFFECTIVE, PURPLE, 0.25))
        tensors = ("bb(C)_0", "bb(C)^\"hom\" (tau)", "bb(C)^\"hom\" (tau + d tau)", "bb(C)^\"hom\"")
        taus = ("tau = 0", "tau", "tau + d tau", "tau = 1")
        y = 128
        for (k, x) in enumerate(xs)
            c = Point(x, y)
            rve_box(c, 140, 140; fill = tones[k])
            label(m(tensors[k]), c + Point(0, 92); size = 17)
            label(m(taus[k]), c + Point(0, 120); size = 16, color = MUTED)
        end
        # the step dτ is the drawn one; the integration before and after, dotted
        for (k, x) in enumerate(xs[1:3])
            a, b = Point(x + 76, y), Point(x + 104, y)
            if k == 2
                then_arrow(a, b)
            else
                @layer (setcolor(MUTED); setline(2.2); setdash([3, 4]); line(a, b - Point(10, 0), :stroke))
                then_arrow(b - Point(10, 0), b)
            end
        end
        # the small amounts d φ_i of every family, added to the current medium
        c = Point(xs[2], y)
        kinds = (Ellipsoidal(), Needle(), Coated(), Ellipsoidal(), Needle(), Coated())
        draw_all(scatter(rng, kinds, in_box(c, 124, 124); s = 0.6, gap = 12); s = 0.6)
        label(m("+ d phi_i"), Point(xs[2], 34); size = 17, color = MUTED)
    end
end

"Cluster model: the positions enter, through the interaction tensor of every pair."
function scheme_cluster()
    return sheet("schemes/scheme_cluster", 360, 300) do
        rng = StableRNG(21)
        c, w, h = Point(180, 150), 330.0, 270.0
        rve_box(c, w, h)
        kinds = repeat([Ellipsoidal(), Needle()], 7)
        items = scatter(rng, kinds, in_box(c, w - 30, h - 30); gap = 26)
        # the inclusion closest to the center and its four nearest neighbors
        a = argmin([distance(p, c) for (_, p, _) in items])
        pa = items[a][2]
        order = sortperm([distance(p, pa) for (_, p, _) in items])
        near = order[2:5]
        @layer begin
            setcolor(VECTOR)
            setline(1.6)
            setdash([5, 4])
            for b in near
                line(pa, items[b][2], :stroke)
            end
        end
        draw_all(items)
        spot(pa, 24)
        label(m("a"), pa + Point(-34, 0); size = 18, color = HIGHLIGHT)
        # the pair drawn: the neighbor whose link is the most horizontal
        b = near[argmin([abs((items[k][2] - pa).y) / distance(items[k][2], pa) for k in near])]
        pb = items[b][2]
        label(m("b"), pb + Point(0, -24); size = 18)
        label(m("bb(T)^(a b)"), (pa + pb) / 2 + Point(0, -14); size = 17, color = VECTOR, bg = MATRIX)
    end
end

"Equivalent inclusion method: a polarization per inclusion, in a periodic cell."
function scheme_eim()
    return sheet("schemes/scheme_eim", 360, 360) do
        rng = StableRNG(17)
        c, L, sc = Point(180, 180), 108.0, 1.5
        items = scatter(rng, (Ellipsoidal(), Ellipsoidal(), Ellipsoidal()), in_box(Point(0, 0), L - 8, L - 8); s = sc, gap = 6)
        for i in -1:1, j in -1:1
            o = c + Point(i * L, j * L)
            setcolor(MATRIX)
            box(o, L, L, :fill)
            for (k, p, θ) in items
                draw_inclusion(k, o + p, θ, sc)
            end
            if !(i == 0 && j == 0)       # the periodic images, faded
                @layer (setcolor(1, 1, 1, 0.6); box(o, L, L, :fill))
            end
            @layer (setcolor(GUIDE); setline(1.0); setdash([4, 3]); box(o, L, L, :stroke))
        end
        @layer (setcolor(INK); setline(2.0); box(c, L, L, :stroke))
        for (n, (_, p, _)) in enumerate(items)
            label(m("bold(tau)^$n"), c + p; size = 16, color = "#ffffff")
        end
    end
end

scheme_figures() = (
    rve_decomposition(), rve_mori_tanaka(), rve_self_consistent(), rve_maxwell(), rve_pcw(),
    scheme_voigt(), scheme_reuss(), scheme_laminated(), scheme_dilute(), scheme_dilute_dual(),
    scheme_mori_tanaka(), scheme_self_consistent(), scheme_asymmetric_self_consistent(),
    scheme_differential(), scheme_cluster(), scheme_eim(),
)
