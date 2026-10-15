# =============================================================================
#  rheology.jl — the spring-and-dashpot networks of the rheological model
#  catalog, ported from MeanFieldHomogenization's
#  docs/assets_src/make_rheology_diagrams.jl to the shared palette and to
#  Typst labels.
#
#  Every diagram corresponds exactly to one model in
#  src/Viscoelasticity/rheology_models.jl and carries the same symbols as that
#  model's constructor and as the formulas on the page. Where an arrangement is
#  not obvious from the transform (fractional Zener, Huet–Sayegh), the
#  derivation is written in a comment above the drawing.
#
#  Layout is computed, never eyeballed: branches sit PITCH apart, labels
#  LABEL_GAP from their own element. Elastic elements are blue, viscous ones
#  orange, fractional (springpot) ones purple.
#  Docs: manual/rheological_models.md, theory/laplace_carson.md,
#        applications/bituminous.md.
# =============================================================================

const COL_SPRING = BLUE
const COL_DASHPOT = ORANGE
const COL_POT = PURPLE
const LW_ELEMENT = 2.0
const LW_WIRE = 1.6

"Vertical distance between two branches of a parallel stack."
const PITCH = 58
"Gap between an element's envelope and its label."
const LABEL_GAP = 8
"Half-height of an element."
const HALF = 11
const RHEO_LABEL = 16

wire(a::Point, b::Point) = @layer (setcolor(INK); setline(LW_WIRE); line(a, b, :stroke))

"An element label, above or below the element centered at `(x, y)`."
function elabel(s::AbstractString, x, y, above::Bool)
    isempty(s) && return nothing
    if above
        label(m(s), Point(x, y - HALF - LABEL_GAP); size = RHEO_LABEL, valign = :bottom)
    else
        label(m(s), Point(x, y + HALF + LABEL_GAP); size = RHEO_LABEL, valign = :top)
    end
    return nothing
end

"A spring: a zigzag between `(x1, y)` and `(x2, y)` with straight leads."
function spring(x1, x2, y; label = "", label_above::Bool = true, lead = 0.16, coils = nothing)
    l = lead * (x2 - x1)
    xa, xb = x1 + l, x2 - l
    # coil density rather than a fixed count: a spring spanning a whole frame
    # must not come out as four enormous zigzags
    n = 2 * (coils === nothing ? clamp(round(Int, (xb - xa) / 26), 4, 12) : coils)
    wire(Point(x1, y), Point(xa, y))
    wire(Point(xb, y), Point(x2, y))
    step = (xb - xa) / n
    pts = vcat(Point(xa, y), [Point(xa + (i - 0.5) * step, y + (isodd(i) ? -HALF : HALF)) for i in 1:n], Point(xb, y))
    @layer (setcolor(COL_SPRING); setline(LW_ELEMENT); poly(pts, :stroke))
    elabel(label, (x1 + x2) / 2, y, label_above)
    return nothing
end

"A dashpot: a cylinder closed on the right, a piston on a rod entering from the left."
function dashpot(x1, x2, y; label = "", label_above::Bool = true, lead = 0.14)
    l = lead * (x2 - x1)
    xa, xb = x1 + l, x2 - l
    body = xb - xa
    xopen, xpiston = xa + 0.2body, xa + 0.5body
    @layer (setcolor(tint(COL_DASHPOT, 0.85)); rect(xopen, y - HALF, xb - xopen, 2HALF, :fill))
    wire(Point(x1, y), Point(xpiston, y))
    wire(Point(xb, y), Point(x2, y))
    @layer begin
        setcolor(COL_DASHPOT)
        setline(LW_ELEMENT)
        poly([Point(xopen, y - HALF), Point(xb, y - HALF), Point(xb, y + HALF), Point(xopen, y + HALF)], :stroke)
        setline(LW_ELEMENT + 1.4)
        line(Point(xpiston, y - HALF + 1.5), Point(xpiston, y + HALF - 1.5), :stroke)
    end
    elabel(label, (x1 + x2) / 2, y, label_above)
    return nothing
end

"""
The parabolic element (springpot, Scott-Blair element): its creep function is
a power law, `J(t) ∝ t^α/Γ(1+α)` — a parabola at `α = 1/2` — and the arc in
the box is that curve, flat at `α = 0` (a spring), straight at `α = 1` (a
dashpot). `order` is the exponent, written in the box.
"""
function springpot(x1, x2, y; label = "", order = "alpha", label_above::Bool = true, lead = 0.2)
    l = lead * (x2 - x1)
    xa, xb = x1 + l, x2 - l
    w = xb - xa
    wire(Point(x1, y), Point(xa, y))
    wire(Point(xb, y), Point(x2, y))
    @layer begin
        setcolor(tint(COL_POT, 0.88))
        rect(xa, y - HALF, w, 2HALF, :fill)
        setcolor(COL_POT)
        setline(LW_ELEMENT)
        rect(xa, y - HALF, w, 2HALF, :stroke)
        setline(LW_ELEMENT - 0.4)
        poly([Point(xa + u * w, y + HALF - 2HALF * sqrt(u)) for u in range(0, 1; length = 40)], :stroke)
    end
    order_label(order, Point(xa + 0.22w, y - 0.35HALF))
    elabel(label, (x1 + x2) / 2, y, label_above)
    return nothing
end

"The exponent of a springpot, in the upper left of its box."
order_label(order::AbstractString, p::Point) = label(m(order), p; size = 13)

"The two bars that put the branches at heights `ys` in parallel, and their leads."
function parallel_frame(x1, x2, ys; y = 0, lead = 16)
    @layer begin
        setcolor(INK)
        setline(LW_WIRE)
        line(Point(x1, minimum(ys)), Point(x1, maximum(ys)), :stroke)
        line(Point(x2, minimum(ys)), Point(x2, maximum(ys)), :stroke)
    end
    wire(Point(x1 - lead, y), Point(x1, y))
    wire(Point(x2, y), Point(x2 + lead, y))
    return nothing
end

"A terminal stub with a filled node, where the network is loaded."
function terminal(x, y; left::Bool = true, len = 18)
    e = Point(left ? x - len : x + len, y)
    wire(Point(x, y), e)
    @layer (setcolor(INK); circle(e, 2.8, :fill))
    return nothing
end

branch_ys(n, y0) = ntuple(i -> y0 + (i - (n + 1) / 2) * PITCH, n)

"A caption line: mathematics (`math = true`) or words."
caption(s, x, y; size = 15, color = INK, math = true) =
    label(math ? m(s) : txt(s), Point(x, y); size = size, color = color, valign = :bottom)
note(s, x, y) = caption(s, x, y; size = 11.5, color = MUTED, math = false)
title_at(s, x, y; math = false) = label(math ? m(s) : txt(s), Point(x, y); size = 16, color = VECTOR, valign = :bottom)

rheo(f, name, w, h) = sheet(f, "rheology/" * name, w, h)

# ── The three elementary elements ─────────────────────────────────────────────

rheo_elements() = rheo("elements", 660, 190) do
    y = 58
    spring(70, 210, y; label = "E")
    dashpot(260, 400, y; label = "eta")
    springpot(450, 590, y; label = "V", order = "alpha")
    caption("\"Spring\"(E)", 140, y + 52)
    caption("\"Dashpot\"(eta)", 330, y + 52)
    caption("\"ScottBlair\"(V, alpha)", 520, y + 52)
    note("elastic · viscous · parabolic. The third is the springpot, the Scott-Blair element and the", 330, 146)
    note("parabolic element of the bituminous models — one object under three names. The arc drawn in it", 330, 163)
    note("is its creep function: flat onto the axis at α = 0 (a spring), straight at α = 1 (a dashpot).", 330, 180)
end

# ── Maxwell and Kelvin units ──────────────────────────────────────────────────

rheo_maxwell_kelvin_units() = rheo("maxwell_kelvin_units", 680, 240) do
    # Maxwell: in series, the same stress in both, the compliances add —
    # J(t) = 1/E + t/η.
    y = 80
    terminal(60, y)
    spring(60, 185, y; label = "E")
    dashpot(185, 310, y; label = "eta")
    terminal(310, y; left = false)
    caption("\"MaxwellUnit\"(E, eta)", 185, y + 98)
    note("in series — a fluid", 185, y + 120)
    # Kelvin-Voigt: in parallel, the same strain in both, the stiffnesses add.
    ya, yb = branch_ys(2, y)
    terminal(420, y)
    spring(436, 576, ya; label = "E")
    dashpot(436, 576, yb; label = "eta", label_above = false)
    parallel_frame(436, 576, (ya, yb); y = y)
    terminal(592, y; left = false)
    caption("\"KelvinUnit\"(E, eta)", 506, y + 98)
    note("in parallel — rigid at t = 0", 506, y + 120)
    note("τ = η/E in both. Each has a Dirac impulse in one of its two functions,", 340, 214)
    note("which is why relaxation throws on the Kelvin unit.", 340, 230)
end

# ── The standard solid, both ways round ───────────────────────────────────────

rheo_zener() = rheo("zener", 720, 270) do
    title_at("The same material, written two ways", 360, 34)
    # zener_maxwell(E_inf, E_1, tau_1): an equilibrium spring in parallel with
    # one Maxwell branch. R(t) = E_∞ + E_1 exp(-t/τ_1).
    y = 112
    ya, yb = branch_ys(2, y)
    terminal(50, y)
    spring(50, 300, ya; label = "E_infinity")
    spring(50, 175, yb; label = "E_1", label_above = false)
    dashpot(175, 300, yb; label = "eta_1 = E_1 tau_1", label_above = false)
    parallel_frame(50, 300, (ya, yb); y = y)
    terminal(316, y; left = false)
    caption("\"zener_maxwell\"(E_infinity, E_1, tau_1)", 175, 220)
    caption("R(t) = E_infinity + E_1 e^(-t\\/tau_1)", 175, 252; color = MUTED)
    # zener_kelvin(E_glassy, E_delayed, tau_1): an instantaneous spring in
    # series with one Kelvin cell.
    terminal(410, y)
    spring(410, 520, y; label = "E_g")
    wire(Point(520, y), Point(536, y))
    spring(536, 666, ya; label = "E_d")
    dashpot(536, 666, yb; label = "E_d tau_1", label_above = false)
    parallel_frame(536, 666, (ya, yb); y = y)
    terminal(682, y; left = false)
    caption("\"zener_kelvin\"(E_g, E_d, tau_1)", 540, 220)
    caption("J(t) = 1\\/E_g + (1 - e^(-t\\/tau_1))\\/E_d", 540, 252; color = MUTED)
end

# ── The two Prony chains ──────────────────────────────────────────────────────

rheo_prony_relaxation() = rheo("prony_relaxation", 660, 300) do
    title_at("\"PronyRelaxation\"(E_infinity, E, tau)", 330, 36; math = true)
    y = 144
    y0, y1, y2 = branch_ys(3, y)
    terminal(60, y)
    spring(60, 500, y0; label = "E_infinity")
    spring(60, 280, y1; label = "E_1", label_above = false)
    dashpot(280, 500, y1; label = "eta_1 = E_1 tau_1", label_above = false)
    spring(60, 280, y2; label = "E_N", label_above = false)
    dashpot(280, 500, y2; label = "eta_N = E_N tau_N", label_above = false)
    parallel_frame(60, 500, (y0, y1, y2); y = y)
    terminal(516, y; left = false)
    caption("R(t) = E_infinity + sum_i E_i e^(-t\\/tau_i)", 330, 262)
    note("The equilibrium spring is what makes the chain a solid; drop it and the chain is a fluid.", 330, 288)
end

rheo_prony_creep() = rheo("prony_creep", 760, 272) do
    title_at("\"PronyCreep\"(J_0, J, tau, phi)", 380, 36; math = true)
    y = 120
    ya, yb = branch_ys(2, y)
    terminal(40, y)
    spring(40, 150, y; label = "1\\/J_0")
    wire(Point(150, y), Point(174, y))
    for (x0, sp, dp) in ((174, "1\\/J_1", "tau_1\\/J_1"), (376, "1\\/J_N", "tau_N\\/J_N"))
        spring(x0, x0 + 140, ya; label = sp)
        dashpot(x0, x0 + 140, yb; label = dp, label_above = false)
        parallel_frame(x0, x0 + 140, (ya, yb); y = y)
    end
    label(m("dots.c"), Point(345, y); size = 20, color = MUTED)
    wire(Point(532, y), Point(566, y))
    dashpot(566, 706, y; label = "1\\/phi")
    terminal(706, y; left = false)
    caption("J(t) = J_0 + sum_i J_i (1 - e^(-t\\/tau_i)) + phi t", 380, 232)
    note("φ is a fluidity, not a viscosity: a solid is φ = 0, which sorts and differentiates; η = ∞ does neither.", 380, 258)
end

# ── Burgers ───────────────────────────────────────────────────────────────────

rheo_burgers() = rheo("burgers", 660, 262) do
    title_at("\"burgers\"(k_s, eta_s, k_p, eta_p)", 330, 36; math = true)
    y = 120
    ya, yb = branch_ys(2, y)
    terminal(50, y)
    spring(50, 175, y; label = "k_s")
    dashpot(175, 300, y; label = "eta_s")
    wire(Point(300, y), Point(344, y))
    spring(360, 500, ya; label = "k_p")
    dashpot(360, 500, yb; label = "eta_p", label_above = false)
    parallel_frame(360, 500, (ya, yb); y = y)
    terminal(516, y; left = false)
    caption("J(t) = 1\\/k_s + t\\/eta_s + (1 - e^(-t k_p\\/eta_p))\\/k_p", 330, 216)
    note("A Maxwell unit in series with a Kelvin cell. The series dashpot never stops, so Burgers is a fluid — and its", 330, 236)
    note("Maxwell form therefore carries one branch more than its Kelvin form.", 330, 253)
end

# ── The fractional family ─────────────────────────────────────────────────────

rheo_fractional() = rheo("fractional", 740, 400) do
    title_at("The fractional family", 370, 34)
    # two parabolic elements in series
    y = 108
    terminal(60, y)
    springpot(60, 195, y; label = "V_a", order = "alpha")
    springpot(195, 330, y; label = "V_b", order = "beta", label_above = false)
    terminal(330, y; left = false)
    caption("\"FractionalMaxwell\"(V_a, alpha, V_b, beta)", 195, y + 92)
    # the same two in parallel
    ya, yb = branch_ys(2, y)
    terminal(450, y)
    springpot(466, 606, ya; label = "V_a", order = "alpha")
    springpot(466, 606, yb; label = "V_b", order = "beta", label_above = false)
    parallel_frame(466, 606, (ya, yb); y = y)
    terminal(622, y; left = false)
    caption("\"FractionalKelvin\"(V_a, alpha, V_b, beta)", 536, y + 92)
    # FractionalZener. Its transform is
    #     R*(p) = E_∞ + (E_0 − E_∞) (pτ)^α / (1 + (pτ)^α),
    # and a spring E_d in series with a parabolic element V p^α gives exactly
    #     E_d V p^α / (E_d + V p^α) = E_d (pτ)^α / (1 + (pτ)^α),   τ^α = V/E_d.
    # So the arrangement is  E_∞  ∥  (spring E_0 − E_∞  —  parabolic α).
    y2 = 282
    y2a, y2b = branch_ys(2, y2)
    terminal(190, y2)
    spring(190, 550, y2a; label = "E_infinity")
    spring(190, 370, y2b; label = "E_0 - E_infinity", label_above = false)
    springpot(370, 550, y2b; order = "alpha", label_above = false)
    parallel_frame(190, 550, (y2a, y2b); y = y2)
    terminal(566, y2; left = false)
    caption("\"FractionalZener\"(E_infinity, E_0, tau, alpha)", 370, 368)
    note("the Cole-Cole model", 370, 390)
end

# ── The bituminous models ─────────────────────────────────────────────────────

rheo_huet_sayegh_2s2p1d() = rheo("huet_sayegh_2s2p1d", 760, 400) do
    title_at("Two springs, two parabolic elements, one dashpot", 380, 34)
    # Huet-Sayegh: E*(p) = E00 + (E0 − E00) / (1 + δ(pτ_E)^{-k} + (pτ_E)^{-h}).
    # The denominator is a sum of compliances, so that branch is a spring and
    # two parabolic elements in series, the whole in parallel with E00.
    y = 110
    ya, yb = branch_ys(2, y)
    terminal(90, y)
    spring(90, 650, ya; label = "E_(00)")
    spring(90, 277, yb; label = "E_0 - E_(00)", label_above = false)
    springpot(277, 464, yb; label = "delta", order = "k", label_above = false)
    springpot(464, 650, yb; order = "h", label_above = false)
    parallel_frame(90, 650, (ya, yb); y = y)
    terminal(666, y; left = false)
    caption("\"HuetSayegh\"(E_(00), E_0, delta, tau_E, k, h) quad \"a solid\"", 380, 196)
    # 2S2P1D adds one linear dashpot in series inside the same branch.
    y2 = 286
    y2a, y2b = branch_ys(2, y2)
    terminal(90, y2)
    spring(90, 650, y2a; label = "E_(00)")
    spring(90, 230, y2b; label = "E_0 - E_(00)", label_above = false)
    springpot(230, 370, y2b; label = "delta", order = "k", label_above = false)
    springpot(370, 510, y2b; order = "h", label_above = false)
    dashpot(510, 650, y2b; label = "beta", label_above = false)
    parallel_frame(90, 650, (y2a, y2b); y = y2)
    terminal(666, y2; left = false)
    caption("\"Model2S2P1D\"(E_(00), E_0, delta, tau_E, k, h, beta)", 380, 372)
    note("The same branch with one linear dashpot added in series — the “1D” of the name, and what turns the solid into a fluid.", 380, 392)
end

# ── The conversion, as a picture ──────────────────────────────────────────────

rheo_kelvin_maxwell_conversion() = rheo("kelvin_maxwell_conversion", 880, 324) do
    title_at("\"maxwell_to_kelvin\" quad arrows.rl quad \"kelvin_to_maxwell\"", 440, 36; math = true)
    # left: a two-branch Maxwell chain
    y = 152
    y0, y1, y2 = branch_ys(3, y)
    terminal(40, y)
    spring(40, 310, y0; label = "E_infinity")
    spring(40, 175, y1; label = "E_1", label_above = false)
    dashpot(175, 310, y1; label = "E_1 tau_1", label_above = false)
    spring(40, 175, y2; label = "E_2", label_above = false)
    dashpot(175, 310, y2; label = "E_2 tau_2", label_above = false)
    parallel_frame(40, 310, (y0, y1, y2); y = y)
    terminal(326, y; left = false)
    caption("generalized Maxwell", 175, 272; math = false)
    caption("R(t)", 175, 294; color = MUTED)
    # the two arrows
    arrow_ab(Point(362, y - 12), Point(430, y - 12); color = VECTOR, lw = 2.0)
    arrow_ab(Point(430, y + 12), Point(362, y + 12); color = VECTOR, lw = 2.0)
    label(txt("exact"), Point(396, y - 30); size = 12.5, color = VECTOR)
    # right: the equivalent two-cell Kelvin chain
    ka, kb = branch_ys(2, y)
    terminal(466, y)
    spring(466, 566, y; label = "1\\/J_0")
    wire(Point(566, y), Point(582, y))
    spring(582, 692, ka; label = "1\\/J_1")
    dashpot(582, 692, kb; label = "sigma_1\\/J_1", label_above = false)
    parallel_frame(582, 692, (ka, kb); y = y)
    wire(Point(708, y), Point(724, y))
    spring(724, 834, ka; label = "1\\/J_2")
    dashpot(724, 834, kb; label = "sigma_2\\/J_2", label_above = false)
    parallel_frame(724, 834, (ka, kb); y = y)
    terminal(850, y; left = false)
    caption("generalized Kelvin", 650, 272; math = false)
    caption("J(t)", 650, 294; color = MUTED)
    note("The retardation times σⱼ interlace the relaxation times — τ₁ < σ₁ < τ₂ < σ₂ — which is what isolates every root before any arithmetic.", 440, 316)
end

rheology_figures() = (
    rheo_elements(), rheo_maxwell_kelvin_units(), rheo_zener(), rheo_prony_relaxation(),
    rheo_prony_creep(), rheo_burgers(), rheo_fractional(), rheo_huet_sayegh_2s2p1d(),
    rheo_kelvin_maxwell_conversion(),
)
