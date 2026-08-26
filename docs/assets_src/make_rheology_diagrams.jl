# =============================================================================
#  make_rheology_diagrams.jl — the spring-and-dashpot networks of the catalog.
#
#  Every diagram corresponds **exactly** to one model in
#  `src/Viscoelasticity/rheology_models.jl`, and carries **the same symbols as
#  that model's constructor signature and as the formulas on the page** — so a
#  reader can put picture, formula and code side by side without translating
#  between three sets of names.  Where an arrangement is not obvious from the
#  transform (the fractional Zener and Huet-Sayegh especially), the derivation
#  is written in a comment above the drawing.
#
#  Labels are `LaTeXString`s, typeset through `MathTeXEngine`: `E_\infty`,
#  `\tau_E`, `V_b`, `E_{00}` all render properly, and several of them cannot be
#  written in Unicode at all — there is no subscript `b`.
#
#  Layout is **computed**, never eyeballed: branches sit `PITCH` apart, labels
#  `LABEL_UP` from their own element, captions `CAPTION_DROP` below.  Hand-tuned
#  offsets are what produce overlapping text.
#
#  Output: SVG in `docs/src/assets/rheology/` — vector, a few kilobytes, and
#  committed, so `Luxor` never becomes a documentation dependency.
#
#  Usage:
#      julia --project=docs/assets_src docs/assets_src/make_rheology_diagrams.jl
# =============================================================================

include(joinpath(@__DIR__, "rheology_symbols.jl"))

const OUT = normpath(joinpath(@__DIR__, "..", "src", "assets", "rheology"))
mkpath(OUT)

const FONT = "DejaVu Sans"
const GREY = "#555555"

"""
    sheet(name, w, h) do … end

Open an SVG of size `w × h` with the origin at its **top-left** corner, so every
coordinate below is a plain distance from that corner.
"""
function sheet(f, name, w, h)
    Drawing(w, h, joinpath(OUT, name * ".svg"))
    origin()
    background("white")
    fontface(FONT)
    translate(-w / 2, -h / 2)
    f()
    finish()
    println("  ", rpad(name * ".svg", 32), w, "×", h)
    return nothing
end

"""
    branch_ys(n, y0)

The `n` branch heights of a parallel stack centered on `y0`, spaced by `PITCH`.
"""
branch_ys(n, y0) = ntuple(i -> y0 + (i - (n + 1) / 2) * PITCH, n)

println("Rheological network diagrams → ", OUT)

# ── The three elementary elements ───────────────────────────────────────────

sheet("elements", 660, 176) do
    y = 52
    spring(70, 210, y; label = L"E")
    dashpot(260, 400, y; label = L"\eta")
    springpot(450, 590, y; label = L"V", order = L"\alpha")

    caption(L"\mathrm{Spring}(E)", 140, y + CAPTION_DROP; size = 12)
    caption(L"\mathrm{Dashpot}(\eta)", 330, y + CAPTION_DROP; size = 12)
    caption(L"\mathrm{ScottBlair}(V,\, \alpha)", 520, y + CAPTION_DROP; size = 12)

    caption(
        "elastic · viscous · parabolic. The third is the springpot, the Scott-Blair element and the",
        330, 132; size = 11, color = GREY
    )
    caption(
        "parabolic element of the bituminous models — one object under three names. The arc drawn in it",
        330, 148; size = 11, color = GREY
    )
    caption(
        "is its creep function: flat onto the axis at α = 0 (a spring), straight at α = 1 (a dashpot).",
        330, 164; size = 11, color = GREY
    )
end

# ── Maxwell and Kelvin units ────────────────────────────────────────────────

sheet("maxwell_kelvin_units", 680, 232) do
    ## Maxwell: in series, so the same stress runs through both and the
    ## compliances add — J(t) = 1/E + t/η.
    y = 78
    terminal(60, y)
    spring(60, 185, y; label = L"E")
    dashpot(185, 310, y; label = L"\eta")
    terminal(310, y; left = false)
    caption(L"\mathrm{MaxwellUnit}(E,\, \eta)", 185, y + PITCH / 2 + CAPTION_DROP + 4; size = 12)
    caption("in series — a fluid", 185, y + PITCH / 2 + CAPTION_DROP + 24; size = 11, color = GREY)

    ## Kelvin-Voigt: in parallel, so the same strain is in both and the
    ## stiffnesses add.
    ya, yb = branch_ys(2, y)
    terminal(420, y)
    spring(436, 576, ya; label = L"E")
    dashpot(436, 576, yb; label = L"\eta", label_above = false)
    parallel_frame(436, 576, (ya, yb); y = y)
    terminal(592, y; left = false)
    caption(L"\mathrm{KelvinUnit}(E,\, \eta)", 506, y + PITCH / 2 + CAPTION_DROP + 4; size = 12)
    caption("in parallel — rigid at t = 0", 506, y + PITCH / 2 + CAPTION_DROP + 24; size = 11, color = GREY)

    caption(
        "τ = η/E in both. Each has a Dirac impulse in one of its two functions, which is why relaxation throws on the Kelvin unit.",
        340, 218; size = 11, color = GREY
    )
end

# ── The standard solid, both ways round ─────────────────────────────────────

sheet("zener", 720, 262) do
    title_at("The same material, written two ways", 360, 26)

    ## zener_maxwell(E_inf, E_1, tau_1): an equilibrium spring in parallel with
    ## one Maxwell branch.  R(t) = E_∞ + E_1 exp(-t/τ_1).
    y = 108
    ya, yb = branch_ys(2, y)
    terminal(50, y)
    spring(50, 300, ya; label = L"E_\infty")
    spring(50, 175, yb; label = L"E_1", label_above = false)
    dashpot(175, 300, yb; label = L"\eta_1 = E_1\tau_1", label_above = false)
    parallel_frame(50, 300, (ya, yb); y = y)
    terminal(316, y; left = false)
    caption(L"\mathrm{zener\_maxwell}(E_\infty,\, E_1,\, \tau_1)", 175, 202; size = 12)
    caption(L"R(t) = E_\infty + E_1 e^{-t/\tau_1}", 175, 232; size = 12, color = GREY)

    ## zener_kelvin(E_glassy, E_delayed, tau_1): an instantaneous spring in
    ## series with one Kelvin cell.
    terminal(410, y)
    spring(410, 520, y; label = L"E_g")
    wire(Point(520, y), Point(536, y))
    spring(536, 666, ya; label = L"E_d")
    dashpot(536, 666, yb; label = L"E_d\tau_1", label_above = false)
    parallel_frame(536, 666, (ya, yb); y = y)
    terminal(682, y; left = false)
    caption(L"\mathrm{zener\_kelvin}(E_g,\, E_d,\, \tau_1)", 540, 202; size = 12)
    caption(L"J(t) = 1/E_g + (1 - e^{-t/\tau_1})/E_d", 540, 232; size = 12, color = GREY)
end

# ── The two Prony chains ────────────────────────────────────────────────────

sheet("prony_relaxation", 660, 290) do
    title_at(L"\mathrm{PronyRelaxation}(E_\infty,\, E,\, \tau)", 330, 28)
    y = 140
    y0, y1, y2 = branch_ys(3, y)
    terminal(60, y)
    spring(60, 500, y0; label = L"E_\infty")
    spring(60, 280, y1; label = L"E_1", label_above = false)
    dashpot(280, 500, y1; label = L"\eta_1 = E_1\tau_1", label_above = false)
    spring(60, 280, y2; label = L"E_N", label_above = false)
    dashpot(280, 500, y2; label = L"\eta_N = E_N\tau_N", label_above = false)
    parallel_frame(60, 500, (y0, y1, y2); y = y)
    terminal(516, y; left = false)
    caption(L"R(t) = E_\infty + \sum_i E_i e^{-t/\tau_i}", 330, 252; size = 12)
    caption(
        "The equilibrium spring is what makes the chain a solid; drop it and the chain is a fluid.",
        330, 278; size = 11, color = GREY
    )
end

sheet("prony_creep", 760, 262) do
    title_at(L"\mathrm{PronyCreep}(J_0,\, J,\, \tau,\, \varphi)", 380, 28)
    y = 116
    ya, yb = branch_ys(2, y)
    terminal(40, y)
    spring(40, 150, y; label = L"1/J_0")
    wire(Point(150, y), Point(174, y))
    for (x0, sp, dp) in (
            (174, L"1/J_1", L"\tau_1/J_1"),
            (376, L"1/J_N", L"\tau_N/J_N"),
        )
        spring(x0, x0 + 140, ya; label = sp)
        dashpot(x0, x0 + 140, yb; label = dp, label_above = false)
        parallel_frame(x0, x0 + 140, (ya, yb); y = y)
    end
    caption("⋯", 345, y + 5; size = 18, color = "#888888")
    wire(Point(532, y), Point(566, y))
    dashpot(566, 706, y; label = L"1/\varphi")
    terminal(706, y; left = false)
    caption(L"J(t) = J_0 + \sum_i J_i (1 - e^{-t/\tau_i}) + \varphi t", 380, 222; size = 12)
    caption(
        "φ is a fluidity, not a viscosity: a solid is φ = 0, which sorts and differentiates; η = ∞ does neither.",
        380, 248; size = 11, color = GREY
    )
end

# ── Burgers ─────────────────────────────────────────────────────────────────

sheet("burgers", 660, 244) do
    title_at(L"\mathrm{burgers}(k_s,\, \eta_s,\, k_p,\, \eta_p)", 330, 28)
    y = 116
    ya, yb = branch_ys(2, y)
    terminal(50, y)
    spring(50, 175, y; label = L"k_s")
    dashpot(175, 300, y; label = L"\eta_s")
    wire(Point(300, y), Point(344, y))
    spring(360, 500, ya; label = L"k_p")
    dashpot(360, 500, yb; label = L"\eta_p", label_above = false)
    parallel_frame(360, 500, (ya, yb); y = y)
    terminal(516, y; left = false)
    caption(L"J(t) = 1/k_s + t/\eta_s + (1 - e^{-t k_p/\eta_p})/k_p", 330, 202; size = 12)
    caption(
        "A Maxwell unit in series with a Kelvin cell. The series dashpot never stops, so Burgers is a fluid — and its",
        330, 226; size = 11, color = GREY
    )
    caption(
        "Maxwell form therefore carries one branch more than its Kelvin form.",
        330, 240; size = 11, color = GREY
    )
end

# ── The fractional family ───────────────────────────────────────────────────

sheet("fractional", 740, 392) do
    title_at("The fractional family", 370, 28)

    ## Two parabolic elements in series.
    y = 104
    terminal(60, y)
    springpot(60, 195, y; label = L"V_a", order = L"\alpha")
    springpot(195, 330, y; label = L"V_b", order = L"\beta", label_above = false)
    terminal(330, y; left = false)
    caption(L"\mathrm{FractionalMaxwell}(V_a,\, \alpha,\, V_b,\, \beta)", 195, y + 86; size = 12)

    ## The same two in parallel.
    ya, yb = branch_ys(2, y)
    terminal(450, y)
    springpot(466, 606, ya; label = L"V_a", order = L"\alpha")
    springpot(466, 606, yb; label = L"V_b", order = L"\beta", label_above = false)
    parallel_frame(466, 606, (ya, yb); y = y)
    terminal(622, y; left = false)
    caption(L"\mathrm{FractionalKelvin}(V_a,\, \alpha,\, V_b,\, \beta)", 536, y + 86; size = 12)

    ## FractionalZener.  Its transform is
    ##     R*(p) = E_∞ + (E_0 − E_∞) (pτ)^α / (1 + (pτ)^α),
    ## and a spring E_d in series with a parabolic element V p^α gives exactly
    ##     E_d V p^α / (E_d + V p^α) = E_d (pτ)^α / (1 + (pτ)^α),   τ^α = V/E_d.
    ## So the arrangement is  E_∞  ∥  (spring E_0 − E_∞  —  parabolic α).
    y2 = 276
    y2a, y2b = branch_ys(2, y2)
    terminal(190, y2)
    spring(190, 550, y2a; label = L"E_\infty")
    spring(190, 370, y2b; label = L"E_0 - E_\infty", label_above = false)
    springpot(370, 550, y2b; label = "", order = L"\alpha", label_above = false)
    parallel_frame(190, 550, (y2a, y2b); y = y2)
    terminal(566, y2; left = false)
    caption(L"\mathrm{FractionalZener}(E_\infty,\, E_0,\, \tau,\, \alpha)", 370, 358; size = 12)
    caption("the Cole-Cole model", 370, 380; size = 11, color = GREY)
end

# ── The bituminous models ───────────────────────────────────────────────────

sheet("huet_sayegh_2s2p1d", 760, 386) do
    title_at("Two springs, two parabolic elements, one dashpot", 380, 28)

    ## Huet-Sayegh:  E*(p) = E00 + (E0 − E00) / (1 + δ(pτ_E)^{-k} + (pτ_E)^{-h}).
    ## The denominator is a sum of *compliances*, so that branch is a spring
    ## and two parabolic elements in series, the whole in parallel with E00.
    y = 106
    ya, yb = branch_ys(2, y)
    terminal(90, y)
    spring(90, 650, ya; label = L"E_{00}")
    spring(90, 277, yb; label = L"E_0 - E_{00}", label_above = false)
    springpot(277, 464, yb; label = L"\delta", order = L"k", label_above = false)
    springpot(464, 650, yb; label = "", order = L"h", label_above = false)
    parallel_frame(90, 650, (ya, yb); y = y)
    terminal(666, y; left = false)
    caption(
        L"\mathrm{HuetSayegh}(E_{00},\, E_0,\, \delta,\, \tau_E,\, k,\, h) \quad \mathrm{a\ solid}",
        380, 188; size = 12
    )

    ## 2S2P1D adds one linear dashpot in series inside the same branch.
    y2 = 278
    y2a, y2b = branch_ys(2, y2)
    terminal(90, y2)
    spring(90, 650, y2a; label = L"E_{00}")
    spring(90, 230, y2b; label = L"E_0 - E_{00}", label_above = false)
    springpot(230, 370, y2b; label = L"\delta", order = L"k", label_above = false)
    springpot(370, 510, y2b; label = "", order = L"h", label_above = false)
    dashpot(510, 650, y2b; label = L"\beta", label_above = false)
    parallel_frame(90, 650, (y2a, y2b); y = y2)
    terminal(666, y2; left = false)
    caption(
        L"\mathrm{Model2S2P1D}(E_{00},\, E_0,\, \delta,\, \tau_E,\, k,\, h,\, \beta)",
        380, 364; size = 12
    )
    caption(
        "The same branch with one linear dashpot added in series — the \"1D\" of the name, and what turns the solid into a fluid.",
        380, 380; size = 11, color = GREY
    )
end

# ── The conversion, as a picture ────────────────────────────────────────────

sheet("kelvin_maxwell_conversion", 880, 312) do
    title_at(L"\mathrm{maxwell\_to\_kelvin} \quad \rightleftarrows \quad \mathrm{kelvin\_to\_maxwell}", 440, 28)

    ## Left: a two-branch Maxwell chain.
    y = 148
    y0, y1, y2 = branch_ys(3, y)
    terminal(40, y)
    spring(40, 310, y0; label = L"E_\infty")
    spring(40, 175, y1; label = L"E_1", label_above = false)
    dashpot(175, 310, y1; label = L"E_1\tau_1", label_above = false)
    spring(40, 175, y2; label = L"E_2", label_above = false)
    dashpot(175, 310, y2; label = L"E_2\tau_2", label_above = false)
    parallel_frame(40, 310, (y0, y1, y2); y = y)
    terminal(326, y; left = false)
    caption("generalized Maxwell", 175, 262; size = 12)
    caption(L"R(t)", 175, 284; size = 12, color = GREY)

    ## The two arrows.
    setcolor("#1F4E79")
    setline(2.0)
    for (x0, x1, ytip) in ((362, 430, y - 12), (430, 362, y + 12))
        line(Point(x0, ytip), Point(x1, ytip), :stroke)
        dx = sign(x1 - x0)
        line(Point(x1 - 8dx, ytip - 5), Point(x1, ytip), :stroke)
        line(Point(x1 - 8dx, ytip + 5), Point(x1, ytip), :stroke)
    end
    caption("exact", 396, y - 26; size = 11, color = "#1F4E79")

    ## Right: the equivalent two-cell Kelvin chain.
    ka, kb = branch_ys(2, y)
    terminal(466, y)
    spring(466, 566, y; label = L"1/J_0")
    wire(Point(566, y), Point(582, y))
    spring(582, 692, ka; label = L"1/J_1")
    dashpot(582, 692, kb; label = L"\sigma_1/J_1", label_above = false)
    parallel_frame(582, 692, (ka, kb); y = y)
    wire(Point(708, y), Point(724, y))
    spring(724, 834, ka; label = L"1/J_2")
    dashpot(724, 834, kb; label = L"\sigma_2/J_2", label_above = false)
    parallel_frame(724, 834, (ka, kb); y = y)
    terminal(850, y; left = false)
    caption("generalized Kelvin", 650, 262; size = 12)
    caption(L"J(t)", 650, 284; size = 12, color = GREY)

    caption(
        "The retardation times σⱼ interlace the relaxation times — τ₁ < σ₁ < τ₂ < σ₂ — which is what isolates every root before any arithmetic.",
        440, 304; size = 11, color = GREY
    )
end

println("done.")
