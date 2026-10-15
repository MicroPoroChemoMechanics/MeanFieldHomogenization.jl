# =============================================================================
#  vignettes.jl — three small sketches of what lies beyond the one-site mean
#  field: positioned particles and their pairwise interactions, a neural
#  surrogate from shape to response, a finite-element mesh of a crack.
#  Docs: manual/particle_assemblies.md, manual/neural_inclusions.md,
#        manual/fe_inclusions.md. The MPCM talk draws them without their panel.
# =============================================================================

"Positioned particles, the closest interacting pair marked."
function vignette_nbody(; path = "schemes/particle_assembly", panel::Bool = true)
    return sheet(path, 320, 200; panel = panel) do
        rng = StableRNG(7)
        c = Point(160, 100)
        rve_box(c, 300, 180)
        items = scatter(rng, repeat([Ellipsoidal(), Needle()], 13), in_box(c, 280, 160); s = 0.85, gap = 9)
        ps = [p for (_, p, _) in items]
        n = length(ps)
        clear(p) = 40 < p.x < 280 && 38 < p.y < 162     # rings drawn whole, away from the edge
        i, j = argmin(Dict((i, j) => distance(ps[i], ps[j]) for i in 1:n for j in (i + 1):n if clear(ps[i]) && clear(ps[j])))
        draw_all(items; s = 0.85)
        @layer (setcolor(VECTOR); setline(2.2); setdash([4, 3]); line(ps[i], ps[j], :stroke))
        spot(ps[i], 18)
        spot(ps[j], 18)
    end
end

"A small neural network, from the shape of an inclusion to its response."
function vignette_nn(; path = "nn/network_sketch", words = ("shape", "response"), panel::Bool = true)
    return sheet(path, 320, 210; panel = panel) do
        layers, xs = (3, 5, 5, 2), (40, 125, 210, 285)
        nodes = [[Point(x, n > 1 ? 25 + (k - 1) * 150 / (n - 1) : 100) for k in 1:n] for (x, n) in zip(xs, layers)]
        @layer begin
            setcolor(tint(GUIDE, 0.45))
            setline(1.1)
            newpath()
            for (a, b) in zip(nodes[1:(end - 1)], nodes[2:end]), p in a, q in b
                move(p)
                line(q)
            end
            strokepath()
        end
        for (layer, col) in zip(nodes, (GREEN, BLUE, BLUE, ORANGE)), p in layer
            disk(p, 10; fill = col, stroke = PANEL, lw = 1.5)
        end
        label(txt(words[1]), Point(40, 192); size = 15, color = MUTED)
        label(txt(words[2]), Point(285, 192); size = 15, color = MUTED)
    end
end

"""
Edges joining two closed rings of nodes, parameterized by `t` increasing on
`[0, 2π)`: the zip triangulation, which advances on the ring whose next node
comes first.
"""
function zip_edges(r1, t1, r2, t2)
    n1, n2 = length(r1), length(r2)
    i, j = 1, 1
    edges = Tuple{Point, Point}[]
    for _ in 1:(n1 + n2)
        push!(edges, (r1[mod1(i, n1)], r2[mod1(j, n2)]))
        ti = t1[mod1(i + 1, n1)] + 2π * fld(i, n1)
        tj = t2[mod1(j + 1, n2)] + 2π * fld(j, n2)
        ti < tj ? (i += 1) : (j += 1)
    end
    return edges
end

"A finite-element mesh of an elliptical crack, refined toward its front."
function vignette_fe(; path = "fe/crack_mesh_sketch", panel::Bool = true)
    return sheet(path, 320, 200; panel = panel) do
        c, A, B = Point(160, 100), 78.0, 46.0
        # rings of nodes, closer and closer to the front s = 1, denser there too
        ss = (0.35, 0.58, 0.76, 0.88, 0.95, 1.0, 1.05, 1.12, 1.25, 1.42, 1.65, 1.95, 2.35, 2.85, 3.4)
        ns = (10, 16, 22, 28, 34, 40, 40, 36, 32, 28, 24, 22, 20, 18, 18)
        ts = [[2π * (i + 0.5 * isodd(k)) / ns[k] for i in 0:(ns[k] - 1)] for k in eachindex(ss)]
        rings = [[c + Point(ss[k] * A * cos(t), ss[k] * B * sin(t)) for t in ts[k]] for k in eachindex(ss)]
        @layer begin
            box(c, 300, 180, 10, :clip)
            setcolor(MATRIX)
            box(c, 300, 180, :fill)
            ell(c, A, B; fill = tint(CRACK, 0.72), stroke = nothing)
            # every edge in one path, stroked once
            setcolor(INK)
            setopacity(0.5)
            setline(0.7)
            newpath()
            foreach(p -> (move(c); line(p)), rings[1])
            for k in eachindex(rings)
                move(rings[k][end])
                foreach(line, rings[k])
                k == length(rings) && break
                for (a, b) in zip_edges(rings[k], ts[k], rings[k + 1], ts[k + 1])
                    move(a)
                    line(b)
                end
            end
            strokepath()
            setopacity(1.0)
            ell(c, A, B; fill = nothing, stroke = CRACK, lw = 2.4)
        end
        @layer (setcolor(INK); setline(1.6); box(c, 300, 180, 10, :stroke))
    end
end

vignette_figures() = (vignette_nbody(), vignette_nn(), vignette_fe())
