# make_thick_cylinder_figures.jl — assets for `docs/src/fe_coupling/`.
#
# Hand-run maintenance script, exactly like `make_doc_figures.jl` next to it:
# the documentation page is static and reads the committed PNG/GIF, so a
# documentation build never re-runs a finite-element solve.
#
#     julia --project=scripts/fe scripts/fe/make_thick_cylinder_figures.jl
#
# Writes into `docs/src/assets/fe_coupling/`:
#
#   mesh.png            the quarter-annulus mesh and the deformed shape
#   lame.png            u_r(r) and σ_θθ(r) against the closed form
#   closure.gif         the crack-closure front advancing with the pressure
#   anisotropy.png      induced anisotropy of C_hom across the wall

using MeanFieldHomogenization
using TensND
using Ferrite
using LinearAlgebra
using Printf
using Plots
# Composite panels crop their outer labels unless the margins are explicit —
# a big enough canvas is not enough on its own.
gr(size = (760, 320), dpi = 130)
const MARGINS = (left_margin = 7Plots.mm, bottom_margin = 7Plots.mm,
    top_margin = 3Plots.mm, right_margin = 3Plots.mm)

const EXT = Base.get_extension(
    MeanFieldHomogenization, :MeanFieldHomogenizationFerriteMaterialExt
)
const OUT = normpath(joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "fe_coupling"))
mkpath(OUT)

const Ri, Ro, PMAX = 0.1, 1.0, 20.0
const C_MATRIX = TensISO{3}(3 * 30.0, 2 * 18.0)

# ── model ────────────────────────────────────────────────────────────────────

grid = EXT.annulus_grid(Ri, Ro, 24, 24)
ip = Lagrange{RefQuadrilateral, 1}()^2
cv = CellValues(QuadratureRule{RefQuadrilateral}(2), ip)
fv = FacetValues(FacetQuadratureRule{RefQuadrilateral}(2), ip)
dh = DofHandler(grid); add!(dh, :u, ip); close!(dh)
ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0, 2))
add!(ch, Dirichlet(:u, getfacetset(grid, "top"), (x, t) -> 0.0, 1))
close!(ch)

coords = [grid.nodes[i].x for i in 1:getnnodes(grid)]
radii = [norm(x) for x in coords]

function pressure_force(p)
    f = zeros(ndofs(dh))
    for fc in FacetIterator(dh, getfacetset(grid, "left"))
        reinit!(fv, fc)
        fe = zeros(getnbasefunctions(fv))
        for q in 1:getnquadpoints(fv)
            n = getnormal(fv, q); dΓ = getdetJdV(fv, q)
            for i in 1:getnbasefunctions(fv)
                fe[i] += (shape_value(fv, q, i) ⋅ (-p * n)) * dΓ
            end
        end
        f[celldofs(fc)] .+= fe
    end
    return f
end

"Newton solve up to pressure `p`, in `nsteps`; returns `(u, states)` per step."
function solve_path(material, nsteps; cache = MaterialCache())
    nqp = getnquadpoints(cv)
    states = EXT.mfh_states(material, getncells(grid), nqp)
    states_old = EXT.mfh_states(material, getncells(grid), nqp)
    u = zeros(ndofs(dh)); K = allocate_matrix(dh, ch)
    nbf = getnbasefunctions(cv); Ke, re = zeros(nbf, nbf), zeros(nbf)
    history = Vector{Tuple{Float64, Vector{Float64}, Vector{Vector{Any}}}}()
    for step in 1:nsteps
        p = PMAX * step / nsteps
        for _ in 1:25
            asm = start_assemble(K, zeros(ndofs(dh))); r = zeros(ndofs(dh))
            for cell in CellIterator(dh)
                reinit!(cv, cell); fill!(Ke, 0); fill!(re, 0)
                ed = celldofs(cell)
                EXT.mfh_element!(
                    Ke, re, cv, material, u[ed],
                    states[cellid(cell)], states_old[cellid(cell)], 0.0; cache = cache
                )
                assemble!(asm, ed, Ke, zeros(nbf)); r[ed] .+= re
            end
            r .-= pressure_force(p)
            apply_zero!(K, r, ch)
            norm(r) < 1.0e-9 && break
            u .-= K \ r; apply!(u, ch)
        end
        for c in eachindex(states), q in eachindex(states[c])
            states_old[c][q] = states[c][q]
        end
        push!(history, (p, copy(u), deepcopy(states)))
    end
    return history
end

radial_disp(u) = begin
    un = evaluate_at_grid_nodes(dh, u, :u)      # dof order != node order
    [(un[i][1] * coords[i][1] + un[i][2] * coords[i][2]) / radii[i] for i in eachindex(radii)]
end

# ── 1. linear composite, against Lamé ────────────────────────────────────────

rve_lin = RVE()
add_phase!(rve_lin, :M, Ellipsoid(1.0), Dict(:C => C_MATRIX); fraction = :rest)
add_phase!(rve_lin, :I, Ellipsoid(1.0), Dict(:C => TensISO{3}(3 * 120.0, 2 * 80.0)); fraction = 0.25)
mat_lin = HomogenizedElastic(rve_lin, MoriTanaka())
k_h, μ_h = k_mu(stiffness(mat_lin))
E_h = 9k_h * μ_h / (3k_h + μ_h); ν_h = (3k_h - 2μ_h) / (2 * (3k_h + μ_h))

hist_lin = solve_path(mat_lin, 1)
u_lin = hist_lin[end][2]
ur_lin = radial_disp(u_lin)

lame_ur(r) = (1 + ν_h) * PMAX * Ri^2 / (E_h * (Ro^2 - Ri^2)) * ((1 - 2ν_h) * r + Ro^2 / r)
lame_σθθ(r) = PMAX * Ri^2 / (Ro^2 - Ri^2) * (1 + Ro^2 / r^2)

# hoop stress at the quadrature points, projected on the radial direction
function hoop_profile(u, material, states)
    rs, σθ = Float64[], Float64[]
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        ue = u[celldofs(cell)]
        for q in 1:getnquadpoints(cv)
            x = spatial_coordinate(cv, q, getcoordinates(cell))
            r = norm(x); eθ = Vec{2}((-x[2] / r, x[1] / r))
            ε = function_symmetric_gradient(cv, q, ue)
            resp = plane_strain_response(material, ε, states[cellid(cell)][q], 0.0)
            push!(rs, r); push!(σθ, eθ ⋅ (resp.σ ⋅ eθ))
        end
    end
    return rs, σθ
end

rs, σθ = hoop_profile(u_lin, mat_lin, hist_lin[end][3])
perm = sortperm(radii)
p1 = plot(
    radii[perm], ur_lin[perm]; label = "FE (MFH material)", lw = 0, marker = :circle,
    ms = 2.2, mc = :steelblue, xlabel = "radius r", ylabel = "radial displacement uᵣ", legend = :topright
)
plot!(p1, range(Ri, Ro; length = 200), lame_ur; label = "Lamé", lw = 2, lc = :black, ls = :dash)
p2 = scatter(
    rs, σθ; label = "FE (quadrature points)", ms = 1.6, mc = :indianred,
    xlabel = "radius r", ylabel = "hoop stress σθθ", legend = :topright, msw = 0
)
plot!(p2, range(Ri, Ro; length = 200), lame_σθθ; label = "Lamé", lw = 2, lc = :black, ls = :dash)
savefig(
    plot(p1, p2; layout = (1, 2), size = (900, 380), MARGINS...),
    joinpath(OUT, "lame.png")
)

# ── 2. mesh and deformed shape ───────────────────────────────────────────────

function mesh_plot(u = nothing; scale = 1.0, color = :grey35, kw...)
    pl = plot(; aspect_ratio = 1, legend = false, axis = false, grid = false, kw...)
    un = u === nothing ? nothing : evaluate_at_grid_nodes(dh, u, :u)
    for c in grid.cells
        nn = collect(c.nodes); push!(nn, nn[1])
        xs = [coords[n][1] + (un === nothing ? 0.0 : scale * un[n][1]) for n in nn]
        ys = [coords[n][2] + (un === nothing ? 0.0 : scale * un[n][2]) for n in nn]
        plot!(pl, xs, ys; lc = color, lw = 0.5)
    end
    return pl
end

savefig(
    plot(
        mesh_plot(; title = "mesh (24×24, no gmsh)"),
        mesh_plot(u_lin; scale = 8.0, color = :steelblue, title = "deformed ×8");
        layout = (1, 2), size = (860, 400), MARGINS...
    ),
    joinpath(OUT, "mesh.png")
)

# ── 3. crack closure front, as a GIF ─────────────────────────────────────────

rve_cr = RVE()
add_phase!(rve_cr, :M, Ellipsoid(1.0), Dict(:C => C_MATRIX); fraction = :rest)
add_phase!(rve_cr, :Fx, PennyCrack(1.0; euler_angles = (π / 2, 0.0)), Dict(:C => C_MATRIX); density = 0.15)
add_phase!(rve_cr, :Fy, PennyCrack(1.0; euler_angles = (π / 2, π / 2)), Dict(:C => C_MATRIX); density = 0.15)
mat_cr = MicrocrackedMaterial(rve_cr, MoriTanaka(); ω₀ = (2.0e-3, 2.0e-3))

cache = MaterialCache()
nsteps = 12
hist_cr = solve_path(mat_cr, nsteps; cache = cache)
@printf("cracked run: %d scheme solves, %d quadrature points\n",
    cache_stats(cache).entries, getncells(grid) * getnquadpoints(cv))

# Per-cell fraction of quadrature points whose radial family has closed.
function closed_fraction(states)
    [count(st -> !open_set(st)[1], states[c]) / length(states[c]) for c in 1:getncells(grid)]
end

anim = @animate for (p, u, states) in hist_cr
    frac = closed_fraction(states)
    pl = plot(; aspect_ratio = 1, legend = false, axis = false, grid = false,
        title = @sprintf("p = %.1f MPa — closed fraction, family normal to e₁", p))
    for (i, c) in enumerate(grid.cells)
        nn = collect(c.nodes); push!(nn, nn[1])
        xs = [coords[n][1] for n in nn]; ys = [coords[n][2] for n in nn]
        col = cgrad(:lajolla)[clamp(frac[i], 0, 1)]
        plot!(pl, Shape(xs, ys); fillcolor = col, lc = :grey70, lw = 0.3)
    end
    pl
end
gif(anim, joinpath(OUT, "closure.gif"); fps = 3)

# ── 4. induced anisotropy across the wall ────────────────────────────────────

# Closure is governed by the ANGLE, not the radius: a crack whose normal is e₁
# sees σ_rr near θ = 0 (compressive, so it closes) and σ_θθ near θ = π/2
# (tensile, so it stays open). A profile against r would hide that entirely, so
# the state is mapped over the domain instead.
states_end = hist_cr[end][3]
function state_map(fam, title)
    xs, ys, cs = Float64[], Float64[], Float64[]
    for cell in CellIterator(dh)
        reinit!(cv, cell)
        for q in 1:getnquadpoints(cv)
            x = spatial_coordinate(cv, q, getcoordinates(cell))
            push!(xs, x[1]); push!(ys, x[2])
            push!(cs, open_set(states_end[cellid(cell)][q])[fam] ? 1.0 : 0.0)
        end
    end
    scatter(
        xs, ys; marker_z = cs, ms = 2.4, msw = 0, aspect_ratio = 1,
        c = cgrad([:indianred, :steelblue]), clims = (0, 1), colorbar = false,
        legend = false, title = title, xlabel = "x", ylabel = "y"
    )
end
savefig(
    plot(
        state_map(1, "family normal to e₁"),
        state_map(2, "family normal to e₂");
        layout = (1, 2), size = (880, 420), MARGINS...,
        plot_title = "blue = open, red = closed (p = 20 MPa)"
    ),
    joinpath(OUT, "anisotropy.png")
)

println("figures written to ", OUT)
