# arma2011_welltest.jl — the simplified well test of Barthelemy & Daniel,
# "Multiscale hydromechanical coupling for finite-element simulations of
# fractured reservoirs", ARMA 11-557, 45th US Rock Mechanics Symposium, 2011,
# section 3.2 and Table 2.
#
# Hand-run maintenance script, like `make_thick_cylinder_figures.jl` next to it:
# the documentation page is static and reads the committed PNG/GIF, so a
# documentation build never runs a coupled reservoir simulation.
#
#     julia --project=scripts/fe scripts/fe/arma2011_welltest.jl
#
# Writes into `docs/src/assets/fe_coupling/arma2011/`:
#
#   welltest_mesh.png          the quarter reservoir and its graded mesh
#   welltest_pressure.gif      3-D pressure field around the well, draw-down + build-up
#   welltest_well_pressure.png well pressure history, M1 vs M2  (paper Fig. 7)
#   welltest_permeability.png  maximum principal permeability at 24 h (paper Fig. 6)

using MeanFieldHomogenization
using TensND
using Ferrite
using Tensors
using LinearAlgebra
using Printf
using Serialization
using Plots
gr()

const EXT = Base.get_extension(
    MeanFieldHomogenization, :MeanFieldHomogenizationFerriteMaterialExt
)
const OUT = normpath(
    joinpath(@__DIR__, "..", "..", "docs", "src", "assets", "fe_coupling", "arma2011")
)
mkpath(OUT)

# ── The microstructure (paper, Table 2) ──────────────────────────────────────
#
# Units: MPa for stresses and pressures, m for lengths, s for time, m² for
# permeability.  The fluid viscosity is then 1e-3 Pa·s = 1e-9 MPa·s, so a
# mobility K/μ comes out in m²/(MPa·s) and a Darcy flux in m/s.

const E_S, NU_S = 3.0e4, 0.3                      # solid: 30 GPa, ν = 0.3
const K_S = E_S / (3 * (1 - 2NU_S))
const MU_S = E_S / (2 * (1 + NU_S))
const C_SOLID = TensISO{3}(3K_S, 2MU_S)

const A_FRAC = 1.0                                 # fracture radius, m
const APERTURE = 1.0e-3                            # initial aperture 2c, m
const OMEGA0 = APERTURE / (2 * A_FRAC)             # aspect ratio ω = c/a
const COND0 = 6.7e-11                              # fracture conductivity, m³
const DENSITY = 0.37                               # Budiansky density, per family
const AZIMUTH = (π / 8, 7π / 8)                    # dip-azimuth 22.5° and 157.5°
const K_MATRIX = 1.0e-18                           # nearly impermeable matrix, m²
const MU_F = 1.0e-9                                # fluid viscosity, MPa·s

"The two vertical fracture families of Table 2, dip 90°, azimuths 22.5°/157.5°."
function welltest_rve(conductivity = COND0)
    props = Dict(:C => C_SOLID, :K => TensISO{3}(K_MATRIX))
    rve = RVE(:M)
    add_matrix!(rve, Ellipsoid(1.0), props)
    for (i, φ) in enumerate(AZIMUTH)
        add_phase!(
            rve, Symbol("F", i),
            ConductiveCrack(A_FRAC; conductivity = conductivity, euler_angles = (π / 2, φ)),
            props; density = DENSITY
        )
    end
    return rve
end

welltest_material(conductivity = COND0) = FracturedPoroelasticRock(
    welltest_rve(conductivity), SelfConsistent();
    ω₀ = ntuple(_ -> OMEGA0, 2), k_matrix = K_MATRIX
)

"Horizontal geometric mean of the effective permeability — what a radial flow sees."
function radial_permeability(conductivity)
    K = to_tensors(transport_property(
        welltest_material(conductivity), initial_state(welltest_material(conductivity))
    ))
    return sqrt(K[1, 1] * K[2, 2])
end

"""
Fracture conductivity giving a prescribed radial permeability.

Past the percolation threshold the estimate is proportional to γ = C/2a, so two
secant steps in the logarithm are enough. This is the knob the paper leaves
open — its § 1.3 allows C⁰ to differ from the Poiseuille value "to account for
tortuosity due to the roughness of fracture lips or to the fracture filling".
"""
function calibrate_conductivity(target; C0 = COND0)
    f(lc) = log(radial_permeability(exp(lc)) / target)
    lc, flc = log(C0), f(log(C0))
    lc2 = lc - flc                                 # slope 1 in log-log
    for _ in 1:6
        f2 = f(lc2)
        abs(f2) < 1.0e-4 && break
        lc, flc, lc2 = lc2, f2, lc2 - f2 * (lc2 - lc) / (f2 - flc)
    end
    return exp(lc2)
end
# ── Geometry and discretization ──────────────────────────────────────────────
#
# A quarter of the reservoir is enough: reflecting about e₁ maps the family at
# 22.5° onto −22.5° ≡ 157.5° (a fracture normal is defined up to a sign) and
# about e₂ maps 22.5° onto 157.5°, so the microstructure — and hence every
# homogenized property — has both symmetry planes.

const R_WELL, R_OUT, HEIGHT = 0.15, 3500.0, 200.0
const NR, NTHETA, NZ = 28, 12, 2
const Q_LIN = 5.0e-4                               # produced flow rate, m³/s per m
const T_DRAW = 24 * 3600.0                         # draw-down duration, s

grid = EXT.cylinder_sector_grid(R_WELL, R_OUT, HEIGHT, NR, NTHETA, NZ)

ipu = Lagrange{RefHexahedron, 1}()^3
ipp = Lagrange{RefHexahedron, 1}()
qr = QuadratureRule{RefHexahedron}(2)
cvu, cvp = CellValues(qr, ipu), CellValues(qr, ipp)
fqr = FacetQuadratureRule{RefHexahedron}(2)
fvp = FacetValues(fqr, ipp)

dh = DofHandler(grid)
add!(dh, :u, ipu)
add!(dh, :p, ipp)
close!(dh)

const URANGE, PRANGE = dof_range(dh, :u), dof_range(dh, :p)

ch = ConstraintHandler(dh)
add!(ch, Dirichlet(:u, getfacetset(grid, "front"), (x, t) -> 0.0, 2))   # θ = 0
add!(ch, Dirichlet(:u, getfacetset(grid, "back"), (x, t) -> 0.0, 1))    # θ = π/2
add!(ch, Dirichlet(:u, getfacetset(grid, "bottom"), (x, t) -> 0.0, 3))  # z = 0
add!(ch, Dirichlet(:u, getfacetset(grid, "right"), (x, t) -> [0.0, 0.0], [1, 2]))
add!(ch, Dirichlet(:p, getfacetset(grid, "right"), (x, t) -> 0.0))      # p = 0 far away
close!(ch)

@printf(
    "mesh: %d hexahedra, %d dofs (u: %d, p: %d)\n",
    getncells(grid), ndofs(dh),
    ndofs(dh) - getnnodes(grid), getnnodes(grid)
)

# Consistent nodal flow of the produced rate, spread over the well wall. The
# outward normal of Ω points into the well, so a production is a positive
# Q·n and enters the mass residual with a plus sign.
function well_flow(q_lin)
    f = zeros(ndofs(dh))
    q_area = q_lin / (2π * R_WELL)                 # m/s, same on a quarter model
    for fc in FacetIterator(dh, getfacetset(grid, "left"))
        reinit!(fvp, fc)
        fe = zeros(getnbasefunctions(fvp))
        for q in 1:getnquadpoints(fvp)
            dΓ = getdetJdV(fvp, q)
            for i in 1:getnbasefunctions(fvp)
                fe[i] += shape_value(fvp, q, i) * q_area * dΓ
            end
        end
        # `fvp` is a scalar facet value: its basis functions are the pressure
        # ones, so they index the p block of the element dof vector.
        dofs = celldofs(fc)
        for (i, a) in enumerate(PRANGE)
            f[dofs[a]] += fe[i]
        end
    end
    return f
end

# ── The coupled solve ────────────────────────────────────────────────────────

"Mobility K/μ at every quadrature point, from the converged states."
function mobilities!(mob, material, states)
    for c in eachindex(states), q in eachindex(states[c])
        K = transport_property(material, states[c][q])
        mob[c][q] = K === nothing ?
            zero(SymmetricTensor{2, 3}) : to_tensors(K) / MU_F
    end
    return mob
end

"""
Backward-Euler coupled (u, p) solve over `times`, producing `q_lin(t)`.

`update_permeability = false` freezes the mobility at its initial value — the
paper's model M1. `true` is M2: the apertures follow the effective stress and
the permeability follows the apertures through the cubic law.
"""
function solve_welltest(
        material, times, q_lin; update_permeability::Bool,
        cache = MaterialCache(), tol = 1.0e-8, maxit = 12
    )
    nqp = getnquadpoints(cvu)
    ncells = getncells(grid)
    states = EXT.mfh_states(material, ncells, nqp)
    states_old = EXT.mfh_states(material, ncells, nqp)
    mob = [[zero(SymmetricTensor{2, 3}) for _ in 1:nqp] for _ in 1:ncells]
    mobilities!(mob, material, states_old)
    K0 = mob[1][1]

    x = zeros(ndofs(dh))
    K = allocate_matrix(dh, ch)
    nbf = ndofs_per_cell(dh)
    Ke, re = zeros(nbf, nbf), zeros(nbf)

    history = NamedTuple[]
    t = 0.0
    for (step, tnew) in enumerate(times)
        Δt = tnew - t
        fext = well_flow(q_lin(tnew)) * Δt
        r0 = 0.0
        for it in 1:maxit
            assembler = start_assemble(K, zeros(ndofs(dh)))
            r = zeros(ndofs(dh))
            for cell in CellIterator(dh)
                reinit!(cvu, cell); reinit!(cvp, cell)
                fill!(Ke, 0); fill!(re, 0)
                eldofs = celldofs(cell)
                xe = x[eldofs]
                EXT.mfh_poro_element!(
                    Ke, re, cvu, cvp, material, xe[URANGE], xe[PRANGE],
                    states[cellid(cell)], states_old[cellid(cell)], Δt,
                    mob[cellid(cell)]; u_range = URANGE, p_range = PRANGE, cache = cache
                )
                assemble!(assembler, eldofs, Ke, zeros(nbf))
                r[eldofs] .+= re
            end
            r .+= fext
            apply_zero!(K, r, ch)
            it == 1 && (r0 = max(norm(r), eps()))
            # The two residuals carry different units, so the criterion is
            # relative to the first residual of the step.
            norm(r) <= tol * r0 && break
            x .-= K \ r
            apply!(x, ch)
            it == maxit && @warn "step $step did not converge" residual = norm(r) / r0
        end
        for c in eachindex(states), q in eachindex(states[c])
            states_old[c][q] = states[c][q]
        end
        update_permeability && mobilities!(mob, material, states_old)
        push!(history, (t = tnew, x = copy(x), mob = deepcopy(mob)))
        t = tnew
    end
    return (history = history, K0 = K0, cache = cache)
end

# ── The two cases ────────────────────────────────────────────────────────────
#
# WHY TWO.  With Table 2 taken literally, the estimate implemented here gives a
# 24-darcy reservoir, hence a draw-down of 33 kPa. The paper's own Fig. 6 reads
# 200–400 (mD) and its Fig. 7 reaches −1.8 MPa, i.e. a permeability some fifty
# times smaller — the difference between this simplified self-consistent
# estimate and the paper's Eq. (12), which carries a matrix concentration factor
# built on the order-2 Hill tensor of the effective medium.
#
# That single number explains the whole gap, because the coupling is LINEAR in
# the draw-down: Δω ∝ ΔΣ' ∝ Δp ∝ 1/K. So the second case calibrates the fracture
# conductivity to the paper's permeability level and lets the same model answer
# at the paper's scale.

# Log-spaced draw-down, then log-spaced build-up.
logsteps(t0, t1, n) = t0 .* (t1 / t0) .^ (range(0, 1; length = n))
const TIMES = vcat(
    logsteps(60.0, T_DRAW, 16),
    T_DRAW .+ logsteps(60.0, 2 * T_DRAW, 14)
)
const N_DRAW = 16
q_schedule(t) = t <= T_DRAW + 1.0e-9 ? Q_LIN : 0.0

const K_TARGET = 3.5e-13                   # m², the level of the paper's Fig. 6
const COND_CAL = calibrate_conductivity(K_TARGET)

for (name, C) in (("Table 2", COND0), ("calibrated", COND_CAL))
    m = welltest_material(C)
    Kh = to_tensors(transport_property(m, initial_state(m)))
    ch_ = homogenize(welltest_rve(C), SelfConsistent())
    pr = poroelastic_parameters(ch_, C_SOLID, 0.0)
    @printf(
        "%-11s C = %.3e m³   K = diag(%.3e, %.3e, %.3e) m²  (radial %.3e)\n",
        name, C, Kh[1, 1], Kh[2, 2], Kh[3, 3], sqrt(Kh[1, 1] * Kh[2, 2])
    )
    name == "Table 2" && @printf(
        "            ℂʰᵒᵐ₃₃₃₃ = %.4g MPa  𝐁 = diag(%.4f, %.4f, %.4f)  1/M = %.4g MPa⁻¹\n",
        to_tensors(ch_)[3, 3, 3, 3], to_tensors(pr.B)[1, 1],
        to_tensors(pr.B)[2, 2], to_tensors(pr.B)[3, 3], pr.inverse_modulus
    )
end

# The four solves cost about forty minutes, almost all of it in the
# self-consistent permeability at every quadrature point. Set
# `MFH_WELLTEST_CACHE` to a path to reuse them while iterating on the figures;
# unset (the default) always solves.
const CACHE_FILE = get(ENV, "MFH_WELLTEST_CACHE", "")

function run_case(conductivity)
    m = welltest_material(conductivity)
    m2 = solve_welltest(m, TIMES, q_schedule; update_permeability = true)
    m1 = solve_welltest(m, TIMES, q_schedule; update_permeability = false)
    return (material = m, M1 = m1, M2 = m2)
end

table2, calibrated = if !isempty(CACHE_FILE) && isfile(CACHE_FILE)
    @info "reusing $CACHE_FILE"
    Serialization.deserialize(CACHE_FILE)
else
    a, b = run_case(COND0), run_case(COND_CAL)
    isempty(CACHE_FILE) || Serialization.serialize(CACHE_FILE, (a, b))
    (a, b)
end
@printf("scheme solves: %d\n", cache_stats(table2.M2.cache).entries)

# ── Post-processing ──────────────────────────────────────────────────────────

coords = [grid.nodes[i].x for i in 1:getnnodes(grid)]
pressure_nodes(x) = evaluate_at_grid_nodes(dh, x, :p)
node_id(i, j, k) = i + (NR + 1) * ((j - 1) + (NTHETA + 1) * (k - 1))

"Pressure at the well wall, mid-height."
function well_pressure(x)
    p = pressure_nodes(x)
    idx = [i for i in eachindex(coords) if norm(coords[i][1:2]) < 1.01R_WELL]
    return sum(p[i] for i in idx) / length(idx)
end

# ── Fig. 7 — well pressure, M1 against M2, for both cases ────────────────────

t_h = [h.t for h in table2.M2.history] ./ 3600
series(case) = ([well_pressure(h.x) for h in case.M1.history],
    [well_pressure(h.x) for h in case.M2.history])
p1a, p2a = series(table2)
p1b, p2b = series(calibrated)
gap(a, b) = 100 * (b[N_DRAW] / a[N_DRAW] - 1)

function history_panel(pw1, pw2, title)
    pl = plot(
        t_h, pw1; label = "M1 — permeability frozen", lw = 2.4, lc = :red,
        xlabel = "t (h)", ylabel = "Δp (MPa)", legend = :bottomright,
        title = title, titlefontsize = 10, bottom_margin = 4Plots.mm
    )
    plot!(pl, t_h, pw2; label = "M2 — permeability follows the apertures",
        lw = 2.4, lc = :blue)
    vline!(pl, [T_DRAW / 3600]; lc = :grey50, ls = :dot, label = "")
    return pl
end

savefig(
    plot(
        history_panel(p1a, p2a, @sprintf(
            "C⁰ = %.1e m³   (K̄ = %.0f D)   M2 %.1f %% deeper",
            COND0, radial_permeability(COND0) / 9.87e-13, gap(p1a, p2a)
        )),
        history_panel(p1b, p2b, @sprintf(
            "C⁰ = %.1e m³   (K̄ = %.0f mD)   M2 %.0f %% deeper",
            COND_CAL, radial_permeability(COND_CAL) / 9.87e-16, gap(p1b, p2b)
        ));
        layout = (1, 2), size = (1020, 430), dpi = 130,
        left_margin = 7Plots.mm, bottom_margin = 6Plots.mm, top_margin = 3Plots.mm
    ),
    joinpath(OUT, "welltest_well_pressure.png")
)
@printf("Table 2   : M1 %.4f MPa, M2 %.4f MPa at 24 h  (%.2f %% deeper)\n",
    p1a[N_DRAW], p2a[N_DRAW], gap(p1a, p2a))
@printf("calibrated: M1 %.4f MPa, M2 %.4f MPa at 24 h  (%.2f %% deeper)\n",
    p1b[N_DRAW], p2b[N_DRAW], gap(p1b, p2b))

# ── A painter's-algorithm renderer for the 3-D mesh ──────────────────────────
#
# GR colors a 3-D surface by its height and by nothing else, so the faces are
# projected, depth-sorted and drawn as filled 2-D shapes here. That is what
# gives the banded isovalues of the paper's Fig. 6, with the mesh drawn on top.

"Orthographic projection onto the screen plane, with a depth key."
function projector(az, el)
    α, β = deg2rad(az), deg2rad(el)
    ex = (-sin(α), cos(α), 0.0)
    ey = (-cos(α) * sin(β), -sin(α) * sin(β), cos(β))
    ed = (cos(α) * cos(β), sin(α) * cos(β), sin(β))
    dot3(a, p) = a[1] * p[1] + a[2] * p[2] + a[3] * p[3]
    return p -> (dot3(ex, p), dot3(ey, p)), p -> dot3(ed, p)
end

"Exterior faces of the displayed sector, as `(node quadruple, value)` pairs."
function sector_faces(F, imax)
    faces = NTuple{4, Int}[]
    push_face!(a, b, c, d) = push!(faces, (a, b, c, d))
    for i in 1:(imax - 1), j in 1:NTHETA          # top and bottom
        for k in (1, NZ + 1)
            push_face!(node_id(i, j, k), node_id(i + 1, j, k),
                node_id(i + 1, j + 1, k), node_id(i, j + 1, k))
        end
    end
    for j in 1:NTHETA, k in 1:NZ                  # well wall and outer cut
        for i in (1, imax)
            push_face!(node_id(i, j, k), node_id(i, j + 1, k),
                node_id(i, j + 1, k + 1), node_id(i, j, k + 1))
        end
    end
    for i in 1:(imax - 1), k in 1:NZ              # the two symmetry planes
        for j in (1, NTHETA + 1)
            push_face!(node_id(i, j, k), node_id(i + 1, j, k),
                node_id(i + 1, j, k + 1), node_id(i, j, k + 1))
        end
    end
    return [(f, sum(F[n] for n in f) / 4) for f in faces]
end

"""
The sector drawn as the paper draws it: filled isovalue bands, mesh on top.

`zscale` exaggerates the 200 m of height against the 3500 m of radius, exactly
as a reservoir post-processor does.
"""
function render_sector(
        F; imax, az = 38, el = 26, zscale = 8.0, clims, nbands = 12,
        title = "", cmap = :turbo, edges = true
    )
    proj, depth = projector(az, el)
    pt(n) = (coords[n][1], coords[n][2], zscale * coords[n][3])
    faces = sector_faces(F, imax)
    order = sortperm([mean(depth(pt(n)) for n in f) for (f, _) in faces])
    lo, hi = clims
    grad = cgrad(cmap, nbands; categorical = true)
    pl = plot(; aspect_ratio = 1, legend = false, axis = false, grid = false,
        ticks = false, title = title)
    for idx in order
        f, v = faces[idx]
        xy = [proj(pt(n)) for n in f]
        band = clamp(1 + floor(Int, nbands * (v - lo) / (hi - lo + eps())), 1, nbands)
        plot!(pl, Shape([q[1] for q in xy], [q[2] for q in xy]);
            fillcolor = grad[band], linewidth = edges ? 0.25 : 0,
            linecolor = edges ? RGBA(0, 0, 0, 0.35) : grad[band])
    end
    return pl
end

mean(x) = sum(x) / length(x)

"Colour-bar companion, since the shapes carry no colour scale of their own."
function bar_plot(clims, nbands; cmap = :turbo, label = "Δp (MPa)")
    lo, hi = clims
    v = range(lo, hi; length = 256)
    return heatmap(
        [0.0], v, reshape(collect(v), :, 1); c = cgrad(cmap, nbands; categorical = true),
        clims = clims, legend = false, xticks = false, ymirror = true,
        ylabel = label, framestyle = :box
    )
end

# ── The animated 3-D pressure field ──────────────────────────────────────────

const ZOOM = 800.0
const IMAX = findlast(i -> norm(coords[node_id(i, 1, 1)][1:2]) <= ZOOM, 1:(NR + 1))

pfields = [pressure_nodes(h.x) for h in calibrated.M2.history]
pclim = (minimum(minimum(f) for f in pfields), 0.0)

# The draw-down is logarithmic in ρ, so most of the *area* of any view sits in
# the outer, nearly undisturbed ring: the sector alone reads as flat however it
# is colored. The radial profile beside it is where the front is legible.
radii_line = [norm(coords[node_id(i, 1, NZ + 1)][1:2]) for i in 1:(NR + 1)]
profile(F) = [F[node_id(i, 1, NZ + 1)] for i in 1:(NR + 1)]

anim = @animate for (n, (h, F)) in enumerate(zip(calibrated.M2.history, pfields))
    phase = h.t <= T_DRAW ? "draw-down" : "build-up"
    sector = render_sector(
        F; imax = IMAX, az = 25 + 2.2n, el = 42, zscale = 1.0, clims = pclim,
        nbands = 12, title = @sprintf("t = %5.1f h   (%s)", h.t / 3600, phase)
    )
    prof = plot(;
        xscale = :log10, xlabel = "ρ (m)", ylabel = "Δp (MPa)", ylims = pclim,
        legend = false, title = "radial profile", titlefontsize = 10
    )
    for m in 1:(n - 1)                        # ghosts of the earlier instants
        plot!(prof, radii_line, profile(pfields[m]); lw = 0.7, lc = :grey80)
    end
    plot!(prof, radii_line, profile(F); lw = 2.6, lc = :steelblue)
    plot(
        sector, prof, bar_plot(pclim, 12);
        layout = Plots.grid(1, 3; widths = [0.50, 0.38, 0.12]),
        size = (1180, 470), dpi = 120,
        left_margin = 6Plots.mm, bottom_margin = 7Plots.mm, right_margin = 9Plots.mm,
        plot_title = "pore pressure — quarter reservoir, ρ ≤ 800 m"
    )
end
gif(anim, joinpath(OUT, "welltest_pressure.gif"); fps = 3)

# ── The mesh ─────────────────────────────────────────────────────────────────

function plan_mesh!(pl; zoom = Inf, lc = :grey45, lw = 0.5)
    for c in grid.cells
        n = collect(c.nodes)
        for (i, j) in ((1, 2), (2, 3), (3, 4), (4, 1))
            a, b = coords[n[i]], coords[n[j]]
            (norm(a[1:2]) > zoom || norm(b[1:2]) > zoom) && continue
            for (sx, sy) in ((1, 1), (-1, 1), (-1, -1), (1, -1))
                plot!(pl, [sx * a[1], sx * b[1]], [sy * a[2], sy * b[2]];
                    lc = lc, lw = lw, label = "")
            end
        end
    end
    return pl
end

pm1 = plot(; aspect_ratio = 1, legend = false, xlabel = "x (m)", ylabel = "y (m)",
    title = "plan view — geometric radial grading")
plan_mesh!(pm1)
pm2 = render_sector(
    zeros(getnnodes(grid)); imax = NR + 1, az = 38, el = 22, zscale = 12.0,
    clims = (-1.0, 1.0), nbands = 12, cmap = :grays,
    title = "the whole sector (R = 3500 m, z × 12)"
)
savefig(
    plot(
        pm1, pm2; layout = (1, 2), size = (1000, 450), dpi = 130,
        left_margin = 7Plots.mm, bottom_margin = 6Plots.mm
    ),
    joinpath(OUT, "welltest_mesh.png")
)

# ── Fig. 6 — maximum principal permeability at the end of the draw-down ──────

function kmax_nodes(entry)
    F = zeros(getnnodes(grid))
    seen = zeros(Int, getnnodes(grid))
    for cell in CellIterator(dh)
        reinit!(cvu, cell)
        k = maximum(
            maximum(eigvals(entry.mob[cellid(cell)][q] * MU_F))
                for q in 1:getnquadpoints(cvu)
        )
        for n in cell.nodes
            F[n] += k
            seen[n] += 1
        end
    end
    return F ./ max.(seen, 1)
end

kend = kmax_nodes(calibrated.M2.history[N_DRAW])
kini = kmax_nodes(calibrated.M2.history[1])
ratio = 100 .* (kend ./ kini .- 1)
kclim = (minimum(ratio), 0.0)
pk = render_sector(
    ratio; imax = IMAX, az = 38, el = 42, zscale = 1.0, clims = kclim,
    nbands = 12, cmap = :turbo, title = "end of the draw-down"
)
savefig(
    plot(pk, bar_plot(kclim, 12; label = "change in kmax (%)");
        layout = Plots.grid(1, 2; widths = [0.88, 0.12]), size = (940, 480), dpi = 130,
        right_margin = 9Plots.mm,
        plot_title = "maximum principal permeability around the well"),
    joinpath(OUT, "welltest_permeability.png")
)
@printf("kmax change at the well face: %.2f %%\n", minimum(ratio))

println("figures written to ", OUT)
