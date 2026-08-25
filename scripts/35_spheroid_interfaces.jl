# # Imperfect interfaces: what they do to the local fields
#
# An imperfect interface is a surface of zero thickness that nonetheless changes
# the physics across it. Two models are available on a [`LayeredSpheroid`](@ref):
# ```math
# \text{LC (Kapitza):}\quad [\![T]\!] = \rho\,q_n,
# \qquad
# \text{HC (surface-conductive):}\quad [\![q_n]\!] = -\beta\,\mathrm{div}_S(\nabla_S T),
# ```
# with ``\rho`` a genuine thermal resistance and ``\beta`` a genuine surface
# conductance. The LC interface impedes the normal flux; the HC one adds a
# tangential short-circuit. Both come from [kushch2015](@cite); the confocal
# ``N``-layer solution used to resolve them is
# [barthelemyBignonnetIJES2020](@cite), and the surface-conductive model goes
# back to [miloh1999](@cite).
#
# Effective properties tell you *how much* an interface matters — that is the
# subject of the two companion pages. This page shows *what it actually does*,
# by reconstructing the temperature and flux **pointwise**:
#
# 1. a two-layer **prolate** particle with a Kapitza (LC) interface at its core,
#    where the fields are read off directly;
# 2. an insulating **oblate** particle with a highly-conducting (HC) skin, swept
#    over ``\beta`` — the configuration of the ECHOES presentation of
#    06/07/2020 — including an animation and an interactive 3-D view.
#
# Theory: [Layered spheroid](../../theory/layered_spheroid.md).

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using MeanFieldHomogenization.LayeredSpheroids: local_temperature, local_gradient, local_flux
using TensND
using Printf
using Plots
gr()

## Interactive-3-D helpers (Plotly through require.js); see the file header for
## why the include goes through `pkgdir` rather than `@__DIR__`.
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))

default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

# ## Shared machinery: tracing a streamline
#
# The flux is shown as **integrated streamlines**, never as arrows on a grid. A
# quiver plot of a channelled field is unreadable: normalizing every arrow to a
# common length throws away the magnitude — precisely the story near an
# interface — while an arrow long enough to be visible is longer than the grid
# spacing, so the arrows overlap and hide the geometry they should reveal.
# Streamlines carry the geometry in their *spacing* and the magnitude in their
# *color*.
#
# The tracer below is written once and reused by both geometries. The geometry
# enters only through `coord`, the map from a meridian point `(x, z)` to the
# spheroid's own confocal coordinates `(q, p)`.

_unit(u, v) = (n = hypot(u, v); n < 1.0e-12 ? (0.0, 0.0) : (u / n, v / n))

function _flux_xz(s, coord, k₀, x, z)
    q, p = coord(max(abs(x), 1.0e-9), z)
    u1, _, u3 = local_flux(s, k₀, q, p, 0.0; H_axial = 1.0, H_trans = 0.0)
    return real(u1) * (x < 0 ? -1.0 : 1.0), real(u3)
end

# Two parameters deserve a word.
#
# `qmin` stops a line where the bulk flux dies out — inside an insulating core.
# An HC interface carries a *surface* current, so the bulk flux inside the
# particle is numerically tiny but not exactly zero; integrating the normalized
# field through it would draw straight lines across the particle that represent
# nothing physical.
#
# `dir` is the integration sense: `+1` follows the flux downstream, `-1` walks
# it back upstream. **Both are needed.** A line released upstream terminates when
# it reaches the particle (that is `qmin` doing its job), so seeding downstream
# only would leave the entire wake blank — the field there is perfectly regular,
# it simply is not reachable by going forward.

function _streamline(
        s, coord, k₀, x0, z0;
        dir = +1, ds = 0.05, nmax = 1400, xmax = 5.0, zmax = 4.5, qmin = 0.0
    )
    f(x, z) = _flux_xz(s, coord, k₀, x, z)
    step(x, z) = (u = f(x, z); _unit(dir * u[1], dir * u[2]))
    xs = Float64[x0]; zs = Float64[z0]; sp = Float64[hypot(f(x0, z0)...)]
    x, z = x0, z0
    for _ in 1:nmax
        hypot(f(x, z)...) < qmin && break
        k1 = step(x, z)
        k1 == (0.0, 0.0) && break
        k2 = step(x + 0.5ds * k1[1], z + 0.5ds * k1[2])
        k3 = step(x + 0.5ds * k2[1], z + 0.5ds * k2[2])
        k4 = step(x + ds * k3[1], z + ds * k3[2])
        x += ds * (k1[1] + 2k2[1] + 2k3[1] + k4[1]) / 6
        z += ds * (k1[2] + 2k2[2] + 2k3[2] + k4[2]) / 6
        (abs(x) ≤ xmax && abs(z) ≤ zmax) || break
        push!(xs, x); push!(zs, z); push!(sp, hypot(f(x, z)...))
    end
    return xs, zs, sp
end

# Seeding both edges is what makes the picture complete: downstream from the
# inflow edge, upstream from the outflow one.

function _draw_streamlines!(plt, s, coord, k₀, seeds, zedge; qmin = 0.0, clims = nothing, ds = 0.05)
    for x0 in seeds, (z0, dir) in ((zedge, +1), (-zedge, -1))
        sx, sz, sp = _streamline(s, coord, k₀, x0, z0; dir, qmin, ds)
        length(sx) > 3 || continue
        if clims === nothing
            plot!(plt, sx, sz; line_z = sp, color = :thermal, lw = 1.1)
        else
            plot!(plt, sx, sz; line_z = sp, color = :thermal, lw = 1.1, clims)
        end
    end
    return plt
end

# Meridian trace of a spheroid: `x²/ρ_t² + z²/ρ_a² = 1`.

function _outline!(plt, ρ_a, ρ_t; color = :black, lw = 1.5)
    tt = range(0, 2π; length = 300)
    return plot!(plt, ρ_t .* cos.(tt), ρ_a .* sin.(tt); color, lw, label = false)
end

# ## Part 1 — Local fields in a two-layer prolate particle
#
# Core and shell of contrasted conductivity, with a **Kapitza** interface at the
# core boundary and a perfect one at the shell/matrix boundary. The
# reference field maps were cross-checked against a finite-element solution.
#
# `(q, p, φ)` are the spheroid's own confocal coordinates (revolution axis
# ``\underline{e}_3``). A meridian point `(x, z)` maps to them through the
# two-foci distances, `q = (r₁+r₂)/(2c)` and `p = (r₂-r₁)/(2c)`, with `r₁`, `r₂`
# the distances to the foci `(0, 0, ±c)`. The sign convention on `p` is fixed by
# `z = c·q·p`, checked on the axis: `z > c ⟹ r₁ = z-c, r₂ = z+c ⟹ p = 1`.

const PRO_KCORE = TensISO{3}(2.0)
const PRO_KSHELL = TensISO{3}(15.0)
const PRO_KM = TensISO{3}(1.0)
const PRO_ρ = 0.15                      # Kapitza resistance at the core interface

const PRO_A_OUT, PRO_B_OUT = 3.0, 1.0
const PRO_FOCAL2 = PRO_A_OUT^2 - PRO_B_OUT^2
const PRO_A_IN = 2.9                    # > √(a_out²-b_out²) for a real confocal shell
const PRO_B_IN = sqrt(PRO_A_IN^2 - PRO_FOCAL2)
const PRO_C = sqrt(PRO_FOCAL2)

s_pro = LayeredSpheroid(
    (PRO_A_IN, PRO_A_OUT), (PRO_B_IN, PRO_B_OUT), (PRO_KCORE, PRO_KSHELL);
    interfaces = (MeanFieldHomogenization.KapitzaInterface(PRO_ρ), MeanFieldHomogenization.PerfectInterface()),
    Nseries = 8,
)

function _coord_prolate(x, z)
    r1 = sqrt(x^2 + (z - PRO_C)^2)
    r2 = sqrt(x^2 + (z + PRO_C)^2)
    return (r1 + r2) / (2PRO_C), clamp((r2 - r1) / (2PRO_C), -1.0, 1.0)
end

# Temperature on a meridian grid. The field is axisymmetric, so the `x < 0` half
# is the mirror image of the computed one.

const PRO_XS = range(1.0e-3, 4.5; length = 60)
const PRO_ZS = range(-4.5, 4.5; length = 90)

T_pro = Matrix{Float64}(undef, length(PRO_ZS), length(PRO_XS))
for (j, z) in enumerate(PRO_ZS), (i, x) in enumerate(PRO_XS)
    q, p = _coord_prolate(x, z)
    T_pro[j, i] = local_temperature(s_pro, PRO_KM, q, p, 0.0; H_axial = 1.0, H_trans = 0.0)
end

@printf "Two-layer prolate particle (a_out=%.1f, b_out=%.1f)\n" PRO_A_OUT PRO_B_OUT
println("─"^70)
@printf "core (a=%.2f, b=%.2f) with Kapitza ρ=%.2f  |  shell (a=%.2f, b=%.2f), perfect\n" PRO_A_IN PRO_B_IN PRO_ρ PRO_A_OUT PRO_B_OUT
@printf "T at the center: %.4f   T far away (x=0, z=10): %.4f\n" local_temperature(
    s_pro, PRO_KM, 1.0 + 1.0e-6, 1.0, 0.0; H_axial = 1.0
) local_temperature(s_pro, PRO_KM, _coord_prolate(1.0e-6, 10.0)..., 0.0; H_axial = 1.0)
println()

p_T = heatmap(
    vcat(-reverse(PRO_XS), PRO_XS), PRO_ZS, hcat(reverse(T_pro; dims = 2), T_pro);
    xlabel = "x", ylabel = "z", title = "Temperature (axial remote gradient)",
    color = :thermal, aspect_ratio = :equal, colorbar_title = "T",
)
_outline!(p_T, PRO_A_IN, PRO_B_IN; color = :white)
_outline!(p_T, PRO_A_OUT, PRO_B_OUT; color = :white)

p_q = plot(;
    xlabel = "x", ylabel = "z", title = "Heat flux streamlines (color = ‖q‖)",
    aspect_ratio = :equal, legend = false, xlims = (-4.6, 4.6), ylims = (-4.5, 4.5),
    colorbar = true, colorbar_title = "‖q‖",
)
_draw_streamlines!(p_q, s_pro, _coord_prolate, PRO_KM, vcat(-4.0:0.4:-0.2, 0.2:0.4:4.0), 4.4; ds = 0.06)
_outline!(p_q, PRO_A_IN, PRO_B_IN)
_outline!(p_q, PRO_A_OUT, PRO_B_OUT)

p_local = plot(p_T, p_q; layout = (1, 2), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (1250, 600))
p_local

# The shell conducts fifteen times better than the matrix, so the flux is drawn
# **into** it — the bright band inside the shell. The Kapitza resistance at the
# core boundary then blocks part of what would otherwise enter the core.

# ## Part 2 — An insulating particle with a highly-conducting skin
#
# Now the opposite situation, and the one that makes the point: an **insulating**
# oblate particle wrapped in an HC interface of surface conductance ``\beta``.
# With no skin the particle is a pure obstacle. As ``\beta`` grows, its skin
# becomes a preferential path and the particle stops behaving like a hole.
#
# Oblate means the revolution semi-axis is the *shorter* one. The two semi-axes
# share a focal **ring** of radius ``\bar c = \sqrt{\rho_t^2 - \rho_a^2}``, and
# the confocal parameter is ``q = \mathrm{i}\,\tau`` with ``\tau = \rho_a/\bar c``
# real — the complex substitution that lets every prolate formula carry over.

const OBL_A = 1.0                       # ρ_a, revolution semi-axis (short)
const OBL_T = 2.0                       # ρ_t, transverse semi-axis (long)
const OBL_CBAR = sqrt(OBL_T^2 - OBL_A^2)
const OBL_KM = 1.0
const OBL_KCORE = 1.0e-6                # insulating core (numerically safe zero)
const OBL_K0 = TensISO{3}(OBL_KM)
const OBL_SEEDS = vcat(-3.6:0.45:-0.2, 0.2:0.45:3.6)
const OBL_ZEDGE = 3.3

# Inverting `x = c̄√(1+τ²)√(1-p²)`, `z = c̄ τ p` gives a quadratic in `u = τ²`:
# ``c̄²u² + u(c̄² - x² - z²) - z² = 0``.

function _coord_oblate(x, z)
    d = x^2 + z^2 - OBL_CBAR^2
    u = (d + sqrt(d^2 + 4 * OBL_CBAR^2 * z^2)) / (2 * OBL_CBAR^2)
    τ = sqrt(max(u, 0.0))
    p = τ > 1.0e-12 ? clamp(z / (OBL_CBAR * τ), -1.0, 1.0) : 0.0
    return im * τ, p
end

_oblate_particle(β) = LayeredSpheroid(
    (OBL_A,), (OBL_T,), (TensISO{3}(OBL_KCORE),);
    interfaces = (MeanFieldHomogenization.SurfaceConductiveInterface(β),),
    Nseries = 5,
)

function _streamplot(β; title = "")
    s = _oblate_particle(β)
    plt = plot(;
        aspect_ratio = :equal, xlims = (-4.2, 4.2), ylims = (-3.4, 3.4),
        xlabel = "x", ylabel = "z", title = title, legend = false,
        colorbar = true, colorbar_title = "‖q‖",
    )
    _draw_streamlines!(
        plt, s, _coord_oblate, OBL_K0, OBL_SEEDS, OBL_ZEDGE;
        qmin = 0.02, clims = (0.0, 2.2)
    )
    ## The interface is drawn with a width proportional to β — the ECHOES
    ## convention, and a reminder that the "layer" has zero thickness but a
    ## finite conductance.
    _outline!(plt, OBL_A, OBL_T; lw = 1 + 2.5β)
    return plt
end

# ### Before and after
#
# At `β = 0` the particle is a pure obstacle: the flux parts around it and
# **piles up at the equator**, the bright spots where ``\|\underline{q}\|`` is
# largest. At `β = 3` that concentration is gone and the streamlines run almost
# undisturbed — the skin accepts the flux tangentially, carries it around, and
# returns it downstream, so the far field barely registers the particle.
#
# Note what the lines do *not* show: they trace the **bulk** flux, and inside an
# insulating core there is none. The current carried by an HC interface is a
# genuine *surface* current on a layer of zero thickness — it cannot appear as a
# streamline. Hence the ``\beta``-proportional line width, and hence the need for
# the quantitative statement below.

p_off = _streamplot(0.0; title = "β = 0  (perfect interface, insulating particle)")
p_on = _streamplot(3.0; title = "β = 3  (highly conducting interface)")
p_pair = plot(p_off, p_on; layout = (1, 2), left_margin = 8Plots.mm, bottom_margin = 8Plots.mm, size = (1250, 520))
p_pair

# ### How much does the skin carry?
#
# The scalar summary: the equivalent particle conductivity
# ``\boldsymbol{k}^{eq} = \boldsymbol{B}_\Omega\cdot\boldsymbol{A}_\Omega^{-1}``,
# where ``\boldsymbol{A}_\Omega`` and ``\boldsymbol{B}_\Omega`` are the
# volume-averaged gradient and flux concentration tensors of the particle,
# defined by ``\langle\nabla T\rangle_\Omega =
# \boldsymbol{A}_\Omega\cdot\underline{H}`` and
# ``\langle\boldsymbol{K}\cdot\nabla T\rangle_\Omega =
# \boldsymbol{B}_\Omega\cdot\underline{H}``.

βs = range(0.0, 3.0; length = 25)
keq_a = Float64[]; keq_t = Float64[]
for β in βs
    s = _oblate_particle(β)
    A = gradient_gradient_loc(s, TensISO{3}(OBL_KCORE), OBL_K0)
    B = flux_gradient_loc(s, TensISO{3}(OBL_KCORE), OBL_K0)
    push!(keq_t, real(B[1, 1] / A[1, 1]))
    push!(keq_a, real(B[3, 3] / A[3, 3]))
end

println("HC interface on an insulating oblate particle (ρ_a=$OBL_A, ρ_t=$OBL_T)")
println("─"^70)
@printf "β = 0.0 :  k_t^eq/k_m = %6.3f   k_a^eq/k_m = %6.3f\n" keq_t[1]/OBL_KM keq_a[1]/OBL_KM
@printf "β = 3.0 :  k_t^eq/k_m = %6.3f   k_a^eq/k_m = %6.3f\n" keq_t[end]/OBL_KM keq_a[end]/OBL_KM
println()

p_keq = plot(
    βs, keq_t ./ OBL_KM; label = "transverse  k_t^eq / k_m", lw = 2.5, color = :steelblue,
    xlabel = "surface conductance β", ylabel = "k^eq / k_m",
    title = "Equivalent particle conductivity vs. interface conductance",
    legend = :topleft, size = (760, 460),
)
plot!(p_keq, βs, keq_a ./ OBL_KM; label = "axial  k_a^eq / k_m", lw = 2.5, color = :darkorange)
hline!(p_keq, [1.0]; color = :grey, ls = :dash, label = "matrix (k_m)")
p_keq

# An **insulating** particle whose equivalent conductivity crosses that of the
# matrix: past ``\beta \approx 0.6`` transversely, the skin more than compensates
# for the hole it wraps.

# ### The sweep, animated

anim = @animate for β in range(0.0, 3.0; length = 10)
    _streamplot(β; title = @sprintf("HC interface,  β = %.2f", β))
end
## No output path on purpose: `gif(anim; fps)` writes to a temporary file that
## Documenter and the notebook embed directly. A path built from `@__DIR__`
## would break both, because in the generated notebook `@__DIR__` has no
## `figures/` subdirectory and ffmpeg fails with a bare exit 254. The standalone
## copy is written in the `#jl` block at the end.
gif(anim; fps = 2)

# ### Interactive 3-D view
#
# The meridian streamlines revolved around the axis: drag to rotate, scroll to
# zoom. The gray surface is the particle, the colored curves are flux lines.

function _plotly_streamlines(β; nrev = 8, uid = "spheroid-3d")
    s = _oblate_particle(β)
    traces = String[]
    for x0 in (-3.2, -2.2, -1.3, -0.6, 0.6, 1.3, 2.2, 3.2),
            krev in 0:(nrev - 1),
            (z0, dir) in ((3.0, +1), (-3.0, -1))

        sx, sz, sp = _streamline(
            s, _coord_oblate, OBL_K0, x0, z0;
            dir, ds = 0.12, nmax = 300, qmin = 0.02
        )
        length(sx) > 4 || continue
        θ = 2π * krev / nrev
        X = [x * cos(θ) for x in sx]
        Y = [x * sin(θ) for x in sx]
        push!(
            traces, """{type:"scatter3d",mode:"lines",
            x:[$(join(round.(X; digits=3), ","))],
            y:[$(join(round.(Y; digits=3), ","))],
            z:[$(join(round.(sz; digits=3), ","))],
            line:{width:3,color:[$(join(round.(sp; digits=3), ","))],
                  colorscale:"Hot",cmin:0,cmax:2.2},showlegend:false}"""
        )
    end
    ## Particle surface, coarse parametric mesh.
    Xs, Ys, Zs = ellipsoid_surface(OBL_T, OBL_T, OBL_A; nu = 24, nv = 16)
    push!(traces, surface_trace(Xs, Ys, Zs; color = "#888888", opacity = 0.35))
    return plotly_scene(
        traces; uid = uid, height = 560,
        title = "Flux lines around an insulating spheroid, β = $β"
    )
end

_plotly_streamlines(3.0; uid = "spheroid-hc-3d")

const figdir = joinpath(@__DIR__, "figures")                        #jl
isdir(figdir) || mkdir(figdir)                                       #jl
savefig(p_local, joinpath(figdir, "35_spheroid_local_fields.png"))    #jl
savefig(p_pair, joinpath(figdir, "35_interface_effect_pair.png"))     #jl
savefig(p_keq, joinpath(figdir, "35_interface_effect_keq.png"))       #jl
gif(anim, joinpath(figdir, "35_interface_effect.gif"); fps = 2, show_msg = false)  #jl
display(p_local)                                                      #jl
@printf "\nSaved : %s\n" joinpath(figdir, "35_*.png")                 #jl
