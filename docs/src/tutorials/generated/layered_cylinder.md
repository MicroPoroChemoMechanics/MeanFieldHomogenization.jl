```@meta
EditURL = "../../../../scripts/72_layered_cylinder.jl"
```

# [Coated fibers: the layered cylinder at work](@id tut-layered-cylinder)

!!! tip "The whole page as one script"
    Every block of this page, in order, is [`scripts/72_layered_cylinder.jl`](https://github.com/MicroPoroChemoMechanics/MeanFieldHomogenization.jl/blob/main/scripts/72_layered_cylinder.jl).
    From a clone of the repository, `julia --project=docs -e 'using Pkg; Pkg.instantiate()'`
    once, then `julia scripts/72_layered_cylinder.jl`, runs it end to end.

!!! info "Before this page"
    The theory page [Layered cylinder](@ref th-layered-cylinder), which
    derives the solution used here, and the tutorial
    [n-layer sphere: volume-averaged concentration tensors](@ref tut-layered-sphere),
    its spherical counterpart.

A unidirectional composite of coated fibers, from one fiber to the effective
medium and its creep. The fiber is transversely isotropic about its axis
``\underline{e}_3``, as a carbon fiber is; a soft isotropic sizing surrounds it,
bonded by a spring interface; the matrix is isotropic. All the moduli are in
GPa and the lengths in µm.

| | ``k`` | ``l`` | ``m`` | ``n`` | ``p`` |
|:--|--:|--:|--:|--:|--:|
| fiber (TI) | 12 | 6 | 5 | 230 | 20 |
| sizing (``E = 2``, ``\nu = 0.4``) | 3.571 | 2.857 | 0.714 | 4.286 | 0.714 |
| matrix (``E = 3.5``, ``\nu = 0.35``) | 4.321 | 3.025 | 1.296 | 5.617 | 1.296 |

``k`` is the plane-strain bulk modulus, ``m`` and ``p`` the transverse and axial
shear moduli, ``n = C_{3333}`` and ``l = C_{1133}`` ([theory, section 1](@ref th-layered-cylinder)).

The fibers are along ``\underline{e}_3``, the default. Fibers of another
direction are given by the `axis` keyword of `LayeredCylinder`, their
transversely isotropic layers being then built about that same axis
([Layered inclusions](@ref man-layered)); every result below is turned with
them and is otherwise unchanged.

````@example layered_cylinder
using MeanFieldHomogenization
using TensND
using LinearAlgebra
using Printf
using QuadGK: gauss
using Plots
gr()
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))
default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

const ez = (0.0, 0.0, 1.0)
hill(k, l, m, n, p) = ti_from_constants(k + m, k - m, l, n, p, ez)
isotropic(E, ν) = iso_stiffness(E / (3 * (1 - 2ν)), E / (2 * (1 + ν)))
const C_fiber = hill(12.0, 6.0, 5.0, 230.0, 20.0)
const C_sizing = isotropic(2.0, 0.4)
const C_matrix = isotropic(3.5, 0.35)
nothing #hide
````

## 1. One coated fiber

A fiber of radius ``a = 3.5`` µm, a sizing of thickness ``e = 0.5`` µm, and a
spring interface between them, of normal and tangential stiffnesses
``k_n = k_t = 20`` GPa/µm.

````@example layered_cylinder
const a, e = 3.5, 0.5
const bond = SpringInterface(20.0, 20.0)
fiber = LayeredCylinder((a, a + e), (C_fiber, C_sizing); interfaces = (bond, PerfectInterface()))
````

Drag to rotate: the fiber and its sizing, cut short to show the layers.

````@example layered_cylinder
function fiber_scene()
    Rz = [0.0 1.0 0.0; 0.0 0.0 1.0; 1.0 0.0 0.0]       # the local axis of `cylinder_surface` along e₃
    items = Any[
        surface_trace(cylinder_surface(a, a, 24.0; R = Rz, center = (0.0, 0.0, 2.0))...; color = "#4063D8", name = "fiber"),
        surface_trace(cylinder_surface(a + e, a + e, 18.0; R = Rz, center = (0.0, 0.0, -1.0))...; color = "#E07B39", opacity = 0.55, name = "sizing"),
        surface_trace(disc_surface(a + e, a + e; center = (0.0, 0.0, -10.0))...; color = "#E07B39", opacity = 0.55),
        surface_trace(disc_surface(a, a; center = (0.0, 0.0, 14.0))...; color = "#4063D8"),
    ]
    append!(items, arrow_trace((0.0, 0.0, 14.0), (0.0, 0.0, 6.0); size = 1.5, label = "e₃"))
    return plotly_scene(items; uid = "coated-fiber-3d", height = 460, title = "Carbon fiber (a = 3.5 µm) and its sizing (e = 0.5 µm)", camera = (1.6, 1.4, 0.6))
end
fiber_scene()
````

## 2. Inside the fiber

Under a remote transverse shear ``\boldsymbol E = \varepsilon\,(\underline{e}_1\otimes\underline{e}_1 -
\underline{e}_2\otimes\underline{e}_2)``, ``\varepsilon = 10^{-3}``, in the matrix, the stress
``\sigma_{11}`` on a section. [`LayeredCylinderFields`](@ref) solves the problem
once; the fields are then evaluated point by point.

````@example layered_cylinder
fields = LayeredCylinderFields(fiber, C_matrix)
E = Tens(1.0e-3 * [1.0 0.0 0.0; 0.0 -1.0 0.0; 0.0 0.0 0.0])
xs = range(-2.2 * (a + e), 2.2 * (a + e); length = 161)
σ₁₁ = [canonical_components(local_flux(fields, (x, y, 0.0), E))[1, 1] for y in xs, x in xs]
circle(r) = (r .* cos.(range(0, 2π; length = 181)), r .* sin.(range(0, 2π; length = 181)))
heatmap(
    xs, xs, 1.0e3 .* σ₁₁; aspect_ratio = 1, color = :balance, clims = (-5, 5), xlims = extrema(xs), ylims = extrema(xs),
    xlabel = "x₁ (µm)", ylabel = "x₂ (µm)", colorbar_title = "σ₁₁ (MPa)",
    title = "σ₁₁ under transverse shear ε = 10⁻³, kₙ = kₜ = 20 GPa/µm", titlefontsize = 10,
)
plot!(circle(a)...; color = :black, lw = 1, label = "")
plot!(circle(a + e)...; color = :black, lw = 1, ls = :dash, label = "")
````

The stress is uniform in the fiber, where the core modes are those of a
homogeneous strain; it varies in the sizing and the matrix. Along the radius
``\theta = 0`` the radial traction is continuous across the spring while the
displacement jumps by ``\sigma_{rr}/k_n``:

````@example layered_cylinder
for side in (:inner, :outer)
    u = local_displacement(fields, a, 0.0, 0.0, E; side)
    σ = canonical_components(local_flux(fields, a, 0.0, 0.0, E; side))
    @printf("%-6s u₁ = %.5f µm   σ₁₁ = %.3f MPa\n", side, u[1], 1.0e3 * σ[1, 1])
end
````

The pointwise field and the layer averages come from the same amplitudes, and
they agree: the strain averaged over the sizing, ``a < r < a + e``, is the one
[`gradient_gradient_concentration`](@ref) gives for its layer. The integral over
the annulus is computed here by quadrature. In ``\theta`` the strain varies as
``\cos 2\theta`` and ``\sin 2\theta``, which 32 equally spaced angles integrate
exactly; in ``r``, 12 Gauss–Legendre points.

````@example layered_cylinder
strain(r, θ) = canonical_components(local_gradient(fields, r, θ, 0.0, E; layer = 2))
rs, ws = gauss(12, a, a + e)
θs = 2π .* (0:31) ./ 32
ε_quadrature = sum(ws[i] * rs[i] * strain(rs[i], θ) for i in eachindex(rs), θ in θs) *
    (2π / 32) / (π * ((a + e)^2 - a^2))
ε_layer = canonical_components(gradient_gradient_concentration(fiber, C_matrix; layer = 2) ⊡ E)
maximum(abs, ε_quadrature - ε_layer) / maximum(abs, ε_layer)
````

## 3. The aligned composite

Fibers at the volume fraction ``\phi`` of carbon in the composite: the coated
fiber occupies ``\phi\,(1 + e/a)^2``. Three estimates: Mori–Tanaka (the
matrix is the reference), the self-consistent scheme (every phase a cylinder,
the matrix a one-layer one, so that the reference stays transversely
isotropic) and the differential scheme. Each is compared with the bare fiber
at the same carbon content.

````@example layered_cylinder
function composite(ϕ, cyl; scheme)
    f = ϕ * (outer_radius(cyl) / a)^2
    rve = RVE()
    add_phase!(rve, :matrix, LayeredCylinder((1.0,), (C_matrix,)), Dict(:C => C_matrix); fraction = :rest)
    add_phase!(rve, :fiber, cyl, Dict(:C => C_fiber); fraction = f)
    return ti_constants(best_fit_ti(homogenize(rve, scheme, :C), ez); form = :engineering, role = :stiffness)
end

contents = vcat(0.0, 0.005, 0.01, 0.02, range(0.04, 0.45; length = 9))
bare = LayeredCylinder((a,), (C_fiber,))
p_ET = plot(; xlabel = "carbon fraction ϕ", ylabel = "E_T (GPa)", legend = :topleft,
    title = "Transverse modulus, a = 3.5 µm, e = 0.5 µm, kₙ = kₜ = 20 GPa/µm", titlefontsize = 10)
for (scheme, name, ls) in ((MoriTanaka(), "Mori-Tanaka", :solid), (SelfConsistent(), "self-consistent", :dash), (DifferentialScheme(), "differential", :dot))
    plot!(p_ET, contents, [composite(ϕ, fiber; scheme)[1] for ϕ in contents]; label = name * ", coated", color = 1, ls)
    plot!(p_ET, contents, [composite(ϕ, bare; scheme)[1] for ϕ in contents]; label = name * ", bare", color = 2, ls)
end
p_ET
````

The first fibers raise ``E_T`` steeply: stiff along the axis, they forbid the
axial Poisson contraction of a transverse tension, and the matrix tends to its
plane-strain modulus ``E/(1 - \nu^2) = 3.99`` GPa. Beyond, the soft sizing and
its spring lower the transverse modulus; the axial one, carried by the fibers,
is the same either way:

````@example layered_cylinder
for cyl in (bare, fiber)
    E₁, E₃, ν₁₂, ν₃₁, G₃₁ = composite(0.45, cyl; scheme = MoriTanaka())
    @printf("%-6s  E_T = %5.2f GPa   E_L = %6.1f GPa   G_LT = %5.2f GPa\n", cyl === bare ? "bare" : "coated", E₁, E₃, G₃₁)
end
````

## 4. The fiber as the limit of a needle

A homogeneous prolate ellipsoid of aspect ratio ``\omega`` tends to the
one-layer cylinder, and so does a confocal layered spheroid of growing focal
distance ``c`` to the layered cylinder with perfect interfaces. An imperfect
interface carries a length: the spheroid then tends to the average of the
cylinders of its sections, not to the cylinder of its largest radius.

````@example layered_cylinder
rel(A, B) = norm(canonical_components(A) - canonical_components(B)) / norm(canonical_components(B))
A_cyl = gradient_gradient_concentration(bare, C_fiber, C_matrix)
ωs = 10 .^ range(1, 4; length = 13)
err_ellipsoid = [rel(gradient_gradient_concentration(Ellipsoid(a, a, ω * a), C_fiber, C_matrix), A_cyl) for ω in ωs]
layersK = (TensISO{3}(10.0), TensISO{3}(1.0))
K₀ = TensISO{3}(0.5)
cyl_K = LayeredCylinder((0.6, 1.0), layersK)
A_K = gradient_gradient_concentration(cyl_K, K₀, K₀)
cs = 10 .^ range(0.5, 2.5; length = 9)
err_spheroid = map(cs) do c
    sp = LayeredSpheroid(map(b -> sqrt(b^2 + c^2), (0.6, 1.0)), (0.6, 1.0), layersK; Nseries = 10)
    rel(gradient_gradient_concentration(sp, K₀, K₀), A_K)
end
plot(ωs, err_ellipsoid; xscale = :log10, yscale = :log10, marker = :circle, label = "ellipsoid → cylinder (elasticity), ω = a₃/a",
    xlabel = "ω, or c/r_N", ylabel = "relative difference", legend = :bottomleft,
    title = "Elongated inclusions tend to the cylinder", titlefontsize = 10)
plot!(cs, err_spheroid; marker = :square, label = "confocal 2-layer spheroid → 2-layer cylinder (conduction)")
````

## 5. Creep of the composite

The matrix creeps and ages: its relaxation kernel is
``\mathbb C_m\,\exp\!\big(-(t - t')/(\tau\,g(t'))\big)\,g(t')`` with
``g(t') = 1 + t'/(1 + t')``, ``\tau = 2`` days; the fiber and the sizing are
elastic. The axial and transverse creep functions under a unit stress applied
at ``t = 0``, Mori–Tanaka estimate, ``f = 0.6``:

````@example layered_cylinder
g(tp) = 1 + tp / (1 + tp)
law_matrix = ViscoLaw((t, tp) -> t < tp ? zero(C_matrix) : g(tp) * exp(-(t - tp) / (2.0 * g(tp))) * C_matrix, :relaxation)
elastic(C) = ViscoLaw((t, tp) -> t < tp ? zero(C) : C, :relaxation)
fiber_alv = LayeredCylinder((a, a + e), (elastic(C_fiber), elastic(C_sizing)); interfaces = (bond, PerfectInterface()))
times = vcat(0.0, 10 .^ range(-2, log10(30.0); length = 24))
rve = RVE()
add_phase!(rve, :matrix, Ellipsoid(1.0), Dict(:C => law_matrix); fraction = :rest)
add_phase!(rve, :fiber, fiber_alv, Dict(:C => elastic(C_fiber)); fraction = 0.6)
R = Matrix(homogenize_alv(rve, MoriTanaka(), :C; times))
J = volterra_inverse(R; block_size = 6)
creep(i) = [sum(J[6(s - 1) + i, 6(r - 1) + i] for r in 1:s) for s in eachindex(times)]
plot(times[2:end], 1.0e3 .* creep(1)[2:end]; xscale = :log10, yscale = :log10, label = "transverse, J₁₁", xlabel = "t (days)",
    ylabel = "creep function (1/TPa)", legend = :topleft, title = "Coated-fiber composite, f = 0.6, Mori-Tanaka", titlefontsize = 10)
plot!(times[2:end], 1.0e3 .* creep(3)[2:end]; label = "axial, J₃₃")
````

The axial creep is held by the elastic fibers; the transverse one follows the
matrix. The aging cylinder is the elastic one evaluated on Volterra matrices
([theory, section 9](@ref th-layered-cylinder)).

---

*This page was generated using [Literate.jl](https://github.com/fredrikekre/Literate.jl).*

