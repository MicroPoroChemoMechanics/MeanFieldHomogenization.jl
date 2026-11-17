# # [Lattices of spheroids by the multipole expansion](@id app-mem-spheroids)
#
# !!! info "Before this page"
#     [The multipole expansion method for spheroids](@ref th-mem-spheroids),
#     where the scheme is derived, and
#     [Particle assemblies and N-body schemes](@ref man-assemblies) for the
#     assemblies it acts on.
#
# [kushch2015](@citet) give the conductivity of oblate spheroids on three
# lattices, with a perfect interface, a low-conducting (Kapitza) one and a
# highly conducting one. This page computes the same quantities with
# [`MultipoleExpansion`](@ref) and sets them beside the published values, read
# from the file below.

import Pkg                                                          #jl
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)                 #jl

using MeanFieldHomogenization
using TensND
using LinearAlgebra
using JSON
using Markdown
using Printf
using Plots
gr()

## Interactive 3D helpers (Plotly through require.js).
include(joinpath(pkgdir(MeanFieldHomogenization), "scripts", "common", "docviz.jl"))

default(; left_margin = 5Plots.mm, bottom_margin = 5Plots.mm)

paper = JSON.parsefile(joinpath(pkgdir(MeanFieldHomogenization), "data", "literature", "kushch2015.json"))
quantity(name) = paper["quantities"][name]["value"]
nothing #hide

#md Main.show_literature("kushch2015") # hide

# ## 1. The lattices
#
# The spheroids are oblate, of semi-axes ``(l_1, l_1, e\,l_1)`` with
# ``e = 1/3``, aligned with ``\underline{e}_3``. The lattices are proportional:
# the cell has the periods ``(1, 1, e)``, and holds one spheroid (SP), two
# (BCP, the second at the center of the cell) or four (FCP, the others at the
# centers of three faces). The fraction ``c`` sets ``l_1``. The interface is
# carried by a one-layer [`LayeredSpheroid`](@ref): a
# [`KapitzaInterface`](@ref) of resistance ``\rho`` for the low-conducting
# interface, of parameter ``\tilde h_c = l_1/(\rho\,k_0)``, and a
# [`SurfaceConductiveInterface`](@ref) of conductance ``k^{\mathrm s}`` for the
# highly conducting one, of parameter ``\tilde h_s = k^{\mathrm s}/(l_1\,k_0)``.

e = quantity("aspect_ratio")
k₀ = TensISO{3}(1.0)
sites = Dict(
    "sp" => [(0.0, 0.0, 0.0)],
    "bcp" => [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5)],
    "fcp" => [(0.0, 0.0, 0.0), (0.5, 0.0, 0.5), (0.5, 0.5, 0.0), (0.0, 0.5, 0.5)],
)
semi_axis(kind, c) = cbrt(3c / (4π * length(sites[kind])))

function lattice(kind, c; k₁, interface = l₁ -> PerfectInterface{Float64}())
    l₁ = semi_axis(kind, c)
    asm = ParticleAssembly(; boundary = PeriodicBox((1.0, 1.0, e)))
    add_matrix!(asm, Dict(:K => k₀))
    for (i, x) in enumerate(sites[kind])
        spheroid = LayeredSpheroid((e * l₁,), (l₁,), (TensISO{3}(k₁),); interfaces = (interface(l₁),))
        add_particle!(asm, Symbol(:p, i), (x[1], x[2], e * x[3]), spheroid, Dict(:K => TensISO{3}(k₁)))
    end
    return asm
end

kapitza(h̃c) = l₁ -> KapitzaInterface(l₁ / h̃c)
surface(h̃s) = l₁ -> SurfaceConductiveInterface(h̃s * l₁)
conductivity(asm, nmax) = (K = canonical_components(homogenize(asm, MultipoleExpansion(; nmax), :K)); (K[1, 1], K[3, 3]))
nothing #hide

# The body-centered cell at ``c = 0.4``, with the spheroids of the neighboring
# cells that overlap it:

function cell_scene(asm; uid)
    traces = String[]
    for name in particle_names(asm), shift in Iterators.product(-1:1, -1:1, -1:1)
        X₀ = particle_center(asm, name) .+ shift .* (1.0, 1.0, e)
        all(-0.6 .≤ X₀ ./ (1.0, 1.0, e) .≤ 0.6) || continue
        a, b = MeanFieldHomogenization.LayeredSpheroids.outer_semiaxes(particle_geometry(asm, name))
        X, Y, Z = ellipsoid_surface(b, b, a; center = X₀, nu = 25, nv = 13)
        push!(traces, surface_trace(X, Y, Z; color = "#4063D8"))
    end
    return plotly_scene(traces; uid, height = 480, title = "oblate spheroids of aspect ratio 1/3 on the BCP lattice, c = 0.4")
end

cell_scene(lattice("bcp", 0.4; k₁ = 1.0); uid = "mem-bcp-spheroids")

# ## 2. Convergence on the simple lattice
#
# At ``c = 0.5`` the spheroids of the simple lattice nearly touch, by a gap of
# ``0.005`` along their axis. The table gives ``(k_{11}, k_{33})/k_0`` against
# the truncation degree, computed and, in parentheses, published, for the
# conductivity ratio ``k_1/k_0 = 1000``. Only the odd degrees enter: every
# spheroid being a center of symmetry of the lattice, the field it receives
# under a uniform gradient is odd about it.

k₁ = quantity("contrast_lc")
perfect = l₁ -> PerfectInterface{Float64}()
columns = (("perfect", "hc_infinity", perfect), ("``\\tilde h_c = 10``", "hc_10", kapitza(10.0)), ("``\\tilde h_c = 1``", "hc_1", kapitza(1.0)))
function convergence_table(rows)
    io = IOBuffer()
    println(io, "| ``n_{\\max}`` | ", join(first.(columns), " | "), " |")
    println(io, "| :-- | ", join(fill(":--", length(columns)), " | "), " |")
    for row in rows
        cells = map(columns) do (_, key, interface)
            k11, k33 = conductivity(lattice("sp", 0.5; k₁, interface), row["tmax"])
            @sprintf("%.3f, %.3f (%.3f, %.3f)", k11, k33, row["k11_$key"], row["k33_$key"])
        end
        println(io, "| ", row["tmax"], " | ", join(cells, " | "), " |")
    end
    return Markdown.parse(String(take!(io)))
end
convergence_table(r for r in paper["tables"]["convergence_sp"] if r["tmax"] ≤ 7)

# The published values agree with the computed ones to the last printed digit
# at every degree: the truncation is the same, and so is the solution. The
# perfect interface converges slowly, as the article notes; a resistance at
# the interface smooths the field and the degree 5 suffices.

# ## 3. Three lattices, three interfaces
#
# Each cell gives ``(k_{11}, k_{33})/k_0`` computed at `nmax = 7` and, in
# parentheses, published.

function lattices_table(table, k₁, interface; cs = (0.1, 0.2), nmax = 7)
    io = IOBuffer()
    println(io, "| ``c`` | SP | BCP | FCP |")
    println(io, "| :-- | :-- | :-- | :-- |")
    for row in paper["tables"][table]
        row["c"] in cs || continue
        cells = map(("sp", "bcp", "fcp")) do kind
            k11, k33 = conductivity(lattice(kind, row["c"]; k₁, interface), nmax)
            @sprintf("%.3f, %.3f (%.3f, %.3f)", k11, k33, row["k11_$kind"], row["k33_$kind"])
        end
        println(io, "| ", row["c"], " | ", join(cells, " | "), " |")
    end
    return Markdown.parse(String(take!(io)))
end
nothing #hide

# **Perfect interface**, ``k_1/k_0 = 1000``:

lattices_table("lattices_perfect", k₁, perfect)

# **Low-conducting interface**, ``\tilde h_c = 10``, ``k_1/k_0 = 1000``:

lattices_table("lattices_lc_10", k₁, kapitza(10.0); cs = (0.2,), nmax = 5)

# **Highly conducting interface**, ``\tilde h_s = 10``, ``k_1/k_0 = 0.1``:

lattices_table("lattices_hc_10", quantity("contrast_hc"), surface(10.0); cs = (0.2,), nmax = 5)

# The arrangement matters: with the perfect interface, the FCP lattice
# conducts a quarter more than the simple one in the plane of the spheroids at
# ``c = 0.2``, and less along their axis. Denser lattices need higher degrees:
# on the simple lattice at ``c = 0.3``, ``k_{33}`` reaches its published value
# 2.016 at `nmax = 13` (2.016 computed), against 1.971 at `nmax = 7`. The random
# arrangements of the article, averaged over a hundred cells, are not
# reproduced here.

# ## 4. The interface parameter
#
# The conductivities of the simple lattice at ``c = 0.5`` against the
# parameter ``\tilde h_c`` of the low-conducting interface, beside the
# [Mori–Tanaka](@ref th-scheme-mori-tanaka) estimate built on the same layered
# spheroid. The latter is the Maxwell estimate of the article, Table 1, which it
# recovers:

function maxwell_table()
    io = IOBuffer()
    println(io, "| ``\\tilde h_c`` | ``(k_{11}, k_{33})/k_0`` at ``c = 0.5`` | published |")
    println(io, "| :-- | :-- | :-- |")
    row = only(r for r in paper["tables"]["maxwell_lc"] if r["c"] == 0.5)
    for (label, key, interface) in (("1", "hc_1", kapitza(1.0)), ("10", "hc_10", kapitza(10.0)), ("∞", "hc_infinity", perfect))
        K = canonical_components(homogenize(lattice("sp", 0.5; k₁, interface), MoriTanaka(), :K))
        println(io, @sprintf("| %s | %.3f, %.3f | %.3f, %.3f |", label, K[1, 1], K[3, 3], row["k11_$key"], row["k33_$key"]))
    end
    return Markdown.parse(String(take!(io)))
end
maxwell_table()

# and the curves:

h̃ = 10 .^ range(-1, 2; length = 7)
mem = [conductivity(lattice("sp", 0.5; k₁, interface = kapitza(h)), 5) for h in h̃]
mt = map(h̃) do h
    K = canonical_components(homogenize(lattice("sp", 0.5; k₁, interface = kapitza(h)), MoriTanaka(), :K))
    (K[1, 1], K[3, 3])
end
p_interface = plot(;
    xscale = :log10, xlabel = "h̃c", ylabel = "k / k₀", legend = :topleft, size = (640, 400),
    title = "SP lattice, c = 0.5, k₁/k₀ = 1000",
)
plot!(p_interface, h̃, first.(mem); marker = :circle, label = "k₁₁, multipole expansion")
plot!(p_interface, h̃, last.(mem); marker = :square, label = "k₃₃, multipole expansion")
plot!(p_interface, h̃, first.(mt); linestyle = :dash, label = "k₁₁, Mori–Tanaka")
plot!(p_interface, h̃, last.(mt); linestyle = :dash, label = "k₃₃, Mori–Tanaka")
p_interface

# The single-particle estimate misses the interactions over the whole range:
# it overestimates ``k_{11}`` and underestimates ``k_{33}``, the two curves of
# ``k_{33}`` crossing only near ``\tilde h_c = 3``, and the gap grows with
# ``\tilde h_c``.

# ## Summary
#
# | result | source | recovered |
# | :-- | :-- | :-- |
# | convergence with the degree, simple lattice, ``c = 0.5`` | [kushch2015](@cite), Table 3 | to the last digit at every degree up to 7 |
# | Maxwell estimate, low-conducting interface | [kushch2015](@cite), Table 1 | Mori–Tanaka on the layered spheroid, to the third decimal |
# | three lattices, perfect interface | [kushch2015](@cite), Table 4 | to the third decimal at `nmax = 7` up to ``c = 0.2``; ``c = 0.3`` on the simple lattice at `nmax = 13` |
# | low- and highly conducting interfaces | [kushch2015](@cite), Tables 6 and 8 | to the third decimal at `nmax = 5`, ``c = 0.2`` |
#
#md # ```@raw html
#md # <details class="source-file"><summary>Show the code of <code>scripts/common/docviz.jl</code>, included above</summary>
#md # ```
#md #
#md # ```@eval
#md # Main.show_source("scripts/common/docviz.jl")
#md # ```
#md #
#md # ```@raw html
#md # </details>
#md # ```

const figdir = joinpath(@__DIR__, "figures")                        #jl
isdir(figdir) || mkdir(figdir)                                       #jl
savefig(p_interface, joinpath(figdir, "99_multipole_interface.png"))  #jl
@printf "\nSaved : %s\n" joinpath(figdir, "99_multipole_interface.png") #jl
