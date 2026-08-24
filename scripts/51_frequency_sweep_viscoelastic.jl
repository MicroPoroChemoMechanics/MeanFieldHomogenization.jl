# =============================================================================
#  51_frequency_sweep_viscoelastic.jl
#
#  Frequency sweep on a 2-phase viscoelastic RVE using the Mori-Tanaka and
#  self-consistent schemes. Each phase has a Maxwell-model relaxation
#  spectrum: G(ω) = G_∞ + G_d * iωτ / (1 + iωτ).
#
#  The script demonstrates that every scheme is `Complex{Float64}`-safe and
#  produces the expected viscoelastic effective-modulus curves.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using Printf
using Plots

# Maxwell-model complex shear modulus: G*(ω) = G_∞ + G_d · iωτ / (1 + iωτ)
function maxwell_G(ω; G_inf = 10.0, G_d = 5.0, τ = 1.0)
    iωτ = im * ω * τ
    return G_inf + G_d * iωτ / (1 + iωτ)
end

# Build the iso 4th-order stiffness from K and G
iso_C(K, G) = TensISO{3}(3K, 2G)

const f_inc = 0.3
ωs = exp10.(range(-2, 2; length = 60))
y_mt = Vector{ComplexF64}(undef, length(ωs))
y_sc = Vector{ComplexF64}(undef, length(ωs))

# Same Maxwell spectrum on both phases but different baseline moduli
for (i, ω) in enumerate(ωs)
    G_m = maxwell_G(ω; G_inf = 10.0, G_d = 5.0, τ = 1.0)
    G_i = maxwell_G(ω; G_inf = 30.0, G_d = 15.0, τ = 1.0)
    K_m = ComplexF64(30.0)
    K_i = ComplexF64(80.0)
    rve = RVE(:M)                                # nothing to declare: the moduli
    add_matrix!(rve, Ellipsoid(1.0), Dict(:C => iso_C(K_m, G_m)))   # carry the
    add_phase!(                                  # complex part, the fraction stays real
        rve, :I, Ellipsoid(1.0),
        Dict(:C => iso_C(K_i, G_i)); fraction = f_inc
    )
    y_mt[i] = get_array(homogenize(rve, MoriTanaka()))[1, 2, 1, 2]    # shear-like component
    y_sc[i] = get_array(homogenize(rve, SelfConsistent(; abstol = 1.0e-10, maxiters = 200)))[1, 2, 1, 2]
end

p1 = plot(
    ωs, real.(y_mt); xscale = :log10, label = "Re — MT", lw = 2, color = :blue,
    xlabel = "ω", ylabel = "C[1212]",
    title = "Frequency sweep — viscoelastic 2-phase, f=$(f_inc)"
)
plot!(p1, ωs, imag.(y_mt); label = "Im — MT", lw = 2, color = :blue, ls = :dash)
plot!(p1, ωs, real.(y_sc); label = "Re — SC", lw = 2, color = :red)
plot!(p1, ωs, imag.(y_sc); label = "Im — SC", lw = 2, color = :red, ls = :dash)

figdir = joinpath(@__DIR__, "figures")
isdir(figdir) || mkdir(figdir)
figpath = joinpath(figdir, "frequency_sweep_viscoelastic.png")
savefig(p1, figpath)
display(p1)
println("Saved : ", figpath)

@printf("\nω        Re(MT)    Im(MT)    Re(SC)    Im(SC)\n")
for (i, ω) in enumerate(ωs)
    if i % 10 == 1
        @printf(
            "  %.2e   %.4f   %.4f   %.4f   %.4f\n",
            ω, real(y_mt[i]), imag(y_mt[i]), real(y_sc[i]), imag(y_sc[i])
        )
    end
end
