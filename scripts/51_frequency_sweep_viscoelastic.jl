# =============================================================================
#  51_frequency_sweep_viscoelastic.jl
#
#  Frequency sweep on a 2-phase viscoelastic RVE using the Mori-Tanaka and
#  self-consistent schemes.  Each phase is a standard solid in each channel,
#  built with `iso_rheology(zener_maxwell(...), zener_maxwell(...))` from the
#  rheological model library.
#
#  The script demonstrates that every scheme is `Complex{Float64}`-safe and
#  produces the expected viscoelastic effective-modulus curves.
#
#  For the same composite taken all the way back to the time domain — and the
#  three-route cross-check that goes with it — see `61_freq_vs_time.jl`.
# =============================================================================

import Pkg
Pkg.activate(joinpath(@__DIR__, "..", "docs"); io = devnull)

using MeanFieldHomogenization
using TensND
using Printf
using Plots

# One model per phase: an elastic bulk channel and a standard-solid shear
# channel.  `carson_relaxation(m, im * ω)` is then the complex stiffness the
# schemes consume — the same object would also give `ViscoLaw(m)` for the
# time-domain route.
const PHASE_M = iso_rheology(Spring(30.0), zener_maxwell(10.0, 5.0, 1.0))
const PHASE_I = iso_rheology(Spring(80.0), zener_maxwell(30.0, 15.0, 1.0))

const f_inc = 0.3
ωs = exp10.(range(-2, 2; length = 60))
y_mt = Vector{ComplexF64}(undef, length(ωs))
y_sc = Vector{ComplexF64}(undef, length(ωs))

# Same Maxwell spectrum on both phases but different baseline moduli
for (i, ω) in enumerate(ωs)
    p = im * ω
    rve = RVE()                                # nothing to declare: the moduli
    add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => carson_relaxation(PHASE_M, p)); fraction = :rest)
    add_phase!(                                  # carry the complex part, the
        rve, :I, Ellipsoid(1.0),                 # fraction stays real
        Dict(:C => carson_relaxation(PHASE_I, p)); fraction = f_inc
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
