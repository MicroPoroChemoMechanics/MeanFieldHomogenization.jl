# # [Drained and undrained poroelasticity](@id ml-ex-biot)
#
# A saturated porous sample is compressed in an oedometer twice: once with its
# fluid free to leave (**drained**), once with the fluid trapped (**undrained**).
# The same law answers both; only the loading path changes. The law couples two
# primal fields, the strain ``\boldsymbol\varepsilon`` and the pore pressure ``p``,
# with their fluxes, the stress ``\boldsymbol\sigma`` and the change of fluid
# content ``\phi`` (see [`BiotPoroelasticity`](@ref)):
#
# ```math
# \boldsymbol\sigma = \mathbb{C}:\boldsymbol\varepsilon - b\,p\,\boldsymbol{1},\qquad
# \phi = b\operatorname{tr}\boldsymbol\varepsilon + \frac{p}{M}.
# ```
#
# ## Two tests, one law
#
using MeanFieldHomogenization

E, ν, b, M = 10e3, 0.25, 0.8, 5e3        # MPa; Biot coefficient and modulus
law = BiotPoroelasticity(; E, ν, b, M)

oedometer = (ε11 = t -> -1e-3 * t, ε22 = 0, ε33 = 0)          # lateral strains held at zero
drained   = point_response(law, LoadingPath(0:0.25:1; oedometer..., p = 0))   # pore pressure held
undrained = point_response(law, LoadingPath(0:0.25:1; oedometer...))        # φ unnamed: zero
[drained[:σ11][end] undrained[:σ11][end]; drained[:p][end] undrained[:p][end]]
#
# In the drained test the pore pressure is prescribed (zero) and the fluid
# content is free; in the undrained test the fluid content is not named, so it is
# held at zero, and the pore pressure is what the law makes it.
#
# ## Against the closed forms
#
# With the oedometric modulus ``M_{oed} = k + 4\mu/3`` and the strain
# ``\varepsilon = -10^{-3}``:
#
# - drained: ``\sigma_{11} = M_{oed}\,\varepsilon``, ``\phi = b\,\varepsilon``;
# - undrained: ``\phi = 0`` gives ``p = -M b\,\varepsilon``, hence
#   ``\sigma_{11} = (M_{oed} + b^2 M)\,\varepsilon``: the trapped fluid stiffens the
#   sample by ``b^2 M``.
#
k, μ = E / (3(1 - 2ν)), E / (2(1 + ν))
Moed, ε = k + 4μ / 3, -1e-3
(drained[:σ11][end] - Moed * ε, drained[:φ][end] - b * ε,
 undrained[:p][end] + M * b * ε, undrained[:σ11][end] - (Moed + b^2 * M) * ε)
#
# ## The same tests on the fractured rock of ARMA 2011
#
# The fractured-reservoir material of [barthelemyARMA2011](@citet), built in
# [Building a fractured-rock material](@ref fe-fractured-rock), is a material of
# the same contract with the same two fields. Its stiffness and its Biot
# parameters come from a homogenization of the rock and its fractures, and its
# fractures close under compression. The driver runs it along the same paths:
#
using TensND
C0 = TensISO{3}(3 * 30.0e3, 2 * 18.0e3)                 # intact rock, MPa
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => C0, :K => TensISO{3}(1e-18)); fraction = :rest)
add_phase!(rve, :F, PennyCrack(1.0), Dict(:C => C0, :K => TensISO{3}(1e-18)); density = 0.1)
rock = FracturedPoroelasticRock(rve, MoriTanaka(); ω₀ = (1e-3,), k_matrix = 1e-18)

oedometer_z = (ε33 = t -> -1e-5 * t, ε11 = 0, ε22 = 0)
rock_drained   = point_response(rock, LoadingPath(0:0.25:1; oedometer_z..., p = 0))
rock_undrained = point_response(rock, LoadingPath(0:0.25:1; oedometer_z...))
rock_drained[:σ33][end], rock_undrained[:σ33][end], rock_undrained[:p][end]
#
# The undrained sample is again stiffer, and the trapped fluid takes a part of
# the load as pore pressure. Here the Biot coefficient and modulus are not
# parameters but results of the homogenization, recomputed as the fractures open
# or close.
