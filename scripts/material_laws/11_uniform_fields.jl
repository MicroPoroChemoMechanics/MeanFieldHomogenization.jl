# # [Uniform fields: Voigt–Taylor and Reuss–Sachs](@id ml-theory-uniform-fields)
#
# The simplest composites of nonlinear phases assume one field to be uniform:
# every phase undergoes the macroscopic strain, or every phase carries the
# macroscopic stress. They ignore the shapes and the arrangement of the phases.
# For linear elastic phases they are the bounds of [voigt1889](@citet) and
# [reuss1929](@citet), between which every estimate of the effective stiffness
# lies [hill1952](@cite). For plastic phases they are the models of Taylor and
# of Sachs, the extremes of the [polycrystal models](@ref ml-theory-polycrystals).
# This page states them for any law of `MaterialLaws`; the
# [secant methods](@ref ml-theory-secant) and the polycrystal interactions sit in
# between. [The example below](@ref ml-ex-uniform-fields) runs them.
#
# ## Uniform strain
#
# Every phase ``r``, of volume fraction ``f_r``, undergoes the macroscopic strain
# and integrates its own law over the step:
#
# ```math
# \boldsymbol{\varepsilon}_r = \boldsymbol{E},
# \qquad
# \boldsymbol{\Sigma} = \sum_r f_r\,\boldsymbol{\sigma}_r\big(\boldsymbol{E}, \mathsf{y}_r\big),
# ```
#
# ``\mathsf{y}_r`` being the internal variables of the phase. The phases interact
# only through the strain they share: each integrates independently, and the
# consistent tangent of the composite is the average of theirs. A uniform strain
# is a kinematically admissible field: for phases deriving from convex
# potentials, it gives an upper bound of the effective potential. For linear
# elastic phases this is the Voigt stiffness ``\sum_r f_r\,\mathbb{C}_r``.
#
# ## Uniform stress
#
# Every phase carries the macroscopic stress, and the strains of the phases,
# unknown, average to the macroscopic one:
#
# ```math
# \boldsymbol{\sigma}_r\big(\boldsymbol{\varepsilon}_r, \mathsf{y}_r\big) = \boldsymbol{\Sigma}
# \quad\text{for every } r,
# \qquad
# \sum_r f_r\,\boldsymbol{\varepsilon}_r = \boldsymbol{E}.
# ```
#
# Under a prescribed macroscopic strain, the unknowns of a step are the
# macroscopic stress, the strain of every phase and their internal variables.
# They are solved together, the equilibrium of every phase with the macroscopic
# stress and the mixture rule of the strains being added to the equations of the
# phase laws. A uniform stress is statically admissible: for convex potentials it
# gives a lower bound. For linear elastic phases this is the Reuss stiffness
# ``\big(\sum_r f_r\,\mathbb{S}_r\big)^{-1}``.
#
# ## As laws of `MaterialLaws`
#
# [`UniformStrain`](@ref) and [`UniformStress`](@ref) build laws whose internal
# variables gather those of the phases, renamed after them (`M_εᵖ`, …), and for
# the uniform stress also the macroscopic stress (`Σ_σ`) and the strains of the
# phases (`M_ε`, …). A phase whose step is elastic keeps its internal variables,
# whatever the others do. The consistent tangent of the composite comes, as for
# any law, from the implicit-function theorem applied to all these equations:
# for the uniform stress it is the inverse of the average of the tangent
# compliances of the phases, which no formula of the code writes. Every phase law
# of the package fits, with internal variables (plasticity, viscosity, damage) or
# with several fields, provided the phases share their fields.
#
# ## Numerical notes
#
# | | |
# |:--|:--|
# | start | the Newton iteration of a uniform-stress step starts with every phase taking the macroscopic increment of the strain, a uniform-strain predictor |
# | strongly nonlinear phases | inverting a phase law (a power law of small exponent) is ill-conditioned: the driver cuts the steps that fail |
# | no finite tangent | a phase without a finite tangent at the current strain (a power law at zero strain) makes the tangent of the composite undefined there (`NaN`), not the update |
#
# ## [Example: two elastoplastic phases](@id ml-ex-uniform-fields)
#
# This example builds the two [uniform-field composites](@ref ml-theory-uniform-fields)
# of a steel and an aluminum alloy, and checks them against the elastic bounds
# and against the closed form of linear hardening.
#
# ### The phases
#
using MeanFieldHomogenization, Plots

steel = J2Plasticity(; E = 200e3, ν = 0.3, σ₀ = 250.0, H = 10e3)     # MPa
alloy = J2Plasticity(; E = 70e3, ν = 0.33, σ₀ = 150.0, H = 2e3)
laws, fractions = (S = steel, A = alloy), (S = 0.3, A = 0.7)
taylor = UniformStrain(laws, fractions)
sachs = UniformStress(laws, fractions)
nothing #hide
#
# | code | theory |
# |:--|:--|
# | `UniformStrain(laws, fractions)` | every phase at the macroscopic strain (Voigt, Taylor) |
# | `UniformStress(laws, fractions)` | every phase at the macroscopic stress (Reuss, Sachs) |
# | `laws`, `fractions` | the law and the volume fraction of each phase, by name |
#
# ### A tensile test
#
path = LoadingPath(range(0, 1; length = 41); ε11 = t -> 6e-3 * t)
rT, rS = point_response(taylor, path), point_response(sachs, path)
p = plot(rT[:ε11] * 1e3, rT[:σ11]; lw = 2, label = "uniform strain (Taylor)",
    xlabel = "axial strain (‰)", ylabel = "axial stress (MPa)", legend = :bottomright)
plot!(p, rS[:ε11] * 1e3, rS[:σ11]; lw = 2, label = "uniform stress (Sachs)")
#
# The uniform strain is the stiffer: the strong steel is strained as much as the
# alloy and carries a large share of the stress. Under a uniform stress the alloy
# yields first and takes most of the strain, which the composite shows as an
# earlier and lower plastic branch. The internal variables of each phase are read
# by name:
#
(alloy_p_taylor = rT[:A_p][end], alloy_p_sachs = rS[:A_p][end], alloy_strain_sachs = rS[:A_ε11][end])
#
# ### Checks
#
# **Linear hardening in uniaxial stress.** Under a uniform stress every phase is
# in uniaxial stress ``\sigma``, and its axial strain is
# ``\sigma/E + \langle\sigma - \sigma_0\rangle/H``. The macroscopic strain is
# their average:
#
axial(σ, E, σ₀, H) = σ / E + max(σ - σ₀, 0) / H
expected = [0.3 * axial(σ, 200e3, 250.0, 10e3) + 0.7 * axial(σ, 70e3, 150.0, 2e3) for σ in rS[:σ11]]
maximum(abs, expected - rS[:ε11])
#
# **The elastic bounds.** In the elastic range the tangents of the two composites
# are the Voigt and Reuss stiffnesses of the package:
#
using TensND, StaticArrays
rve = RVE()
add_phase!(rve, :S, Ellipsoid(1.0), Dict(:C => iso_stiffness_young_poisson(200e3, 0.3)); fraction = 0.3)
add_phase!(rve, :A, Ellipsoid(1.0), Dict(:C => iso_stiffness_young_poisson(70e3, 0.33)); fraction = :rest)
E = Mandel{Float64}(1e-4, 0, 0, 0, 0, 0)
(point_update(taylor, E, initial_state(taylor), 1.0)[2] ≈ mandel(homogenize(rve, Voigt(), :C)),
 point_update(sachs, E, initial_state(sachs), 1.0)[2] ≈ mandel(homogenize(rve, Reuss(), :C)))
