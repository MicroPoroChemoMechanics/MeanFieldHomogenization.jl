# # [The quasi-elastic affine formulation](@id ml-theory-quasi-elastic)
#
# The [affine formulation for elasto-viscoplastic composites](@ref ml-theory-affine-evp)
# solves, at every step, a thermoelastic composite at each collocation point of
# a numerical inversion of the Laplace–Carson transform. [brenner2002](@citet)
# replace that inversion by a direct one, a single value of the transform
# standing for the function at a given time. The comparison composite then
# becomes an elastic one with an eigenstrain, in time: a "quasi-elastic"
# composite, solved once per step. This page states it for a matrix with
# inclusions and the Mori–Tanaka scheme. [The example below](@ref ml-ex-quasi-elastic)
# compares it with the formulation it approximates.
#
# ## The phases
#
# Each phase is a nonlinear viscoelastic solid of Maxwell type,
# ``\dot{\boldsymbol{\varepsilon}} = \mathbb{s}:\dot{\boldsymbol{\sigma}} + \boldsymbol{g}(\boldsymbol{\sigma})``,
# with an elastic compliance ``\mathbb{s}`` and a viscous strain rate that
# depends on the stress alone, such as [`NortonViscoplasticity`](@ref).
#
# ## The linearization and the direct inversion
#
# At the time ``\zeta``, the viscous rate is linearized about the current stress
# of the phase, ``\mathbb{m} = \partial\boldsymbol{g}/\partial\boldsymbol{\sigma}``, and the rest of
# it is an eigenstrain rate that follows the actual history:
#
# ```math
# \boldsymbol{\varepsilon}(t) = \big[(\mathbb{s} + \mathbb{m}\,t) * \boldsymbol{\sigma}\big](t) + \boldsymbol{\varepsilon}^0(t),
# \qquad
# \boldsymbol{\varepsilon}^0(t) = \int_0^t \big(\boldsymbol{g}(\boldsymbol{\sigma}) - \mathbb{m}:\boldsymbol{\sigma}\big)\,\mathrm{d}\theta,
# ```
#
# ``*`` the Stieltjes convolution. In the Laplace–Carson domain the creep
# function ``\mathbb{s} + \mathbb{m}\,t`` becomes ``\mathbb{s} + \mathbb{m}/p``. The
# direct inversion takes a transform at ``p = h(t)/t`` for the function at
# ``t``, which turns the convolution into a product: at ``t = \zeta`` the phase is
# an elastic solid of compliance ``\mathbb{s} + (\zeta/h(\zeta))\,\mathbb{m}`` with the
# eigenstrain ``\boldsymbol{\varepsilon}^0(\zeta)``. The paper shows on linear Maxwell
# phases that ``h = 1`` makes the inversion exact for the creep of a single
# phase, and exact for a composite at the start and in the steady regime; and
# that ``h(t) = t\,e^{-t/\bar\tau}/(\bar\tau\,(1 - e^{-t/\bar\tau}))`` makes it exact for
# the relaxation, and for a constant strain rate, of a single phase of
# relaxation time ``\bar\tau``, an average relaxation time of the phases standing
# for it in a composite.
#
# ## The composite
#
# The two quasi-elastic phases form a thermoelastic composite that the
# Mori–Tanaka scheme homogenizes as in the
# [affine formulation](@ref ml-theory-affine-evp), the Eshelby tensor in the
# isotropic part of the modulus of the matrix. The average strain of the
# inclusions is a fixed point: the stresses of the phases set their
# linearizations, which set the concentration. The eigenstrain needs only two
# integrals of the history of each phase, its inelastic strain and the integral
# of its stress, taken by the backward Euler scheme over the steps.
#
# ## Scope
#
# | | |
# |:--|:--|
# | phases | a matrix and inclusions of Maxwell type without internal variable: [`NortonViscoplasticity`](@ref) |
# | inversion | ``h = 1`` (creep), or ``h(t)`` built on a `relaxation_time` (relaxation, constant strain rate) |
# | checks | a single phase is exact in creep and, with ``h(t)``, in relaxation; linear Maxwell phases in creep give the elastic and the viscous Mori–Tanaka estimates at the start and in the steady regime |
# | cost | one thermoelastic composite per iteration of a step, against one per collocation point for [`AffineEVPComposite`](@ref) |
#
# ## [Example: creep of a two-phase composite](@id ml-ex-quasi-elastic)
#
# A matrix with 20 % of stiffer, more viscous spheres, both of Norton type of
# exponent 3, crept under a constant uniaxial stress:
#
using MeanFieldHomogenization, TensND, Plots

f = 0.2
laws = (M = NortonViscoplasticity(; E = 70e3, ν = 0.3, K = 1000.0, n = 3),
        P = NortonViscoplasticity(; E = 400e3, ν = 0.25, K = 3000.0, n = 3))
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest)
add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = f)
creep = LoadingPath(vcat(0.0, exp10.(range(-3, 1; length = 41))); σ11 = 100.0)
nothing #hide
#
# | code | theory |
# |:--|:--|
# | `QuasiElasticComposite(rve, laws)` | the quasi-elastic affine formulation, ``h = 1`` |
# | `AffineEVPComposite(rve, laws; relaxation_times)` | the affine formulation with the collocation inversion |
#
p = plot(; xlabel = "time (s)", ylabel = "axial strain (%)", xscale = :log10, legend = :topleft)
for (c, label) in ((QuasiElasticComposite(rve, laws), "quasi-elastic"),
                   (AffineEVPComposite(rve, laws; relaxation_times = exp10.(range(-3, 3; length = 20))), "affine, collocation"))
    r = point_response(c, creep)
    plot!(p, r.t[2:end], r[:ε11][2:end] * 100; lw = 2, label)
end
p
#
# Both start from the elastic response of the Mori–Tanaka estimate and creep
# at nearly the same rate; the quasi-elastic formulation solves one composite
# per iteration instead of one per collocation point.
