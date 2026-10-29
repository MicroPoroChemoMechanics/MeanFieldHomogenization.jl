# # [The affine formulation for elasto-viscoplastic composites](@id ml-theory-affine-evp)
#
# The [affine formulation](@ref ml-theory-affine) linearizes phases whose stress
# is a function of the strain alone. [pierardDoghri2006](@citet) extend it to
# phases with a history: elastic, viscoplastic, and hardening with their
# accumulated strain. Linearized in time, such a phase is a linear viscoelastic
# solid with an eigenstrain, which the Laplace–Carson transform turns into a
# thermoelastic one, so that a linear scheme can concentrate the strain. This
# page states the formulation for a matrix with inclusions and the Mori–Tanaka
# scheme, as in the paper. [The example below](@ref ml-ex-affine-evp) runs it.
#
# ## The phases
#
# Each phase ``r`` (``r = 0`` the matrix, ``r = 1`` the inclusions, of volume
# fractions ``f_r``) is elasto-viscoplastic in rate form,
#
# ```math
# \dot{\boldsymbol{\varepsilon}} = \mathbb{S}_r:\dot{\boldsymbol{\sigma}} + \dot{\boldsymbol{\varepsilon}}^{in}(\boldsymbol{\sigma}, p),
# \qquad
# \dot p = \dot p(\boldsymbol{\sigma}, p),
# ```
#
# with ``\mathbb{S}_r`` its elastic compliance and ``p`` an accumulated strain
# that hardens it, as in [`PerzynaViscoplasticity`](@ref); a phase without
# hardening, such as [`NortonViscoplasticity`](@ref), has no ``p``.
#
# ## The linearization at the end of a step
#
# At the end ``\tau`` of a step, the rates are expanded about the current stress
# and accumulated strain of the phase,
#
# ```math
# \dot{\boldsymbol{\varepsilon}}^{in}(t) \approx \dot{\boldsymbol{\varepsilon}}^{in}_\tau + \mathbb{m}:(\boldsymbol{\sigma}(t) - \boldsymbol{\sigma}_\tau) + \boldsymbol{n}\,(p(t) - p_\tau),
# \qquad
# \dot p(t) \approx \dot p_\tau + \boldsymbol{l}:(\boldsymbol{\sigma}(t) - \boldsymbol{\sigma}_\tau) + q\,(p(t) - p_\tau),
# ```
#
# ``\mathbb{m}``, ``\boldsymbol{n}``, ``\boldsymbol{l}``, ``q`` the derivatives of the rates
# in the stress and in ``p`` at ``\tau``. Integrating the second equation and
# carrying ``p`` into the first gives a linear viscoelastic solid. The paper
# applies this linear solid to the whole time axis, with an eigenstrain that
# makes it follow the actual inelastic rate of the phase up to ``\tau`` and the
# linearized one after it. In the Laplace–Carson domain, of variable ``s``, the
# linear solid is a thermoelastic one,
#
# ```math
# \boldsymbol{\varepsilon}^{*}(s) = \mathbb{S}^{*}(s):\boldsymbol{\sigma}^{*}(s) + \boldsymbol{J}(s),
# \qquad
# \mathbb{S}^{*}(s) = \mathbb{S}_r + \frac{\mathbb{m}}{s} + \frac{\boldsymbol{n}\otimes\boldsymbol{l}}{s\,(s - q)} .
# ```
#
# The compliance is equation (47) of the paper. Writing the eigenstrain out,
# the terms in which the past enters through ``e^{q(t-u)}`` cancel between before
# and after ``\tau``, and what remains needs only two integrals of the history,
# ``I_{\sigma}(s) = \int_0^\tau \boldsymbol{\sigma}(t)\,e^{-st}\,\mathrm{d}t`` and
# ``I_{in}(s) = \int_0^\tau \dot{\boldsymbol{\varepsilon}}^{in}(t)\,e^{-st}\,\mathrm{d}t``:
#
# ```math
# \boldsymbol{J}(s) = I_{in}(s) - \mathbb{m}:I_{\sigma}(s) + \boldsymbol{n}\,\frac{\boldsymbol{l}:I_{\sigma}(s)}{q - s}
#   + \big(\dot{\boldsymbol{\varepsilon}}^{in}_\tau - \mathbb{m}:\boldsymbol{\sigma}_\tau\big)\frac{e^{-s\tau}}{s}
#   + \boldsymbol{n}\,\big(\dot p_\tau - \boldsymbol{l}:\boldsymbol{\sigma}_\tau\big)\frac{e^{-s\tau}}{s\,(s - q)} .
# ```
#
# This is ``1/s`` times equation (48) of the paper, which transforms the rate of
# the eigenstrain rather than the eigenstrain, for a history starting from
# rest. For a linear phase, ``\dot{\boldsymbol{\varepsilon}}^{in} = \mathbb{m}:\boldsymbol{\sigma}``
# without ``p``, it vanishes: the phase is exactly its linear viscoelastic
# solid.
#
# ## The comparison composite
#
# The two thermoelastic phases, of moduli ``\mathbb{C}^{*}_r = (\mathbb{S}^{*}_r)^{-1}`` and
# eigenstrains ``\boldsymbol{J}_r``, form a composite that the Mori–Tanaka scheme
# homogenizes at each ``s``. With the Eshelby tensor ``\mathbb{E}`` of the
# inclusions, computed in the isotropic part of ``\mathbb{C}^{*}_0`` as the paper
# does after [doghriOuaar2003](@citet), and
# ``\mathbb{P} = \mathbb{E}:(\mathbb{C}^{*}_0)^{-1}``, the inclusion in the matrix strain
# ``\boldsymbol{\varepsilon}_0`` strains as
#
# ```math
# \boldsymbol{\varepsilon}_1 = \mathbb{B}:\big(\boldsymbol{\varepsilon}_0 + \mathbb{P}:(\mathbb{C}^{*}_1:\boldsymbol{J}_1 - \mathbb{C}^{*}_0:\boldsymbol{J}_0)\big),
# \qquad
# \mathbb{B} = \big[\mathbb{I} + \mathbb{P}:(\mathbb{C}^{*}_1 - \mathbb{C}^{*}_0)\big]^{-1},
# ```
#
# ``\mathbb{B}`` being equation (20) of the paper and the second term the effect
# of the eigenstrains, the polarization of the inclusion relative to the matrix.
# With ``f_0\boldsymbol{\varepsilon}_0 + f_1\boldsymbol{\varepsilon}_1 = \boldsymbol{E}``,
#
# ```math
# \bar{\boldsymbol{\varepsilon}}^{*}_1 = \mathbb{A}^{*}:\boldsymbol{E}^{*} + \boldsymbol{a}^{*},
# \qquad
# \mathbb{A}^{*} = \mathbb{B}:(f_1\mathbb{B} + f_0\mathbb{I})^{-1},
# \qquad
# \boldsymbol{a}^{*} = f_0\,\mathbb{A}^{*}:\mathbb{P}:(\mathbb{C}^{*}_1:\boldsymbol{J}_1 - \mathbb{C}^{*}_0:\boldsymbol{J}_0),
# ```
#
# the same concentration as the paper's equations (17)–(19), written without
# the inverse of ``\mathbb{C}^{*}_1 - \mathbb{C}^{*}_0``: two identical phases are
# allowed, and then ``\mathbb{A}^{*} = \mathbb{I}`` and ``\boldsymbol{a}^{*} = 0``.
#
# ## Back to time
#
# ``\mathbb{A}^{*}`` and ``\boldsymbol{a}^{*}`` are brought back to time by the
# collocation of the paper's appendix B: a function
# ``f(t) = A + B\,t + \sum_k b_k\,(1 - e^{-t/\theta_k})``, of transform
# ``A + B/s + \sum_k b_k/(1 + s\theta_k)``, matching the transform at
# ``s_l = 1/\theta_l`` for the given relaxation times ``\theta_k``, with ``A`` its
# value at ``s \to \infty`` (the elastic concentration for ``\mathbb{A}^{*}``, zero
# for ``\boldsymbol{a}^{*}``) and ``B`` the limit of ``s f^{*}(s)`` at zero, read by
# extrapolation from two points far below the collocation points. The average
# strain of the inclusions at ``\tau`` is then
#
# ```math
# \bar{\boldsymbol{\varepsilon}}_1(\tau) = \int_0^\tau \mathbb{A}(\tau - u):\dot{\boldsymbol{E}}(u)\,\mathrm{d}u + \boldsymbol{a}(\tau),
# ```
#
# whose integral needs only the memories ``\int_0^\tau e^{-(\tau-u)/\theta_k}\,\dot{\boldsymbol{E}}\,\mathrm{d}u``.
# The integrals of the history are kept at the collocation points and updated
# step by step, a fixed amount of memory whatever the length of the history.
#
# ## The step
#
# The phases are integrated over the step at the average strains
# ``\bar{\boldsymbol{\varepsilon}}_1`` and ``(\boldsymbol{E} - f_1\bar{\boldsymbol{\varepsilon}}_1)/f_0``,
# which sets their linearizations and so the concentration; the strain of the
# inclusions it returns must be the one they were integrated at. The package
# iterates this fixed point by successive substitutions, as the paper does,
# accelerated as [anderson1965](@citet) proposes. The macroscopic stress is the
# average of the stresses of the phases, the first of the two choices of the
# paper (its section 3.4). The consistent tangent follows by the
# implicit-function theorem, from the tangents of the phases and the
# derivative of the concentration.
#
# ## Scope
#
# | | |
# |:--|:--|
# | phases | a matrix and inclusions of laws in rate form: [`PerzynaViscoplasticity`](@ref), [`NortonViscoplasticity`](@ref) |
# | scheme | Mori–Tanaka, the Eshelby tensor in the isotropic part of the matrix modulus (`isotropic = false` takes the whole modulus) |
# | checks | two identical phases give the phase; elastic phases the elastic Mori–Tanaka estimate; linear Maxwell phases the linear viscoelastic one, computed in closed form, which the steps approach at the first order of their backward Euler scheme |
# | not covered | the second moments of the strain as reference of the linearization, the macroscopic stress from the viscoelastic relation of the comparison composite (the second choice of the paper), more than two phases |
#
# ## [Example: a particle-reinforced elasto-viscoplastic composite](@id ml-ex-affine-evp)
#
# A matrix holds 20 % of stiffer and stronger spherical particles, both of
# Perzyna type with linear hardening, pulled at a constant strain rate with
# free lateral faces.
#
using MeanFieldHomogenization, TensND, Plots

f = 0.2
matrix = PerzynaViscoplasticity(; E = 70e3, ν = 0.3, σY = 100.0, η = 1e5, m = 2.0, hardening = p -> 2e3 * p)
particles = PerzynaViscoplasticity(; E = 400e3, ν = 0.25, σY = 500.0, η = 1e6, m = 2.0, hardening = p -> 1e4 * p)
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = :rest)
add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(1.0, 1.0)); fraction = f)
laws = (M = matrix, P = particles)
θ = exp10.(range(-3, 3; length = 20))                          # relaxation times of the collocation, s
nothing #hide
#
# | code | theory |
# |:--|:--|
# | `AffineEVPComposite(rve, laws; relaxation_times = θ)` | the affine formulation of Pierard and Doghri, Mori–Tanaka on the comparison composite |
# | `IncrementalTangent(rve, laws)` | the [incremental tangent approach](@ref ml-theory-incremental-tangent), the consistent tangents projected on the isotropic tensors |
# | `UniformStrain`, `UniformStress` | the [uniform-field composites](@ref ml-theory-uniform-fields) |
#
# ### Tension
#
# A strain rate of ``10^{-3}\,\mathrm{s}^{-1}`` up to 1 %:
#
pull = LoadingPath(range(0, 10; length = 41); ε11 = t -> 1e-3 * t)
p = plot(; xlabel = "axial strain (%)", ylabel = "axial stress (MPa)", legend = :bottomright)
for (c, label, ls) in ((UniformStrain(laws, (M = 1 - f, P = f)), "uniform strain", :dash),
                       (IncrementalTangent(rve, laws), "incremental tangent", :solid),
                       (AffineEVPComposite(rve, laws; relaxation_times = θ), "affine", :solid),
                       (UniformStress(laws, (M = 1 - f, P = f)), "uniform stress", :dash))
    r = point_response(c, pull)
    plot!(p, r[:ε11] * 100, r[:σ11]; lw = 2, ls, label)
end
p
#
# The affine estimate lies below the incremental tangent one, which
# [pierardDoghri2006](@citet) found too stiff when the tangent of a time step is
# used for elasto-viscoplastic phases, and between the uniform-field bounds.
#
# ### Check
#
# Two identical phases are the phase itself:
#
c = AffineEVPComposite(rve, (M = matrix, P = matrix); relaxation_times = θ)
point_response(c, pull)[:σ11] ≈ point_response(matrix, pull)[:σ11]
