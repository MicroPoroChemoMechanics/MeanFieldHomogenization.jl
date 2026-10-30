# # [Strength of a frictional matrix with a non-associated flow rule](@id ml-theory-frictional-strength)
#
# Soils, rocks, mortars and concretes have a strength that depends on the mean
# stress, and their plastic flow is usually less dilatant than the normality rule
# would make it. This page computes the macroscopic strength of a composite whose
# matrix obeys such a law, a Drucker–Prager criterion with a non-associated flow
# rule, around pores or rigid inclusions. The method is that of
# [maghous2009](@citet), which extends to non-associated plasticity the
# approach of [barthelemyDormieux2003](@citet) for a von Mises matrix
# ([the strength of a porous plastic solid](@ref ml-theory-strength)). The
# equation numbers below are those of the 2009 paper.
# [The example below](@ref ml-ex-frictional-strength) runs it.
#
# ## The matrix
#
# The matrix can carry the stresses whose mean stress
# ``\sigma_m = \operatorname{tr}\boldsymbol{\sigma}/3`` and the norm of the deviator
# ``\sigma_d = \sqrt{\boldsymbol{\sigma}^d:\boldsymbol{\sigma}^d}``, with
# ``\boldsymbol{\sigma}^d = \boldsymbol{\sigma} - \sigma_m\,\boldsymbol{1}``, satisfy
#
# ```math
# \sigma_d + T\,(\sigma_m - h) \le 0 ,
# ```
#
# eq. (1): ``h`` is the isotropic tensile strength and ``T`` the friction
# coefficient. Its plastic strain rate follows the potential
# ``\sigma_d + t\,\sigma_m``, eq. (15), so that away from the apex
# ``d_v = t\,d_d``, eq. (16), with the volume strain rate
# ``d_v = \operatorname{tr}\boldsymbol{d}`` and the norm of the strain-rate deviator
# ``d_d = \sqrt{\boldsymbol{d}^d:\boldsymbol{d}^d}``,
# ``\boldsymbol{d}^d = \boldsymbol{d} - \tfrac13\,d_v\,\boldsymbol{1}``, the definitions of the
# paper (no factor 1/2 in the norms): with them
# ``\boldsymbol{\sigma}:\boldsymbol{d} = \sigma_m\,d_v + \boldsymbol{\sigma}^d:\boldsymbol{d}^d``. The dilatancy coefficient ``t`` lies between 0 and
# ``T``; the normality rule is ``t = T``. When ``t < T`` the theorems of limit
# analysis no longer hold, and the macroscopic limit states have to be defined
# otherwise.
#
# ## A viscous matrix that tends to the plastic one
#
# The limit states are defined as limits of the states of a sequence of viscous
# matrices indexed by ``a > 0`` (§3.2 of the paper). Writing
# ``Y = d_v - t\,d_d``, the matrix of index ``a`` has the stress
#
# ```math
# \boldsymbol{\sigma} = \frac{\partial \psi_a}{\partial \boldsymbol{d}} + \sigma_0(Y)\,\boldsymbol{1},
# \qquad
# \psi_a = f_a(Y) + h\,d_v ,
# ```
#
# eq. (17), where ``f_a`` is a convex, twice differentiable function, zero for
# ``Y \ge 0``, decreasing, and infinite at ``Y = -a``. Its mean stress is
# ``f_a'(Y) + h + \sigma_0`` and the norm of its deviator ``-t\,f_a'(Y)``. The
# prestress ``\sigma_0 = f_a'(Y)\,(t/T - 1)``, eq. (19), is the one that places
# every stress on the criterion: ``\sigma_d + T(\sigma_m - h) = -t f_a' + t f_a' = 0``.
# Since ``f_a'(Y) \ne 0`` only for ``-a < Y < 0``, the flow rule ``d_v = t\,d_d`` is
# reached as ``a \to 0``.
#
# The same stress reads as a linear law with strain-rate dependent coefficients,
# eqs. (20)–(22):
#
# ```math
# \boldsymbol{\sigma} = 3k\,\mathbb{J}:\boldsymbol{d} + 2\mu\,\mathbb{K}:\boldsymbol{d} + \sigma^p\,\boldsymbol{1},
# \qquad
# k = \frac{f_a'(Y)}{Y},\quad
# \mu = -\frac{t\,f_a'(Y)}{2\,d_d},\quad
# \sigma^p = h + k\Big(\frac{t}{T}\,Y - d_v\Big),
# ```
#
# with positive moduli, since ``Y < 0`` and ``f_a' < 0``. One checks that
# ``k\,d_v + \sigma^p = h + t f_a'/T``, the mean stress found above.
#
# The package takes ``f_a(Y) = c\,(-Y)^3/(a + Y)`` for ``-a < Y < 0`` (and zero
# beyond), with ``c = h/a``, which has the required properties; the limit states do
# not depend on this choice.
#
# ## The modified secant method
#
# The moduli and the prestress vary through the matrix with the local strain
# rate. As in the [secant methods](@ref ml-theory-secant), the matrix is given
# uniform values, taken at effective strain rates: the square roots of the second
# moments of ``d_v`` and ``d_d`` over the matrix, eq. (25),
# ``d_v^e = \sqrt{\langle d_v^2\rangle_m}``, ``d_d^e = \sqrt{\langle d_d^2\rangle_m}``. The
# matrix is then a linear viscous material with a uniform prestress, and the
# composite a linear one, whose second moments follow from the derivatives of its
# effective potential ``W(\boldsymbol{D})`` with respect to the moduli of the matrix,
# at fixed prestress:
#
# ```math
# \frac{1-\varphi}{2}\,\big(d_v^e\big)^2 = \frac{\partial W}{\partial k},
# \qquad
# (1-\varphi)\,\big(d_d^e\big)^2 = \frac{\partial W}{\partial \mu},
# ```
#
# ``\varphi`` being the fraction of the heterogeneities. Both equations form a
# fixed point on ``(d_v^e, d_d^e)``, which the package solves by Newton's method.
#
# The effective potential of the composite with a prestressed matrix is exact in
# the two cases of the paper, for any linear scheme:
#
# - **Pores.** A uniform prestress ``\sigma^p\boldsymbol{1}`` in the matrix is a uniform
#   eigenstrain ``-\sigma^p/(3k)\,\boldsymbol{1}`` there. The same eigenstrain in the
#   pores changes nothing, since they carry no stress: it is a uniform eigenstrain
#   of the whole cell, which only shifts the strain rate. Hence
#   ``W = \tfrac12\,\boldsymbol{D}':\mathbb{C}^{\mathrm{hom}}:\boldsymbol{D}' - (1-\varphi)\,(\sigma^p)^2/(2k)``
#   with ``\boldsymbol{D}' = \boldsymbol{D} + \sigma^p/(3k)\,\boldsymbol{1}``, and the macroscopic stress is
#   ``\boldsymbol{\Sigma} = \mathbb{C}^{\mathrm{hom}}:\boldsymbol{D}'``. Its derivatives are those of
#   eq. (45) of the paper, which is recovered for the Mori–Tanaka scheme.
# - **Rigid inclusions.** The strain rate vanishes in the inclusions, so that the
#   average of ``\sigma^p d_v`` over the cell is ``\sigma^p D_v``,
#   ``D_v = \operatorname{tr}\boldsymbol{D}`` the macroscopic volume strain rate:
#   ``W = \tfrac12\,\boldsymbol{D}:\mathbb{C}^{\mathrm{hom}}:\boldsymbol{D} + \sigma^p D_v`` and
#   ``\boldsymbol{\Sigma} = \mathbb{C}^{\mathrm{hom}}:\boldsymbol{D} + \sigma^p\boldsymbol{1}``, eqs. (33)–(34).
#
# The derivatives of ``\mathbb{C}^{\mathrm{hom}}`` come from automatic differentiation
# through the scheme, so that nothing specific to a scheme or to a shape is
# written.
#
# ## The limit states
#
# As ``a \to 0``, ``Y^e = d_v^e - t\,d_d^e \to 0`` and ``\mu/k \to 0``: the matrix of the
# comparison composite becomes nearly incompressible about the dilatant volume
# change ``t\,d_d^e``. The macroscopic stresses of the regularized composites tend to
# the limit states, at first order in ``a``. They are written with the macroscopic
# invariants defined as the local ones: the mean stress
# ``\Sigma_m = \operatorname{tr}\boldsymbol{\Sigma}/3``, the norm of the stress deviator
# ``\Sigma_d = \sqrt{\boldsymbol{\Sigma}^d:\boldsymbol{\Sigma}^d}``, the volume strain rate
# ``D_v = \operatorname{tr}\boldsymbol{D}`` and the norm of the strain-rate deviator
# ``D_d = \sqrt{\boldsymbol{D}^d:\boldsymbol{D}^d}``. For spherical heterogeneities and the
# Mori–Tanaka scheme the paper finds the limit in closed form:
#
# - **porous medium**: the elliptic domain of eq. (50),
#
#   ```math
#   \frac{1 + 2\varphi/3}{T^2}\,\Sigma_d^2 + \Big(\frac{3\varphi}{2T^2} - 1\Big)\,\Sigma_m^2 + 2(1-\varphi)\,h\,\Sigma_m - (1-\varphi)^2 h^2 = 0 ,
#   ```
#
#   which does not depend on the dilatancy ``t``, while the macroscopic flow rule
#   does: it derives from the potential of eqs. (53) and (55), the same ellipse
#   with ``T^2`` replaced by ``T\,t``. The flow is associated only if ``t = T``.
# - **rigid inclusions**: a Drucker–Prager cone, eq. (40),
#   ``\Sigma_d = T^{\mathrm{hom}}(h - \Sigma_m)``, reached by strain rates that follow the
#   macroscopic flow rule ``D_v = t^{\mathrm{hom}} D_d``, eq. (39), with
#
#   ```math
#   t^{\mathrm{hom}} = t\,\sqrt{\frac{1 + \tfrac32\varphi}{1 - \tfrac23\varphi\,t^2}},
#   \qquad
#   T^{\mathrm{hom}} = T\,\frac{\sqrt{(1 + \tfrac32\varphi)(1 - \tfrac23\varphi\,t^2)}}{1 - \tfrac23\varphi\,t\,T}.
#   ```
#
#   Every point of the cone has the same normal, so that a strain rate fixes the
#   point only through its distance, of order ``a``, to the flow rule. A strain
#   rate off the flow rule has no limit state.
#
# The tests of the package recover these closed forms from the general fixed
# point, and the von Mises limit (``h \to \infty``, ``T \to 0`` with ``T h`` fixed, eq. (56))
# from the [rigid–plastic secant composite](@ref ml-theory-strength).
#
# ## Numerical notes and scope
#
# | | |
# |:--|:--|
# | regularization | ``a`` is taken relative to ``t\,\lvert\boldsymbol{D}\rvert``: ``d_v^e = t\,d_d^e + Y`` is nonnegative, so ``a`` must stay below ``t\,d_d^e``. A dilatancy ``t = 0`` is the limit of small ``t`` and is refused |
# | convergence | first order in ``a``; two values of ``a`` and a Richardson extrapolation remove most of the error; below a relative ``a`` of about ``10^{-3}`` rounding takes over |
# | heterogeneities | pores are given moduli ``10^{-8}`` times the smallest modulus of the matrix, rigid inclusions ``10^{8}`` times the largest, since ``k/\mu`` grows as ``1/a`` |
# | no limit state | a strain rate without one (rigid inclusions off the flow rule) drives the fixed point to the edge ``Y = -a`` of the regularized matrix, and is refused |
# | cost | about 40 µs per limit state for a two-phase Mori–Tanaka cell |
# | scope | a matrix and one kind of heterogeneity, pores or rigid inclusions, any scheme and shape of the package; a single effective strain rate for the whole matrix, which, as the paper notes, underestimates the strength of a porous medium in compression near the porosity ``2T^2/3`` |
#
# ## [Example: porous and reinforced frictional solids](@id ml-ex-frictional-strength)
#
# This example computes the macroscopic limit states of composites with a
# Drucker–Prager matrix whose flow is less dilatant than the normality rule, by
# the [method of Maghous, Dormieux and Barthélémy](@ref ml-theory-frictional-strength),
# and compares them with the closed forms of [maghous2009](@citet).
#
# ### A porous frictional solid
#
# A solid with a tensile strength ``h = 1`` and a friction coefficient
# ``T = 0.3`` holds 15 % of spherical pores. Three dilatancy coefficients are
# compared, from the normality rule ``t = T`` to a nearly isochoric flow:
#
using MeanFieldHomogenization, TensND, Plots

φ, h, T = 0.15, 1.0, 0.3
rve = RVE()
add_phase!(rve, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3.0, 2.0)); fraction = :rest)
add_phase!(rve, :P, Ellipsoid(1.0), Dict(:C => TensISO{3}(3.0, 2.0)); fraction = φ)
composite(t; a = 1e-2) = FrictionalComposite(rve, DruckerPrager(; h, friction = T, dilatancy = t);
    heterogeneities = Pores(), regularization = a)
nothing #hide
#
# | code | theory |
# |:--|:--|
# | `DruckerPrager(; h, friction, dilatancy)` | the criterion ``\sigma_d + T(\sigma_m - h) \le 0`` and the flow potential ``\sigma_d + t\,\sigma_m`` |
# | `Pores()` | the phases other than the matrix carry no stress |
# | `regularization` | the parameter ``a`` of the viscous matrix, relative to ``t\lvert\boldsymbol{D}\rvert`` |
# | `limit_stress(composite, D)` | the limit state reached under the macroscopic strain rate ``\boldsymbol{D}`` |
#
# Each direction of the strain rate, from compaction to expansion, gives a point
# of the boundary. Two regularizations and a Richardson extrapolation remove the
# first-order error in ``a``:
#
𝟏 = Mandel{Float64}(1, 1, 1, 0, 0, 0)
shear = Mandel{Float64}(1, -0.5, -0.5, 0, 0, 0)
mean_dev(σ) = (m = sum(σ[1:3]) / 3; (m, sqrt(sum(abs2, σ - m * 𝟏))))
limit(t, D) = 2 * limit_stress(composite(t; a = 5e-3), D).σ - limit_stress(composite(t; a = 1e-2), D).σ
rates = [sin(θ) * 𝟏 / 3 + cos(θ) * shear for θ in range(-1.4, 1.4; length = 25)]

## The closed-form domain, eq. (50): A Σm² + B Σm + C + D Σd² = 0.
A, B, C, Dd = 3φ / (2T^2) - 1, 2(1 - φ) * h, -(1 - φ)^2 * h^2, (1 + 2φ / 3) / T^2
Σm_line = range(-(B + sqrt(B^2 - 4A * C)) / (2A), (-B + sqrt(B^2 - 4A * C)) / (2A); length = 300)
p = plot(Σm_line, [sqrt(max(-(A * x^2 + B * x + C) / Dd, 0)) for x in Σm_line]; lw = 2, color = :black,
    label = "closed form, eq. (50)", xlabel = "Σ_m / h", ylabel = "Σ_d / h", legend = :topleft)
for (t, marker) in ((0.3, :circle), (0.15, :diamond), (0.05, :utriangle))
    pts = mean_dev.(limit.(t, rates))
    scatter!(p, first.(pts), last.(pts); marker, ms = 4, label = "t = $t")
end
p
#
# The three dilatancies give the same domain, as the paper predicts, and the
# points fall on the closed form. The domain is closed in compression: a porous
# solid ends up crushing its pores.
#
# ### The flow rule depends on the dilatancy
#
# The domain does not depend on ``t``, but the direction of the plastic flow at a
# given stress does. With the normality rule, the strain rate is normal to the
# domain; with ``t < T``, it is normal to the plastic potential of eqs. (53) and
# (55), the same ellipse with ``T^2`` replaced by ``T t``:
#
θs = range(-1.2, 1.2; length = 9)
q = plot(Σm_line, [sqrt(max(-(A * x^2 + B * x + C) / Dd, 0)) for x in Σm_line]; lw = 2, color = :black,
    label = "", xlabel = "Σ_m / h", ylabel = "Σ_d / h", aspect_ratio = :equal)
for (t, color) in ((0.3, :steelblue), (0.05, :firebrick))
    for θ in θs
        D = sin(θ) * 𝟏 / 3 + cos(θ) * shear
        Σm, Σd = mean_dev(limit(t, D))
        Dv, Ddev = sum(D[1:3]), sqrt(sum(abs2, D - sum(D[1:3]) / 3 * 𝟏))
        n = 0.15 / hypot(Dv, Ddev)
        plot!(q, [Σm, Σm + n * Dv], [Σd, Σd + n * Ddev]; color, lw = 2, label = θ == first(θs) ? "flow direction, t = $t" : "")
    end
end
q
#
# The arrows give the macroscopic strain rate ``(D_v, D_d)`` at each limit state.
# For ``t = T`` they are normal to the domain. For a small dilatancy they lean
# toward the deviatoric axis: the porous solid flows with less volume change than
# the normality rule would give.
#
# ### Rigid inclusions: a frictional cone
#
# With rigid inclusions instead of pores, the limit states form a
# Drucker–Prager cone, eq. (40), whose friction coefficient ``T^{\mathrm{hom}}`` grows
# with the fraction of inclusions and depends on the dilatancy. The strain rates
# that reach it follow the macroscopic flow rule ``D_v = t^{\mathrm{hom}} D_d``,
# eq. (39); the point on the cone depends on their distance to it, of the order
# of the regularization:
#
t_hom(φ, t) = t * sqrt((1 + 1.5φ) / (1 - 2φ * t^2 / 3))
T_hom(φ, t) = T * sqrt((1 + 1.5φ) * (1 - 2φ * t^2 / 3)) / (1 - 2φ * t * T / 3)
function friction(φ, t; offset = 0.4)
    cell = RVE()
    add_phase!(cell, :M, Ellipsoid(1.0), Dict(:C => TensISO{3}(3.0, 2.0)); fraction = :rest)
    add_phase!(cell, :R, Ellipsoid(1.0), Dict(:C => TensISO{3}(3.0, 2.0)); fraction = φ)
    D_on = t_hom(φ, t) * sqrt(1.5) * 𝟏 / 3 + shear
    T_of(a) = begin
        fc = FrictionalComposite(cell, DruckerPrager(; h, friction = T, dilatancy = t); heterogeneities = RigidInclusions(), regularization = a)
        Σm, Σd = mean_dev(limit_stress(fc, D_on - offset * a * t * sqrt(sum(abs2, D_on)) * 𝟏 / 3).σ)
        Σd / (h - Σm)
    end
    return 2T_of(5e-3) - T_of(1e-2)
end
fractions = range(0.05, 0.5; length = 10)
r = plot(; xlabel = "fraction of rigid inclusions", ylabel = "macroscopic friction T^hom", legend = :topleft)
for (t, color) in ((0.3, :steelblue), (0.05, :firebrick))
    plot!(r, fractions, T_hom.(fractions, t); color, lw = 2, label = "closed form, t = $t")
    scatter!(r, fractions, friction.(fractions, t); color, ms = 4, label = "computed, t = $t")
end
r
#
# The inclusions raise the friction coefficient, more with the normality rule
# than with a nearly isochoric flow, and the computed values meet the closed form
# of eq. (40).
#
# ### Going further
#
# - Any scheme of the package replaces Mori–Tanaka (`scheme = SelfConsistent()`),
#   and any shape of heterogeneity, with no closed form to compare with.
# - The [rigid–plastic secant composite](@ref ml-ex-strength) is the von Mises
#   limit of this one, ``h \to \infty`` and ``T \to 0`` with ``T h`` fixed.
