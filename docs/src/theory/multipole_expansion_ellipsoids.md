# [The multipole expansion method for ellipsoids](@id th-mem-ellipsoids)

!!! info "Before this page"
    [The multipole expansion method for spheroids](@ref th-mem-spheroids), whose
    structure is kept here: the response of a particle over its own harmonics,
    the translation by projection on the receiving surface, the far images of a
    lattice through the spherical harmonics.

The particles are now ellipsoids of three different semi-axes, all oriented
alike, with a perfect or a Kapitza interface, in an isotropic matrix of
conductivity ``k_0`` [kushch2020](@cite). The harmonics adapted to an ellipsoid
are those of Lamé, built here from the theory of [dassios2012](@citet). An
assembly of [`Ellipsoid`](@ref)s, or of [`InterfacedEllipsoid`](@ref)s for an
imperfect interface, selects this page.

## 1. Ellipsoidal coordinates and Lamé functions

Let ``a_1 > a_2 > a_3`` be the semi-axes, along ``x_1, x_2, x_3``, and
``h_1^2 = a_2^2 - a_3^2``, ``h_2^2 = a_1^2 - a_3^2``, ``h_3^2 = a_1^2 - a_2^2``. The
ellipsoidal coordinates ``(\rho, \mu, \nu)``, with
``0 \le \nu^2 \le h_3^2 \le \mu^2 \le h_2^2 \le \rho^2``, are the three roots ``t`` of

```math
\frac{x_1^2}{t} + \frac{x_2^2}{t - h_3^2} + \frac{x_3^2}{t - h_2^2} = 1 , \qquad t \in \{\rho^2, \mu^2, \nu^2\} ,
```

and the surface of the particle is ``\rho = a_1`` [dassios2012](@cite). Laplace's
equation separates as ``E(\rho)\,E(\mu)\,E(\nu)``, ``E`` being a Lamé function of
degree ``n``, a solution of

```math
(x^2 - h_3^2)(x^2 - h_2^2)\,E'' + x\,(2x^2 - h_3^2 - h_2^2)\,E' + \big[(h_3^2 + h_2^2)\,p - n(n+1)\,x^2\big]\,E = 0
```

of the form ``\psi(x)\,P(x)``: ``P`` a polynomial of every other power and ``\psi`` one
of ``1``, ``\sqrt{|x^2-h_3^2|}``, ``\sqrt{|x^2-h_2^2|}`` and their product, the four
classes K, L, M, N. Writing ``P = \sum_k c_k\,x^{d-2k}`` turns the equation into a
three-term recurrence on the ``c_k`` whose truncation is a tridiagonal
eigenproblem for the separation constant ``p`` [dassios2012](@cite); its
eigenvalues give the ``2n+1`` Lamé functions of the degree, ``c_0 = 1``. The
square roots take the signs of the octant of the point.

The interior and exterior ellipsoidal harmonics are

```math
\mathbb{E}_n^m = E_n^m(\rho)\,E_n^m(\mu)\,E_n^m(\nu) , \qquad
\mathbb{F}_n^m = F_n^m(\rho)\,E_n^m(\mu)\,E_n^m(\nu) , \qquad
F(\rho) = (2n+1)\,E(\rho)\int_\rho^\infty \frac{\mathrm{d}t}{E(t)^2\sqrt{t^2-h_3^2}\sqrt{t^2-h_2^2}} ,
```

the first a polynomial of degree ``n``, the second decaying as ``\rho^{-n-1}``. The
integral is computed as ``\rho\int_0^1 \mathrm{d}s/[E(\rho/s)^2\sqrt{\rho^2-h_3^2s^2}\sqrt{\rho^2-h_2^2s^2}]``,
whose integrand is smooth. At the degree 1 the three classes K, L, M give
``\mathbb{E}_1 = h_2h_3\,x_1``, ``h_1h_3\,x_2``, ``h_1h_2\,x_3``, and the
corresponding ``\mathbb{F}_1`` tend to the same constants times ``x_j/r^3``.

On a surface ``\rho = \mathrm{const}`` the surface harmonics ``S_n^m = E_n^m(\mu)\,E_n^m(\nu)``
are orthogonal for the weight

```math
\mathrm{d}\Omega = \frac{(\mu^2 - \nu^2)\,\mathrm{d}\mu\,\mathrm{d}\nu}
  {\sqrt{\mu^2-h_3^2}\sqrt{h_2^2-\mu^2}\sqrt{h_3^2-\nu^2}\sqrt{h_2^2-\nu^2}} ,
\qquad \int S_n^m S_{n'}^{m'}\,\mathrm{d}\Omega = \gamma_n^m\,\delta_{nn'}\delta_{mm'} .
```

The substitutions ``\mu^2 = h_3^2 + (h_2^2 - h_3^2)\sin^2\alpha``, ``\alpha \in [0, \pi]``, and
``\nu = h_3\sin\beta``, ``\beta \in [0, 2\pi)``, cover the eight octants with their signs
and remove the singular factors: ``\mathrm{d}\Omega = (\mu^2 - \nu^2)/(\mu\sqrt{h_2^2-\nu^2})\,\mathrm{d}\alpha\,\mathrm{d}\beta``,
integrated by Gauss–Legendre in ``\alpha`` and the trapezoidal rule in ``\beta``. At
the degree 1, ``\gamma_1^m = (4\pi/3)\,H_m^2`` with ``H_m = h_1h_2h_3/h_m``.

## 2. Response of an ellipsoid

In the regular field ``\sum a_n^m\,\mathbb{E}_n^m`` the ellipsoid sends
``\sum b_n^m\,\mathbb{F}_n^m``. On ``\rho = a_1`` both are ``E(a_1)\,S`` and ``F(a_1)\,S``, and the
normal derivative is ``h_\rho^{-1}\,\partial_\rho`` with the same metric coefficient
``h_\rho`` on both sides, so that a perfect interface answers harmonic by harmonic:

```math
b_n^m = \frac{\tilde k - 1}{F'/E' - \tilde k\,F/E}\,a_n^m , \qquad \tilde k = k_1/k_0 ,
```

the functions being taken at ``a_1``. A Kapitza interface of resistance
``\rho_{\mathrm{K}}``, ``[\![T]\!] = -\rho_{\mathrm{K}}\,q_n``, keeps the flux continuous harmonic by
harmonic, ``d = (b\,F' + a\,E')/(\tilde k\,E')`` inside, while the jump, projected on
``S_j``, couples the harmonics through the metric:

```math
\gamma_j\,\big(b_j F_j + a_j E_j - d_j E_j\big)
  = \rho_{\mathrm{K}}\,k_1 \sum_k \Big(\int \frac{S_j\,S_k}{h_\rho}\,\mathrm{d}\Omega\Big)\,E'_k\,d_k ,
```

a linear system that gives the response matrix ``\mathbf{Z}``, ``b = \mathbf{Z}\,a``.
[kushch2020](@citet) tabulates the Maxwell estimate ``k_{11}/k_0`` of ellipsoids of
semi-axes ``(1.5, 1, 0.5)`` at the fraction 0.5, ``k_1/k_0 = 100``, against the
conductance ``1/\rho_{\mathrm{K}}`` of the interface and the degree; for
``1/\rho_{\mathrm{K}} = 1`` it gives 1.5541708 at the degree 1 and 1.6079410 at the degree 3.
One ellipsoid in a domain of its own shape is that estimate:

```@example mem_ell
using MeanFieldHomogenization, TensND, LinearAlgebra
semi = (1.5, 1.0, 0.5)
asm = ParticleAssembly(; boundary = MixedBC(Ellipsoid((semi ./ cbrt(0.5))...)))
add_matrix!(asm, Dict(:K => TensISO{3}(1.0)))
add_particle!(asm, :p, (0.0, 0.0, 0.0), InterfacedEllipsoid(semi..., KapitzaInterface(1.0)), Dict(:K => TensISO{3}(100.0)))
[round(canonical_components(homogenize(asm, MultipoleExpansion(; nmax), :K))[1, 1]; digits = 7) for nmax in (1, 3)]
```

## 3. From one ellipsoid to another

The coefficients of the field of particle ``J`` on the interior harmonics of
particle ``I`` are read on the surface of ``I``, as for spheroids:

```math
\eta_{(n,m)(k,l)} = \frac{1}{E_n^m(a_1)\,\gamma_n^m}\int \mathbb{F}_k^l(\underline{x} - \underline{X}_J)\,S_n^m\,\mathrm{d}\Omega
```

([kushch2020](@cite), Eq. (33)), which holds whatever the distance as long as the
particles do not overlap. Far apart, the exterior harmonics of ``J`` are expanded
on the decaying spherical ones about its center, by projection on a sphere of
radius ``2a_1``, which holds the focal ellipse; these are translated as in
[the method for spheres](@ref th-mem), and the regular spherical harmonics are
expanded on the ``\mathbb{E}`` of ``I`` by projection on its surface, exactly since
they are polynomials ([kushch2020](@cite), Eqs. (38), (41)). The series of this
far form converges as that of spheroids; it is used beyond
``\tfrac52\,(h_{2,I} + h_{2,J})`` with thirty spherical degrees beyond ``n_{\max}``,
and agrees there with the projection to ``10^{-9}``. On a periodic cell the far
images go through the lattice sums of [the spheroids](@ref th-mem-spheroids),
with the same background.

The coefficients of the translation along ``\underline{e}_3`` tabulated by
[kushch2020](@citet) are recovered by the projection to five digits; at
``|\underline{R}| = 3a_1`` the table gives ``\eta_{(1,1)(1,1)} = 0.917124\times 10^{-2}``:

```@example mem_ell
A = MeanFieldHomogenization.Assemblies
d = A._mem_ellipsoid_data(semi, 2.0, 7, false)
i₁ = A._mem_index(1, -1)
round(real(A._mem_eta_near(d, d, [0.0, 0.0, 4.5], 7)[i₁, i₁]); sigdigits = 6)
```

## 4. Orientation and effective conductivity

The ellipsoids of an assembly share their orientation, their axes along those
of the assembly; the problem is solved in the frame of their axes ordered by
decreasing semi-axis, and the effective conductivity turned back to the frame
of the assembly. The dipole of a particle is read on its harmonics of degree 1,
``\underline{d} = (h_2h_3\,b_1^{\mathrm K}, h_1h_3\,b_1^{\mathrm L}, h_1h_2\,b_1^{\mathrm M})``, and the
formulas of the [spheres](@ref th-mem) apply. The constituents are isotropic;
anisotropic ones [kushch2017](@cite) and the elasticity of ellipsoids are listed
in the [roadmap](@ref dev-roadmap).
