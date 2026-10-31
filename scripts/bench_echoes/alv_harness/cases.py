"""Cases of the aging linear viscoelastic (ALV) comparison harness, Echoes side.

Every case is a plain dict, fully specified, so that MeanFieldHomogenization.jl can rebuild
it exactly from the JSON written by `run_echoes.py` (each result file embeds its case).

Keys of a case
--------------
name        unique identifier, also the file name `results/<name>.json`
order       4 (elasticity, 6x6 Kelvin-Mandel blocks) or 2 (conduction, 3x3 blocks)
scheme      "dilute" | "mori_tanaka" | "self_consistent"
times       the time grid t_0 < ... < t_{n-1} (n <= 30)
epsrel      relative tolerance of the self-consistent fixed point (Frobenius norm)
maxnb       maximum number of fixed-point iterations
phases      list of phases; the FIRST one is the matrix (reference of DIL and MT)

Keys of a phase
---------------
name        phase name
fraction    volume fraction (zero for a crack family)
crack       True for a family of flat elliptic cracks (absent otherwise); its semi-axes are
            (a, b, c) with a >= b and c << b the flatness Echoes needs (its compliance is the
            limit c -> 0), its law the zero kernel of a traction-free crack
density     crack density in Echoes' convention, N a^3: Echoes adds (4 pi/3) density eta H_E,
            its H_E normalized by a (H_E = eta H); the same family in MeanFieldHomogenization
            has the density eta^2 * density = N a b^2, eta = b/a, its H being normalized by b
axes        semi-axes (a1, a2, a3) of the ellipsoid along its local axes, None for a sphere
euler       Echoes angles (theta, phi, psi): the local frame is the columns of
            P = Rz(phi) Ry(theta) Rz(psi) (echoes_cpp/src/echoes/tensor_angles.cpp:71-81),
            so the local axis 3 is (sin(theta)cos(phi), sin(theta)sin(phi), cos(theta)):
            theta is the inclination from e3, phi the azimuth, psi the spin about the axis
symmetrize  None, {"sym": "ISO"} (isotropic orientation average), or
            {"sym": "TI", "theta": .., "phi": ..} (average over rotations about the axis
            (sin th cos ph, sin th sin ph, cos th))
law         the kernel, see LAWS below; or {"type": "from_case", "case": <name>}: the
            effective discretized relaxation operator of another case on the same grid,
            used as this phase's discretized operator (multiscale)

Kernels (closed forms; Delta = t - t', g(t') = 1 - a exp(-t'/t_a) the aging factor;
J = (1/3) 1 x 1 and K = I - J the isotropic projectors, Kelvin-Mandel components)
---------------------------------------------------------------------------------
"elastic_iso"            RELAXATION  C = 3k J + 2mu K
"maxwell_aging_iso"      RELAXATION  C(t,t') = g(t') [3(k_inf + k1 e^{-Delta/tau_k}) J
                                                    + 2(mu_inf + mu1 e^{-Delta/tau_mu}) K]
"kelvin_aging_iso"       CREEP       J(t,t') = [1/(3k0) + (1 - e^{-Delta/tau_k})/(3 k1 g(t'))] J
                                             + [1/(2mu0) + (1 - e^{-Delta/tau_mu})/(2 mu1 g(t'))] K
"maxwell_aging_ti"       RELAXATION  C(t,t') = g(t') R6 [CA e^{-Delta/tau} + CB] R6^T, with CA, CB
                                     transversely isotropic about the LOCAL axis 3, given by
                                     (C11, C12, C13, C33, C44) in Kelvin-Mandel:
                                     [[C11,C12,C13,0,0,0],[C12,C11,C13,0,0,0],[C13,C13,C33,0,0,0],
                                      [0,0,0,2C44,0,0],[0,0,0,0,2C44,0],[0,0,0,0,0,C11-C12]],
                                     and R6 the Kelvin-Mandel image of P(euler_law)
"elastic_cond_iso"       RELAXATION  K = k I3
"maxwell_aging_cond_iso" RELAXATION  K(t,t') = g(t') (k_inf + k1 e^{-Delta/tau}) I3
"kelvin_aging_cond_iso"  CREEP       R(t,t') = [1/k0 + (1 - e^{-Delta/tau})/(k1 g(t'))] I3
"maxwell_aging_cond_ti"  RELAXATION  K(t,t') = g(t') P diag(kt, kt, ka) P^T (k_inf' + e^{-Delta/tau})
                                     with k_inf' = "w_inf", P = P(euler_law)

A CREEP law is discretized by Echoes as a creep matrix and inverted numerically to give the
relaxation matrix used in the schemes (see README).
"""

import numpy as np

# A short, geometrically graded grid: 12 instants, the first at t_0 = 0.1 so that the aging
# factor g(t') varies over the grid.
TIMES = [float(t) for t in np.geomspace(0.1, 20.0, 12)]
AGE = dict(a=0.5, t_a=1.0)

# --- laws -----------------------------------------------------------------------------------
def elastic_iso(k, mu):
    return dict(type="elastic_iso", k=k, mu=mu)


def maxwell_iso(k1, mu1, tau_k, tau_mu, k_inf=0.0, mu_inf=0.0, a=AGE["a"], t_a=AGE["t_a"]):
    return dict(type="maxwell_aging_iso", k1=k1, mu1=mu1, tau_k=tau_k, tau_mu=tau_mu,
                k_inf=k_inf, mu_inf=mu_inf, a=a, t_a=t_a)


def kelvin_iso(k0, mu0, k1, mu1, tau_k, tau_mu, a=AGE["a"], t_a=AGE["t_a"]):
    return dict(type="kelvin_aging_iso", k0=k0, mu0=mu0, k1=k1, mu1=mu1,
                tau_k=tau_k, tau_mu=tau_mu, a=a, t_a=t_a)


def maxwell_ti(CA, CB, tau, euler, a=AGE["a"], t_a=AGE["t_a"]):
    return dict(type="maxwell_aging_ti", CA=list(CA), CB=list(CB), tau=tau,
                euler_law=list(euler), a=a, t_a=t_a)


def elastic_cond(k):
    return dict(type="elastic_cond_iso", k=k)


def maxwell_cond(k1, tau, k_inf=0.0, a=AGE["a"], t_a=AGE["t_a"]):
    return dict(type="maxwell_aging_cond_iso", k1=k1, tau=tau, k_inf=k_inf, a=a, t_a=t_a)


def kelvin_cond(k0, k1, tau, a=AGE["a"], t_a=AGE["t_a"]):
    return dict(type="kelvin_aging_cond_iso", k0=k0, k1=k1, tau=tau, a=a, t_a=t_a)


def maxwell_cond_ti(kt, ka, tau, w_inf, euler, a=AGE["a"], t_a=AGE["t_a"]):
    return dict(type="maxwell_aging_cond_ti", kt=kt, ka=ka, tau=tau, w_inf=w_inf,
                euler_law=list(euler), a=a, t_a=t_a)


def from_case(name):
    return dict(type="from_case", case=name)


def phase(name, law, fraction, axes=None, euler=(0.0, 0.0, 0.0), symmetrize=None):
    return dict(name=name, law=law, fraction=fraction,
                axes=None if axes is None else list(axes), euler=list(euler),
                symmetrize=symmetrize)


# Echoes takes the volume (4 pi/3) density eta flat of the flat ellipsoid off the matrix
# fraction: an error proportional to `flat` (1.2e-5 at 1e-4 on f3, 1.2e-8 at 1e-7), which
# MeanFieldHomogenization, whose cracks carry no volume, does not make.
def crack_family(name, density, axes, euler=(0.0, 0.0, 0.0), symmetrize=None, flat=1e-12):
    a, b = axes
    return dict(name=name, law=elastic_iso(0.0, 0.0), fraction=0.0, crack=True,
                density=density, axes=[a, b, flat * b], euler=list(euler),
                symmetrize=symmetrize)


def case(name, order, scheme, phases, times=TIMES, epsrel=1e-10, maxnb=200):
    return dict(name=name, order=order, scheme=scheme, times=list(times),
                epsrel=epsrel, maxnb=maxnb, phases=phases)


ISO = {"sym": "ISO"}
ROT = (0.4, 0.9, 0.2)            # non-canonical orientation (theta, phi, psi)
PROLATE = (1.0, 1.0, 5.0)        # spheroid, axis = local 3
OBLATE = (1.0, 1.0, 0.2)
TRIAXIAL = (2.0, 1.0, 0.4)

# Matrix: isotropic aging Maxwell (relaxes to zero), plus a small elastic floor for the SC
# cases so that the operator stays invertible at late times.
M_MAX = maxwell_iso(k1=20.0, mu1=9.0, tau_k=3.0, tau_mu=1.0)
M_MAX_FLOOR = maxwell_iso(k1=20.0, mu1=9.0, tau_k=3.0, tau_mu=1.0, k_inf=2.0, mu_inf=1.0)
I_EL = elastic_iso(60.0, 30.0)
I_MAX = maxwell_iso(k1=50.0, mu1=25.0, tau_k=10.0, tau_mu=4.0, k_inf=5.0, mu_inf=2.0)
I_KEL = kelvin_iso(k0=40.0, mu0=20.0, k1=30.0, mu1=10.0, tau_k=2.0, tau_mu=1.5)
# TI local constants (C11, C12, C13, C33, C44), positive definite
TI_A = (40.0, 10.0, 8.0, 80.0, 12.0)
TI_B = (8.0, 2.0, 1.5, 15.0, 3.0)

CASES = [
    # a. order 4, spheres -------------------------------------------------------------------
    case("a1_sph_maxwell_elastic_DIL", 4, "dilute",
         [phase("M", M_MAX, 0.8), phase("I", I_EL, 0.2)]),
    case("a2_sph_maxwell_elastic_MT", 4, "mori_tanaka",
         [phase("M", M_MAX, 0.7), phase("I", I_EL, 0.3)]),
    case("a3_sph_maxwell_elastic_SC", 4, "self_consistent",
         [phase("M", M_MAX_FLOOR, 0.6), phase("I", I_EL, 0.4)]),
    case("a4_sph_maxwell_kelvin_MT", 4, "mori_tanaka",
         [phase("M", M_MAX, 0.7), phase("I", I_KEL, 0.3)]),
    case("a5_sph_maxwell_maxwell_SC", 4, "self_consistent",
         [phase("M", M_MAX_FLOOR, 0.5), phase("I", I_MAX, 0.3), phase("J", I_KEL, 0.2)]),
    # b. order 4, spheroids ---------------------------------------------------------------
    case("b1_prolate_e3_MT", 4, "mori_tanaka",
         [phase("M", M_MAX, 0.75), phase("I", I_MAX, 0.25, axes=PROLATE)]),
    case("b2_oblate_e3_MT", 4, "mori_tanaka",
         [phase("M", M_MAX, 0.8), phase("I", I_EL, 0.2, axes=OBLATE)]),
    case("b3_prolate_rot_MT", 4, "mori_tanaka",
         [phase("M", M_MAX, 0.75), phase("I", I_MAX, 0.25, axes=PROLATE, euler=ROT)]),
    case("b4_oblate_rot_MT", 4, "mori_tanaka",
         [phase("M", M_MAX, 0.8), phase("I", I_EL, 0.2, axes=OBLATE, euler=ROT)]),
    case("b5_oblate_rot_DIL", 4, "dilute",
         [phase("M", M_MAX, 0.9), phase("I", I_KEL, 0.1, axes=OBLATE, euler=ROT)]),
    case("b6_triaxial_rot_MT", 4, "mori_tanaka",
         [phase("M", M_MAX, 0.8), phase("I", I_MAX, 0.2, axes=TRIAXIAL, euler=ROT)]),
    case("b7_prolate_isoavg_MT", 4, "mori_tanaka",
         [phase("M", M_MAX, 0.75), phase("I", I_MAX, 0.25, axes=PROLATE, symmetrize=ISO)]),
    case("b8_oblate_isoavg_SC", 4, "self_consistent",
         [phase("M", M_MAX_FLOOR, 0.6), phase("I", I_EL, 0.4, axes=OBLATE, symmetrize=ISO)]),
    case("b9_triaxial_tiavg_MT", 4, "mori_tanaka",
         [phase("M", M_MAX, 0.8),
          phase("I", I_MAX, 0.2, axes=TRIAXIAL, euler=ROT,
                symmetrize={"sym": "TI", "theta": ROT[0], "phi": ROT[1]})]),
    # c. order 4, transversely isotropic phase law, axis not along e3 ---------------------
    case("c1_sph_tilaw_rot_MT", 4, "mori_tanaka",
         [phase("M", M_MAX, 0.7), phase("I", maxwell_ti(TI_A, TI_B, 2.0, ROT), 0.3)]),
    case("c2_prolate_tilaw_rot_MT", 4, "mori_tanaka",
         [phase("M", M_MAX, 0.75),
          phase("I", maxwell_ti(TI_A, TI_B, 2.0, ROT), 0.25, axes=PROLATE, euler=ROT)]),
    case("c3_prolate_tilaw_misaligned_DIL", 4, "dilute",
         [phase("M", M_MAX, 0.9),
          phase("I", maxwell_ti(TI_A, TI_B, 2.0, (1.1, 0.3, 0.0)), 0.1, axes=PROLATE,
                euler=ROT)]),
    # d. order 2 --------------------------------------------------------------------------
    case("d1_cond_sph_MT", 2, "mori_tanaka",
         [phase("M", maxwell_cond(5.0, 2.0, k_inf=0.5), 0.7),
          phase("I", elastic_cond(20.0), 0.3)]),
    case("d2_cond_sph_SC", 2, "self_consistent",
         [phase("M", maxwell_cond(5.0, 2.0, k_inf=0.5), 0.6),
          phase("I", kelvin_cond(10.0, 4.0, 1.5), 0.4)]),
    case("d3_cond_prolate_rot_MT", 2, "mori_tanaka",
         [phase("M", maxwell_cond(5.0, 2.0, k_inf=0.5), 0.75),
          phase("I", maxwell_cond(30.0, 5.0, k_inf=3.0), 0.25, axes=PROLATE, euler=ROT)]),
    case("d4_cond_oblate_rot_tilaw_DIL", 2, "dilute",
         [phase("M", maxwell_cond(5.0, 2.0, k_inf=0.5), 0.9),
          phase("I", maxwell_cond_ti(4.0, 25.0, 3.0, 0.2, (1.1, 0.3, 0.0)), 0.1,
                axes=OBLATE, euler=ROT)]),
    # e. multiscale: the effective operator of a case used as a phase of another ----------
    # e1: the dense anisotropic b3 result as spherical inclusions in an aging Maxwell matrix
    case("e1_multiscale_b3_as_inclusion_MT", 4, "mori_tanaka",
         [phase("M", M_MAX, 0.6), phase("H", from_case("b3_prolate_rot_MT"), 0.4)]),
    # e2: the isotropic a3 (SC) result as the matrix of a rotated-oblate MT
    case("e2_multiscale_a3_as_matrix_MT", 4, "mori_tanaka",
         [phase("H", from_case("a3_sph_maxwell_elastic_SC"), 0.8),
          phase("I", I_EL, 0.2, axes=OBLATE, euler=ROT)]),
    # e3: order 2, the d1 result as the matrix of a rotated-prolate MT
    case("e3_multiscale_d1_as_matrix_cond_MT", 2, "mori_tanaka",
         [phase("H", from_case("d1_cond_sph_MT"), 0.75),
          phase("I", elastic_cond(50.0), 0.25, axes=PROLATE, euler=ROT)]),
    # f. order 4, elliptic cracks of any orientation in an aging matrix -------------------
    case("f1_ellcrack_rot_DIL", 4, "dilute",
         [phase("M", M_MAX, 1.0), crack_family("C", 0.1, (1.0, 0.4), euler=ROT)]),
    case("f2_ellcrack_rot_MT", 4, "mori_tanaka",
         [phase("M", M_MAX, 0.8), phase("I", I_EL, 0.2),
          crack_family("C", 0.08, (1.0, 0.15), euler=ROT)]),
    case("f3_penny_rot_MT", 4, "mori_tanaka",
         [phase("M", M_MAX, 0.85), phase("I", I_MAX, 0.15),
          crack_family("C", 0.15, (1.0, 1.0), euler=ROT)]),
    case("f4_ellcrack_isoavg_SC", 4, "self_consistent",
         [phase("M", M_MAX_FLOOR, 0.7), phase("I", I_EL, 0.3),
          crack_family("C", 0.05, (1.0, 0.4), symmetrize=ISO)]),
    case("f5_ellcrack_two_families_MT", 4, "mori_tanaka",
         [phase("M", M_MAX, 1.0), crack_family("C1", 0.05, (1.0, 0.4), euler=ROT),
          crack_family("C2", 0.05, (1.0, 0.6), euler=(1.2, 0.3, 0.8))]),
]

CASES_BY_NAME = {c["name"]: c for c in CASES}
assert len(CASES_BY_NAME) == len(CASES)
assert all(len(c["times"]) <= 30 for c in CASES)
