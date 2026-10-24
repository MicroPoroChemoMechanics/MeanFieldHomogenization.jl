"""The Echoes side of `benchmark_schemes_speed.jl`: the cases, and their homogenization.

The cases are defined here once. `benchmark_schemes_speed.jl` imports this module through
PyCall, builds the same RVEs in MeanFieldHomogenization.jl from `CASES`, and times both
codes; run on its own, this module times the Echoes side alone:

    python3 -I scripts/bench_echoes/schemes_speed_echoes.py

with `python3` the interpreter of the environment where echoes is installed.

Tolerances, equal on both sides: numerical Hill tensors to a relative 1e-6 (absolute 1e-8),
fixed points to a relative 1e-8. Echoes integrates the differential scheme in explicit
first-order steps (1000 here), MeanFieldHomogenization.jl adaptively to a relative 1e-8.

A phase is a dict: `name`, `value` (Kelvin-Mandel 6x6 stiffness or 3x3 conductivity, canonical
components), `fraction`, `axes` (semi-axes, None for a sphere), `euler` (ZYZ angles, the
convention both codes share) and `iso` (isotropic orientation average). The first phase is
the matrix.
"""

import math
import timeit

import echoes
import numpy as np

HILL = dict(algo=echoes.NUMINT3D, epsabs=1e-8, epsrel=1e-6, maxnb=1000000, epsroots=1e-8)
FIXED_POINT_RTOL = 1e-8
SCHEMES = {"dilute": echoes.DIL, "mori_tanaka": echoes.MT, "self_consistent": echoes.SC,
           "differential": echoes.DIFF}


def iso_km(k, mu):
    return np.asarray(echoes.stiff_kmu(k, mu).array)


def aniso_km(seed, scale):
    a = np.array([[math.sin(seed * i + 0.3 * j) + math.cos(1.7 * seed * j - i)
                   for j in range(1, 7)] for i in range(1, 7)])
    return scale * (a @ a.T / 6 + 3.0 * np.eye(6))


def spd3(seed, scale):
    a = np.array([[math.sin(seed * i + 0.7 * j) for j in range(1, 4)] for i in range(1, 4)])
    return scale * (a @ a.T / 3 + np.eye(3))


def phase(name, value, fraction, axes=None, euler=(0.0, 0.0, 0.0), iso=False):
    return dict(name=name, value=np.asarray(value).tolist(), fraction=fraction,
                axes=None if axes is None else list(axes), euler=list(euler), iso=iso)


# (label, elastic, phases, schemes)
CASES = [
    ("isotropic matrix, spheres", True,
     [phase("M", iso_km(20.0, 9.0), 0.7), phase("I", iso_km(60.0, 30.0), 0.3)],
     ["mori_tanaka", "self_consistent", "differential"]),
    ("isotropic matrix, aligned prolate spheroids 5:1:1", True,
     [phase("M", iso_km(20.0, 9.0), 0.75), phase("I", iso_km(60.0, 30.0), 0.25, axes=(5.0, 1.0, 1.0))],
     ["mori_tanaka", "self_consistent"]),
    ("isotropic matrix, oblate pores 1:1:0.1, isotropic average", True,
     [phase("M", iso_km(20.0, 9.0), 0.8),
      phase("P", iso_km(1e-3, 1e-3), 0.2, axes=(1.0, 1.0, 0.1), iso=True)],
     ["mori_tanaka", "self_consistent"]),
    ("anisotropic matrix, two rotated triaxial families", True,
     [phase("M", aniso_km(1.0, 10.0), 0.8),
      phase("A", aniso_km(2.0, 40.0), 0.12, axes=(2.0, 1.0, 0.5), euler=(0.7, 0.3, 1.1)),
      phase("B", aniso_km(3.0, 2.0), 0.08, axes=(0.8, 0.8, 3.0), euler=(1.9, 2.4, 0.4))],
     ["dilute", "mori_tanaka"]),
    ("conduction, isotropic matrix, spheres", False,
     [phase("M", 2.0 * np.eye(3), 0.7), phase("I", 10.0 * np.eye(3), 0.3)],
     ["mori_tanaka", "self_consistent"]),
    ("conduction, anisotropic matrix, rotated triaxial ellipsoids", False,
     [phase("M", spd3(1.0, 2.0), 0.8),
      phase("A", spd3(2.0, 8.0), 0.2, axes=(2.0, 1.0, 0.5), euler=(0.7, 0.3, 1.1))],
     ["dilute", "mori_tanaka", "self_consistent"]),
]


def _prop(value, elastic):
    v = np.asarray(value)
    return echoes.tensor(np.asarray([v[i, j] for i in range(6) for j in range(i, 6)])) if elastic \
        else echoes.tensor(v)


def make_rve(phases, elastic):
    key = "C" if elastic else "K"
    ver = echoes.rve(matrix=phases[0]["name"])
    for p in phases:
        shape = echoes.spherical if p["axes"] is None else \
            echoes.ellipsoidal(np.array(list(p["axes"]) + list(p["euler"])))
        kw = dict(symmetrize=[echoes.ISO]) if p["iso"] else {}
        ver[p["name"]] = echoes.ellipsoid(shape=shape, fraction=p["fraction"],
                                          prop={key: _prop(p["value"], elastic)}, **kw)
    ver.set_param_eshelby(**HILL)
    return ver


def maxnb(scheme):
    return 1000 if scheme == "differential" else 500


def homogenize(phases, elastic, scheme):
    """The effective tensor, canonical components (6x6 Kelvin-Mandel or 3x3)."""
    ver = make_rve(phases, elastic)
    out = echoes.homogenize(prop="C" if elastic else "K", rve=ver, scheme=SCHEMES[scheme],
                            verbose=False, epsrel=FIXED_POINT_RTOL, maxnb=maxnb(scheme))
    return np.asarray(out.array)


def homogenize_case(i, j):
    """Case `i` of `CASES` (0-based) with its scheme `j`: what the Julia side times, so
    that no conversion of the phases enters the timing."""
    _, elastic, phases, schemes = CASES[i]
    return homogenize(phases, elastic, schemes[j])


# Aging linear viscoelasticity: a Maxwell matrix, relaxation kernel
#   C(t, t') = 3k exp(-(t - t')/tau_k) J + 2mu exp(-(t - t')/tau_mu) K,
# with elastic inclusions (constant kernels), on a time grid. A phase is a dict:
# `name`, `law` (("maxwell", k, mu, tau_k, tau_mu) or ("elastic", k, mu)),
# `fraction`, `axes`, `iso`, as above. Echoes evaluates the kernels through the
# Python functions given here, at each pair of instants of the grid.
ALV_TIMES = np.linspace(0.0, 5.0, 11)


def alv_phase(name, law, fraction, axes=None, iso=False):
    return dict(name=name, law=list(law), fraction=fraction,
                axes=None if axes is None else list(axes), iso=iso)


# (label, phases, schemes)
ALV_CASES = [
    ("ALV: Maxwell matrix, elastic spheres",
     [alv_phase("M", ("maxwell", 20.0, 9.0, 1.0, 0.5), 0.7),
      alv_phase("I", ("elastic", 60.0, 30.0), 0.3)],
     ["mori_tanaka", "self_consistent"]),
    ("ALV: Maxwell matrix, oblate soft inclusions 1:1:0.1, isotropic average",
     [alv_phase("M", ("maxwell", 20.0, 9.0, 1.0, 0.5), 0.8),
      alv_phase("P", ("elastic", 0.5, 0.3), 0.2, axes=(1.0, 1.0, 0.1), iso=True)],
     ["mori_tanaka", "self_consistent"]),
]


def _alv_kernel(law):
    if law[0] == "maxwell":
        _, k, mu, tau_k, tau_mu = law
        return lambda t, tp: 3 * k * math.exp(-(t - tp) / tau_k) * echoes.J4 \
            + 2 * mu * math.exp(-(t - tp) / tau_mu) * echoes.K4
    _, k, mu = law
    c = 3 * k * echoes.J4 + 2 * mu * echoes.K4
    return lambda t, tp: c


def make_rve_alv(phases):
    ver = echoes.rve(matrix=phases[0]["name"])
    for p in phases:
        shape = echoes.spherical if p["axes"] is None else \
            echoes.ellipsoidal(np.array(list(p["axes"]) + [0.0, 0.0, 0.0]))
        kw = dict(symmetrize=[echoes.ISO]) if p["iso"] else {}
        law = p["law"]
        ver[p["name"]] = echoes.ellipsoid(
            shape=shape, fraction=p["fraction"],
            prop={"C": echoes.stiff_kmu(law[1], law[2])},
            visco_prop={"C": (_alv_kernel(law), echoes.RELAXATION)}, **kw)
    ver.set_param_eshelby(**HILL)
    # The starting point of the fixed point: the relaxation matrix of the matrix.
    ver.set_visco_mat("C", echoes.visco_law(_alv_kernel(phases[0]["law"]), echoes.RELAXATION)
                      .relaxation_mat(ALV_TIMES))
    return ver


def homogenize_alv(phases, scheme):
    """The effective relaxation matrix on `ALV_TIMES`, 6n x 6n Kelvin-Mandel blocks."""
    ver = make_rve_alv(phases)
    return np.asarray(echoes.homogenize_visco(prop="C", rve=ver, time_series=ALV_TIMES,
                                              scheme=SCHEMES[scheme], verbose=False,
                                              epsrel=FIXED_POINT_RTOL, maxnb=maxnb(scheme)))


def homogenize_alv_case(i, j):
    """Case `i` of `ALV_CASES` (0-based) with its scheme `j`, as `homogenize_case`."""
    _, phases, schemes = ALV_CASES[i]
    return homogenize_alv(phases, schemes[j])


def best_time(f, repeat=5):
    """The minimum over `repeat` runs, in seconds, as BenchmarkTools' minimum on the Julia side."""
    return min(timeit.repeat(f, number=1, repeat=repeat))


if __name__ == "__main__":
    for label, elastic, phases, schemes in CASES:
        for s in schemes:
            t = best_time(lambda: homogenize(phases, elastic, s))
            print(f"{label:62s} {s:16s} {t:10.3g} s", flush=True)
    for label, phases, schemes in ALV_CASES:
        for s in schemes:
            t = best_time(lambda: homogenize_alv(phases, s))
            print(f"{label:62s} {s:16s} {t:10.3g} s", flush=True)
