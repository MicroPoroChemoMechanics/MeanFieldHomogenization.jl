"""Run the ALV harness cases of `cases.py` with Echoes and write `results/<case>.json`.

    python3 -I run_echoes.py            # every case, in order (multiscale sources first)
    python3 -I run_echoes.py b3_prolate_rot_MT e1_multiscale_b3_as_inclusion_MT

with `python3` the interpreter of the environment where echoes is installed. The script
puts its own directory on `sys.path`, so `-I` is fine.

Each result file holds: the case itself (`case`), the time grid, the effective discretized
relaxation operator (`effective`, an (n u) x (n u) matrix, u = 6 or 3, row-major nested
list, time-major 6x6 / 3x3 blocks, Kelvin-Mandel canonical components), the discretized
operator Echoes used for every phase (`phase_mats`), the check of those against an
independent trapezoidal discretization of the closed-form kernels (`discretization_check`),
the conventions, and the wall time of `homogenize_visco` (minimum of 3 runs).
"""

import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import echoes  # noqa: E402
import numpy as np  # noqa: E402

from cases import CASES, CASES_BY_NAME  # noqa: E402

RESULTS = os.path.join(HERE, "results")
SCHEMES = {"dilute": echoes.DIL, "mori_tanaka": echoes.MT, "self_consistent": echoes.SC}
HILL = dict(algo=echoes.NUMINT3D, epsabs=1e-10, epsrel=1e-8, maxnb=1000000, epsroots=1e-8)
NREPEAT = 3

# Kelvin-Mandel isotropic projectors, order (11, 22, 33, sqrt2*23, sqrt2*31, sqrt2*12)
J6 = np.zeros((6, 6))
J6[:3, :3] = 1.0 / 3.0
K6 = np.eye(6) - J6
I3 = np.eye(3)


def rot3(euler):
    return np.asarray(echoes.rot3(*[float(x) for x in euler]))


def mandel_rotation(P):
    """Kelvin-Mandel image R6 of a rotation P: (P e P^T) in Mandel = R6 (e in Mandel)."""
    idx = [(0, 0), (1, 1), (2, 2), (1, 2), (2, 0), (0, 1)]
    s = math.sqrt(2.0)
    R = np.zeros((6, 6))
    for a, (i, j) in enumerate(idx):
        for b, (k, l) in enumerate(idx):
            v = P[i, k] * P[j, l] if b < 3 else (P[i, k] * P[j, l] + P[i, l] * P[j, k]) / s
            R[a, b] = s * v if a >= 3 else v
    return R


def ti_mandel(c):
    c11, c12, c13, c33, c44 = c
    return np.array([[c11, c12, c13, 0, 0, 0], [c12, c11, c13, 0, 0, 0],
                     [c13, c13, c33, 0, 0, 0], [0, 0, 0, 2 * c44, 0, 0],
                     [0, 0, 0, 0, 2 * c44, 0], [0, 0, 0, 0, 0, c11 - c12]], dtype=float)


def kernel(law):
    """(function (t, t') -> matrix, 'RELAXATION' | 'CREEP') of a closed-form law."""
    typ = law["type"]
    g = (lambda tp: 1.0 - law["a"] * math.exp(-tp / law["t_a"])) if "a" in law else None
    if typ == "elastic_iso":
        C = 3 * law["k"] * J6 + 2 * law["mu"] * K6
        return (lambda t, tp: C.copy()), "RELAXATION"
    if typ == "maxwell_aging_iso":
        def f(t, tp):
            d = t - tp
            return g(tp) * (3 * (law["k_inf"] + law["k1"] * math.exp(-d / law["tau_k"])) * J6
                            + 2 * (law["mu_inf"] + law["mu1"] * math.exp(-d / law["tau_mu"])) * K6)
        return f, "RELAXATION"
    if typ == "kelvin_aging_iso":
        def f(t, tp):
            d = t - tp
            return ((1 / (3 * law["k0"]) + (1 - math.exp(-d / law["tau_k"])) / (3 * law["k1"] * g(tp))) * J6
                    + (1 / (2 * law["mu0"]) + (1 - math.exp(-d / law["tau_mu"])) / (2 * law["mu1"] * g(tp))) * K6)
        return f, "CREEP"
    if typ == "maxwell_aging_ti":
        R = mandel_rotation(rot3(law["euler_law"]))
        CA = R @ ti_mandel(law["CA"]) @ R.T
        CB = R @ ti_mandel(law["CB"]) @ R.T
        return (lambda t, tp: g(tp) * (CA * math.exp(-(t - tp) / law["tau"]) + CB)), "RELAXATION"
    if typ == "elastic_cond_iso":
        return (lambda t, tp: law["k"] * I3), "RELAXATION"
    if typ == "maxwell_aging_cond_iso":
        return (lambda t, tp: g(tp) * (law["k_inf"] + law["k1"] * math.exp(-(t - tp) / law["tau"])) * I3), \
            "RELAXATION"
    if typ == "kelvin_aging_cond_iso":
        return (lambda t, tp: (1 / law["k0"] + (1 - math.exp(-(t - tp) / law["tau"])) / (law["k1"] * g(tp))) * I3), \
            "CREEP"
    if typ == "maxwell_aging_cond_ti":
        P = rot3(law["euler_law"])
        Kloc = P @ np.diag([law["kt"], law["kt"], law["ka"]]) @ P.T
        return (lambda t, tp: g(tp) * (law["w_inf"] + math.exp(-(t - tp) / law["tau"])) * Kloc), "RELAXATION"
    raise ValueError(f"unknown law type {typ}")


def trapezoidal(f, times, u):
    """Independent re-implementation of Echoes' discretization (visco_law.h:102-127)."""
    n = len(times)
    M = np.zeros((n * u, n * u))
    B = lambda i, j: (slice(i * u, (i + 1) * u), slice(j * u, (j + 1) * u))  # noqa: E731
    M[B(0, 0)] = f(times[0], times[0])
    for i in range(1, n):
        M[B(i, 0)] = 0.5 * (f(times[i], times[0]) - f(times[i], times[1]))
        M[B(i, i)] = 0.5 * (f(times[i], times[i - 1]) + f(times[i], times[i]))
    for i in range(2, n):
        for j in range(1, i):
            M[B(i, j)] = 0.5 * (f(times[i], times[j - 1]) - f(times[i], times[j + 1]))
    return M


def phase_operator(p, times, u, computed):
    """What is handed to Echoes for phase `p`, and the expected relaxation matrix."""
    law = p["law"]
    if law["type"] == "from_case":
        M = np.asarray(computed[law["case"]], dtype=float)
        return M.copy(), M
    f, kind = kernel(law)
    M = trapezoidal(f, times, u)
    expected = M if kind == "RELAXATION" else np.linalg.inv(M)
    vlt = echoes.RELAXATION if kind == "RELAXATION" else echoes.CREEP
    return (f, vlt), expected


def make_rve(c, computed):
    u = 6 if c["order"] == 4 else 3
    prop = "C" if c["order"] == 4 else "K"
    times = np.asarray(c["times"], dtype=float)
    ver = echoes.rve(matrix=c["phases"][0]["name"])
    expected = {}
    # Echoes BORROWS the buffer of a numpy array handed as a discretized operator
    # (interface/python/array_wrapper.h:159-167, neverDeleteData unless the array has a
    # single reference) and stores it by reference (inclusion.h:277-279): the array must
    # outlive the RVE, or the phase silently reads freed memory.
    keepalive = []
    for p in c["phases"]:
        shape = echoes.spherical if p["axes"] is None else \
            echoes.ellipsoidal(np.array(list(p["axes"]) + list(p["euler"]), dtype=float))
        sym = p["symmetrize"]
        kw = {}
        if sym is not None:
            kw["symmetrize"] = [echoes.ISO] if sym["sym"] == "ISO" else \
                [echoes.TI, float(sym["theta"]), float(sym["phi"])]
        given, expected[p["name"]] = phase_operator(p, times, u, computed)
        keepalive.append(given)
        ver[p["name"]] = echoes.ellipsoid(shape=shape, fraction=p["fraction"],
                                          visco_prop={prop: given}, **kw)
    ver.set_param_eshelby(**HILL)
    return ver, prop, u, times, expected, keepalive


def run_case(c, computed):
    timings, eff = [], None
    for _ in range(NREPEAT):
        ver, prop, u, times, expected, keepalive = make_rve(c, computed)
        t0 = time.perf_counter()
        out = echoes.homogenize_visco(rve=ver, scheme=SCHEMES[c["scheme"]], prop=prop,
                                      time_series=times, unitsize=u, maxnb=c["maxnb"],
                                      epsrel=c["epsrel"], verbose=False)
        timings.append(time.perf_counter() - t0)
        out = np.array(out, dtype=float)
        if eff is None:
            eff = out
            phase_mats = {p["name"]: np.array(ver[p["name"]].visco_mat(prop), dtype=float)
                          for p in c["phases"]}
        else:
            assert np.array_equal(out, eff), "non-deterministic result across repetitions"
        del ver, keepalive
    if not np.all(np.isfinite(eff)):
        raise FloatingPointError(f"{c['name']}: non-finite effective operator")
    n = len(times)
    upper = max(float(np.abs(eff[i * u:(i + 1) * u, (i + 1) * u:]).max(initial=0.0)) for i in range(n))
    disc = {name: float(np.abs(phase_mats[name] - expected[name]).max()
                        / np.abs(expected[name]).max()) for name in phase_mats}
    return eff, dict(
        case=c,
        times=list(c["times"]),
        unitsize=u,
        shape=list(eff.shape),
        effective=eff.tolist(),
        phase_mats={k: v.tolist() for k, v in phase_mats.items()},
        discretization_check=disc,
        max_abs_strict_upper_block=upper,
        wall_time_s=min(timings),
        wall_times_s=timings,
        conventions=dict(
            discretization="Echoes trapezoidal (Stieltjes) rule, visco_law.h:102-127: "
                           "M[0,0]=R(t0,t0); M[i,0]=(R(ti,t0)-R(ti,t1))/2; "
                           "M[i,i]=(R(ti,ti-1)+R(ti,ti))/2; M[i,j]=(R(ti,tj-1)-R(ti,tj+1))/2 "
                           "for 0<j<i; zero above the block diagonal",
            creep_laws="CREEP kernels: the trapezoidal matrix of J is built, then inverted "
                       "numerically (visco_law.cpp:22-31, inclusion.h:360-363)",
            layout="time-major: block (i,j) = rows u*i:u*(i+1), cols u*j:u*(j+1) "
                   "(tensor_conversion.h:680-685), sigma_vec = M eps_vec with "
                   "eps_vec = (eps(t0), eps(t1), ...)",
            components="Kelvin-Mandel (11,22,33,sqrt2*23,sqrt2*31,sqrt2*12), "
                       "tensor_conversion.h:295-299 (vectors), 422-445 (4th order)" if u == 6 else "3x3 canonical",
            euler="Echoes (theta, phi, psi), P = Rz(phi) Ry(theta) Rz(psi), tensor_angles.cpp:71-81",
            echoes_module=echoes.__file__,
        ),
    )


def order_with_sources(names):
    """Add the source cases a multiscale case needs, before it."""
    seen, out = set(), []

    def visit(name):
        if name in seen:
            return
        for p in CASES_BY_NAME[name]["phases"]:
            if p["law"]["type"] == "from_case":
                visit(p["law"]["case"])
        seen.add(name)
        out.append(name)

    for nm in names:
        visit(nm)
    return out


def main(argv):
    os.makedirs(RESULTS, exist_ok=True)
    names = order_with_sources(argv if argv else [c["name"] for c in CASES])
    computed = {}
    for name in names:
        c = CASES_BY_NAME[name]
        eff, res = run_case(c, computed)
        computed[name] = eff
        with open(os.path.join(RESULTS, f"{name}.json"), "w") as fh:
            json.dump(res, fh)
        print(f"{name:42s} {c['scheme']:16s} {eff.shape[0]:4d}x{eff.shape[1]:<4d} "
              f"t={res['wall_time_s']:.3g} s  disc={max(res['discretization_check'].values()):.1e}  "
              f"|eff|max={np.abs(eff).max():.4g}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
