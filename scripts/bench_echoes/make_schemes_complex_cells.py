"""Every homogenization scheme of echoes on complex cells, in elasticity and in conduction.

Writes `test/Schemes/echoes_schemes_complex_cells.json`, read by
`test/Schemes/test_schemes_complex_cells.jl`.

The companion of `make_schemes_random_bases.py`, which covers the dilute, dilute dual,
Mori-Tanaka and both self-consistent schemes on two families of inclusions. Here every
scheme of echoes (Voigt, Reuss, dilute, dilute dual, Mori-Tanaka, Maxwell, Ponte
Castañeda-Willis, self-consistent, asymmetric self-consistent, differential) runs on
fixed cells, so that a case can be named and reproduced: a matrix, isotropic or
transversely isotropic about a tilted axis, a family of stiff anisotropic triaxial
ellipsoids, a family of soft prolate spheroids, and two crack families, an ellipse and
a penny, each with its own orientation off the axes. Maxwell and Ponte Castañeda-Willis
take a tilted oblate distribution ellipsoid (the shape of the RVE in echoes).

Conventions:
- orientations are ZYZ Euler angles, shared by `Ellipsoid(a, b, c; euler_angles)`;
- a crack family is an ellipsoid of semi-axes (a, b, 1e-12 b) of zero property in
  echoes, of density N a^3 there and N a b^2 in MeanFieldHomogenization; Voigt and
  Reuss pass cracks over (no volume), and the Reuss reference is computed without them,
  echoes forming zero times an infinite compliance;
- echoes integrates the differential scheme by `maxnb` explicit Euler steps: its result
  here is the Richardson extrapolation from `DIFF_STEPS` and `2 DIFF_STEPS` steps,
  second order, and only on the isotropic matrix (an anisotropic one would need a
  numerical Hill tensor at every step).

Run with the Python of the environment where echoes is installed:

    python3 -I make_schemes_complex_cells.py
"""

import json
import pathlib
import sys

import echoes
import numpy as np

HILL = dict(algo=echoes.NUMINT3D, epsabs=1e-12, epsrel=1e-10, maxnb=2_000_000, epsroots=1e-10)
FIXED_POINT = dict(epsrel=1e-12, maxnb=500)
DIFF_STEPS = 400
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
    pathlib.Path(__file__).resolve().parents[2] / "test" / "Schemes" / "echoes_schemes_complex_cells.json"
MANDEL = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
SCHEMES = {"voigt": echoes.VOIGT, "reuss": echoes.REUSS, "dilute": echoes.DIL,
           "dilute_dual": echoes.DILD, "mori_tanaka": echoes.MT, "maxwell": echoes.MAX,
           "pcw": echoes.PCW, "self_consistent": echoes.SC,
           "asymmetric_self_consistent": echoes.ASC, "differential": echoes.DIFF}
ROT = [0.4, 0.9, 0.2]
ROT2 = [1.2, 0.3, 0.8]
TI_AXIS = [0.7, 0.4, 0.0]
DISTRIBUTION = dict(semi_axes=[1.0, 1.0, 0.6], euler=[0.7, 0.4, 0.0])


def rot_zyz(th, ph, ps):
    def rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    c, s = np.cos(th), np.sin(th)
    ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return rz(ph) @ ry @ rz(ps)


def km_to_array(km):
    a = np.zeros((3, 3, 3, 3))
    for I, (i, j) in enumerate(MANDEL):
        for J, (k, l) in enumerate(MANDEL):
            v = km[I, J] / ((1.0 if I < 3 else np.sqrt(2)) * (1.0 if J < 3 else np.sqrt(2)))
            for (p, q) in ((i, j), (j, i)):
                for (r, s) in ((k, l), (l, k)):
                    a[p, q, r, s] = v
    return a


def array_to_km(a):
    km = np.zeros((6, 6))
    for I, (i, j) in enumerate(MANDEL):
        for J, (k, l) in enumerate(MANDEL):
            km[I, J] = (1.0 if I < 3 else np.sqrt(2)) * (1.0 if J < 3 else np.sqrt(2)) * a[i, j, k, l]
    return km


def rotated_km(km, R):
    return array_to_km(np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, R, R, km_to_array(km)))


def km_upper(km):
    return np.asarray([float(km[i, j]) for i in range(6) for j in range(i, 6)])


def km_iso(k, mu):
    return np.asarray(echoes.stiff_kmu(k, mu).array)


def km_ti(c1111, c1122, c1133, c3333, c2323):
    km = np.zeros((6, 6))
    km[0, 0] = km[1, 1] = c1111
    km[0, 1] = km[1, 0] = c1122
    km[0, 2] = km[2, 0] = km[1, 2] = km[2, 1] = c1133
    km[2, 2] = c3333
    km[3, 3] = km[4, 4] = 2 * c2323
    km[5, 5] = c1111 - c1122
    return km


# A fixed anisotropic stiffness and conductivity, symmetric positive definite.
_A6 = np.array([[1.0, 0.3, -0.2, 0.1, 0.0, 0.2], [0.3, 1.2, 0.1, -0.1, 0.2, 0.0],
                [-0.2, 0.1, 0.9, 0.0, 0.1, -0.1], [0.1, -0.1, 0.0, 0.8, 0.2, 0.1],
                [0.0, 0.2, 0.1, 0.2, 1.1, -0.2], [0.2, 0.0, -0.1, 0.1, -0.2, 1.0]])
_A3 = np.array([[1.0, 0.4, -0.3], [0.4, 1.3, 0.2], [-0.3, 0.2, 0.8]])
KM_STIFF = 40.0 * (_A6 @ _A6.T + 0.5 * np.eye(6))
K_STIFF = 8.0 * (_A3 @ _A3.T + 0.5 * np.eye(3))

MATRICES = {
    "isotropic": dict(C_km=km_iso(10.0, 6.0), K=2.0 * np.eye(3)),
    "transversely isotropic": dict(
        C_km=rotated_km(km_ti(30.0, 8.0, 6.0, 45.0, 7.0), rot_zyz(*TI_AXIS)),
        K=rot_zyz(*TI_AXIS) @ np.diag([1.5, 1.5, 3.5]) @ rot_zyz(*TI_AXIS).T),
}
INCLUSIONS = [
    dict(name="A", semi_axes=[2.0, 1.0, 0.4], euler=ROT, fraction=0.12,
         C_km=KM_STIFF, K=K_STIFF),
    dict(name="B", semi_axes=[1.0, 1.0, 5.0], euler=ROT2, fraction=0.08,
         C_km=km_iso(2.0, 1.0), K=0.1 * np.eye(3)),
]
CRACKS = [
    dict(name="C1", semi_axes=[1.0, 0.4], euler=ROT, density=0.05),
    dict(name="C2", semi_axes=[1.0, 1.0], euler=ROT2, density=0.04),
]


def medium(prop, value):
    return echoes.tensor(km_upper(value) if prop == "C" else value)


def rve(matrix, prop, cracks):
    ver = echoes.rve(matrix="M")
    f_m = 1.0 - sum(i["fraction"] for i in INCLUSIONS)
    key = "C_km" if prop == "C" else "K"
    ver["M"] = echoes.ellipsoid(shape=echoes.spherical, fraction=f_m, prop={prop: medium(prop, matrix[key])})
    for i in INCLUSIONS:
        ver[i["name"]] = echoes.ellipsoid(shape=echoes.ellipsoidal(np.array(i["semi_axes"] + i["euler"])),
                                          fraction=i["fraction"], prop={prop: medium(prop, i[key])})
    zero = np.zeros((6, 6)) if prop == "C" else np.zeros((3, 3))
    for c in cracks:
        a, b = c["semi_axes"]
        ver[c["name"]] = echoes.crack(shape=echoes.ellipsoidal(np.array([a, b, 1e-12 * b] + c["euler"])),
                                      density=c["density"], prop={prop: medium(prop, zero)})
    ver.shape = echoes.ellipsoidal(np.array(DISTRIBUTION["semi_axes"] + DISTRIBUTION["euler"]))
    ver.set_param_eshelby(**HILL)
    return ver


def homogenize(matrix, prop, scheme, maxnb=FIXED_POINT["maxnb"]):
    # A crack has no volume, so the Reuss bound passes it over, as Voigt does; echoes
    # forms its zero fraction times its infinite compliance instead, hence a cell
    # without its cracks for that scheme alone.
    cracks = [] if scheme == "reuss" else CRACKS
    out = echoes.homogenize(prop=prop, rve=rve(matrix, prop, cracks), scheme=SCHEMES[scheme], verbose=False,
                            epsrel=FIXED_POINT["epsrel"], maxnb=maxnb)
    return np.asarray(out.array)


def write(cases, path):
    path.write_text(json.dumps(dict(
        generator="scripts/bench_echoes/make_schemes_complex_cells.py",
        hill_tolerances={k: v for k, v in HILL.items() if k != "algo"},
        fixed_point=FIXED_POINT, differential_steps=DIFF_STEPS,
        matrices={n: {k: np.asarray(v).tolist() for k, v in m.items()} for n, m in MATRICES.items()},
        inclusions=[{k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in i.items()}
                    for i in INCLUSIONS],
        cracks=CRACKS, distribution=DISTRIBUTION, cases=cases)))


# The numerical Hill tensors of the anisotropic matrix make a run long: each case is
# saved to a partial file as soon as it is done, and a rerun resumes from it.
PARTIAL = OUT.with_suffix(".partial.json")
cases = json.loads(PARTIAL.read_text())["cases"] if PARTIAL.exists() else []
done = {(c["matrix"], c["prop"], c["scheme"]) for c in cases}
for matrix_name, matrix in MATRICES.items():
    for prop in ("C", "K"):
        for scheme in SCHEMES:
            if (matrix_name, prop, scheme) in done or (scheme == "differential" and matrix_name != "isotropic"):
                continue
            if scheme == "differential":
                eff = 2 * homogenize(matrix, prop, scheme, 2 * DIFF_STEPS) - homogenize(matrix, prop, scheme, DIFF_STEPS)
            else:
                eff = homogenize(matrix, prop, scheme)
            cases.append(dict(prop=prop, matrix=matrix_name, scheme=scheme, effective=eff.tolist()))
            write(cases, PARTIAL)
            print(f"{matrix_name:24s} {prop} {scheme:28s} done", flush=True)

write(cases, OUT)
PARTIAL.unlink()
print(f"wrote {OUT} ({len(cases)} cases)")
