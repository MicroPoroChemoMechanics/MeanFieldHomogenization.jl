"""Reference homogenization schemes from echoes, with phases drawn at random in every basis.

Writes `test/Schemes/echoes_schemes_random_bases.json`, read by
`test/Schemes/test_schemes_random_bases.jl`.

The companion of `make_hill_random_bases.py`, one level up: where that file checks the
Hill tensor of one ellipsoid, this one checks what the schemes build from several. Each
RVE has a spherical matrix and two families of ellipsoidal inclusions, each with its own
semi-axes, its own orientation (ZYZ Euler angles, R = Rz(φ) Ry(θ) Rz(ψ), the convention
MFH's `Ellipsoid(a, b, c; euler_angles)` shares with echoes' `ellipsoidal`) and its own
medium, written in a frame drawn independently. The matrix is isotropic, transversely
isotropic about a random axis or fully anisotropic. Every medium is handed to echoes as
canonical components (21 Kelvin-Mandel ones in elasticity, a 3x3 matrix in conduction),
so the reference relies on no symmetry convention of echoes.

Schemes: dilute, dilute dual, Mori-Tanaka, self-consistent and asymmetric self-consistent,
each with the inclusions as they are and, where the reference allows it, orientation-
averaged: isotropically (symmetrize=ISO) or about an axis (symmetrize=TI with its
angles). An average is generated only when the Eshelby reference has its symmetry
(isotropic for ISO, transversely isotropic about the same axis, or isotropic, for TI):
only then do averaging the concentrations, as echoes does, and averaging the
contributions, as MFH does, define the same scheme.
The differential scheme is left out: echoes integrates it in explicit first-order steps,
which on an anisotropic matrix would need thousands of numerical Hill tensors to converge
(see `benchmark_porous.jl`); Maxwell and Ponte Castañeda-Willis need a distribution
shape, which the companion porous benchmark covers.

echoes returns canonical components: a 6x6 Kelvin-Mandel matrix (11, 22, 33, 23, 13, 12)
for elasticity, a 3x3 matrix for conduction.

Run with the Python of the environment where echoes is installed:

    ~/miniconda3/envs/fenicsx-env/bin/python3 -I make_schemes_random_bases.py
"""

import json
import pathlib
import sys

import echoes
import numpy as np

SEED = 20261023
HILL = dict(algo=echoes.NUMINT3D, epsabs=1e-12, epsrel=1e-10, maxnb=2_000_000, epsroots=1e-10)
FIXED_POINT = dict(epsrel=1e-12, maxnb=500)
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
    pathlib.Path(__file__).resolve().parents[2] / "test" / "Schemes" / "echoes_schemes_random_bases.json"
MANDEL = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
SCHEMES = {"dilute": echoes.DIL, "dilute_dual": echoes.DILD, "mori_tanaka": echoes.MT,
           "self_consistent": echoes.SC, "asymmetric_self_consistent": echoes.ASC}

rng = np.random.default_rng(SEED)


def rot_zyz(th, ph, ps):
    def rz(a):
        c, s = np.cos(a), np.sin(a)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    c, s = np.cos(th), np.sin(th)
    ry = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    return rz(ph) @ ry @ rz(ps)


def random_euler():
    return [float(np.arccos(2 * rng.random() - 1)), float(2 * np.pi * rng.random()),
            float(2 * np.pi * rng.random())]


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


def rotated_km(km_material, R):
    a = np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, R, R, km_to_array(km_material))
    return array_to_km(a)


def km_upper(km):
    return np.asarray([float(km[i, j]) for i in range(6) for j in range(i, 6)])


def km_iso(k, mu):
    return np.asarray(echoes.stiff_kmu(k, mu).array)


def km_ti_material():
    c1111, c1122, c1133 = 30 + 5 * rng.random(), 8 + 2 * rng.random(), 6 + 2 * rng.random()
    c3333, c2323 = 45 + 10 * rng.random(), 7 + 3 * rng.random()
    km = np.zeros((6, 6))
    km[0, 0] = km[1, 1] = c1111
    km[0, 1] = km[1, 0] = c1122
    km[0, 2] = km[2, 0] = km[1, 2] = km[2, 1] = c1133
    km[2, 2] = c3333
    km[3, 3] = km[4, 4] = 2 * c2323
    km[5, 5] = c1111 - c1122
    return km


def km_aniso(scale):
    a = rng.standard_normal((6, 6))
    return scale * (a @ a.T / 6 + 3.0 * np.eye(6))


def spd3(scale):
    a = rng.standard_normal((3, 3))
    return scale * (a @ a.T / 3 + np.eye(3))


def ti_matrix():
    eu = [*random_euler()[:2], 0.0]
    return rotated_km(km_ti_material(), rot_zyz(*eu)), eu


# Each matrix comes with the orientation averages its symmetry allows: an average
# over a group the Eshelby reference is not invariant under is not defined (the
# projection of the concentrations and that of the contributions then differ),
# so it is not generated. `ti` averages about the axis of `ti_euler`.
MATRICES = {
    "isotropic": (lambda: (km_iso(20 + 10 * rng.random(), 8 + 4 * rng.random()),
                           [*random_euler()[:2], 0.0]), ("none", "iso", "ti")),
    "transversely isotropic": (ti_matrix, ("none", "ti")),
    "anisotropic": (lambda: (km_aniso(10.0), None), ("none",)),
}

SHAPES = [
    lambda: [2.0 + rng.random(), 1.0, 0.4 + 0.3 * rng.random()],     # triaxial
    lambda: [0.7 + 0.2 * rng.random()] * 2 + [2.5 + rng.random()],    # prolate
    lambda: [1.8 + rng.random()] * 2 + [0.3 + 0.2 * rng.random()],    # oblate
]


def homogenize(phases, prop, scheme, average, ti_euler):
    f_m = 1.0 - sum(f for n, (_, _, f) in phases.items() if n != "M")
    sym = {"none": None, "iso": [echoes.ISO], "ti": [echoes.TI, *(ti_euler or [])]}[average]
    ver = echoes.rve(matrix="M")
    for name, (shape, value, frac) in phases.items():
        kw = dict(symmetrize=sym) if (sym and name != "M") else {}
        ver[name] = echoes.ellipsoid(shape=shape, fraction=f_m if name == "M" else frac,
                                     prop={prop: value}, **kw)
    ver.set_param_eshelby(**HILL)
    out = echoes.homogenize(prop=prop, rve=ver, scheme=SCHEMES[scheme], verbose=False,
                            **FIXED_POINT)
    return np.asarray(out.array).tolist()


def conductivity(matrix, ti_euler):
    if matrix == "isotropic":
        return 2.0 * np.eye(3)
    if matrix == "transversely isotropic":
        R = rot_zyz(*ti_euler)
        a, b = 1.5 + rng.random(), 3.0 + rng.random()
        return R @ np.diag([a, a, b]) @ R.T
    return spd3(2.0)


cases = []
for draw in range(2):
    for matrix, (make, averages) in MATRICES.items():
        # Elasticity: the matrix and two families, a stiff one and a soft one.
        incl = []
        for family, scale in (("A", 4.0), ("B", 0.2)):
            ax, eu = SHAPES[(draw + len(incl)) % 3](), random_euler()
            km = km_aniso(10.0 * scale)
            incl.append(dict(name=family, semi_axes=ax, euler=eu, C_km=km.tolist(),
                             fraction=0.12 if family == "A" else 0.08))
        km_m, ti_euler = make()
        phases = {"M": (echoes.spherical, echoes.tensor(km_upper(km_m)), None)}
        for i in incl:
            phases[i["name"]] = (echoes.ellipsoidal(np.array(i["semi_axes"] + i["euler"])),
                                 echoes.tensor(km_upper(np.asarray(i["C_km"]))), i["fraction"])
        for scheme in SCHEMES:
            for average in averages:
                cases.append(dict(kind="elastic", matrix=matrix, C_m_km=km_m.tolist(),
                                  inclusions=incl, scheme=scheme, average=average,
                                  ti_euler=ti_euler,
                                  C_eff_km=homogenize(phases, "C", scheme, average, ti_euler)))
        # Conduction: the same shapes and orientations, conductivities drawn afresh,
        # the matrix with the symmetry (and axis) of the elastic one.
        k_m = conductivity(matrix, ti_euler)
        incl_k = [dict(i, K=spd3(8.0 if i["name"] == "A" else 0.1).tolist()) for i in incl]
        phases = {"M": (echoes.spherical, echoes.tensor(k_m), None)}
        for i in incl_k:
            phases[i["name"]] = (echoes.ellipsoidal(np.array(i["semi_axes"] + i["euler"])),
                                 echoes.tensor(np.asarray(i["K"])), i["fraction"])
        for scheme in SCHEMES:
            for average in averages:
                cases.append(dict(kind="conduction", matrix=matrix, K_m=k_m.tolist(),
                                  inclusions=incl_k, scheme=scheme, average=average,
                                  ti_euler=ti_euler,
                                  K_eff=homogenize(phases, "K", scheme, average, ti_euler)))
        print(f"draw {draw}, {matrix} matrix: {len(cases)} cases", flush=True)

OUT.write_text(json.dumps(dict(
    generator="scripts/bench_echoes/make_schemes_random_bases.py", seed=SEED,
    hill_tolerances={k: v for k, v in HILL.items() if k != "algo"},
    fixed_point=FIXED_POINT, cases=cases)))
print(f"wrote {OUT} ({len(cases)} cases)")
