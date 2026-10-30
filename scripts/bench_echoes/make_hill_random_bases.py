"""Reference Hill tensors from echoes, for ellipsoids and reference media drawn at random.

Writes `test/Elasticity/echoes_hill_random_bases.json`, read by
`test/Elasticity/test_hill_random_bases.jl`.

Every input is drawn at random: the semi-axes, the orientation of the ellipsoid (ZYZ
Euler angles (θ, φ, ψ), the convention MFH's `Ellipsoid(a, b, c; euler_angles)` shares
with echoes' `ellipsoidal([a, b, c, θ, φ, ψ])`, R = Rz(φ) Ry(θ) Rz(ψ)), and the reference
medium. Structured media (transversely isotropic, orthotropic, cubic) are built in a
material frame and rotated here, as 4th-order tensors, by a rotation drawn independently;
echoes receives their 21 canonical Kelvin-Mandel components, so the reference relies on
no symmetry convention of echoes. Round angles are avoided on purpose: a component read in
the wrong frame can coincide with the right one there.

The shapes cover the sphere, prolate and oblate spheroids (written (r, r, h): their axis
is the third column of R), triaxial ellipsoids, and the limits where closed forms and
quadratures part ways: a nearly spherical ellipsoid, a very flat one, a needle. A
transversely isotropic medium is taken both about a random axis and about the spheroid's
own axis (the coaxial closed form of MFH).

echoes takes and returns canonical components: 6x6 Kelvin-Mandel matrices (11, 22, 33,
23, 13, 12) for elasticity, 3x3 matrices for conduction.

Run with the Python of the environment where echoes is installed:

    python3 -I make_hill_random_bases.py
"""

import json
import pathlib
import sys

import echoes
import numpy as np

SEED = 20261021
EPS = dict(epsabs=1e-12, epsrel=1e-10, maxnb=2_000_000, epsroots=1e-10)
OUT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else \
    pathlib.Path(__file__).resolve().parents[2] / "test" / "Elasticity" / "echoes_hill_random_bases.json"
MANDEL = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))

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
    """Canonical KM of the tensor whose components are `km_material` in the frame R."""
    a = np.einsum("ip,jq,kr,ls,pqrs->ijkl", R, R, R, R, km_to_array(km_material))
    return array_to_km(a)


def km_ti(c1111, c1122, c1133, c3333, c2323):
    """Material KM of a TI tensor about e3."""
    c1212 = (c1111 - c1122) / 2
    km = np.zeros((6, 6))
    km[0, 0] = km[1, 1] = c1111
    km[0, 1] = km[1, 0] = c1122
    km[0, 2] = km[2, 0] = km[1, 2] = km[2, 1] = c1133
    km[2, 2] = c3333
    km[3, 3] = km[4, 4] = 2 * c2323
    km[5, 5] = 2 * c1212
    return km


def km_ortho():
    """Material KM and the nine constants (C11, C22, C33, C12, C13, C23, C44, C55, C66),
    Voigt C44 = C2323 etc., the order of TensND's `TensOrtho`."""
    d = 20 + 20 * rng.random(3)
    o = 3 + 4 * rng.random(3)
    s = 4 + 6 * rng.random(3)
    km = np.diag(np.concatenate([d, 2 * s]))
    km[0, 1] = km[1, 0] = o[0]
    km[0, 2] = km[2, 0] = o[1]
    km[1, 2] = km[2, 1] = o[2]
    return km, [float(x) for x in (*d, *o, *s)]


def km_cubic():
    """Material KM and (C11, C12, C44), the arguments of TensND's `cubic_from_constants`."""
    c11, c12, c44 = 30 + 10 * rng.random(), 8 + 4 * rng.random(), 6 + 6 * rng.random()
    km = np.diag([c11, c11, c11, 2 * c44, 2 * c44, 2 * c44])
    for i, j in ((0, 1), (0, 2), (1, 2)):
        km[i, j] = km[j, i] = c12
    return km, [float(c11), float(c12), float(c44)]


def km_iso():
    k, mu = 20 + 20 * rng.random(), 5 + 10 * rng.random()
    return np.asarray(echoes.stiff_kmu(k, mu).array), [float(k), float(mu)]


def km_aniso():
    a = rng.standard_normal((6, 6))
    return a @ a.T + 20.0 * np.eye(6)


def km_upper(km):
    return np.asarray([float(km[i, j]) for i in range(6) for j in range(i, 6)])


def hill(ax, eu, c):
    return np.asarray(echoes.hill(echoes.ellipsoidal(np.array(ax + eu)), c, algo=echoes.NUMINT3D, **EPS))


def elastic(shape, ax, eu, medium, km, material=None):
    """`material`: how the medium is built in TensND's structured types (its constants
    and the Euler angles of its frame), or None for a general anisotropic medium."""
    c = echoes.tensor(km_upper(km))
    return dict(kind="elastic", shape=shape, medium=medium, semi_axes=ax, euler=eu,
                material=material, C_km=np.asarray(c.array).tolist(), P_km=hill(ax, eu, c).tolist())


def conduction(shape, ax, eu, medium, k, material=None):
    return dict(kind="conduction", shape=shape, medium=medium, semi_axes=ax, euler=eu,
                material=material, K=k.tolist(), P=hill(ax, eu, echoes.tensor(k)).tolist())


def ti_material():
    """Material KM of a TI tensor about e3, and (C1111, C1122, C1133, C3333, C2323), the
    arguments of TensND's `ti_from_constants`."""
    p = [30 + 5 * rng.random(), 8 + 2 * rng.random(), 6 + 2 * rng.random(),
         45 + 10 * rng.random(), 7 + 3 * rng.random()]
    return km_ti(*p), [float(x) for x in p]


def spd3(shift):
    a = rng.standard_normal((3, 3))
    return a @ a.T + shift * np.eye(3)


SHAPES = {
    "sphere": lambda: [1.3] * 3,
    "prolate spheroid": lambda: [0.6 + 0.3 * rng.random()] * 2 + [2.0 + rng.random()],
    "oblate spheroid": lambda: [1.5 + rng.random()] * 2 + [0.2 + 0.3 * rng.random()],
    "triaxial": lambda: [1.5 + 1.5 * rng.random(), 0.6 + 0.6 * rng.random(), 0.1 + 0.4 * rng.random()],
    "nearly spherical": lambda: [1.0 + 2e-4, 1.0, 1.0 - 3e-4],
    "very flat": lambda: [1.4, 0.9, 0.01],
    "needle": lambda: [3.0, 0.06, 0.04],
}

cases = []
for draw in range(2):
    for shape, make in SHAPES.items():
        ax, eu = make(), random_euler()
        R = rot_zyz(*eu)
        km, p = km_iso()
        cases.append(elastic(shape, ax, eu, "isotropic", km, dict(type="iso", params=p)))
        km, p = ti_material(); em = random_euler()
        cases.append(elastic(shape, ax, eu, "transversely isotropic, random axis",
                             rotated_km(km, rot_zyz(*em)), dict(type="TI", params=p, euler=em)))
        if "spheroid" in shape:
            # About the spheroid's own axis, the third column of R.
            km, p = ti_material()
            cases.append(elastic(shape, ax, eu, "transversely isotropic, coaxial",
                                 rotated_km(km, R), dict(type="TI", params=p, euler=eu)))
        km, p = km_ortho(); em = random_euler()
        cases.append(elastic(shape, ax, eu, "orthotropic", rotated_km(km, rot_zyz(*em)),
                             dict(type="ortho", params=p, euler=em)))
        km, p = km_cubic(); em = random_euler()
        cases.append(elastic(shape, ax, eu, "cubic", rotated_km(km, rot_zyz(*em)),
                             dict(type="cubic", params=p, euler=em)))
        cases.append(elastic(shape, ax, eu, "anisotropic", km_aniso()))
        k = 1 + 4 * rng.random()
        cases.append(conduction(shape, ax, eu, "isotropic", k * np.eye(3), dict(type="iso", params=[float(k)])))
        cases.append(conduction(shape, ax, eu, "anisotropic", spd3(1.0)))
        print(f"draw {draw}, {shape}: {len(cases)} cases", flush=True)

OUT.write_text(json.dumps(dict(
    generator="scripts/bench_echoes/make_hill_random_bases.py",
    seed=SEED, tolerances=EPS, algorithm="NUMINT3D",
    note="canonical components; 6x6 Kelvin-Mandel (11, 22, 33, 23, 13, 12) for elasticity",
    cases=cases,
), indent=1) + "\n")
print(f"{len(cases)} cases written to {OUT}")
