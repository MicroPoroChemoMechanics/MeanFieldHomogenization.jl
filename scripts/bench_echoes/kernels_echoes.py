"""The Echoes side of `benchmark.jl`: media, cases and kernel calls.

Hill tensors (elasticity and conduction), crack compliance contributions and Hill
derivatives. The media and the cases are defined here once: `benchmark.jl` imports this
module through PythonCall, builds the same tensors in MeanFieldHomogenization.jl and times
both codes. Run on its own, this module times the Echoes side alone:

    python3 -I scripts/bench_echoes/kernels_echoes.py

with `python3` the interpreter of the environment where echoes is installed.

Tolerances of the numerical algorithms: relative and absolute 1e-6, roots 1e-6, at most
200 000 integrand evaluations (`maxnb`). Echoes takes closed forms whenever the reference is
isotropic, whatever algorithm is declared.
"""

import math
import timeit

import echoes
import numpy as np

_ALGO = {
    'DEFAULT': echoes.DEFAULT,
    'RESIDUES': echoes.RESIDUES,
    'NUMINT3D': echoes.NUMINT3D,
}

# ── Media ────────────────────────────────────────────────────────────────────

E_ISO, NU_ISO = 210.0e3, 0.3
K_ISO_MOD = E_ISO / (3 * (1 - 2 * NU_ISO))        # bulk modulus
MU_ISO = E_ISO / (2 * (1 + NU_ISO))               # shear modulus
C11C, C12C, C44C = 250.0e3, 100.0e3, 80.0e3       # cubic crystal, Voigt notation

C_TRIC_KM = np.array([
    [0.388487, 0.200301, 0.13255, -0.0803777, -0.249878, 0.038079],
    [0.200301, 1.09373, 0.178878, -0.369538, -0.161806, 0.0734051],
    [0.13255, 0.178878, 0.387019, -0.210259, -0.249375, 0.0735958],
    [-0.0803777, -0.369538, -0.210259, 0.655779, 0.123902, -0.227447],
    [-0.249878, -0.161806, -0.249375, 0.123902, 0.442613, -0.120333],
    [0.038079, 0.0734051, 0.0735958, -0.227447, -0.120333, 0.448281],
])

K_COND = 5.0
_R_COND = np.array([[math.cos(math.pi / 4), -math.sin(math.pi / 4), 0.0],
                    [math.sin(math.pi / 4), math.cos(math.pi / 4), 0.0],
                    [0.0, 0.0, 1.0]])
K_ISO_MAT = K_COND * np.eye(3)
K_ANISO_MAT = _R_COND @ np.diag([100.0, 20.0, 50.0]) @ _R_COND.T   # diag(100, 20, 50) turned by pi/4


def cubic_km():
    """Kelvin-Mandel matrix of the cubic crystal (shear terms 2 C44)."""
    km = np.full((3, 3), C12C)
    np.fill_diagonal(km, C11C)
    out = np.zeros((6, 6))
    out[:3, :3] = km
    out[3, 3] = out[4, 4] = out[5, 5] = 2 * C44C
    return out


def stiffness_tensor(km):
    """An Echoes stiffness from its 6x6 Kelvin-Mandel matrix (21 upper-triangle components)."""
    km = np.asarray(km)
    return echoes.tensor(np.asarray([km[i, j] for i in range(6) for j in range(i, 6)], dtype=float))


def conductivity_tensor(mat33):
    """An Echoes 2nd-order tensor from a 3x3 matrix; `echoes.hill` dispatches on the rank."""
    return echoes.tensor(np.asarray(mat33, dtype=float))


MEDIA = {
    "iso": stiffness_tensor(np.asarray(echoes.stiff_kmu(K_ISO_MOD, MU_ISO).array)),
    "cubic": stiffness_tensor(cubic_km()),
    "triclinic": stiffness_tensor(C_TRIC_KM),
    "K iso": conductivity_tensor(K_ISO_MAT),
    "K aniso": conductivity_tensor(K_ANISO_MAT),
}

# ── Kernel calls (None when Echoes raises) ───────────────────────────────────


def py_hill(a, b, c, C, algo, epsroots=1e-6, epsrel=1e-6, epsabs=1e-6,
            maxnb=200000, theta=0.0, phi=0.0, psi=0.0):
    ell = echoes.ellipsoidal(np.array([a, b, c, theta, phi, psi]))
    try:
        H = echoes.hill(ell, C, algo=_ALGO[algo], epsrel=epsrel, epsabs=epsabs,
                        maxnb=maxnb, epsroots=epsroots)
        return np.asarray(H)
    except Exception:
        return None


def py_crack(C, algo, epsroots=1e-6, epsrel=1e-6, epsabs=1e-6, maxnb=200000):
    try:
        H = echoes.crack_compliance(echoes.spheroidal(0.0), C, algo=_ALGO[algo],
                                    epsrel=epsrel, epsabs=epsabs, maxnb=maxnb, epsroots=epsroots)
        return np.asarray(H)
    except Exception:
        return None


def py_crack_ell(C, a, b, c, algo='DEFAULT', epsroots=1e-6, epsrel=1e-6, epsabs=1e-6, maxnb=200000):
    # Flat TRIAXIAL ellipsoid (a >= b >> c): the elliptic-crack case, which discriminates the
    # two H normalization conventions. `spheroidal` cannot express it (one aspect ratio only).
    ell = echoes.ellipsoidal(np.array([a, b, c, 0.0, 0.0, 0.0]))
    try:
        H = echoes.crack_compliance(ell, C, algo=_ALGO[algo], epsrel=epsrel, epsabs=epsabs,
                                    maxnb=maxnb, epsroots=epsroots)
        return np.asarray(H)
    except Exception:
        return None


def py_hill_derivative_iso(a, b, c, k, mu, index, epsrel=1e-6, epsabs=1e-6, maxnb=200000, epsroots=1e-6):
    ell = echoes.ellipsoidal(np.array([a, b, c, 0.0, 0.0, 0.0]))
    C = echoes.stiff_kmu(float(k), float(mu))
    try:
        dP = echoes.hill_derivative(ell, C, index, echoes.ISO, epsrel=epsrel, epsabs=epsabs,
                                    maxnb=maxnb, epsroots=epsroots)
        return np.asarray(dP)
    except Exception:
        return None


# ORTHO parameters in Echoes' order: (C11, C12, C13, C22, C23, C33, mu23, mu13, mu12).
ECHOES_ORTHO_BASE = [C11C, C12C, C12C, C11C, C12C, C11C, C44C, C44C, C44C]
ORTHO_LABELS = ("C11", "C22", "C33", "C12", "C13", "C23", "mu23", "mu13", "mu12")
JL_TO_ECHOES_ORTHO = (0, 3, 5, 1, 2, 4, 6, 7, 8)   # MFH TensOrtho order -> Echoes order


def py_hill_derivative_ortho(a, b, c, params9, index, algo, epsrel=1e-6, epsabs=1e-6,
                             maxnb=200000, epsroots=1e-6):
    ell = echoes.ellipsoidal(np.array([a, b, c, 0.0, 0.0, 0.0]))
    C = echoes.tensor(list(params9), [0.0, 0.0, 0.0])
    try:
        dP = echoes.hill_derivative(ell, C, index, echoes.ORTHO, algo=_ALGO[algo],
                                    epsrel=epsrel, epsabs=epsabs, maxnb=maxnb, epsroots=epsroots)
        return np.asarray(dP)
    except Exception:
        return None


# ── Cases ────────────────────────────────────────────────────────────────────
#
# `algos` are Echoes' algorithms ("DEFAULT" for the closed form of an isotropic reference),
# `jl_method` the MeanFieldHomogenization.jl method it is compared with.

ROT = (math.pi / 4, math.pi / 6, math.pi / 3)


def _case(label, axes, medium, algos, jl_method="auto", euler=(0.0, 0.0, 0.0)):
    return dict(label=label, axes=list(axes), medium=medium, algos=list(algos),
                jl_method=jl_method, euler=list(euler))


HILL_CASES = [
    _case("Sphere a=b=c=1 / ISO", (1.0, 1.0, 1.0), "iso", ["DEFAULT"]),
    _case("Prolate a=5,b=c=1 / ISO", (5.0, 1.0, 1.0), "iso", ["DEFAULT"]),
    _case("Oblate a=b=5,c=1 / ISO", (5.0, 5.0, 1.0), "iso", ["DEFAULT"]),
    _case("Prolate a=3,b=c=1 / cubic", (3.0, 1.0, 1.0), "cubic", ["NUMINT3D"], "residues"),
    _case("Prolate a=3,b=c=1 / cubic (Julia :decuhr)", (3.0, 1.0, 1.0), "cubic", ["NUMINT3D"], "decuhr"),
    _case("Prolate a=3,b=c=1 / cubic (Julia :nestedquadgk)", (3.0, 1.0, 1.0), "cubic", ["NUMINT3D"], "nestedquadgk"),
    _case("Triaxial a=2,b=1,c=0.5 / triclinic", (2.0, 1.0, 0.5), "triclinic", ["RESIDUES", "NUMINT3D"], "residues"),
    _case("Triaxial a=2,b=1,c=0.5 / triclinic (Julia :decuhr)", (2.0, 1.0, 0.5), "triclinic", ["RESIDUES", "NUMINT3D"], "decuhr"),
    _case("Triaxial a=2,b=1,c=0.5 / triclinic (Julia :nestedquadgk)", (2.0, 1.0, 0.5), "triclinic", ["RESIDUES", "NUMINT3D"], "nestedquadgk"),
    _case("Rotated prolate a=5,b=c=1 / ISO  (pi/4,pi/6,pi/3)", (5.0, 1.0, 1.0), "iso", ["DEFAULT"], euler=ROT),
    _case("Rotated triaxial a=2,b=1,c=0.5 / triclinic (pi/4,pi/6,pi/3)", (2.0, 1.0, 0.5), "triclinic", ["RESIDUES", "NUMINT3D"], "residues", ROT),
    _case("Rotated triaxial a=2,b=1,c=0.5 / triclinic :decuhr (pi/4,pi/6,pi/3)", (2.0, 1.0, 0.5), "triclinic", ["RESIDUES", "NUMINT3D"], "decuhr", ROT),
]

CRACK_CASES = [
    dict(label="Penny / ISO", medium="iso", algos=["DEFAULT"], jl_method="auto"),
    dict(label="Penny / cubic", medium="cubic", algos=["NUMINT3D"], jl_method="residues"),
    dict(label="Penny / triclinic", medium="triclinic", algos=["RESIDUES", "NUMINT3D"], jl_method="residues"),
]
ELLIPTIC_CRACK_ETAS = [1.0, 0.7, 0.5, 0.3, 0.1]
ELLIPTIC_CRACK_FLATNESS = 1e-5

COND_CASES = [
    _case("Sphere a=b=c=1 / K iso", (1.0, 1.0, 1.0), "K iso", ["DEFAULT"]),
    _case("Prolate a=5,b=c=1 / K iso", (5.0, 1.0, 1.0), "K iso", ["DEFAULT"]),
    _case("Oblate a=b=5,c=1 / K iso", (5.0, 5.0, 1.0), "K iso", ["DEFAULT"]),
    _case("Triaxial a=2,b=1,c=0.5 / K aniso", (2.0, 1.0, 0.5), "K aniso", ["RESIDUES", "NUMINT3D"]),
    _case("Rotated triaxial a=2,b=1,c=0.5 / K aniso (pi/4,pi/6,pi/3)", (2.0, 1.0, 0.5), "K aniso", ["RESIDUES", "NUMINT3D"], euler=ROT),
]

DERIV_ISO_CASES = [
    dict(label="Sphere a=b=c=1 / ISO", axes=[1.0, 1.0, 1.0]),
    dict(label="Prolate a=5,b=c=1 / ISO", axes=[5.0, 1.0, 1.0]),
    dict(label="Oblate a=b=5,c=1 / ISO", axes=[5.0, 5.0, 1.0]),
]
DERIV_ORTHO_CASES = [
    dict(label="Prolate a=3,b=c=1 / cubic-as-ORTHO", axes=[3.0, 1.0, 1.0], algos=["NUMINT3D"]),
]

# ── Case runners: what the Julia side times, so that no conversion enters it ──


def run_hill(i, algo):
    c = HILL_CASES[i]
    return py_hill(*c["axes"], MEDIA[c["medium"]], algo, theta=c["euler"][0], phi=c["euler"][1], psi=c["euler"][2])


def run_crack(i, algo):
    return py_crack(MEDIA[CRACK_CASES[i]["medium"]], algo)


def run_cond(i, algo):
    c = COND_CASES[i]
    return py_hill(*c["axes"], MEDIA[c["medium"]], algo, theta=c["euler"][0], phi=c["euler"][1], psi=c["euler"][2])


def run_deriv_iso(i, index):
    return py_hill_derivative_iso(*DERIV_ISO_CASES[i]["axes"], K_ISO_MOD, MU_ISO, index)


def run_deriv_ortho(i, echoes_index, algo):
    return py_hill_derivative_ortho(*DERIV_ORTHO_CASES[i]["axes"], ECHOES_ORTHO_BASE, echoes_index, algo)


def best_time(f, repeat=20):
    """The minimum over `repeat` single runs, in seconds (BenchmarkTools' minimum on the Julia side)."""
    return min(timeit.repeat(f, number=1, repeat=repeat))


if __name__ == "__main__":
    def show(section, label, algo, f):
        t = best_time(f)
        print(f"{section:6s} {label:72s} {algo:9s} {t * 1e3:10.4f} ms", flush=True)

    for i, c in enumerate(HILL_CASES):
        for a in c["algos"]:
            show("hill", c["label"], a, lambda: run_hill(i, a))
    for i, c in enumerate(CRACK_CASES):
        for a in c["algos"]:
            show("crack", c["label"], a, lambda: run_crack(i, a))
    for i, c in enumerate(COND_CASES):
        for a in c["algos"]:
            show("cond", c["label"], a, lambda: run_cond(i, a))
    for i, c in enumerate(DERIV_ISO_CASES):
        for idx, name in ((0, "kappa"), (1, "eta")):
            show("dP", c["label"] + " | " + name, "ISO", lambda: run_deriv_iso(i, idx))
    for i, c in enumerate(DERIV_ORTHO_CASES):
        for k, name in enumerate(ORTHO_LABELS):
            for a in c["algos"]:
                show("dP", c["label"] + " | " + name, a, lambda: run_deriv_ortho(i, JL_TO_ECHOES_ORTHO[k], a))
