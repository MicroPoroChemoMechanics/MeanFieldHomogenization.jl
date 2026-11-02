"""
make_doubled_instants_reference.py — Echoes references on time grids that hold an
instant twice, for `test/Viscoelasticity/test_alv_jumps_evaluation.jl`.

An aging Maxwell matrix with spherical elastic inclusions (Mori-Tanaka), and the same
inclusions precipitated at t = 1 (`tp >= 1`), on a strict grid, a grid with 1 and 2
doubled, and a grid with its first instant doubled. Written: the isotropic parameters
(3k, 2mu) of the law of the phase alone (a one-phase Voigt cell) and of the estimate.

Echoes evaluates a law at the instants themselves, both nodes of a doubled instant
included; MeanFieldHomogenization reads the first node as the left limit. The two agree
wherever the law is continuous at the doubled instant, and, for the precipitation,
when MFH places the switch just before 1.

Run from this directory with the interpreter of the environment where `echoes` imports:
    python3 make_doubled_instants_reference.py
"""
from numpy import *
from echoes import *
import json, os

k, mu, eta, gam = 10.0, 6.0, 3.0, 1.0
Ci = stiff_kmu(30.0, 20.0)

def aging(t, tp):
    return 3*k*(1 + 0.5*exp(-tp))*exp(-(t - tp)/eta)*J4 + 2*mu*exp(-(t - tp)/gam)*K4

def precipitated_at_1(t, tp):
    return Ci.array if tp >= 1.0 else 0.0*J4

GRIDS = {"strict": [0.5, 0.7, 1.0, 1.5, 2.0, 4.0],
         "doubled": [0.5, 0.7, 1.0, 1.0, 1.5, 2.0, 2.0, 4.0],
         "first_doubled": [0.5, 0.5, 1.0, 1.0, 2.0, 4.0, 4.0]}

def iso(M):
    a, b = visco_paramsym(array(M), ISO)
    return array(a).tolist(), array(b).tolist()

out = {}
for gname, T in GRIDS.items():
    T = array(T)
    for lname, inc in (("aging", None), ("step", precipitated_at_1)):
        one = rve(matrix="ONE")
        one["ONE"] = ellipsoid(spherical, fraction=1.0, prop={"C": Ci},
                               visco_prop={"C": (aging if inc is None else inc, RELAXATION)})
        L = homogenize_visco(prop="C", rve=one, time_series=T, scheme=VOIGT, maxnb=10, epsrel=1e-10, verbose=False)
        ver = rve(matrix="MAT")
        ver["MAT"] = ellipsoid(spherical, fraction=0.7, prop={"C": Ci}, visco_prop={"C": (aging, RELAXATION)})
        if inc is None:
            ver["INC"] = ellipsoid(spherical, fraction=0.3, prop={"C": Ci})
        else:
            ver["INC"] = ellipsoid(spherical, fraction=0.3, prop={"C": Ci}, visco_prop={"C": (inc, RELAXATION)})
        H = homogenize_visco(prop="C", rve=ver, time_series=T, scheme=MT, maxnb=10, epsrel=1e-10, verbose=False)
        (a, b), (ah, bh) = iso(L), iso(H)
        out[gname + "/" + lname] = {"times": T.tolist(), "alpha": a, "beta": b, "alpha_MT": ah, "beta_MT": bh}

path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "test", "Viscoelasticity",
                    "echoes_doubled_instants.json")
with open(path, "w") as f:
    json.dump({"source": "scripts/bench_echoes/alv_harness/make_doubled_instants_reference.py", "cases": out}, f)
print("wrote", os.path.relpath(path))
