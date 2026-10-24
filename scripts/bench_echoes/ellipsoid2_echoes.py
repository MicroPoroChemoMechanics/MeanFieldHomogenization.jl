"""Echoes side of `bench_ellipsoid2.jl`: the ellipsoid-2 aging-creep setup.

Imported from Julia with PythonCall (`pyimport("ellipsoid2_echoes")`), the
directory of this file being put on `sys.path` first.
"""
from echoes import (rve, ellipsoid, ISO, CREEP, MT, DIFF, SC, ASC, MAX, PCW,
                     spherical, spheroidal, stiff_Enu, tensor, tId4,
                     visco_law, homogenize_visco, homogenize)
import numpy as np

Cs = stiff_Enu(1., 0.2); ks, mus = Cs.kmu
def Js_py(t, tp):
    return ((0.5*np.exp(-tp/20.)+0.5)/(3*ks) + 1e-1/ks*np.log(1.+(t-tp)/2.)) * tens_J4_arr() \
         + ((0.5*np.exp(-tp/20.)+0.5)/(2*mus) + 1e-1/mus*np.log(1.+(t-tp)/1.)) * tens_K4_arr()
def tens_J4_arr():
    from echoes import J4
    return J4
def tens_K4_arr():
    from echoes import K4
    return K4

def make_Js_py():
    from echoes import J4, K4
    def Js_py(t, tp):
        return ((0.5*np.exp(-tp/20.)+0.5)/(3*ks) + 1e-1/ks*np.log(1.+(t-tp)/2.)) * J4 \
             + ((0.5*np.exp(-tp/20.)+0.5)/(2*mus) + 1e-1/mus*np.log(1.+(t-tp)/1.)) * K4
    return Js_py

Ci = 1.e+6 * tId4
def Ji_py(t, tp):
    return np.linalg.inv(Ci.array)

def trapz_creep(T):
    Js_py = make_Js_py()
    return np.asarray(visco_law(Js_py, CREEP).creep_mat(T))

def trapz_relax(T):
    Js_py = make_Js_py()
    return np.asarray(visco_law(Js_py, CREEP).relaxation_mat(T))

_SCH = {"MT": MT, "DIFF": DIFF, "SC": SC, "ASC": ASC, "MAX": MAX, "PCW": PCW}

def homogenize_py(omega, T, sch_name, frac):
    Js_py = make_Js_py()
    ver = rve(matrix="SOLID")
    sh_inc = spherical if omega == 1.0 else spheroidal(omega)
    ver["SOLID"] = ellipsoid(shape=spherical, symmetrize=[ISO],
                              prop={"C": Cs},
                              visco_prop={"C": (Js_py, CREEP)},
                              fraction=1.0 - frac)
    ver["PORE"]  = ellipsoid(shape=sh_inc, symmetrize=[ISO],
                              prop={"C": Ci},
                              visco_prop={"C": (Ji_py, CREEP)},
                              fraction=frac)
    # Python writes set_visco_mat with the discrete relaxation matrix.
    ver.set_visco_mat("C", visco_law(Js_py, CREEP).relaxation_mat(np.asarray(T)))
    def run(nb):
        return np.asarray(homogenize_visco(prop="C", rve=ver,
                                           time_series=np.asarray(T),
                                           scheme=_SCH[sch_name],
                                           epsrel=1.e-10, maxnb=nb, verbose=False))
    if sch_name == "DIFF":
        # The differential scheme in `maxnb` explicit steps is first order:
        # the reference is the Richardson extrapolation of 1000 and 10000.
        V1, V2 = run(1000), run(10000)
        return V2 + (V2 - V1) / 9
    return run(1000)
