"""Echoes side of `15_cracks_iso_interface.jl`: cracks with an interface, in an isotropic matrix.

Imported from Julia with PythonCall (`pyimport("echoes_cracks_iso_interface")`),
the directory of this file being put on `sys.path` first.
"""
from numpy import *
from echoes import *

_SCHEMES = {"MT": MT, "SC": SC, "ASC": ASC, "DIFF": DIFF, "PCW": PCW, "MAX": MAX}

def run_echoes(dens, scheme_name, omega, ko, kt, kn, E_s, nu_s):
    sch = _SCHEMES[scheme_name]
    Cs = stiff_Enu(E_s, nu_s)
    ver = rve(matrix="SOLID")
    ver["SOLID"] = ellipsoid(shape=spheroidal(1),
                              prop={"C":Cs, "K":ko*tId2})
    ver["CRACK"] = crack(shape=spheroidal(omega), symmetrize=[ISO],
                          prop={"C":tZ4, "K":tensor(kt,kt,kn)})
    ver["CRACK"].density = float(dens)
    try:
        C = homogenize(prop="C", rve=ver, scheme=sch, maxnb=200,
                       epsrel=1.e-5, select_best=True, verbose=False)
        k_eff = float(C.k); mu_eff = float(C.mu)
    except Exception:
        k_eff = float('nan'); mu_eff = float('nan')
    try:
        K = homogenize(prop="K", rve=ver, scheme=sch, maxnb=200,
                       epsrel=1.e-5, select_best=True, verbose=False)
        K_eff = float(K.param[0])
    except Exception:
        K_eff = float('nan')
    return k_eff, mu_eff, K_eff
