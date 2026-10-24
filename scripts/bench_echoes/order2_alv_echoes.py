"""Echoes side of `bench_order2_alv.jl`: the order-2 Maxwell aging-creep setup.

Imported from Julia with PythonCall (`pyimport("order2_alv_echoes")`), the
directory of this file being put on `sys.path` first.
"""
import numpy as np
from echoes import rve, ellipsoid, ISO, CREEP, MT, MAX, DIL, REUSS, VOIGT, homogenize_visco


def py_Rs(t, tp):
    r0, r1, r2 = 1., 2., 3.
    tau1, tau2 = 2., 10.
    je = r0 + np.exp(-(tp/30.)**2)*(r1*(1.-np.exp(-(t-tp)/tau1))+r2*(1.-np.exp(-(t-tp)/tau2)))
    return je*np.eye(3)

def py_Ri(t, tp):
    r0, r1, r2 = 0.2, 0.3, 1.2
    tau1, tau2 = 1., 15.
    je = np.exp(-(tp/30.)**2)*r0 + np.exp(-(tp/15.)**2)*(r1*(1.-np.exp(-(t-tp)/tau1))+r2*(1.-np.exp(-(t-tp)/tau2)))
    return je*np.eye(3)


_SCH_MAP = {"MT": MT, "MAX": MAX, "DIL": DIL, "REUSS": REUSS, "VOIGT": VOIGT}

def run_order2(sh_M, sh_I, T, sch_name, frac, py_Rs, py_Ri):
    ver = rve(matrix="MAT")
    ver["MAT"] = ellipsoid(sh_M, symmetrize=[ISO],
                            visco_prop={"Y": (py_Rs, CREEP)},
                            fraction=1.0 - frac)
    ver["INC"] = ellipsoid(sh_I, symmetrize=[ISO],
                            visco_prop={"Y": (py_Ri, CREEP)},
                            fraction=frac)
    T_np = np.asarray(T, dtype=np.float64)
    sch = _SCH_MAP[sch_name]
    V = homogenize_visco(prop="Y", rve=ver, time_series=T_np,
                         unitsize=3, scheme=sch,
                         epsrel=1.e-6, maxnb=100, verbose=False)
    return np.asarray(V)
