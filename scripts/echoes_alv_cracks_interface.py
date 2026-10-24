"""Echoes side of `60_alv_cracks_interface.jl`: aging viscoelastic cracks with an interface.

Imported from Julia with PythonCall (`pyimport("echoes_alv_cracks_interface")`),
the directory of this file being put on `sys.path` first.
"""
from numpy import *
from echoes import *

_SCHEMES = {"MT": MT, "SC": SC, "PCW": PCW}

def run_echoes(t0, T, density, scheme_name):
    sch = _SCHEMES[scheme_name]
    k0=5.; mu0=2.; kinf=3.; muinf=1.; tau=1.
    C0=stiff_kmu(k0,mu0); Cinf=stiff_kmu(kinf,muinf)
    V=lambda t,tp: Cinf.array + (C0*(1.+0.2*sqrt(max(tp, 0.0))) - Cinf).array * exp(-(t-tp)/tau)
    kn=10.; kt=5.; kninf=5.; ktinf=2.5; taun=2.; taut=3.
    Rn=lambda t,tp:1.*(1.+0.1*tp**0.4)*(kninf+(kn-kninf)*exp(-(t-tp)/taun))
    Rt=lambda t,tp:1.*(1.+0.1*tp**0.2)*(ktinf+(kt-ktinf)*exp(-(t-tp)/taut))
    ver=rve(matrix="MATRIX")
    ver["MATRIX"]=ellipsoid(shape=spherical, symmetrize=[ISO],
                            prop={"C":tensor(V(t0,t0)), "Cinf":Cinf},
                            visco_prop={"C":(V,RELAXATION)})
    ver["CRACK"]=crack(shape=spheroidal(1.e-3), symmetrize=[ISO],
                       interf_prop={"C":[Rn(t0,t0), Rt(t0,t0)],
                                    "Cinf":[kninf, ktinf]},
                       interf_visco_prop={"C":[(Rn,RELAXATION),(Rt,RELAXATION)]},
                       density=density)
    iV = linalg.inv(homogenize_visco(prop="C", rve=ver, time_series=T,
                                     scheme=sch, epsrel=1.e-6, verbose=False))
    S = zeros(len(T)*6)
    for i in range(len(T)):
        S[6*i] = 1.0
    return iV.dot(S)[::6]
