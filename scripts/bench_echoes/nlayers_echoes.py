"""Echoes side of `benchmark_nlayers.jl`: multilayer spheres and their concentrations.

Imported from Julia with PythonCall (`pyimport("nlayers_echoes")`), the directory
of this file being put on `sys.path` first.
"""
import echoes
import numpy as np
from echoes import rot6, sphere_nlayers, NODISC, PRIMALDISC, DUALDISC

def py_stiff_kmu(K, mu):
    return echoes.stiff_kmu(float(K), float(mu))

def py_make_nlayers(radii, props, Cref):
    radii_np = np.asarray(radii, dtype=float)
    spn = echoes.sphere_nlayers(radii=radii_np, prop={'C': props})
    spn.set_ref('C', Cref)
    return spn

def py_layer_eE(spn, k):
    # ECHOES is 0-indexed.
    return np.asarray(spn.layer_eE(k))

def py_eE(spn):
    return np.asarray(spn.eE)

def py_sE(spn):
    return np.asarray(spn.sE)

def py_layer_fraction(spn, k):
    return float(spn.layer_fraction(k))

def py_conc_sS(spn, r, theta, phi):
    return np.asarray(spn.loc_sS(r, theta, phi))

# The four pointwise concentrations, rows rotated to the CANONICAL basis.
# ECHOES returns rows in the local spherical basis (e_theta, e_phi, e_r);
# rot6(theta, phi) brings them back to global Cartesian, which is what
# MeanFieldHomogenization returns.  `layer` is 0-based, -1 = auto.
def py_conc_all(spn, r, theta, phi, layer):
    P = rot6(theta, phi)
    kw = {} if layer < 0 else {"layer": layer}
    return (np.asarray(P @ spn.loc_eE(r, theta, phi, **kw)),
            np.asarray(P @ spn.loc_sE(r, theta, phi, **kw)),
            np.asarray(P @ spn.loc_eS(r, theta, phi, **kw)),
            np.asarray(P @ spn.loc_sS(r, theta, phi, **kw)))

# Interface descriptors are built ENTIRELY on the Python side.  Handing a
# Julia vector of `[NODISC]` lists across the Python bridge converts the enum members to
# plain integers and ECHOES then silently returns NaN rather than raising, so
# the tag is passed as a string and the list is assembled here.
def _itf(tag, p1, p2):
    if tag == "perfect":
        return [NODISC]
    if tag == "spring":
        return [float(p1), float(p2), PRIMALDISC]
    if tag == "membrane":
        return [float(p1), float(p2), DUALDISC]
    raise ValueError("unknown interface tag " + str(tag))

def py_make_nlayers_itf(radii, props, ref, tags, p1s, p2s):
    interf = [_itf(t, a, b) for t, a, b in zip(tags, p1s, p2s)]
    spn = sphere_nlayers(radii=np.asarray(radii, dtype=float),
                         prop={"C": props}, interf_prop={"C": interf})
    spn.set_ref("C", ref)
    return spn
