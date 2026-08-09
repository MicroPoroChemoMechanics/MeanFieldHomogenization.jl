"""What the interface offers, and where each part comes from.

The catalog has two halves, and keeping them apart is what lets the interface
stay usable when Julia is not.

*Form definitions* — which fields a spheroid needs, what a layer looks like,
which lenses exist — are user-interface concerns. They live here, in Python,
and are available immediately.

*Introspected facts* — the list of schemes and the solver options each one
actually reads — are properties of the installed MeanFieldHomogenization and can only come
from the sidecar. They are merged in when it is ready.

The fallback scheme list below exists so the forms render before Julia has
finished loading; it is replaced wholesale by the introspected one, never
merged with it, so a scheme that MFH drops disappears from the interface.
"""

from __future__ import annotations

from .model import LAMINATE_SCHEMES

#: The two kinds of homogenization cell. A laminate is a *cell*, not an
#: inclusion: it is a unit of homogenization, and embedding one in a matrix
#: would need its Hill tensor, which is a different and open problem. So the
#: interface offers it beside the RVE rather than inside the shape list, where
#: it would promise something MeanFieldHomogenization cannot do.
CELL_KINDS = [
    {
        "name": "rve", "label": "RVE (matrix + inclusions)",
        "doc": "A random morphology: one matrix phase and the inclusions "
               "dispersed in it. Estimated by the mean-field schemes.",
    },
    {
        "name": "laminate", "label": "Laminate (periodic stack)",
        "doc": "A periodic unit cell of parallel layers of common normal — no "
               "matrix, no reference medium, and an exact effective behavior "
               "rather than an estimate.",
        "schemes": list(LAMINATE_SCHEMES),
    },
]

GEOMETRIES = [
    {
        "name": "Spheroid", "kind": "spheroid", "dim": 3,
        "doc": "Axisymmetric ellipsoid; ω < 1 oblate, ω > 1 prolate, ω = 1 sphere.",
        "fields": [{"name": "omega", "label": "ω (aspect ratio)", "type": "number", "default": 1.0}],
        "angles": 2,
    },
    {
        "name": "Ellipsoid", "kind": "ellipsoid", "dim": 3,
        "doc": "General ellipsoid of semi-axes (a, b, c).",
        "fields": [
            {"name": "a", "label": "a", "type": "number", "default": 1.0},
            {"name": "b", "label": "b", "type": "number", "default": 1.0},
            {"name": "c", "label": "c", "type": "number", "default": 1.0},
        ],
        "angles": 3,
    },
    {
        "name": "Cylinder", "kind": "cylinder", "dim": 3,
        "doc": "Infinite elliptic cylinder of transverse semi-axes (b, c).",
        "fields": [
            {"name": "b", "label": "b", "type": "number", "default": 1.0},
            {"name": "c", "label": "c", "type": "number", "default": 1.0},
        ],
        "angles": 3,
    },
    {
        "name": "PennyCrack", "kind": "penny_crack", "dim": 3,
        "doc": "Circular flat crack of radius a. Enters the RVE with a crack density.",
        "fields": [{"name": "a", "label": "a (radius)", "type": "number", "default": 1.0}],
        "angles": 2, "amount": "density",
    },
    {
        "name": "EllipticCrack", "kind": "elliptic_crack", "dim": 3,
        "doc": "Elliptic flat crack, semi-axes a ≥ b.",
        "fields": [
            {"name": "a", "label": "a", "type": "number", "default": 1.0},
            {"name": "b", "label": "b", "type": "number", "default": 0.5},
        ],
        "angles": 3, "amount": "density",
    },
    {
        "name": "RibbonCrack", "kind": "ribbon_crack", "dim": 3,
        "doc": "Tunnel crack of half-width b, unbounded along its length.",
        "fields": [{"name": "b", "label": "b (half-width)", "type": "number", "default": 1.0}],
        "angles": 3, "amount": "density",
    },
    {
        "name": "LayeredSphere", "kind": "layered_sphere", "dim": 3,
        "doc": "Concentric layers, ascending radii, r = 0 implicit at the center.",
        "fields": [], "layered": True, "angles": 0,
    },
    {
        "name": "LayeredSpheroid", "kind": "layered_spheroid", "dim": 3,
        "doc": "Confocal spheroidal layers, given by volume fraction. "
               "Conduction only; ω = 1 is a LayeredSphere.",
        "fields": [
            {"name": "omega", "label": "ω (outer aspect ratio)", "type": "number", "default": 0.5},
            {"name": "radius", "label": "outer axial semi-axis", "type": "number", "default": 1.0},
            {"name": "Nseries", "label": "N (series order)", "type": "integer", "default": 5},
        ],
        "layered": True, "layer_by": "fraction", "layer_property": "iso_conduction",
        "angles": 2, "conduction_only": True,
    },
]

# `iso_stiffness(k, μ)` takes *physical* moduli while the raw `TensISO{3}(a, b)`
# constructor takes `(3k, 2μ)`. Only the former is offered, which is how that
# trap is removed rather than merely documented.
PROPERTIES = [
    {
        "name": "iso_kmu", "label": "Isotropic (k, μ)", "order": 4,
        "builder": "iso_stiffness",
        "fields": [
            {"name": "k", "label": "k (bulk)", "type": "number", "default": 10.0},
            {"name": "mu", "label": "μ (shear)", "type": "number", "default": 5.0},
        ],
    },
    {
        "name": "iso_Enu", "label": "Isotropic (E, ν)", "order": 4,
        "builder": "iso_stiffness_E_nu",
        "fields": [
            {"name": "E", "label": "E (Young)", "type": "number", "default": 30.0},
            {"name": "nu", "label": "ν (Poisson)", "type": "number", "default": 0.2},
        ],
    },
    {
        "name": "ti_hoenig", "label": "Transversely isotropic (Hoenig)", "order": 4,
        "builder": "hoenig_stiffness",
        # h = 1 with ν₁ = ν₂ and γ = 1 is not a transversely isotropic material
        # at all — it is the isotropic point of the parametrization, written in
        # a TI type. Shipping it as the default put every new anisotropic model
        # on that degenerate corner, where the axis carries no information and
        # the numerics have nothing to grip. The defaults below are genuinely
        # transversely isotropic (E₃/E₁ = 0.3, γ = 0.5).
        "doc": "E₁ in the transverse plane, h = E₃/E₁, γ = G₁₃ / G₁₂. "
               "h = 1 with ν₁ = ν₂ and γ = 1 is the isotropic point.",
        "fields": [
            {"name": "E1", "label": "E₁", "type": "number", "default": 30.0},
            {"name": "h", "label": "h = E₃/E₁", "type": "number", "default": 0.3},
            {"name": "nu1", "label": "ν₁", "type": "number", "default": 0.2},
            {"name": "nu2", "label": "ν₂", "type": "number", "default": 0.25},
            {"name": "gamma", "label": "γ", "type": "number", "default": 0.5},
        ],
        "orientation": 2,
    },
    {
        "name": "iso_conduction", "label": "Isotropic conductivity", "order": 2,
        "builder": "TensISO{3}",
        "doc": "A single argument to TensISO{dim} gives the 2nd-order tensor.",
        "fields": [{"name": "k", "label": "κ", "type": "number", "default": 1.0}],
    },
    {
        "name": "ti_conduction", "label": "Transversely isotropic conductivity",
        "order": 2, "builder": "TensTI2",
        "doc": "κₜ acts across the axis, κₐ along it.",
        "fields": [
            {"name": "kt", "label": "κₜ (transverse)", "type": "number", "default": 1.0},
            {"name": "ka", "label": "κₐ (axial)", "type": "number", "default": 5.0},
        ],
        "orientation": 2,
    },
    {
        "name": "ortho_conduction", "label": "Orthotropic conductivity",
        "order": 2, "builder": "TensDiag2",
        "doc": "The three principal conductivities, in the phase's own frame.",
        "fields": [
            {"name": "k1", "label": "κ₁", "type": "number", "default": 1.0},
            {"name": "k2", "label": "κ₂", "type": "number", "default": 2.0},
            {"name": "k3", "label": "κ₃", "type": "number", "default": 5.0},
        ],
        "orientation": 3,
    },
    {
        "name": "ortho_stiffness", "label": "Orthotropic (9 constants)",
        "order": 4, "builder": "TensOrtho",
        "doc": "The nine constants in the material frame; the frame itself is "
               "the Orientation block below.",
        "fields": [
            {"name": "C11", "label": "C₁₁", "type": "number", "default": 120.0},
            {"name": "C22", "label": "C₂₂", "type": "number", "default": 90.0},
            {"name": "C33", "label": "C₃₃", "type": "number", "default": 70.0},
            {"name": "C12", "label": "C₁₂", "type": "number", "default": 40.0},
            {"name": "C13", "label": "C₁₃", "type": "number", "default": 35.0},
            {"name": "C23", "label": "C₂₃", "type": "number", "default": 30.0},
            {"name": "C44", "label": "C₄₄", "type": "number", "default": 25.0},
            {"name": "C55", "label": "C₅₅", "type": "number", "default": 22.0},
            {"name": "C66", "label": "C₆₆", "type": "number", "default": 20.0},
        ],
        "orientation": 3,
    },
    # ── viscoelastic laws ────────────────────────────────────────────────
    #
    # A phase carries one of these under the same key as a stiffness would be;
    # `homogenize_alv` finds it there. The signatures below are the ones
    # MeanFieldHomogenization actually declares — `maxwell_iso` takes two relaxation
    # times, not one, and `kelvin_iso` takes whole branch vectors.
    {
        "name": "maxwell_iso", "label": "Viscoelastic — Maxwell (relaxation)",
        "order": 4, "builder": "maxwell_iso", "visco": True, "mode": "relaxation",
        "doc": "R(t,t′) = 3k·e^(−Δt/η_k)·𝕁 + 2μ·e^(−Δt/η_μ)·𝕂",
        "fields": [
            {"name": "k", "label": "k", "type": "number", "default": 10.0},
            {"name": "mu", "label": "μ", "type": "number", "default": 5.0},
            {"name": "eta_k", "label": "η_k (bulk time)", "type": "number", "default": 1.0},
            {"name": "eta_mu", "label": "η_μ (shear time)", "type": "number", "default": 1.0},
        ],
    },
    {
        "name": "kelvin_iso", "label": "Viscoelastic — Kelvin chain (creep)",
        "order": 4, "builder": "kelvin_iso", "visco": True, "mode": "creep",
        "doc": "Instantaneous (k₀, μ₀) plus one Kelvin branch.",
        "fields": [
            {"name": "k0", "label": "k₀", "type": "number", "default": 10.0},
            {"name": "mu0", "label": "μ₀", "type": "number", "default": 5.0},
            {"name": "k1", "label": "k branch", "type": "number", "default": 20.0},
            {"name": "mu1", "label": "μ branch", "type": "number", "default": 10.0},
            {"name": "tau_k", "label": "τ_k", "type": "number", "default": 1.0},
            {"name": "tau_mu", "label": "τ_μ", "type": "number", "default": 1.0},
        ],
    },
    {
        "name": "visco_elastic", "label": "Viscoelastic — elastic (Heaviside)",
        "order": 4, "builder": "heaviside_law", "visco": True, "mode": "relaxation",
        "doc": "A non-ageing elastic phase inside a viscoelastic RVE.",
        "fields": [
            {"name": "k", "label": "k", "type": "number", "default": 10.0},
            {"name": "mu", "label": "μ", "type": "number", "default": 5.0},
        ],
    },
    {
        "name": "visco_custom", "label": "Viscoelastic — custom J(t, t′)",
        "order": 4, "builder": "ViscoLaw", "visco": True, "mode": "creep",
        "doc": "Any Julia expression in t and t′ returning a 4th-order tensor.",
        "fields": [
            {"name": "expr", "label": "expression in t, t′", "type": "code",
             "default": "iso_stiffness(1 / (1 / 10 + 0.01 * (t - t\u2032)), 5.0)"},
            {"name": "mode", "label": "mode (creep or relaxation)", "type": "text",
             "default": "creep"},
        ],
    },
    {
        "name": "void", "label": "Void / pore (near-zero)", "order": 4,
        "builder": "iso_stiffness",
        "doc": "Kept slightly non-zero so the tensor stays invertible.",
        "fields": [
            {"name": "k", "label": "k", "type": "number", "default": 1.0e-6},
            {"name": "mu", "label": "μ", "type": "number", "default": 1.0e-6},
        ],
    },
]

# `symmetrize` is an exact rotational average applied inside the kernel;
# `best_fit_*` is a least-squares projection used for reporting. Conflating
# them changes the numbers, so they are two separate lists.
SYMMETRIZE = [
    {"name": "none", "label": "None", "emit": "nothing"},
    {
        "name": "iso", "label": "Isotropic orientation average",
        "emit": "IsoSymmetrize()",
        "doc": "Uniform distribution of orientations, averaged exactly in the kernel.",
    },
    {
        "name": "ti", "label": "Transversely isotropic average",
        "emit": "TISymmetrize()",
        "doc": "Orientations uniformly distributed about an axis.",
    },
]

PROJECTIONS = [
    {"name": "none", "label": "As computed", "emit": None},
    {"name": "iso", "label": "Best isotropic fit", "emit": "best_fit_iso"},
    {"name": "ti", "label": "Best TI fit", "emit": "best_fit_ti"},
    {"name": "ortho", "label": "Best orthotropic fit", "emit": "best_fit_ortho"},
]

INTERFACES = [
    {"name": "PerfectInterface", "label": "Perfect", "fields": []},
    {
        "name": "SpringInterface", "label": "Spring (displacement jump)",
        "fields": [
            {"name": "kn", "label": "kₙ", "type": "number", "default": 1.0},
            {"name": "kt", "label": "kₜ", "type": "number", "default": 1.0},
        ],
    },
    {
        "name": "MembraneInterface", "label": "Membrane (traction jump)",
        "fields": [
            {"name": "k2D", "label": "k₂D", "type": "number", "default": 1.0},
            {"name": "mu2D", "label": "μ₂D", "type": "number", "default": 1.0},
        ],
    },
    {
        "name": "KapitzaInterface", "label": "Kapitza (thermal)",
        "fields": [{"name": "h", "label": "h", "type": "number", "default": 1.0}],
        "order": 2,
    },
    {
        "name": "SurfaceConductiveInterface", "label": "Surface conductive",
        "fields": [{"name": "ks", "label": "κₛ", "type": "number", "default": 1.0}],
        "order": 2,
    },
    # ── laminate-only, anisotropic ───────────────────────────────────────
    #
    # These take a tensor, not two scalars, so the field is a Julia expression
    # rather than a pair of numbers. They exist only on a laminate: a layered
    # sphere's interface is a sphere, where "anisotropic in the interface
    # plane" has no fixed frame to be written in. There is deliberately no
    # anisotropic Kapitza — `[T] = ρ qₙ` is already fully general.
    {
        "name": "AnisotropicSpringInterface", "label": "Spring, anisotropic",
        "laminate_only": True,
        "doc": "Symmetric 3×3 compliance in the layer frame (ℓ, m, n).",
        "fields": [{
            "name": "compliance", "label": "compliance (3×3)", "type": "code",
            "default": "[1.0e-3 0.0 0.0; 0.0 2.0e-3 0.0; 0.0 0.0 5.0e-4]",
        }],
    },
    {
        "name": "AnisotropicMembraneInterface", "label": "Membrane, anisotropic",
        "laminate_only": True,
        "doc": "In-plane Kelvin-Mandel block on (ℓ⊗ℓ, m⊗m, √2 ℓ⊗ˢm); "
               "entry [3,3] is 2 Cˢ₁₂₁₂.",
        "fields": [{
            "name": "stiffness", "label": "stiffness (3×3 KM)", "type": "code",
            "default": "[1.0 0.3 0.0; 0.3 1.0 0.0; 0.0 0.0 0.7]",
        }],
    },
    {
        "name": "AnisotropicSurfaceConductiveInterface",
        "label": "Surface conductive, anisotropic",
        "laminate_only": True, "order": 2,
        "doc": "In-plane conductance tensor in the layer frame.",
        "fields": [{
            "name": "conductance", "label": "conductance (3×3)", "type": "code",
            "default": "[1.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 0.0]",
        }],
    },
]

#: `cells` says which kind of cell a lens exists on. `amount` is not merely
#: unhelpful on a laminate — `AmountParameter` raises there, pointing at
#: `ThicknessParameter` — and the two laminate lenses have no meaning on an RVE.
LENSES = [
    {
        "name": "amount", "label": "Phase amount (fraction or density)",
        "args": ["phase"], "cells": ["rve"],
        "doc": "The matrix amount is derived (1 − Σ f) and cannot be set.",
    },
    {
        "name": "property", "label": "Property component",
        "args": ["phase", "property", "index"], "cells": ["rve", "laminate"],
        "doc": "On a laminate the phase name is a layer name.",
    },
    {
        "name": "geometry", "label": "Geometry field",
        "args": ["phase", "field", "index"], "cells": ["rve"],
        "doc": "Changing a semi-axis reclassifies the shape trait.",
    },
    {
        "name": "shape_param", "label": "Distribution shape",
        "args": ["field", "index"], "cells": ["rve"],
    },
    {
        "name": "thickness", "label": "Layer thickness", "args": ["phase"],
        "cells": ["laminate"],
        "doc": "Not the same as a volume fraction: changing hᵢ moves the "
               "period too, hence the 1/L weight of every imperfect interface.",
    },
    {
        "name": "interface_param", "label": "Interface field",
        "args": ["index", "field"], "cells": ["laminate"],
        "doc": "Interface k sits on top of layer k in stacking order.",
        "fields": ["kn", "kt", "κs", "μs", "resistance", "conductance"],
    },
    {
        "name": "nested", "label": "Through a nested scale",
        "args": ["member", "property", "inner"], "cells": ["rve", "laminate"],
        "doc": "Reaches into a Homogenized inner cell; crosses scales for AD.",
    },
]

#: The four autodiff wrappers, as the Sensitivity panel offers them.
SENSITIVITIES = [
    {
        "name": "derivative", "label": "Derivative (one parameter)",
        "doc": "f′(x₀) of one scalar reading of the effective tensor.",
    },
    {
        "name": "gradient", "label": "Gradient (several parameters)",
        "doc": "∇f(x₀) of one scalar reading, one entry per parameter.",
    },
    {
        "name": "jacobian", "label": "Jacobian (whole tensor)",
        "doc": "The effective tensor flattened, differentiated against every "
               "parameter. No scalar extraction, so no isotropy needed.",
    },
]

VISCO = [
    {
        "name": "maxwell_iso", "label": "Maxwell (relaxation)", "mode": "relaxation",
        "fields": [
            {"name": "k", "label": "k", "type": "number", "default": 10.0},
            {"name": "mu", "label": "μ", "type": "number", "default": 5.0},
            {"name": "tau", "label": "τ (relaxation time)", "type": "number", "default": 1.0},
        ],
    },
    {
        "name": "kelvin_iso", "label": "Kelvin-Voigt (creep)", "mode": "creep",
        "fields": [
            {"name": "k", "label": "k", "type": "number", "default": 10.0},
            {"name": "mu", "label": "μ", "type": "number", "default": 5.0},
            {"name": "tau", "label": "τ (creep time)", "type": "number", "default": 1.0},
        ],
    },
    {
        "name": "heaviside", "label": "Elastic (Heaviside)", "mode": "relaxation",
        "fields": [
            {"name": "k", "label": "k", "type": "number", "default": 10.0},
            {"name": "mu", "label": "μ", "type": "number", "default": 5.0},
        ],
    },
    {
        "name": "custom", "label": "Custom J(t, t′) or R(t, t′)", "mode": "creep",
        "fields": [{
            "name": "expr", "label": "Julia expression in t, t′", "type": "code",
            "default": "1.0 / 10.0 * (1 + log(1 + (t - t′)))",
        }],
    },
]

#: Used until the sidecar answers. Replaced wholesale, never merged, so a
#: scheme MeanFieldHomogenization drops does not linger in the interface.
FALLBACK_SCHEMES = [
    {"name": n, "options": [], "singleton": True}
    for n in (
        "AsymmetricSelfConsistent", "DifferentialScheme", "Dilute", "DiluteDual",
        "Laminated", "Maxwell", "MoriTanaka", "PonteCastanedaWillis", "Reuss",
        "SelfConsistent", "Voigt",
    )
]


def base_catalog() -> dict:
    """Everything the interface can offer without Julia."""
    return {
        "mfh_version": None,
        "julia_version": None,
        "introspected": False,
        "schemes": FALLBACK_SCHEMES,
        "cell_kinds": CELL_KINDS,
        "geometries": GEOMETRIES,
        "properties": PROPERTIES,
        "symmetrize": SYMMETRIZE,
        "projections": PROJECTIONS,
        "interfaces": INTERFACES,
        "lenses": LENSES,
        "sensitivities": SENSITIVITIES,
        "visco": VISCO,
        "hill_methods": ["auto", "analytical", "residues", "decuhr", "nestedquadgk"],
        "constraints": {
            "no_multiscale_in_alv": True,
            "matrix_amount_is_derived": True,
            "laminate_schemes": list(LAMINATE_SCHEMES),
        },
    }


def merge(introspected: dict | None) -> dict:
    """The form definitions, with the live facts folded in when available."""
    cat = base_catalog()
    if not introspected:
        return cat
    cat["schemes"] = introspected.get("schemes") or cat["schemes"]
    cat["mfh_version"] = introspected.get("mfh_version")
    cat["julia_version"] = introspected.get("julia_version")
    cat["introspected"] = True
    if introspected.get("constraints"):
        cat["constraints"].update(introspected["constraints"])
    return cat
