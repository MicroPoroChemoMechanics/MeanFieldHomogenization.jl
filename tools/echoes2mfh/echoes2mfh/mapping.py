"""Declarative Echoes -> MeanFieldHomogenization correspondence tables.

This module is the single source of truth for the translation. It is data,
not logic: `extract.py` and `emit.py` consult these tables, and
`check_drift.py` validates them against both live APIs.

Every Echoes symbol that the corpus uses must appear in exactly one of:
  * a mapping table below (it translates), or
  * `REFUSED` (it does not, with a reason the user can act on).

A symbol in neither is a hole, and `check-drift` reports it as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Scheme constants.
#
# Echoes selects a scheme with a module-level constant and passes solver
# options as loose `homogenize` keywords. MFH attaches solver options to the
# scheme *instance*. `solver_kw` lists which Echoes homogenize-keywords must
# migrate into the constructor call for this scheme; anything not listed is
# dropped (with a note) because the corresponding MFH scheme ignores it.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchemeSpec:
    ctor: str
    solver_kw: tuple[str, ...] = ()
    note: str = ""


SCHEMES: dict[str, SchemeSpec] = {
    "VOIGT": SchemeSpec("Voigt()"),
    "REUSS": SchemeSpec("Reuss()"),
    "DIL": SchemeSpec("Dilute()"),
    "DILD": SchemeSpec("DiluteDual()"),
    "MT": SchemeSpec("MoriTanaka()"),
    "MAX": SchemeSpec("Maxwell()"),
    "PCW": SchemeSpec("PonteCastanedaWillis()"),
    "SC": SchemeSpec(
        "SelfConsistent",
        ("abstol", "reltol", "maxiters", "damping", "select_best", "verbose"),
    ),
    "ASC": SchemeSpec(
        "AsymmetricSelfConsistent",
        ("abstol", "reltol", "maxiters", "damping", "select_best", "verbose"),
        note="Echoes SC and MFH AsymmetricSelfConsistent are different fixed "
        "points for cracked media; see docs/src/developer/validation.md",
    ),
    "DIFF": SchemeSpec(
        "DifferentialScheme",
        ("nsteps", "abstol", "reltol"),
        note="Echoes maxnb maps to nsteps (step count), not an iteration cap",
    ),
    "LAM": SchemeSpec("Laminated()"),
}

# Echoes `homogenize` keyword -> MFH scheme-constructor keyword.
SOLVER_KW_RENAME: dict[str, str] = {
    "epsrel": "abstol",
    "epsabs": "abstol",
    "maxnb": "maxiters",
    "select_best": "select_best",
    "damping": "damping",
}

# `homogenize` keywords that carry no MFH counterpart and are simply dropped.
HOMOGENIZE_KW_DROPPED: dict[str, str] = {
    "verbose": "MFH homogenize is quiet; use @show or the logging macros",
    "concentration": "localization tensors are obtained on demand via "
    "strain_strain_loc(...) rather than a homogenize flag",
    "weight": "no MFH counterpart",
    "prop": "becomes the positional property Symbol",
    "rve": "becomes the first positional argument",
    "scheme": "becomes the second positional argument",
    "fractions": "pass amounts through add_phase! / set_param instead",
}

# ---------------------------------------------------------------------------
# Eshelby / Hill algorithm selectors.
# Echoes: rve.set_param_eshelby(algo=..., epsabs=, epsrel=, maxnb=, epsroots=)
# MFH:    hill_tensor(...; method = :..., abstol =, reltol =, maxiters =)
# ---------------------------------------------------------------------------

ESHELBY_ALGO: dict[str, str] = {
    "DEFAULT": ":auto",
    "RESIDUES": ":residues",
    "RESIDUES2D": ":residues",
    "RESIDUES3D": ":residues",
    "NUMINT": ":decuhr",
    "NUMINT2D": ":decuhr",
    "NUMINT3D": ":decuhr",
    "NESTEDQUAD": ":nestedquadgk",
    "NESTEDQUAD3D": ":nestedquadgk",
}

ESHELBY_KW_RENAME: dict[str, str] = {
    "epsabs": "abstol",
    "epsrel": "reltol",
    "maxnb": "maxiters",
}

# ---------------------------------------------------------------------------
# Symmetry classes and orientation-averaging.
#
# Two families that Echoes spells alike but MFH separates deliberately:
#   * `symmetrize=[ISO]` on a phase  -> an exact rotational average applied
#     *inside* the kernel            -> IsoSymmetrize() / TISymmetrize()
#   * `.paramsym(ISO)` on a result   -> a least-squares reporting projection
#                                    -> best_fit_iso / best_fit_ti
# Conflating them changes the numbers, so they get separate tables.
# ---------------------------------------------------------------------------

SYMMETRIZE: dict[str, str] = {
    "ISO": "IsoSymmetrize()",
    "TI": "TISymmetrize()",
}

PARAMSYM: dict[str, str] = {
    "ISO": "best_fit_iso",
    "TI": "best_fit_ti",
    "ORTHO": "best_fit_ortho",
}

# Exact rotational averages (kernel-side, not reporting projections).
ROTATIONAL_AVERAGE: dict[str, str] = {
    "isotropify": "isotropify",
    "transverse_isotropify": "transverse_isotropify",
}

INTERFACE_TYPE: dict[str, str] = {
    "NODISC": "PerfectInterface()",
    "PRIMALDISC": "SpringInterface",
    "DUALDISC": "MembraneInterface",
}

VISCO_LAW_TYPE: dict[str, str] = {
    "CREEP": ":creep",
    "RELAXATION": ":relaxation",
}

# ---------------------------------------------------------------------------
# Geometry constructors.
# `arity` is the number of positional Echoes arguments consumed; `emit` is a
# format string over the already-translated Julia argument expressions.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeomSpec:
    emit: str
    note: str = ""


GEOMETRY: dict[str, GeomSpec] = {
    # `spherical` is a module *attribute* in Echoes, used bare.
    "spherical": GeomSpec("Spheroid(1.0)"),
    "spheroidal": GeomSpec("Spheroid({0})"),
    "ellipsoidal": GeomSpec("Ellipsoid({0}, {1}, {2})"),
}

# Echoes `spheroidal(w, theta, phi)` / `ellipsoidal(a,b,c,theta,phi,psi)`
# carry Euler angles as trailing positionals; MFH takes `euler_angles = (...)`.
GEOMETRY_ANGLE_START: dict[str, int] = {
    "spheroidal": 1,
    "ellipsoidal": 3,
}

# ---------------------------------------------------------------------------
# Tensor construction and parameter conversion.
# ---------------------------------------------------------------------------

TENSOR_BUILDERS: dict[str, str] = {
    "stiff_kmu": "iso_stiffness({0}, {1})",
    "stiff_Enu": "iso_stiffness_E_nu({0}, {1})",
    "stiff_lambdamu": "iso_stiffness({0} + 2 * {1} / 3, {1})",
    "stiff_TI": "hoenig_stiffness({0}, {1}, {2}, {3}, {4})",
}

SCALAR_CONVERSIONS: dict[str, str] = {
    "kmu_from_Enu": "k_mu(iso_stiffness_E_nu({0}, {1}))",
    "Enu_from_kmu": "E_nu(iso_stiffness({0}, {1}))",
    "k_from_Enu": "k_mu(iso_stiffness_E_nu({0}, {1}))[1]",
    "mu_from_Enu": "k_mu(iso_stiffness_E_nu({0}, {1}))[2]",
    "E_from_kmu": "E_nu(iso_stiffness({0}, {1}))[1]",
    "nu_from_kmu": "E_nu(iso_stiffness({0}, {1}))[2]",
}

# Echoes module constants -> MFH/TensND equivalents.
CONSTANTS: dict[str, str] = {
    "tZ4": "iso_stiffness(1.0e-6, 1.0e-6)",  # near-zero, kept invertible
    "Z4": "zero(TensISO{3})",
    "tId4": "one(TensISO{3})",
    "Id4": "one(TensISO{3})",
    "tJ4": "TensISO{3}(1.0, 0.0)",
    "J4": "TensISO{3}(1.0, 0.0)",
    "tK4": "TensISO{3}(0.0, 1.0)",
    "K4": "TensISO{3}(0.0, 1.0)",
    "tId2": "one(TensISO{2, 3})",
    "Id2": "one(TensISO{2, 3})",
    "tZ2": "zero(TensISO{2, 3})",
    "Z2": "zero(TensISO{2, 3})",
    "infinity": "Inf",
    "infini": "Inf",
}

# Result / phase attribute accessors.
#
# The four Echoes localization tensors follow the "<local field><macroscopic
# field>" convention: eE = strain given macroscopic Strain, sE = stress given
# Strain, eS = strain given Stress, sS = stress given Stress. MFH names them
# <local>_<macroscopic>_loc, so the correspondence is positional and exact.
PHASE_ACCESSORS: dict[str, str] = {
    "eE": "strain_strain_loc",
    "sE": "stress_strain_loc",
    "eS": "strain_stress_loc",
    "sS": "stress_stress_loc",
    "factor": "volume_fraction",
    "fraction": "volume_fraction",
    "density": "crack_density",
}

# Transport analogues, selected when the phase property is 2nd order.
PHASE_ACCESSORS_TRANSPORT: dict[str, str] = {
    "eE": "gradient_gradient_loc",
    "sE": "flux_gradient_loc",
    "eS": "gradient_flux_loc",
    "sS": "flux_flux_loc",
}

# Attribute reads on a tensor result.
TENSOR_ATTRS: dict[str, str] = {
    "k": "k_mu({0})[1]",
    "mu": "k_mu({0})[2]",
    "kmu": "k_mu({0})",
    "E": "E_nu({0})[1]",
    "nu": "E_nu({0})[2]",
    "Enu": "E_nu({0})",
    "array": "Array({0})",
    "a": "Array({0})",
    # `.param` / `.p` returns the independent components of a symmetric tensor;
    # TensND stores exactly those, so `get_data` is the counterpart.
    "param": "TensND.get_data({0})",
    "p": "TensND.get_data({0})",
    "sym": "material_symmetry({0})",
    "angles": "TensND.angles({0})",
}

# Kelvin-Mandel conversion. TensND provides the direct counterparts: `KM`
# builds the Mandel representation, `inv_KM` reads one back into a tensor,
# dispatching on the array's shape exactly as Echoes does.
KM_FUNCTIONS: dict[str, str] = {
    "KM": "KM",
    "invKM": "inv_KM",
}

# ---------------------------------------------------------------------------
# Free functions.
# ---------------------------------------------------------------------------

FUNCTIONS: dict[str, str] = {
    "hill": "hill_tensor",
    "hill_dual": "hill_tensor",
    "eshelby": "eshelby_tensor",
    "crack_compliance": "cod_tensor",
    "inv": "inv",
    "isotropify": "isotropify",
    "transverse_isotropify": "transverse_isotropify",
    "eigen": "eigen",
    "sif": "sif",
    "dif": "dif",
}

# ---------------------------------------------------------------------------
# Viscoelasticity.
#
# Echoes carries an ageing linear viscoelastic (ALV) law as a callable of
# (t, t') plus a mode flag, and evaluates it into a lower-triangular Volterra
# block matrix over a time series. MFH keeps the same two ingredients but
# names them differently and takes the time series at the call site rather
# than storing it on the RVE.
# ---------------------------------------------------------------------------

VISCO_FUNCTIONS: dict[str, str] = {
    "visco_law": "ViscoLaw",
    "relaxation_mat": "trapezoidal_matrix",
    "creep_mat": "trapezoidal_matrix",
    "build_visco_mat": "trapezoidal_matrix",
    "invert_tri_inf": "volterra_inverse",
    "visco_isotropify": "isotropify",
    "visco_hill": "hill_kernel",
    "visco_crack_compliance": "cod_kernel_alv",
}

# `.mat(T)` evaluates a visco law over a time series; `.inv_mat()` inverts the
# resulting Volterra matrix.
VISCO_METHODS: dict[str, str] = {
    "mat": "trapezoidal_matrix",
    "inv_mat": "volterra_inverse",
}

VISCO_PARAMSYM: dict[str, str] = {
    "ISO": "iso_params_from_blocks",
    "TI": "ti_params_from_blocks",
    "ORTHO": "ortho_params_from_blocks",
}

# ---------------------------------------------------------------------------
# numpy / math -> Julia. Only what the corpus actually uses.
# ---------------------------------------------------------------------------

NUMPY: dict[str, str] = {
    "linspace": "range({0}, {1}; length = {2})",
    "logspace": "(10 .^ range({0}, {1}; length = {2}))",
    "arange": "range({0}, {1}; step = {2})",
    "array": "{0}",
    "zeros": "zeros({0})",
    "ones": "ones({0})",
    "eye": "I({0})",
    "identity": "I({0})",
    "sqrt": "sqrt",
    "exp": "exp",
    "log": "log",
    "log10": "log10",
    "log2": "log2",
    "cos": "cos",
    "sin": "sin",
    "tan": "tan",
    "arccos": "acos",
    "arcsin": "asin",
    "arctan": "atan",
    "arctan2": "atan",
    "cosh": "cosh",
    "sinh": "sinh",
    "tanh": "tanh",
    "fabs": "abs",
    "absolute": "abs",
    "trace": "tr",
    "pow": "{0}^{1}",
    "transpose": "transpose",
    "concatenate": "vcat",
    "power": "^",
    "floor": "floor",
    "ceil": "ceil",
    "sign": "sign",
    "maximum": "maximum",
    "minimum": "minimum",
    "mean": "sum({0}) / length({0})",
    "real": "real",
    "imag": "imag",
    "conj": "conj",
}

NUMPY_ATTR: dict[str, str] = {
    "linalg.inv": "inv",
    "linalg.solve": "\\",
    "linalg.det": "det",
    "linalg.norm": "norm",
    "linalg.eig": "eigen",
    "linalg.eigvals": "eigvals",
    "linalg.pinv": "pinv",
}

MATH_CONSTANTS: dict[str, str] = {
    "pi": "pi",
    "e": "exp(1)",
    "inf": "Inf",
    "nan": "NaN",
}

# ---------------------------------------------------------------------------
# matplotlib -> Plots.jl.
#
# Emission is stateful (Plots.jl `plot` vs `plot!`), so `emit.py` owns the
# bang decision; this table records the name and keyword translation only.
# ---------------------------------------------------------------------------

PLOT_FUNCS: dict[str, str] = {
    "plot": "plot",
    "semilogx": "plot",
    "semilogy": "plot",
    "loglog": "plot",
    "scatter": "scatter",
    "errorbar": "plot",
    "fill_between": "plot",
    "axhline": "hline",
    "axvline": "vline",
    "step": "plot",
}

PLOT_SCALE_KW: dict[str, str] = {
    "semilogx": "xscale = :log10",
    "semilogy": "yscale = :log10",
    "loglog": "xscale = :log10, yscale = :log10",
}

PLOT_SETTERS: dict[str, str] = {
    "xlabel": "xlabel",
    "ylabel": "ylabel",
    "title": "title",
    "xlim": "xlims",
    "ylim": "ylims",
    "xlims": "xlims",
    "ylims": "ylims",
    "xscale": "xscale",
    "yscale": "yscale",
}

# matplotlib format strings ('r--', 'k-', 'g-d') -> Plots.jl keywords.
PLOT_COLORS: dict[str, str] = {
    "b": ":blue",
    "g": ":green",
    "r": ":red",
    "c": ":cyan",
    "m": ":magenta",
    "y": ":gold",
    "k": ":black",
    "w": ":white",
}

PLOT_LINESTYLES: dict[str, str] = {
    "-": ":solid",
    "--": ":dash",
    "-.": ":dashdot",
    ":": ":dot",
}

PLOT_MARKERS: dict[str, str] = {
    "+": ":cross",
    "o": ":circle",
    "*": ":star5",
    "d": ":diamond",
    "s": ":square",
    "^": ":utriangle",
    "v": ":dtriangle",
    "x": ":xcross",
    ".": ":circle",
}

# ---------------------------------------------------------------------------
# Refusals: Echoes features with a known, verified reason not to translate.
#
# A refusal is a *feature*, not a failure. It produces an `#= UNTRANSLATED =#`
# block carrying the original Python verbatim plus a runtime `error(...)`, so a
# partially translated script can never quietly produce wrong numbers.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Refusal:
    reason: str
    suggestion: str = ""


REFUSED: dict[str, Refusal] = {
    "polynomial": Refusal(
        "Echoes polynomial algebra is not part of the MFH public API",
        "MFH uses Polynomials.jl internally; port the algebra by hand",
    ),
    "polynomialc": Refusal("see `polynomial`"),
    "rational": Refusal("see `polynomial`"),
    "rationalc": Refusal("see `polynomial`"),
    "residue_log_I": Refusal(
        "residue-algorithm internals are not exported by MFH",
        "the residue path is reachable via hill_tensor(...; method = :residues)",
    ),
    "residue_log_z": Refusal("see `residue_log_I`"),
    "find_roots_positive_imag": Refusal("see `residue_log_I`"),
    "add_root": Refusal("see `residue_log_I`"),
    "perm6": Refusal(
        "Kelvin-Mandel index permutation is implicit in TensND storage",
        "index TensND tensors directly rather than permuting a 6x6 matrix",
    ),
    "inv_perm6": Refusal("see `perm6`"),
    "import_rve": Refusal(
        "reads an Echoes-specific .rve data file",
        "rebuild the RVE with add_phase! from the same data",
    ),
}

# Third-party imports that place a script out of scope.
OUT_OF_SCOPE_IMPORTS: dict[str, str] = {
    "nlopt": "nonlinear fits: port to Optim.jl or NonlinearSolve.jl by hand",
    "getfem": "GetFEM++ coupling: MFH has its own FE gates "
    "(FEEllipticCrack, FEExcenteredSphere) with a different contract",
    "sympy": "symbolic work: MFH has a SymPy extension, but the script's "
    "symbolic algebra needs manual review",
    "openpyxl": "spreadsheet I/O: use XLSX.jl and re-point the data path",
    "pandas": "dataframes: use DataFrames.jl and re-point the data path",
}


# ---------------------------------------------------------------------------
# Complex-valued twins.
#
# Echoes exposes a parallel `...c` family (`rvec`, `ellipsoidc`, `tensorc`, …)
# because its C++ templates are instantiated separately for real and complex
# scalars. Julia needs no such split: the very same constructors work, and the
# element type follows from the numbers. So the `c` suffix is dropped and the
# RVE is declared with a complex eltype.
# ---------------------------------------------------------------------------

COMPLEX_SUFFIX = "c"

#: names whose complex twin is spelled with a trailing `c`
COMPLEX_TWINS: set[str] = {
    "rve",
    "ellipsoid",
    "ellipsoidal",
    "spheroidal",
    "crack",
    "sphere_nlayers",
    "spheroid_nlayers",
    "tensor",
    "inclusion",
    "inclusion_generic_ellipsoid",
    "user_inclusion",
    "polynomial",
    "rational",
    "homogenize",
    "hill",
    "hill_dual",
    "eshelby",
    "crack_compliance",
}


def strip_complex(name: str) -> tuple[str, bool]:
    """`ellipsoidc` -> `('ellipsoid', True)`; anything else unchanged."""
    if name.endswith(COMPLEX_SUFFIX):
        base = name[: -len(COMPLEX_SUFFIX)]
        if base in COMPLEX_TWINS:
            return base, True
    return name, False


def classify(name: str) -> str:
    """Return the table a bare Echoes name belongs to, or 'unknown'."""
    for table, label in (
        (SCHEMES, "scheme"),
        (ESHELBY_ALGO, "eshelby_algo"),
        (SYMMETRIZE, "symmetry"),
        (INTERFACE_TYPE, "interface"),
        (VISCO_LAW_TYPE, "visco_law_type"),
        (GEOMETRY, "geometry"),
        (TENSOR_BUILDERS, "tensor_builder"),
        (SCALAR_CONVERSIONS, "scalar_conversion"),
        (CONSTANTS, "constant"),
        (FUNCTIONS, "function"),
        (REFUSED, "refused"),
    ):
        if name in table:
            return label
    return "unknown"


def all_mapped_names() -> set[str]:
    """Every Echoes name this module knows about, mapped or refused."""
    names: set[str] = set()
    for table in (
        SCHEMES,
        ESHELBY_ALGO,
        SYMMETRIZE,
        PARAMSYM,
        ROTATIONAL_AVERAGE,
        INTERFACE_TYPE,
        VISCO_LAW_TYPE,
        GEOMETRY,
        TENSOR_BUILDERS,
        SCALAR_CONVERSIONS,
        CONSTANTS,
        PHASE_ACCESSORS,
        TENSOR_ATTRS,
        FUNCTIONS,
        REFUSED,
    ):
        names |= set(table)
    return names
