# Running a script: in a fresh module, so that one run leaves nothing behind
# for the next, with what it prints and what it warns kept for the page.

"""
    check_report() -> Vector{String}

The lines `mfhstudio(check = true)` prints: the versions the scripts run with.
"""
check_report() = [
    "Julia $VERSION, MeanFieldHomogenization $(pkgversion(MeanFieldHomogenization)), " *
        "TensND $(pkgversion(TensND)).",
]
