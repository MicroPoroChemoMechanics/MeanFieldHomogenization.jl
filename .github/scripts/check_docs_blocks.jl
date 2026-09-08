# Two checks, one in each direction, both of which Documenter only performs
# during a full build (~90 min here) and both of which have broken a build:
#
#   forward  — every name listed in an `@docs` block carries a docstring;
#   reverse  — every **exported** name that carries a docstring is listed in
#              some `@docs` block, which is what `checkdocs = :exports` demands.
#
# The reverse one is the subtler failure: a new exported type with a good
# docstring and no `@docs` entry is reported as a missing docstring *and* makes
# every bare `@ref` to it unresolvable, so the build fails on a cross-reference
# far from the actual omission.
using MeanFieldHomogenization
const MFH = MeanFieldHomogenization

"Every module reachable from `MFH`, itself included."
function all_modules()
    seen = Set{Module}([MFH])
    queue = Module[MFH]
    while !isempty(queue)
        m = pop!(queue)
        for n in names(m; all = true)
            isdefined(m, n) || continue
            v = try getfield(m, n) catch; continue end
            if v isa Module && !(v in seen) && startswith(string(v), "MeanFieldHomogenization")
                push!(seen, v); push!(queue, v)
            end
        end
    end
    return collect(seen)
end

const MODS = all_modules()

"Is `path` (bare or dotted) documented in any reachable module?"
function has_doc(path)
    parts = split(path, ".")
    sym = Symbol(parts[end])
    owners = Module[]
    if length(parts) > 1
        m = try
            foldl((mm, s) -> getfield(mm, Symbol(s)), parts[1:(end - 1)]; init = Main)
        catch
            nothing
        end
        m isa Module && push!(owners, m)
    end
    append!(owners, MODS)
    for m in owners
        b = Base.Docs.Binding(m, sym)
        haskey(Base.Docs.meta(m), b) && return true
        # `const B_tensor = cod_tensor`: Documenter follows the alias, so must we.
        a = try Base.Docs.aliasof(b) catch; nothing end
        if a !== nothing && a != b
            for mm in owners
                haskey(Base.Docs.meta(mm), a) && return true
            end
        end
    end
    return false
end

function scan()
    bad, n = String[], 0
    for (root, _, files) in walkdir(joinpath(pkgdir(MFH), "docs", "src"))
        for f in files
            endswith(f, ".md") || continue
            inblock = false
            for line in eachline(joinpath(root, f))
                s = strip(line)
                if startswith(s, "```@docs")
                    inblock = true; continue
                elseif inblock && startswith(s, "```")
                    inblock = false; continue
                end
                inblock || continue
                (isempty(s) || startswith(s, "#")) && continue
                name = first(split(s, r"[\s(]"))
                occursin(r"^[A-Za-z_][A-Za-z0-9_.!]*$", name) || continue
                n += 1
                has_doc(name) || push!(bad, "$(relpath(joinpath(root, f), pkgdir(MFH)))  ->  $name")
            end
        end
    end
    return bad, n
end

"""
Every name that appears in any `@docs` block, bare or dotted, as a `Set` of the
final path segments — which is what a `@docs` entry has to match by.
"""
function listed_names()
    out = Set{Symbol}()
    for (root, _, files) in walkdir(joinpath(pkgdir(MFH), "docs", "src"))
        for f in files
            endswith(f, ".md") || continue
            inblock = false
            for line in eachline(joinpath(root, f))
                s = strip(line)
                if startswith(s, "```@docs")
                    inblock = true; continue
                elseif inblock && startswith(s, "```")
                    inblock = false; continue
                end
                inblock || continue
                (isempty(s) || startswith(s, "#")) && continue
                name = first(split(s, r"[\s(]"))
                occursin(r"^[A-Za-z_][A-Za-z0-9_.!]*$", name) || continue
                push!(out, Symbol(last(split(name, "."))))
            end
        end
    end
    return out
end

"""
Exported names carrying a docstring that no `@docs` block lists.

Only docstrings **owned** by a reachable `MeanFieldHomogenization` module count.
That restriction is the whole difficulty: `MFH` re-exports a large part of
`TensND`, and those bindings have docstrings that belong to `TensND` and are
documented there. Asking `names(MFH)` instead reports twenty of them and the
list becomes useless.
"""
function unlisted_exports()
    listed = listed_names()
    bad = Tuple{Module, Symbol}[]
    seen = Set{Symbol}()
    for m in MODS, s in names(m)          # exported only: no `all = true`
        s in seen && continue
        isdefined(m, s) || continue
        b = Base.Docs.Binding(m, s)
        haskey(Base.Docs.meta(m), b) || continue   # not owned here
        push!(seen, s)
        s in listed || push!(bad, (m, s))
    end
    return sort(bad; by = t -> (string(t[1]), string(t[2])))
end

bad, n = scan()
println("checked $n names in @docs blocks across ", length(MODS), " modules")
isempty(bad) ? println("OK: every one carries a docstring") :
    (println("MISSING DOCSTRING ($(length(bad))):"); foreach(b -> println("  ", b), bad))

missing_entry = unlisted_exports()
println("checked ", length(names(MFH)), " exports of MFH plus its sub-modules'")
isempty(missing_entry) ? println("OK: every documented export is listed in a @docs block") :
    (
    println("NOT IN ANY @docs BLOCK ($(length(missing_entry))):");
        foreach(t -> println("  ", t[1], ".", t[2]), missing_entry)
)

exit(isempty(bad) && isempty(missing_entry) ? 0 : 1)
