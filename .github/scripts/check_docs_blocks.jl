# Every name listed in an `@docs` block must carry a docstring somewhere
# Documenter can reach. Documenter only discovers a missing one during a full
# build (~90 min here); this costs seconds and fails on the same condition.
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

bad, n = scan()
println("checked $n names in @docs blocks across ", length(MODS), " modules")
isempty(bad) ? println("OK: every one carries a docstring") :
    (println("MISSING ($(length(bad))):"); foreach(b -> println("  ", b), bad))
exit(isempty(bad) ? 0 : 1)
