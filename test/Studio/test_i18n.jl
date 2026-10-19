# The page in French: every English text the page shows (a `t("…")` of its
# scripts, an element of index.html marked `data-i18n`, a label or a note of the
# catalog sections the page displays) has its French in i18n/fr.json, and
# fr.json translates nothing else, so that it follows the page as it changes.

"The string literals of a JavaScript concatenation `\"a\" + \"b\"` at the start of `s`, joined."
function js_concatenation(s::AbstractString)
    parts = String[]
    rest = lstrip(s)
    while (m = match(r"^\"((?:[^\"\\]|\\.)*)\"\s*", rest)) !== nothing
        push!(parts, unescape_string(m[1]))
        rest = rest[(ncodeunits(m.match) + 1):end]
        (n = match(r"^\+\s*", rest)) === nothing && break
        rest = rest[(ncodeunits(n.match) + 1):end]
    end
    return join(parts)
end

squash(s) = strip(replace(s, r"\s+" => " "))

"Every English text the page asks to translate."
function page_words()
    words = Set{String}()
    for f in ("app.js", "graph.js", "picker.js")
        src = read(joinpath(MS.WEB_DIR, f), String)
        # `t("…")`, not `t(x.label)` (the catalog's, below) nor `split(`
        for m in eachmatch(r"(?<![\w.$])t\((?=\s*\")", src)
            push!(words, js_concatenation(src[(m.offset + 2):end]))
        end
    end
    html = read(joinpath(MS.WEB_DIR, "index.html"), String)
    for m in eachmatch(r"<\w+[^>]*\sdata-i18n(?=[\s>])[^>]*>([^<]*)<"s, html)
        push!(words, squash(m[1]))
    end
    for (attr, mark) in (("title", "data-i18n-title"), ("placeholder", "data-i18n-placeholder"))
        for m in eachmatch(Regex("<\\w+[^>]*\\s$mark(?=[\\s>])[^>]*>"), html)
            push!(words, squash(match(Regex("\\s$attr=\"([^\"]*)\""), m.match)[1]))
        end
    end
    # what the page shows of the catalog: labels and notes holding a word
    catalog = MS.catalog()
    walk(x) = nothing
    walk(x::AbstractVector) = foreach(walk, x)
    function walk(x::Union{NamedTuple, AbstractDict})
        for (k, v) in pairs(x)
            if Symbol(k) in (:label, :doc) && v isa AbstractString
                occursin(r"[A-Za-z]{3}", v) && push!(words, v)
            else
                walk(v)
            end
        end
        return nothing
    end
    foreach(s -> walk(catalog[s]), ("geometries", "properties", "symmetrize", "projections", "interfaces", "lenses", "sensitivities"))
    return words
end

words = page_words()
french = JSON.parse(read(joinpath(MS.WEB_DIR, "i18n", "fr.json"), String); dicttype = Dict{String, Any})

@test !isempty(words) && all(v -> v isa String && !isempty(v), values(french))
@test isempty(sort!([w for w in words if !haskey(french, w)]))       # nothing left in English
@test isempty(sort!([w for w in keys(french) if !(w in words)]))     # nothing stale
# every `{name}` survives the translation
marks(s) = sort!([m.match for m in eachmatch(r"\{\w+\}", s)])
@test isempty([en for (en, fr) in french if marks(en) != marks(fr)])
