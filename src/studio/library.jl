# The worked examples of the library (src/studio/library): ten models to open
# and take apart, each a running script and a studio model. An example is
# written by the code generation from the model it embeds, with a note under its
# banner saying what it shows and after which demo script: regenerating it from
# its own model, note kept, must give the file back, so that the examples follow
# the code generation and never drift from what the studio writes.

const LIBRARY_DIR = joinpath(STUDIO_ROOT, "library")

"The example scripts of the library, in their order."
library_files() = sort!(filter(f -> endswith(f, ".jl"), readdir(LIBRARY_DIR; join = true)))

"""
    regenerate_example(text) -> String

The example `text`, written again by the code generation from the model it
embeds, its note (the comment block between the banner and the code) kept.
"""
function regenerate_example(text::AbstractString)
    embedded = extract_embedded(text)
    embedded === nothing && throw(ArgumentError("the example embeds no model"))
    lines = split(text, '\n')
    # the note sits after the second rule of the banner, up to the first blank line
    banner_end = findall(l -> startswith(l, "# ===="), lines)[2]
    note_end = findnext(isempty, lines, banner_end + 1)
    note = lines[(banner_end + 1):(note_end - 1)]
    generated = split(generate(model_from_dict(embedded)), '\n')
    at = findall(l -> startswith(l, "# ===="), generated)[2]
    return join(vcat(generated[1:at], note, generated[(at + 1):end]), '\n')
end

"""
    regenerate_library!(dir = LIBRARY_DIR) -> Vector{String}

Write every example of `dir` again from its own model (after a change of the
code generation), and return the files that changed.
"""
function regenerate_library!(dir::AbstractString = LIBRARY_DIR)
    changed = String[]
    for f in sort!(filter(f -> endswith(f, ".jl"), readdir(dir; join = true)))
        old = read(f, String)
        new = regenerate_example(old)
        new == old && continue
        write(f, new)
        push!(changed, f)
    end
    return changed
end
