# The reading path of a chapter, checked against its page tree.
#
# A chapter of `pages.jl` that opens with an `index.md` presents its pages there,
# group by group. This file checks that every page of such a chapter is linked
# from that index by an `@ref` to the identifier of its title: a page added to
# the tree and left out of the index is found by no reader who follows the
# reading path.
#
# `make.jl` includes this file and calls `check_page_tree` once the generated
# pages exist. Run alone, it checks the pages present in the source tree:
#
#     julia docs/page_tree.jl

"The page files of a node of the page tree, in order."
page_files(page::AbstractString) = [String(page)]
page_files(node::Pair) = page_files(node.second)
page_files(nodes::AbstractVector) = reduce(vcat, map(page_files, nodes); init = String[])

"The identifier of the title of a page, or `nothing` when its title carries none."
function title_id(file::AbstractString)
    m = match(r"^# \[.*\]\(@id ([^)\s]+)\)"m, read(file, String))
    return m === nothing ? nothing : String(m.captures[1])
end

"""
    unlisted_pages(pages, srcdir) -> Vector{String}

The pages of the chapters that open with an `index.md` and are not linked from
it. A page file absent from `srcdir` (a generated page, before its generation)
is not judged.
"""
function unlisted_pages(pages, srcdir::AbstractString)
    missing_pages = String[]
    for chapter in pages
        chapter isa Pair && chapter.second isa AbstractVector || continue
        files = page_files(chapter.second)
        index = first(files)
        endswith(index, "/index.md") || continue
        text = read(joinpath(srcdir, index), String)
        for page in files[2:end]
            file = joinpath(srcdir, page)
            isfile(file) || continue
            id = title_id(file)
            id === nothing && error("$page: its title has no `@id`, so no index can link to it")
            occursin(Regex("@ref\\s+\\Q$id\\E\\)"), text) || push!(missing_pages, "$page ($id), absent from $index")
        end
    end
    return missing_pages
end

function check_page_tree(pages, srcdir::AbstractString)
    missing_pages = unlisted_pages(pages, srcdir)
    isempty(missing_pages) || error(
        "$(length(missing_pages)) page(s) of the page tree are not presented in the index of " *
            "their chapter. List each one there, in the section of its group:\n  " *
            join(missing_pages, "\n  ")
    )
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    include(joinpath(@__DIR__, "pages.jl"))
    check_page_tree(pages, joinpath(@__DIR__, "src"))
    println("page tree: every page is presented in the index of its chapter")
end
