# Opening the browser on the studio's page. On a remote session (SSH, a VS Code
# remote window) the browser to open is on the user's own machine, which this
# process cannot reach: the URL is printed with the way to forward the port,
# and `$BROWSER`, which VS Code sets to a helper that does reach it, is honored.

"`true` inside an SSH session or a VS Code remote window."
is_remote_session() = haskey(ENV, "SSH_CONNECTION") || haskey(ENV, "VSCODE_IPC_HOOK_CLI")

"""
    browser_command(url) -> Union{Nothing, Cmd}

The command that opens `url`: `\$BROWSER` when it is set, otherwise the
system's opener. `nothing` on a remote session without `\$BROWSER`, where no
local opener would show the page to the user.
"""
function browser_command(url::AbstractString)
    custom = get(ENV, "BROWSER", "")
    isempty(custom) || return Cmd([Base.shell_split(custom)..., url])
    is_remote_session() && return nothing
    Sys.isapple() && return `open $url`
    # `rundll32` takes the URL as one argument, where `cmd /c start` would
    # have to be taught the quoting of `&` and `?`.
    Sys.iswindows() && return `rundll32 url.dll,FileProtocolHandler $url`
    return `xdg-open $url`
end

"""
    open_browser(url) -> Bool

Ask the browser to open `url`; `false` when there is no way to (the URL is
printed in any case).
"""
function open_browser(url::AbstractString)
    cmd = browser_command(url)
    cmd === nothing && return false
    try
        run(pipeline(cmd; stdout = devnull, stderr = devnull); wait = false)
    catch err
        # no opener installed (a minimal Linux without xdg-open)
        err isa Base.IOError || rethrow()
        return false
    end
    return true
end

"The lines printed at start: the URL, and on a remote session how to reach it."
function banner(name::AbstractString, url::AbstractString, port::Integer)
    lines = ["$name → $url"]
    is_remote_session() && push!(
        lines,
        "  (remote session: open that URL on your own machine; VS Code forwards the port," *
            " and so does `ssh -L $port:127.0.0.1:$port`)",
    )
    return lines
end
