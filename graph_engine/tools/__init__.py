"""Tools: the only channels through which the graph touches the outside world.

Nodes never call `open()`, `subprocess` or `git` directly. Everything goes
through these wrappers so path validation, the write allowlist, backups and
command allowlisting are enforced in one place instead of at every call site.
"""
