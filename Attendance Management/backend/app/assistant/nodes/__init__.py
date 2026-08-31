"""Graph nodes. One responsibility each, one file per concern.

Every node has the same shape::

    def node(state: Mapping, ctx: RunContext) -> Mapping   # partial update

so each is callable in a unit test with a hand-built dict and no database.
"""
