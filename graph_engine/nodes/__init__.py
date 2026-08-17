"""Graph nodes. Signature is `(state, ctx) -> partial update`, as in the runtime.

Deterministic by default. Where a node can delegate judgement to an LLM it does
so only when `ctx.llm` is configured, and the deterministic path remains the
fallback — so the engine is fully executable with no model available.
"""
