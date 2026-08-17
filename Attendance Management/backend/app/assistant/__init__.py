"""HRM assistant: a stateful workflow graph over the existing HRMS services.

The assistant does NOT re-implement any HRM business logic. Every node reads
through `app.services.*` exactly as the REST routes do, so attendance rules,
holiday handling and IST date semantics stay in one place.

Public entry point is `app.api.routes.assistant`, which authenticates with the
normal `get_current_user` dependency and hands an already-authorised actor to
the graph. See `docs/assistant-graph.md` for the architecture.
"""
