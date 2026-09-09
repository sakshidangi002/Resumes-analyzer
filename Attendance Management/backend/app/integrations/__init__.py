"""Adapters to external systems this application does not own.

Anything in here talks over the network to somebody else's service (an MCP
server, a calendar provider, a webhook sink). Business rules never live here —
those belong in ``app.services``. An integration module should be replaceable
without any HR logic changing.
"""
