"""Graph Engineering: an executable software-engineering workflow over this repo.

    review -> bug analysis -> fix -> test -> verify
                                 ^            |
                                 |            v
                          failure analysis <- (tests failed)

This is a *developer tool*, not part of the served HRM application. It imports
no HRM business logic; it treats the repository as files, runs the real test
suite in a subprocess, and edits source through a path-validated, backup-taking
writer.

Run it with::

    python -m graph_engine --goal "..." --scope "path/under/repo"

See `graph_engine/README.md` for the full architecture.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
