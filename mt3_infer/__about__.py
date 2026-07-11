"""Single source of truth for the package version.

Read by both `pyproject.toml` (via `[tool.hatch.version] path = ...`, so the
built wheel/sdist metadata always matches) and `mt3_infer/__init__.py` (so
`mt3_infer.__version__` matches too). Previously these were two separately
hand-maintained "0.1.3" literals that could drift.
"""

__version__ = "0.2.0"
