#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Single source of truth for every filesystem location used by this project.

The original code carried absolute paths tied to one HPC account
(an HPC home directory and a separate work volume). Those are replaced here by
values derived from the repository's own location, each overridable through an
environment variable so the pipeline runs unchanged on another machine:

======================  ==========================  ==================================
Environment variable    Default                     Holds
======================  ==========================  ==================================
``AP7_PROJECT_ROOT``    repository root             code, paper, results
``AP7_DATA_DIR``        ``<root>/dades``            input CSVs (not redistributed)
``AP7_EXPERIMENTS_DIR`` ``<root>/experiments``      model artefacts + simulations
``AP7_RESULTS_DIR``     ``<root>/reports/results``  regenerated result tables
``AP7_FIGURES_DIR``     ``<root>/reports/figures``  regenerated figures
``AP7_TABLES_DIR``      ``<root>/reports/tables``   regenerated LaTeX tables
======================  ==========================  ==================================

Experiment artefacts are large (hundreds of GB) and are normally kept off the
repository volume; point ``AP7_EXPERIMENTS_DIR`` at that storage.

Usage::

    from src.paths import EXPERIMENTS_ROOT, DATA_DIR
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(var: str, default: Path) -> Path:
    value = os.environ.get(var, "").strip()
    return Path(value).expanduser().resolve() if value else default


# src/paths.py -> src -> repository root
PROJECT_ROOT: Path = _env_path(
    "AP7_PROJECT_ROOT", Path(__file__).resolve().parent.parent
)

DATA_DIR: Path = _env_path("AP7_DATA_DIR", PROJECT_ROOT / "dades")
EXPERIMENTS_ROOT: Path = _env_path(
    "AP7_EXPERIMENTS_DIR", PROJECT_ROOT / "experiments"
)
# Regenerated outputs go under reports/ (not tracked), so re-running an
# analysis never overwrites the published copies in paper/ and results/.
RESULTS_DIR: Path = _env_path("AP7_RESULTS_DIR", PROJECT_ROOT / "reports" / "results")
FIGURES_DIR: Path = _env_path("AP7_FIGURES_DIR", PROJECT_ROOT / "reports" / "figures")
TABLES_DIR: Path = _env_path("AP7_TABLES_DIR", PROJECT_ROOT / "reports" / "tables")

# String aliases: the pipeline predates pathlib use and joins these with
# os.path.join, which rejects Path objects in a few older call sites.
PROJECT_ROOT_STR = str(PROJECT_ROOT)
DATA_DIR_STR = str(DATA_DIR)
EXPERIMENTS_ROOT_STR = str(EXPERIMENTS_ROOT)

# Names the original modules used, kept so imports resolve unchanged.
PROD_ENV_ROOT = PROJECT_ROOT_STR
PROD_ROOT = PROJECT_ROOT_STR


def ensure_dirs() -> None:
    """Create the writable output directories if they do not yet exist."""
    for d in (EXPERIMENTS_ROOT, RESULTS_DIR, FIGURES_DIR, TABLES_DIR):
        d.mkdir(parents=True, exist_ok=True)


__all__ = [
    "PROJECT_ROOT", "DATA_DIR", "EXPERIMENTS_ROOT", "RESULTS_DIR",
    "FIGURES_DIR", "TABLES_DIR", "PROJECT_ROOT_STR", "DATA_DIR_STR",
    "EXPERIMENTS_ROOT_STR", "PROD_ENV_ROOT", "PROD_ROOT", "ensure_dirs",
]
