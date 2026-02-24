"""Shared test fixtures for MIMIC-IV Prescriptive ICU test suite."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent
RESEARCH_ROOT = PROJECT_ROOT.parent.parent  # research/
_src_path = str(PROJECT_ROOT / "src")
_research_path = str(RESEARCH_ROOT)
if _src_path not in sys.path:
    sys.path.insert(0, _src_path)
if _research_path not in sys.path:
    sys.path.insert(0, _research_path)

CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"


@pytest.fixture(scope="session")
def config():
    """Load the default configuration."""
    from extract import load_config

    return load_config(CONFIGS_DIR / "default.yaml")


@pytest.fixture(scope="session")
def tables(config):
    """Load all MIMIC-IV tables (cached for session)."""
    from extract import load_tables

    return load_tables(config)


@pytest.fixture(scope="session")
def dataset(config):
    """Build the primary dataset (cached for session)."""
    from extract import build_dataset

    return build_dataset(config, cohort_type="primary")


@pytest.fixture()
def rng():
    """Reproducible random number generator."""
    return np.random.default_rng(42)
