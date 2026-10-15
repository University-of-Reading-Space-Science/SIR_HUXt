"""
Shared pytest configuration for the SIR_HUXt ``src`` test suite.

Run from the ``src`` folder:

    pytest tests                      # fast tests (seconds)
    pytest tests --run-slow           # also the regression test that solves real SURF models (minutes)
    pytest tests --update-golden      # (re)write the regression baselines in tests/data/

Tests that need a package which is not installed (astropy, xarray, surf, huxt, ...) are skipped,
not failed, so the pure-numpy tests can run anywhere.
"""
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")  # never open plot windows during tests

import pytest

TESTS_DIR = pathlib.Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent
DATA_DIR = TESTS_DIR / "data"

# Make the src modules (flat layout) and the test helpers importable
for p in (str(SRC_DIR), str(TESTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False,
        help="run slow tests that solve real SURF models",
    )
    parser.addoption(
        "--update-golden", action="store_true", default=False,
        help="(re)write the regression baselines in tests/data/ instead of comparing against them",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: solves real SURF models (run with --run-slow)")
    config.addinivalue_line("markers", "regression: compares against a stored baseline in tests/data/")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow"):
        return
    skip_slow = pytest.mark.skip(reason="slow test: run with --run-slow")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)


@pytest.fixture
def update_golden(request) -> bool:
    """True when the run should (re)write the regression baselines."""
    return request.config.getoption("--update-golden")


@pytest.fixture(autouse=True)
def _close_figures():
    """Close any matplotlib figures a test left open."""
    yield
    import matplotlib.pyplot as plt
    plt.close("all")


@pytest.fixture
def in_tmp_dir(tmp_path, monkeypatch):
    """Run the test with tmp_path as the working directory (some modules write relative files)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path
