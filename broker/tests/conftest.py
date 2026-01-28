import pytest


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", default=False, help="run slow lifecycle tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full lifecycle test (provisions real GPU, costs money)")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="pass --slow to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
