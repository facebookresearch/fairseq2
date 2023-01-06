import pytest
import torch

from tests.common import TestCase

# Hack around of https://github.com/pytorch/tnt/issues/285
if not hasattr(torch.optim.lr_scheduler, "LRScheduler"):
    setattr(torch.optim.lr_scheduler, "LRScheduler", torch.optim.lr_scheduler._LRScheduler)  # type: ignore


def pytest_addoption(parser: pytest.Parser, pluginmanager: None) -> None:
    parser.addoption(
        "--device",
        default="cpu",
        help="device on which to run tests (default: %(default)s)",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    # This is only required because of the tests using TestCase
    device = torch.device(session.config.getoption("device"))
    TestCase.device = device


@pytest.fixture(scope="session")
def device(pytestconfig: pytest.Config) -> torch.device:
    return torch.device(pytestconfig.getoption("device"))
