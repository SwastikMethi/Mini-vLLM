import json
from pathlib import Path

import pytest

DATA = Path(__file__).parent / "data"


@pytest.fixture(scope="session")
def cpu_model():
    from mini_vllm.model import load

    return load("cpu")


@pytest.fixture(scope="session")
def reference_cpu():
    return json.loads((DATA / "reference_cpu.json").read_text())["cases"]
