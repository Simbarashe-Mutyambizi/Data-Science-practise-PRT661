import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def data():
    from lthc_dashboard.data import get_data

    return get_data()


@pytest.fixture(scope="session")
def model():
    from lthc_dashboard.model import get_model

    return get_model()


@pytest.fixture(scope="session")
def local_client():
    from lthc_dashboard.inference import InferenceClient

    return InferenceClient(api_url="")
