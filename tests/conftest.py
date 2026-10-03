import pytest

from forecast.data.ingest import load_m5_long
from forecast.data.synthetic import make_synthetic_m5


@pytest.fixture(scope="session")
def raw_dir(tmp_path_factory):
    return make_synthetic_m5(tmp_path_factory.mktemp("raw"), n_days=300)


@pytest.fixture(scope="session")
def long_df(raw_dir):
    return load_m5_long(raw_dir, state="CA", category="FOODS")
