import sys

from falcon import testing
import pytest

sys.argv = [sys.argv[0]]
from gamutrf import api


@pytest.fixture(scope='module')
def client():
    app = api.API(start_app=False)
    return testing.TestClient(app.create_app())

def test_routes(client):
    result = client.simulate_get('/v1')
    assert result.status_code == 200
    result = client.simulate_get('/v1/info')
    assert result.status_code == 200
    result = client.simulate_get('/v1/record/100000000/20000000/20000000')
    assert result.status_code == 200
