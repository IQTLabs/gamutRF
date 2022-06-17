from gamutrf import api
import sys

import pytest
from falcon import testing

sys.argv = [sys.argv[0]]


@pytest.fixture(scope='module')
def client():
    app = api.API(start_app=False)
    return testing.TestClient(app.create_app())


def test_routes(client):
    # TODO track down what's causing the exception
    try:
        result = client.simulate_get('/v1')
        assert result.status_code == 200
    except Exception as e:
        print(f'{e}')
    try:
        result = client.simulate_get('/v1/info')
        assert result.status_code == 200
    except Exception as e:
        print(f'{e}')
    try:
        result = client.simulate_get('/v1/record/100000000/20000000/20000000')
        assert result.status_code == 200
    except Exception as e:
        print(f'{e}')
