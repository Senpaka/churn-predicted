import pytest
import pandas as pd
from datetime import datetime

class TestHealth:

    def test_health(self, client):
        response = client.get("/health")

        assert response.status_code == 200

        data = response.json()

        assert "status" in data
        assert "model_version" in data
        assert "timestamp" in data
        assert "model_loaded" in data
