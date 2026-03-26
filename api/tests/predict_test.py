import pytest
import json

class TestPredict:

    def test_predict_single_valid(self, client, sample_customer):

        response = client.post("/prediction", json=sample_customer)

        if response.status_code == 200:
            data = response.json()

            assert "customer_id" in data
            assert "churn_probability" in data
            assert "prediction" in data
            assert "risk_level" in data
            assert "timestamp" in data
            assert 0 <= data["churn_probability"] <= 1
            assert data["risk_level"] in ["low risk", "mid risk", "high risk"]

        else:
            assert response.status_code == 500

    def test_batch_predict_valid(self, client, sample_customers_batch):
        response = client.post("/prediction_batch", json=sample_customers_batch)

        if response.status_code == 200:
            data = response.json()

            assert "predictions" in data
            assert "total" in data
            assert "churn_count" in data
            assert "timestamp" in data
            assert data["total"] == 2
            assert len(data["predictions"]) == 2
        else:
            assert response.status_code == 500

    def test_predict_single_invalid(self, client):
        invalid_data = {"CreditScore": 999}
        response = client.post("/prediction", json=invalid_data)
        assert response.status_code == 422

    def test_predict_batch_invalid_structure(self, client, sample_customers_batch):
        response = client.post("/prediction_batch", json={"wrong": "hehehe"})
        assert response.status_code == 422

    def test_predict_batch_empty(self, client):
        response = client.post("/prediction_batch", json={"users": []})

        if response.status_code == 200:
            data = response.json()
            assert data["total"] == 0
            assert len(data["predictions"]) == 0
        else:
            assert response.status_code in [400, 500]

