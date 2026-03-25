import pytest
from pydantic import ValidationError
from api.schemas import PredictionResponse, UserFeatures

class TestUserFeatures():

    def test_valid_user(self, sample_customer):
        user = UserFeatures(**sample_customer)
        assert user.Age == 42
        assert user.Gender == "Female"

    def test_invalid_user_age(self, sample_customer):
        sample_customer["Age"] = 130
        with pytest.raises(ValidationError) as excinfo:
            UserFeatures(**sample_customer)
        assert "Age" in str(excinfo.value)

    def test_invalid_user_credit_score_to_low(self, sample_customer):
        sample_customer["CreditScore"] = 100
        with pytest.raises(ValidationError) as excinfo:
            UserFeatures(**sample_customer)
        assert "CreditScore" in str(excinfo.value)

    def test_invalid_user_balance(self, sample_customer):
        sample_customer["Balance"] = -123412
        with pytest.raises(ValidationError) as excinfo:
            UserFeatures(**sample_customer)
        assert "Balance" in str(excinfo.value)

    def test_value_of_geography(self, sample_customer):
        geography = ["France", "Germany", "Spain"]

        for geo in geography:
            sample_customer["Geography"] = geo
            user = UserFeatures(**sample_customer)
            assert user.Geography == geo

        geography = "Russia"
        sample_customer["Geography"] = geography

        with pytest.raises(ValidationError):
            UserFeatures(**sample_customer)

    def test_value_of_gender(self, sample_customer):
        gender = ["Male", "Female"]

        for gender in gender:
            sample_customer["Gender"] = gender
            user = UserFeatures(**sample_customer)
            assert user.Gender == gender

        gender = "nonbinary"
        sample_customer["Gender"] = gender

        with pytest.raises(ValidationError):
            UserFeatures(**sample_customer)