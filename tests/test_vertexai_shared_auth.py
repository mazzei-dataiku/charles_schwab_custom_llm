import pytest

from vertexai_shared.auth import build_credentials


def test_build_credentials_adc_default_returns_none():
    assert build_credentials({"auth_method": "ADC"}) is None
    assert build_credentials({}) is None


def test_build_credentials_invalid_auth_method():
    with pytest.raises(TypeError) as exc:
        build_credentials({"auth_method": "NOPE"})
    assert "Invalid plugin setting: auth_method must be one of ADC or SERVICE_ACCOUNT" in str(
        exc.value
    )


def test_build_credentials_service_account_missing_json():
    with pytest.raises(TypeError) as exc:
        build_credentials({"auth_method": "SERVICE_ACCOUNT"})
    assert (
        "Missing plugin setting: service_account_json is required for auth_method=SERVICE_ACCOUNT"
        in str(exc.value)
    )


def test_build_credentials_service_account_invalid_json_string():
    with pytest.raises(TypeError) as exc:
        build_credentials(
            {"auth_method": "SERVICE_ACCOUNT", "service_account_json": "{not json"}
        )
    assert "Invalid plugin setting: service_account_json must be valid JSON" in str(exc.value)


def test_build_credentials_service_account_requires_fields():
    with pytest.raises(TypeError) as exc:
        build_credentials(
            {
                "auth_method": "SERVICE_ACCOUNT",
                "service_account_json": {"client_email": "x"},
            }
        )
    assert (
        "Invalid plugin setting: service_account_json is missing required fields" in str(exc.value)
    )

