import json


def _service_account_module():
    from google.oauth2 import service_account

    return service_account


_CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"


def build_credentials(plugin_config: dict):
    """Build google-auth credentials based on plugin configuration.

    Mirrors the behavior and validation from `python-llms/vertex-ai/llm.py`.

    Returns:
        Credentials instance or None (for ADC)
    """

    auth_method = (plugin_config.get("auth_method") or "ADC").strip().upper()
    if auth_method not in {"ADC", "SERVICE_ACCOUNT"}:
        raise TypeError(
            "Invalid plugin setting: auth_method must be one of ADC or SERVICE_ACCOUNT"
        )

    if auth_method == "ADC":
        return None

    raw_service_account = plugin_config.get("service_account_json")
    if not raw_service_account:
        raise TypeError(
            "Missing plugin setting: service_account_json is required for auth_method=SERVICE_ACCOUNT"
        )

    if isinstance(raw_service_account, dict):
        service_account_info = raw_service_account
    elif isinstance(raw_service_account, str):
        try:
            service_account_info = json.loads(raw_service_account)
        except json.JSONDecodeError as exc:
            raise TypeError(
                "Invalid plugin setting: service_account_json must be valid JSON"
            ) from exc
    else:
        raise TypeError(
            "Invalid plugin setting: service_account_json must be a JSON string"
        )

    if not isinstance(service_account_info, dict):
        raise TypeError(
            "Invalid plugin setting: service_account_json must be a JSON object"
        )

    if not service_account_info.get("client_email") or not service_account_info.get(
        "private_key"
    ):
        raise TypeError(
            "Invalid plugin setting: service_account_json is missing required fields"
        )

    service_account = _service_account_module()
    return service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=[_CLOUD_PLATFORM_SCOPE],
    )
