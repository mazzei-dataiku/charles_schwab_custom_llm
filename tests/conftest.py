import sys
from pathlib import Path

import pytest


python_lib = Path(__file__).resolve().parents[1] / "python-lib"
sys.path.insert(0, str(python_lib))


@pytest.fixture(autouse=True)
def _stub_google_modules(monkeypatch):
    """Provide lightweight stubs so tests don't need google deps installed."""

    class _ServiceAccountCredentials:
        @classmethod
        def from_service_account_info(cls, info, scopes):
            return {"info": info, "scopes": scopes}

    google = type(sys)("google")
    google_oauth2 = type(sys)("google.oauth2")
    google_oauth2_service_account = type(sys)("google.oauth2.service_account")
    google_oauth2_service_account.Credentials = _ServiceAccountCredentials

    google_cloud = type(sys)("google.cloud")
    google_cloud_aiplatform = type(sys)("google.cloud.aiplatform")

    def _noop_init(**kwargs):
        return None

    google_cloud_aiplatform.init = _noop_init
    google_cloud_aiplatform.Endpoint = object
    google_cloud_aiplatform.PrivateEndpoint = object

    monkeypatch.setitem(sys.modules, "google", google)
    # Stub google.auth and its request transport (used by vertexai_shared.gemini)
    google_auth = type(sys)("google.auth")
    google_auth_transport = type(sys)("google.auth.transport")
    google_auth_transport_requests = type(sys)("google.auth.transport.requests")

    class _AuthRequest:
        pass

    def _default(scopes=None):
        class _Creds:
            valid = False
            token = None

            def refresh(self, request):
                self.valid = True
                self.token = "token_from_adc"

        return _Creds(), "project"

    google_auth.default = _default
    google_auth_transport_requests.Request = _AuthRequest

    monkeypatch.setitem(sys.modules, "google.auth", google_auth)
    monkeypatch.setitem(sys.modules, "google.auth.transport", google_auth_transport)
    monkeypatch.setitem(
        sys.modules, "google.auth.transport.requests", google_auth_transport_requests
    )

    monkeypatch.setitem(sys.modules, "google.oauth2", google_oauth2)
    monkeypatch.setitem(
        sys.modules, "google.oauth2.service_account", google_oauth2_service_account
    )
    monkeypatch.setitem(sys.modules, "google.cloud", google_cloud)
    monkeypatch.setitem(sys.modules, "google.cloud.aiplatform", google_cloud_aiplatform)

    # Stub requests for REST calls (used by vertexai_shared.gemini)
    requests = type(sys)("requests")

    class _Response:
        ok = True
        status_code = 200
        text = ""

        def __init__(self, payload=None):
            self._payload = payload or {}

        def json(self):
            return self._payload

    def _post(url, headers=None, json=None, timeout=None):
        return _Response({})

    requests.post = _post
    monkeypatch.setitem(sys.modules, "requests", requests)


@pytest.fixture(autouse=True)
def _stub_dataiku_basellm(monkeypatch):
    """Stub Dataiku BaseLLM to avoid requiring DSS runtime deps."""

    dataiku = type(sys)("dataiku")
    dataiku_llm = type(sys)("dataiku.llm")
    dataiku_llm_python = type(sys)("dataiku.llm.python")

    class BaseLLM:  # minimal stub for import
        pass

    dataiku_llm_python.BaseLLM = BaseLLM

    monkeypatch.setitem(sys.modules, "dataiku", dataiku)
    monkeypatch.setitem(sys.modules, "dataiku.llm", dataiku_llm)
    monkeypatch.setitem(sys.modules, "dataiku.llm.python", dataiku_llm_python)
