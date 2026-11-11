from unittest.mock import MagicMock

import vertexai_shared.client as client


def test_build_endpoint_private_psc(monkeypatch):
    private_endpoint_cls = MagicMock()
    aiplatform = MagicMock(PrivateEndpoint=private_endpoint_cls)
    monkeypatch.setattr(client, "_aiplatform_module", lambda: aiplatform)

    client.build_endpoint(endpoint_id="projects/x/locations/y/endpoints/123", endpoint_type="PRIVATE_PSC")
    private_endpoint_cls.assert_called_once_with(endpoint_name="projects/x/locations/y/endpoints/123")


def test_build_endpoint_dedicated(monkeypatch):
    endpoint_cls = MagicMock()
    aiplatform = MagicMock(Endpoint=endpoint_cls)
    monkeypatch.setattr(client, "_aiplatform_module", lambda: aiplatform)

    client.build_endpoint(endpoint_id="projects/x/locations/y/endpoints/123", endpoint_type="DEDICATED")
    endpoint_cls.assert_called_once_with(endpoint_name="projects/x/locations/y/endpoints/123")


def test_predict_psc_routes_endpoint_override():
    endpoint = MagicMock()
    instances = [{"prompt": "x"}]
    client.predict(
        endpoint=endpoint,
        endpoint_type="PRIVATE_PSC",
        instances=instances,
        psc_address="10.0.0.1",
    )
    endpoint.predict.assert_called_once_with(instances=instances, endpoint_override="10.0.0.1")


def test_predict_dedicated_routes_use_dedicated_endpoint():
    endpoint = MagicMock()
    instances = [{"prompt": "x"}]
    client.predict(
        endpoint=endpoint,
        endpoint_type="DEDICATED",
        instances=instances,
        psc_address=None,
    )
    endpoint.predict.assert_called_once_with(instances=instances, use_dedicated_endpoint=True)
