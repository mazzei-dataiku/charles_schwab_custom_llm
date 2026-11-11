def _aiplatform_module():
    from google.cloud import aiplatform

    return aiplatform


def init_aiplatform(*, project_id: str, location: str, credentials=None) -> None:
    """Initialize Vertex AI SDK exactly as the reference implementation."""

    if credentials is None:
        _aiplatform_module().init(project=project_id, location=location)
    else:
        _aiplatform_module().init(
            project=project_id, location=location, credentials=credentials
        )


def build_endpoint(*, endpoint_id: str, endpoint_type: str):
    endpoint_type = (endpoint_type or "PRIVATE_PSC").strip().upper()
    if endpoint_type == "PRIVATE_PSC":
        return _aiplatform_module().PrivateEndpoint(endpoint_name=endpoint_id)
    return _aiplatform_module().Endpoint(endpoint_name=endpoint_id)


def predict(*, endpoint, endpoint_type: str, instances, psc_address=None):
    endpoint_type = (endpoint_type or "PRIVATE_PSC").strip().upper()

    if endpoint_type == "PRIVATE_PSC":
        return endpoint.predict(instances=instances, endpoint_override=psc_address)
    return endpoint.predict(instances=instances, use_dedicated_endpoint=True)
