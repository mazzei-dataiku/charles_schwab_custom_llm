# Dataiku Vertex AI LLM Plugin

**Author:** Stephen Mazzei  
**Organization:** Dataiku  
**Version:** 1.0.2  
**Status:** Active Development  
**Platform:** Dataiku DSS / LLM Mesh  
**Integration:** Google Cloud Vertex AI

> **Author Notes**
>
> This plugin is intended to extend Dataiku's native LLM Mesh capabilities by providing custom integrations with models hosted on Google Cloud Vertex AI.
>
> Development focuses on maintaining compatibility with Dataiku Agents, Cobuild, and other LLM Mesh features while keeping the implementation lightweight, modular, and maintainable.
>
> This is a custom plugin implementation and should not be assumed to be an officially supported Dataiku product integration.
>
> **Version 1.0.2:** Improved Gemini tool-calling compatibility, token accounting, finish-reason reporting, and Cobuild integration. Additional compatibility testing and enhancements are ongoing.

---

## Overview

The **Dataiku Vertex AI LLM Plugin** provides a Python-based integration between Dataiku DSS and Google Cloud Vertex AI.

It allows Dataiku to interact with supported Vertex AI-hosted language models through the Dataiku LLM Mesh interface.

The plugin is designed to support:

- LLM completion requests
- Multi-turn conversations
- Dataiku Agents
- Function and tool calling
- Dataiku Cobuild
- Token usage reporting
- Model finish-reason handling
- Streaming-compatible responses

Model capabilities may vary depending on the implementation and the underlying Vertex AI endpoint.

## Architecture

The integration uses Dataiku's Python LLM plugin interface.

```text
Dataiku DSS
    |
    +-- LLM Mesh
          |
          +-- Python LLM Plugin
                |
                +-- Model Implementation
                |     |
                |     +-- Request Configuration
                |     +-- Message Conversion
                |     +-- Tool Definitions
                |     +-- Response Formatting
                |
                +-- Shared Vertex AI Libraries
                      |
                      +-- Authentication
                      +-- Gemini API Integration
                      +-- Tool Schema Conversion
                      +-- Usage Metadata
                      +-- Streaming Helpers
                            |
                            v
                    Google Cloud Vertex AI
                            |
                            v
                       Gemini Model
```

The architecture separates model-specific implementations from reusable Vertex AI functionality.

## Repository Structure

The following represents the known core components of the plugin.

```text
.
├── plugin.json
├── README.md
├── AGENTS.md
│
├── python-llms/
│   └── gemini-flash/
│       ├── llm.py
│       └── llm.json
│
├── python-lib/
│   └── vertexai_shared/
│       ├── auth.py
│       ├── gemini.py
│       └── streaming.py
│
└── tests/
    ├── test_gemini_flash_llm.py
    └── test_vertexai_shared_gemini.py
```

Additional model implementations and configuration files may be present depending on the repository version.

### Key Components

| Component | Purpose |
|---|---|
| `plugin.json` | Dataiku plugin metadata |
| `python-llms/` | Dataiku LLM implementations |
| `llm.py` | Dataiku `BaseLLM` integration |
| `llm.json` | Model configuration interface |
| `auth.py` | Google Cloud authentication |
| `gemini.py` | Gemini API and request/response conversion |
| `streaming.py` | Dataiku streaming response helpers |
| `tests/` | Unit and regression tests |

## Requirements

### Dataiku

- Dataiku DSS with Python LLM plugin support
- LLM Mesh enabled and configured
- Appropriate permissions to install and configure plugins

Cobuild integration has been successfully tested using **Dataiku DSS 14.7.3**.

### Google Cloud

- Google Cloud project
- Vertex AI API enabled
- Access to the intended Vertex AI model
- Appropriate IAM permissions for model inference
- Network connectivity between Dataiku DSS and Vertex AI

### Python

The plugin requires a compatible Python code environment with the dependencies used by its authentication and Vertex AI integration modules.

Refer to the plugin's code environment configuration for the exact Python version and dependency requirements.

## Installation

1. Install the plugin into Dataiku DSS using the plugin management interface or your established plugin deployment process.
2. Configure or update the plugin's Python code environment.
3. Verify the required Python dependencies are installed.
4. Configure Google Cloud authentication.
5. Create and configure an LLM connection using the plugin's model implementation.
6. Select the desired Vertex AI model and configure its supported parameters.
7. Test a basic completion request before enabling advanced workflows.

Configuration options depend on the model implementation.

## Authentication

The plugin uses shared authentication functionality located in:

`python-lib/vertexai_shared/auth.py`

Authentication must provide valid Google Cloud credentials with permission to invoke the configured Vertex AI endpoint.

Use the authentication method appropriate for the deployment environment and security requirements.

**Security recommendations:**

- Prefer managed credentials or workload identity where available.
- Avoid embedding service account keys directly in source code.
- Do not commit credentials, access tokens, or private keys.
- Apply least-privilege IAM permissions.
- Keep sensitive information out of application logs.

## Supported Capabilities

The following capabilities have been implemented or validated for the Gemini Flash integration.

| Capability | Status |
|---|---|
| Basic LLM completions | Implemented |
| Multi-turn conversation history | Implemented |
| Gemini function declarations | Implemented |
| Function-call response parsing | Implemented |
| Tool-result handling | Implemented |
| Multiple tool-calling rounds | Validated |
| Tool-choice configuration | Implemented |
| Prompt token reporting | Implemented |
| Completion token reporting | Implemented |
| Finish-reason reporting | Implemented |
| Streaming-compatible responses | Implemented |
| Dataiku Cobuild | Validated on DSS 14.7.3 |
| Structured output | Not yet validated |
| True incremental token streaming | Not implemented |

Streaming-compatible responses currently use a completed model response and emit it through the Dataiku streaming interface. This should not be confused with incremental token streaming from Vertex AI.

## Dataiku Cobuild Integration

The Gemini Flash implementation has been tested with Dataiku Cobuild on DSS 14.7.3.

### Validated Scenario: Web Application Creation

**Test prompt:**

```text
Create a simple HTML web application that allows a user
to enter their name and displays a personalized welcome
message.
```

**Result:** Successful

Cobuild successfully:

1. Communicated with the custom Gemini model through LLM Mesh.
2. Requested and executed Dataiku-managed tools.
3. Handled tool results across multiple model calls.
4. Created a Dataiku web application.
5. Updated HTML, JavaScript, and CSS files.
6. Completed the workflow and returned the created application.
7. Produced a functional web application verified in the browser.

The execution trace confirmed successful tool execution and populated token usage metadata.

This establishes compatibility for the tested workflow, but does not guarantee compatibility with every Cobuild operation or tool schema.

### Additional Testing

Further scenarios include:

- Reading and inspecting project datasets
- Building or modifying Dataiku Flow objects
- Updating existing web applications
- Interacting with database-backed datasets
- Handling additional Cobuild tool schemas
- Recovering from tool execution errors

## Tool Calling and JSON Schema Compatibility

Dataiku provides tool definitions using JSON Schema.

The plugin converts these definitions into the schema format expected by Gemini function declarations.

This conversion is implemented in:

`python-lib/vertexai_shared/gemini.py`

Some JSON Schema keywords accepted by Dataiku may not be supported by the target Vertex AI API.

Schema compatibility must be handled carefully to preserve tool argument structure and avoid unexpected failures.

For example, unsupported schema validation keywords may require explicit translation or omission.

Changes to schema normalization should include regression tests.

## Token Accounting and Finish Reasons

The Gemini implementation maps Vertex AI response metadata to the Dataiku LLM interface.

### Token Usage

| Vertex AI | Dataiku |
|---|---|
| `usageMetadata.promptTokenCount` | `promptTokens` |
| `usageMetadata.candidatesTokenCount` | `completionTokens` |

### Finish Reasons

Gemini finish reasons are translated into the corresponding Dataiku completion status where supported.

This includes normal completion, tool-calling responses, output-length limits, and safety-related termination.

Accurate token reporting and finish-reason handling are important for LLM Mesh observability and Cobuild orchestration.

## Troubleshooting

### Empty Responses or Output Token Limits

A Vertex AI response may complete without returning text or function calls if the model exhausts its output token budget.

Gemini thinking tokens can consume part of this budget.

For workloads involving long prompts or complex tool calling, verify the configured maximum output token limit.

A previous Cobuild test encountered `MAX_TOKENS` with a 1,024-token output limit. Increasing the configured limit to 8,192 was followed by a successful test.

### Unsupported JSON Schema Keywords

An error similar to:

```text
Unsupported JSON schema keyword 'maxItems'
```

indicates that a Dataiku tool declaration contains a schema keyword not accepted by the plugin's current schema conversion logic.

Inspect the schema path reported by the exception.

Verify compatibility with the target Vertex AI API before modifying the normalizer.

### Authentication Failures

Verify:

- Google Cloud credentials
- IAM permissions
- Vertex AI API access
- Configured project and region
- Endpoint connectivity

### Tool-Calling Failures

Review:

- Tool declarations sent to Gemini
- Tool-choice configuration
- Gemini function-call responses
- Dataiku tool-call identifiers
- Tool execution results
- Subsequent conversation history

Use Dataiku LLM Mesh traces to identify the failing stage.

## Development and Testing

The repository includes tests for the Gemini integration and Dataiku LLM implementation.

Example:

```bash
python -m pytest tests/
```

Run targeted tests before the full suite when working on a specific component.

Development conventions and Codex instructions are documented in:

`AGENTS.md`

### Development Guidelines

- Audit existing code before making changes.
- Prefer minimal, targeted modifications.
- Preserve Dataiku and Vertex AI integration contracts.
- Avoid unnecessary dependencies or refactoring.
- Add regression tests for compatibility fixes.
- Do not silently suppress errors.
- Do not expose credentials or sensitive request data in logs.

## Version History

### 1.0.2 — Current

- Improved Dataiku Cobuild compatibility.
- Added Gemini tool-choice handling.
- Added prompt and completion token accounting.
- Improved finish-reason mapping.
- Enhanced empty-response diagnostics.
- Validated multi-round tool execution.
- Successfully tested Cobuild web application creation on DSS 14.7.3.
- Continued testing of additional tool schema compatibility.

### Earlier Versions

Historical changes for versions before 1.0.2 have not yet been documented.

---

## Maintenance

**Maintainer:** Stephen Mazzei  
**Organization:** Dataiku  
**Current Version:** 1.0.2

This repository is under active development. Features and compatibility should be validated against the specific Dataiku DSS version, Vertex AI endpoint, and model configuration in use.

## Disclaimer

This is a custom integration intended for development, testing, and use in appropriately managed Dataiku environments.

It is not an official statement of Dataiku product support or Google Cloud model compatibility. Validate functionality, security, and operational requirements before production deployment.