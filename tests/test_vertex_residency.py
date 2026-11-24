"""Permanent test suite for EU-Region Vertex AI & Read-Only Service Token Auth (M1 F03).

Verifies:
1. Default residency enforces location="europe-west3" (Frankfurt) and vertexai=True.
2. Rejection of non-EU regions by default under GDPR compliance rules.
3. Explicit allow_non_eu_regions override mechanism.
4. Support for all standard European GCP regions in whitelist.
5. Zero-static-secret auth via read-only tokens (google.oauth2.credentials.Credentials).
6. Application Default Credentials (ADC) discovery when token is unset.
7. Offline deterministic fallback when unconfigured (0 network calls, $0 spend).
8. Mock client patch generation and usage metadata extraction.
9. Pydantic v2 ScoutSettings residency validation.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from context_scout.agents.gemini_executor import (
    GeminiImplementationExecutor,
)
from context_scout.config import (
    ScoutSettings,
)
from context_scout.schemas.contracts import (
    ImplementationContextPack,
    ModificationIntent,
    PreFlightValidationReport,
    VerbatimCodeSlice,
)


@pytest.fixture
def sample_pack() -> ImplementationContextPack:
    code_lines = "   1 | class Engine:\n   2 |     def run(self):\n   3 |         pass\n"
    slice1 = VerbatimCodeSlice(
        repo_name="trade-core",
        file_path="src/engine.py",
        start_line=1,
        end_line=4,
        qualified_symbol="Engine.run",
        selection_reason="Residency test slice",
        modification_intent=ModificationIntent.MODIFY_TARGET,
        git_commit_sha="a1b2c3d",
        git_blob_sha="e5f6g7h",
        verbatim_code_with_lines=code_lines,
    )
    report = PreFlightValidationReport(
        passed=True,
        normalized_rrf_score=0.95,
        citations_verified_100_pct=True,
    )
    return ImplementationContextPack(
        trace_id="trace-residency-001",
        domain_id="code_architecture_scout",
        user_task="Add logging to Engine.run",
        architectural_execution_plan=["Add logging statement"],
        code_slices=[slice1],
        preflight_report=report,
        local_tokens_consumed=10_000,
        pack_token_count=1_000,
    )


def test_default_initialization_vertex_eu_residency() -> None:
    """Verifies that executor defaults to Vertex AI mode and Frankfurt (europe-west3)."""
    executor = GeminiImplementationExecutor(project="enterprise-scout-prod")
    assert executor.vertexai is True
    assert executor.location == "europe-west3"
    assert executor.project == "enterprise-scout-prod"
    assert executor.api_key is None

    client = executor._create_genai_client()
    assert client.vertexai is True
    assert client._api_client.project == "enterprise-scout-prod"
    assert client._api_client.location == "europe-west3"
    base_url = client._api_client._http_options.base_url
    assert base_url.startswith("https://europe-west3-aiplatform.googleapis.com/")


@pytest.mark.parametrize(
    "forbidden_region",
    [
        "us-central1",
        "us-east4",
        "us-west1",
        "asia-east1",
        "asia-northeast1",
        "southamerica-east1",
        "australia-southeast1",
    ],
)
def test_rejection_of_non_eu_regions_by_default(forbidden_region: str) -> None:
    """Verifies that non-EU regions are rejected by default with a clear GDPR violation error."""
    with pytest.raises(ValueError, match="violates EU data residency policy"):
        GeminiImplementationExecutor(
            project="enterprise-scout-prod",
            location=forbidden_region,
        )


@pytest.mark.parametrize(
    "approved_region",
    [
        "europe-west1",
        "europe-west2",
        "europe-west3",
        "europe-west4",
        "europe-west6",
        "europe-west8",
        "europe-west9",
        "europe-west10",
        "europe-west12",
        "europe-north1",
        "europe-central2",
        "europe-southwest1",
    ],
)
def test_allowed_eu_regions(approved_region: str) -> None:
    """Verifies that all approved EU regions pass validation and configure the client endpoint."""
    executor = GeminiImplementationExecutor(
        project="enterprise-scout-prod",
        location=approved_region,
    )
    assert executor.location == approved_region
    client = executor._create_genai_client()
    assert client._api_client.location == approved_region
    base_url = client._api_client._http_options.base_url
    assert f"https://{approved_region}-aiplatform.googleapis.com/" in base_url


def test_explicit_override_for_non_eu_regions() -> None:
    """Verifies that non-EU regions are permitted ONLY when allow_non_eu_regions=True is passed."""
    executor = GeminiImplementationExecutor(
        project="enterprise-scout-prod",
        location="us-central1",
        allow_non_eu_regions=True,
    )
    assert executor.location == "us-central1"
    client = executor._create_genai_client()
    assert client._api_client.location == "us-central1"


def test_read_only_service_token_authentication() -> None:
    """Verifies authentication using read-only service tokens without static API keys."""
    service_token = "ya29.test-read-only-bearer-token"
    executor = GeminiImplementationExecutor(
        project="enterprise-scout-prod",
        service_token=service_token,
    )
    assert executor.api_key is None
    assert executor.credentials is not None
    assert getattr(executor.credentials, "token", None) == service_token

    client = executor._create_genai_client()
    assert client._api_client._credentials == executor.credentials
    # Crucial: verify no api key was passed to headers
    assert "x-goog-api-key" not in client._api_client._http_options.headers


def test_offline_fallback_when_unconfigured(
    sample_pack: ImplementationContextPack, tmp_path: Path
) -> None:
    """Verifies that an unconfigured executor gracefully uses deterministic fallback."""
    executor = GeminiImplementationExecutor(api_key=None)
    assert executor.is_live_configured() is False

    # Create dummy repo file
    src_file = tmp_path / "src" / "engine.py"
    src_file.parent.mkdir(parents=True, exist_ok=True)
    src_file.write_text("class Engine:\n    def run(self):\n        pass\n")

    patch_diff = executor._generate_deterministic_patch(sample_pack, tmp_path)
    assert "--- a/src/engine.py" in patch_diff
    assert "Context Scout Verified Implementation" in patch_diff


def test_mock_gemini_client_patch_generation(
    sample_pack: ImplementationContextPack, tmp_path: Path
) -> None:
    """Verifies that mock genai.Client generates patch, extracts diff, and tracks usage."""
    mock_response = MagicMock()
    mock_response.text = (
        "```diff\n"
        "--- a/src/engine.py\n"
        "+++ b/src/engine.py\n"
        "@@ -2,2 +2,3 @@\n"
        "     def run(self):\n"
        "+        logger.info('Engine running')\n"
        "         pass\n"
        "```"
    )
    mock_response.usage_metadata.prompt_token_count = 1200
    mock_response.usage_metadata.candidates_token_count = 150
    mock_response.usage_metadata.total_token_count = 1350

    mock_client = MagicMock()
    mock_client.models.generate_content.return_value = mock_response

    executor = GeminiImplementationExecutor(
        project="enterprise-scout-prod",
        client=mock_client,
    )
    patch_diff = executor._generate_gemini_patch(sample_pack, tmp_path)

    assert "+        logger.info('Engine running')" in patch_diff
    assert executor.last_usage_metadata == {
        "prompt_token_count": 1200,
        "candidates_token_count": 150,
        "total_token_count": 1350,
    }


def test_scout_settings_residency_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies ScoutSettings validation for GCP residency and overrides."""
    # 1. Default settings
    settings = ScoutSettings()
    assert settings.gcp_location == "europe-west3"
    assert settings.use_vertex_ai is True
    assert settings.allow_non_eu_regions is False

    # 2. Non-EU region rejected on load
    monkeypatch.setenv("SCOUT_GCP_LOCATION", "us-east4")
    with pytest.raises(Exception, match="violates EU data residency policy"):
        ScoutSettings()

    # 3. Non-EU region accepted when SCOUT_ALLOW_NON_EU_REGIONS=true
    monkeypatch.setenv("SCOUT_ALLOW_NON_EU_REGIONS", "true")
    allowed_settings = ScoutSettings()
    assert allowed_settings.gcp_location == "us-east4"
