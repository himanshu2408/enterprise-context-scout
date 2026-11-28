"""Keyless ADC Google Cloud Model Armor guardrail provider."""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Mapping
from typing import Any

import httpx

from context_scout.config import ScoutSettings, validate_vertex_location
from context_scout.schemas.contracts import PrincipalContext
from context_scout.security.guardrails_pkg.injection_rules import (
    DeterministicRuleGuardrailProvider,
    PromptInjectionSanitizer,
)
from context_scout.security.guardrails_pkg.models_and_types import (
    _MODEL_ARMOR_SCOPE,
    SafetyCategory,
    ShieldGemmaDecision,
)

logger = logging.getLogger(__name__)


class GoogleCloudModelArmorProvider:
    """Optional keyless ADC Google Cloud Model Armor guardrail plugin (VPC PSC / EU GDPR).

    Authenticates strictly via Application Default Credentials (`google.auth.default`)
    or Workload Identity Federation; static API keys and service-account JSON key files
    are rejected. Requires both `enable_gcp_model_armor=True` and either VPC Private
    Service Connect (`vpc_private_service_connect=True` / `vpc_psc_endpoint`) or explicit
    `allow_external_cloud_egress=True`.
    """

    def __init__(
        self,
        *,
        template_id: str,
        project: str | None = None,
        location: str = "europe-west3",
        enable_gcp_model_armor: bool = False,
        vpc_private_service_connect: bool = False,
        vpc_psc_endpoint: str | None = None,
        allow_external_cloud_egress: bool = False,
        allow_non_eu_regions: bool = False,
        timeout_ms: float = 250.0,
        fail_closed: bool = True,
        credentials: Any | None = None,
        http_client: httpx.Client | None = None,
        armor_client: Any | None = None,
        api_key: str | None = None,
        service_account_key_path: str | None = None,
        settings: ScoutSettings | None = None,
    ) -> None:
        if api_key is not None or service_account_key_path is not None:
            raise ValueError(
                "GoogleCloudModelArmorProvider is strictly keyless (ADC / Workload Identity only); "
                "static API keys and service-account JSON key files are prohibited."
            )

        cfg = settings or ScoutSettings()
        cleaned_template = template_id.strip()
        if not cleaned_template:
            raise ValueError("Google Cloud Model Armor template_id cannot be empty.")

        resolved_enable = bool(enable_gcp_model_armor or cfg.enable_gcp_model_armor)
        if not resolved_enable:
            raise PermissionError(
                "Google Cloud Model Armor integration is disabled (enable_gcp_model_armor=False)."
            )

        resolved_psc_endpoint = (
            vpc_psc_endpoint.strip().rstrip("/")
            if vpc_psc_endpoint and vpc_psc_endpoint.strip()
            else (
                cfg.model_armor_vpc_psc_endpoint.strip().rstrip("/")
                if cfg.model_armor_vpc_psc_endpoint
                else None
            )
        )
        resolved_psc = bool(vpc_private_service_connect or resolved_psc_endpoint)
        resolved_egress = bool(allow_external_cloud_egress or cfg.allow_external_cloud_egress)
        if not resolved_psc and not resolved_egress:
            raise PermissionError(
                "Google Cloud Model Armor requires VPC Private Service Connect or "
                "ModelGovernancePolicy allow_external_cloud_egress=True."
            )

        self._location = validate_vertex_location(
            location or cfg.gcp_location,
            allow_non_eu=allow_non_eu_regions or cfg.allow_non_eu_regions,
        )
        self._project = (project or cfg.gcp_project or "").strip() or None
        self._template_id = cleaned_template
        self._vpc_private_service_connect = resolved_psc
        self._vpc_psc_endpoint = resolved_psc_endpoint
        self._timeout_ms = max(10.0, float(timeout_ms))
        self._fail_closed = fail_closed
        self._credentials = credentials
        self._http_client = http_client
        self._armor_client = armor_client
        self._cred_lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return f"gcp-model-armor/{self._location}/{self._template_id}"

    @property
    def provider_name(self) -> str:
        return "gcp_model_armor"

    @property
    def location(self) -> str:
        return self._location

    @property
    def uses_keyless_adc(self) -> bool:
        return True

    def _template_resource(self) -> str:
        project_id = self._project or "default-project"
        return f"projects/{project_id}/locations/{self._location}/templates/{self._template_id}"

    def _endpoint_url(self) -> str:
        base = (
            self._vpc_psc_endpoint
            if self._vpc_psc_endpoint is not None
            else f"https://modelarmor.{self._location}.rep.googleapis.com/v1"
        )
        return f"{base}/{self._template_resource()}:sanitizeUserPrompt"

    def _resolve_auth_headers(self) -> dict[str, str]:
        with self._cred_lock:
            creds = self._credentials
            if creds is None:
                import google.auth

                creds, discovered_project = google.auth.default(scopes=[_MODEL_ARMOR_SCOPE])
                self._credentials = creds
                if self._project is None and discovered_project:
                    self._project = str(discovered_project)

            valid = getattr(creds, "valid", False)
            token = getattr(creds, "token", None)
            if not valid or not token:
                from google.auth.transport.requests import Request as GoogleAuthRequest

                creds.refresh(GoogleAuthRequest())
                token = getattr(creds, "token", None)

        if not token:
            raise RuntimeError(
                "Keyless GCP Application Default Credentials did not yield an access token."
            )
        headers: dict[str, str] = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        if self._project:
            headers["x-goog-user-project"] = self._project
        return headers

    @staticmethod
    def _parse_sanitization_response(
        payload: Mapping[str, Any],
    ) -> tuple[bool, list[SafetyCategory], str]:
        result = payload.get("sanitizationResult")
        if not isinstance(result, Mapping):
            result = payload

        invocation_result = str(
            result.get("invocationResult") or result.get("invocation_result") or ""
        ).upper()
        match_state = str(
            result.get("filterMatchState") or result.get("filter_match_state") or "NO_MATCH_FOUND"
        ).upper()
        filter_results = result.get("filterResults") or result.get("filter_results") or {}

        categories: list[SafetyCategory] = []
        matched_filters: list[str] = []

        if invocation_result in {"FAILURE", "ERROR"}:
            categories.append(SafetyCategory.POLICY_VIOLATION)
            matched_filters.append(f"invocationResult={invocation_result}")

        filter_category_map: tuple[tuple[tuple[str, ...], SafetyCategory], ...] = (
            (("piAndJailbreakFilterResult", "pi_and_jailbreak"), SafetyCategory.PROMPT_INJECTION),
            (("sdpFilterResult", "sdp"), SafetyCategory.SECRET_LEAKAGE),
            (("maliciousUrisFilterResult", "malicious_uris"), SafetyCategory.MALICIOUS_INPUT),
            (
                ("raiFilterResult", "rai", "csamFilterResult", "csam"),
                SafetyCategory.POLICY_VIOLATION,
            ),
        )

        if isinstance(filter_results, Mapping):
            for keys, category in filter_category_map:
                for key in keys:
                    entry = filter_results.get(key)
                    if not isinstance(entry, Mapping):
                        continue
                    inner = next(
                        (v for v in entry.values() if isinstance(v, Mapping)),
                        entry,
                    )
                    state = str(inner.get("matchState") or inner.get("match_state") or "").upper()
                    if state == "MATCH_FOUND":
                        matched_filters.append(key)
                        if category not in categories:
                            categories.append(category)

        if match_state == "MATCH_FOUND" and not categories:
            categories.append(SafetyCategory.POLICY_VIOLATION)
            matched_filters.append("filterMatchState=MATCH_FOUND")

        passed = match_state != "MATCH_FOUND" and len(categories) == 0
        rationale = (
            "Passed Google Cloud Model Armor sanitization."
            if passed
            else f"Google Cloud Model Armor blocked content: {matched_filters}"
        )
        return passed, categories, rationale

    def evaluate(self, text: str, principal: PrincipalContext | None = None) -> ShieldGemmaDecision:
        """Evaluates text against Google Cloud Model Armor using keyless ADC."""
        del principal
        t0 = time.perf_counter()
        text_bytes = len(text.encode("utf-8"))
        if text_bytes > DeterministicRuleGuardrailProvider.MAX_PAYLOAD_BYTES:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
            return ShieldGemmaDecision(
                passed=False,
                score=0.0,
                violated_categories=[SafetyCategory.POLICY_VIOLATION],
                rationale=(
                    f"Payload size ({text_bytes} bytes) exceeds safety cap "
                    f"of {DeterministicRuleGuardrailProvider.MAX_PAYLOAD_BYTES} bytes."
                ),
                sanitized_content=text[: DeterministicRuleGuardrailProvider.MAX_PAYLOAD_BYTES],
                redacted_tokens=[],
                used_fallback=False,
                provider_name=self.provider_name,
                model_id=self.model_id,
                deterministic_latency_ms=0.0,
                neural_latency_ms=elapsed_ms,
                total_latency_ms=elapsed_ms,
                stage_evaluated="model_armor",
                model_armor_invoked=False,
            )

        normalized_text = PromptInjectionSanitizer.strip_invisible_unicode(text)
        try:
            if self._armor_client is not None:
                raw_resp = self._armor_client.sanitize_user_prompt(
                    template=self._template_resource(),
                    user_prompt_data={"text": normalized_text},
                )
                if not isinstance(raw_resp, Mapping):
                    raise TypeError("armor_client.sanitize_user_prompt must return a Mapping.")
            else:
                headers = self._resolve_auth_headers()
                timeout_s = max(0.01, self._timeout_ms / 1000.0)
                client = self._http_client or httpx.Client(timeout=timeout_s)
                should_close = self._http_client is None
                try:
                    resp = client.post(
                        self._endpoint_url(),
                        headers=headers,
                        json={"userPromptData": {"text": normalized_text}},
                        timeout=timeout_s,
                    )
                    resp.raise_for_status()
                    parsed = resp.json()
                    if not isinstance(parsed, Mapping):
                        raise ValueError("Model Armor response was not a JSON object.")
                    raw_resp = parsed
                finally:
                    if should_close:
                        client.close()

            passed, categories, rationale = self._parse_sanitization_response(raw_resp)
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
            return ShieldGemmaDecision(
                passed=passed,
                score=1.0 if passed else 0.0,
                violated_categories=categories,
                rationale=rationale,
                sanitized_content=normalized_text,
                redacted_tokens=[],
                used_fallback=False,
                provider_name=self.provider_name,
                model_id=self.model_id,
                deterministic_latency_ms=0.0,
                neural_latency_ms=elapsed_ms,
                total_latency_ms=elapsed_ms,
                stage_evaluated="model_armor",
                model_armor_invoked=True,
            )
        except Exception as exc:
            elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 4)
            if not self._fail_closed:
                raise
            return ShieldGemmaDecision(
                passed=False,
                score=0.0,
                violated_categories=[SafetyCategory.POLICY_VIOLATION],
                rationale=(
                    f"Google Cloud Model Armor check failed closed ({type(exc).__name__}): {exc}"
                ),
                sanitized_content=normalized_text,
                redacted_tokens=[],
                used_fallback=False,
                provider_name=self.provider_name,
                model_id=self.model_id,
                deterministic_latency_ms=0.0,
                neural_latency_ms=elapsed_ms,
                total_latency_ms=elapsed_ms,
                stage_evaluated="model_armor_fail_closed",
                model_armor_invoked=True,
            )
