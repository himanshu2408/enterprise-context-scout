"""Pluggable EmbeddingProviderSPI keeping google/embeddinggemma-2 (768d MRL) as local default.

Supports:
1. Local SentenceTransformers (`google/embeddinggemma-2` 768d Matryoshka default,
   `Qwen/Qwen3-Embedding-0.6B/4B/8B` MRL, `BAAI/bge-m3` 1024d) on Apple Silicon MPS / CUDA / CPU.
2. Self-hosted `vLLM --task embed` and Hugging Face `TEI` (`/v1/embeddings` & `/embed`).
3. Optional keyless GCP Vertex AI (`google.genai.Client(vertexai=True)` via ADC) when
   `allow_external_cloud_egress=True` and EU residency policy permit.
4. Strict zero-norm / NaN rejection and Matryoshka (MRL) L2 re-normalization.
"""

from __future__ import annotations

import fnmatch
import logging
import math
import os
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

import httpx

from context_scout.config import (
    ModelGovernancePolicy,
    ScoutSettings,
    validate_vertex_location,
)

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_GOOGLE_EMBEDDING_MODEL: Final[str] = "google/embeddinggemma-2"
DEFAULT_EMBEDDING_SPACE_ID: Final[str] = "embeddinggemma-2-768"
EXPECTED_EMBEDDING_DIM: Final[int] = 768
_ZERO_NORM_EPSILON: Final[float] = 1e-9

_TORCH_EMBED_LOCK: Final[threading.Lock] = threading.Lock()
_EMBEDDER_MODEL_CACHE: dict[tuple[str, str], SentenceTransformer] = {}
_TEXT_VECTOR_CACHE_LOCK: Final[threading.Lock] = threading.Lock()
_TEXT_VECTOR_CACHE: dict[tuple[str, int, bool, str], list[float]] = {}
_PROVIDER_CACHE_LOCK: Final[threading.Lock] = threading.Lock()
_SHARED_LOCAL_PROVIDERS: dict[tuple[str, int, str], LocalSemanticEmbedder] = {}


@dataclass(frozen=True)
class EmbeddingModelSpec:
    """Metadata describing an approved embedding model's dimensions and Matryoshka support."""

    pattern: str
    family: str
    native_dimensions: int
    max_dimensions: int
    supports_matryoshka: bool
    query_instruction_prefix: str | None = None


EMBEDDING_MODEL_CATALOG: Final[tuple[EmbeddingModelSpec, ...]] = (
    EmbeddingModelSpec(
        pattern="google/embeddinggemma-2*",
        family="embeddinggemma-2",
        native_dimensions=768,
        max_dimensions=768,
        supports_matryoshka=True,
    ),
    EmbeddingModelSpec(
        pattern="Qwen/Qwen3-Embedding-0.6B*",
        family="Qwen3-Embedding",
        native_dimensions=1024,
        max_dimensions=1024,
        supports_matryoshka=True,
        query_instruction_prefix=(
            "Instruct: Given a technical search query, retrieve relevant code symbols "
            "and documentation passages\nQuery: "
        ),
    ),
    EmbeddingModelSpec(
        pattern="Qwen/Qwen3-Embedding-4B*",
        family="Qwen3-Embedding",
        native_dimensions=2560,
        max_dimensions=2560,
        supports_matryoshka=True,
        query_instruction_prefix=(
            "Instruct: Given a technical search query, retrieve relevant code symbols "
            "and documentation passages\nQuery: "
        ),
    ),
    EmbeddingModelSpec(
        pattern="Qwen/Qwen3-Embedding-8B*",
        family="Qwen3-Embedding",
        native_dimensions=4096,
        max_dimensions=4096,
        supports_matryoshka=True,
        query_instruction_prefix=(
            "Instruct: Given a technical search query, retrieve relevant code symbols "
            "and documentation passages\nQuery: "
        ),
    ),
    EmbeddingModelSpec(
        pattern="Qwen/Qwen3-Embedding-*",
        family="Qwen3-Embedding",
        native_dimensions=1024,
        max_dimensions=4096,
        supports_matryoshka=True,
        query_instruction_prefix=(
            "Instruct: Given a technical search query, retrieve relevant code symbols "
            "and documentation passages\nQuery: "
        ),
    ),
    EmbeddingModelSpec(
        pattern="BAAI/bge-m3*",
        family="BAAI/bge-m3",
        native_dimensions=1024,
        max_dimensions=1024,
        supports_matryoshka=False,
    ),
    EmbeddingModelSpec(
        pattern="gemini-embedding-*",
        family="gemini-embedding",
        native_dimensions=768,
        max_dimensions=3072,
        supports_matryoshka=True,
    ),
)


def resolve_embedding_model_spec(
    model_id: str,
    requested_dimensions: int | None = None,
) -> tuple[EmbeddingModelSpec, int]:
    """Resolves model catalog metadata and validates target embedding dimensions."""
    cleaned = model_id.strip()
    cleaned_lower = cleaned.lower()
    matched: EmbeddingModelSpec | None = None
    for spec in EMBEDDING_MODEL_CATALOG:
        if fnmatch.fnmatchcase(cleaned_lower, spec.pattern.lower()):
            matched = spec
            break
    if matched is None:
        matched = EmbeddingModelSpec(
            pattern=cleaned,
            family="custom",
            native_dimensions=requested_dimensions or EXPECTED_EMBEDDING_DIM,
            max_dimensions=max(requested_dimensions or EXPECTED_EMBEDDING_DIM, 4096),
            supports_matryoshka=True,
        )

    target_dim = (
        requested_dimensions if requested_dimensions is not None else matched.native_dimensions
    )
    if target_dim < 128 or target_dim > 4096:
        raise ValueError(
            f"Embedding dimensions must be between 128 and 4096; got {target_dim} for '{cleaned}'."
        )
    if target_dim > matched.max_dimensions:
        raise ValueError(
            f"Requested dimensions ({target_dim}) exceed maximum ({matched.max_dimensions}) "
            f"for model '{cleaned}'."
        )
    if not matched.supports_matryoshka and target_dim != matched.native_dimensions:
        raise ValueError(
            f"Model '{cleaned}' does not support Matryoshka dimension truncation; "
            f"expected native {matched.native_dimensions}d, got {target_dim}d."
        )
    return matched, target_dim


def validate_non_zero_embedding(
    vec: Sequence[float],
    *,
    expected_dim: int | None = None,
) -> list[float]:
    """Validates that a vector is non-empty, finite, matches expected_dim, and has non-zero norm."""
    if not vec:
        raise ValueError("Embedding vector cannot be empty.")
    float_list = [float(v) for v in vec]
    if expected_dim is not None and len(float_list) != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {len(float_list)}."
        )
    if any(not math.isfinite(x) for x in float_list):
        raise ValueError("Embedding vector contains NaN or infinite values.")
    norm_sq = sum(x * x for x in float_list)
    if norm_sq <= _ZERO_NORM_EPSILON:
        raise ValueError("Zero-norm embedding vector ([0.0]*D) is prohibited.")
    return float_list


def normalize_embedding(
    vec: Sequence[float],
    *,
    target_dim: int | None = None,
    supports_matryoshka: bool = True,
) -> list[float]:
    """Applies Matryoshka slicing and L2 unit normalization, failing closed on zero norm."""
    if not vec:
        raise ValueError("Embedding vector cannot be empty.")
    float_list = [float(v) for v in vec]
    if any(not math.isfinite(x) for x in float_list):
        raise ValueError("Embedding vector contains NaN or infinite values.")

    if target_dim is not None:
        if len(float_list) > target_dim:
            if not supports_matryoshka:
                raise ValueError(
                    f"Cannot truncate non-Matryoshka vector from "
                    f"{len(float_list)}d to {target_dim}d."
                )
            float_list = float_list[:target_dim]
        elif len(float_list) < target_dim:
            raise ValueError(
                f"Embedding vector length ({len(float_list)}) is smaller "
                f"than target_dim ({target_dim})."
            )

    norm = math.sqrt(sum(x * x for x in float_list))
    if norm <= _ZERO_NORM_EPSILON:
        raise ValueError("Zero-norm embedding vector ([0.0]*D) is prohibited.")
    normalized = [round(x / norm, 6) for x in float_list]
    if sum(x * x for x in normalized) <= _ZERO_NORM_EPSILON:
        raise ValueError("Normalized embedding vector collapsed to zero norm.")
    return normalized


@runtime_checkable
class EmbeddingProviderSPI(Protocol):
    """Protocol for local, self-hosted (vLLM/TEI), and cloud-gated embedding providers."""

    @property
    def model_id(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def provider_name(self) -> str: ...

    @property
    def supports_matryoshka(self) -> bool: ...

    def embed_text(self, text: str, is_query: bool = False) -> list[float]: ...

    def embed_batch(
        self,
        texts: list[str],
        is_query: bool = False,
        batch_size: int = 32,
    ) -> list[list[float]]: ...

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


def resolve_torch_device() -> str:
    """Selects embedding torch device, honoring SCOUT_EMBEDDING_DEVICE & MLX lock."""
    explicit = os.environ.get("SCOUT_EMBEDDING_DEVICE", "").strip().lower()
    if explicit == "cpu":
        return "cpu"
    if explicit in ("mps", "cuda"):
        try:
            import torch

            if (
                explicit == "mps"
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            ):
                return "mps"
            if explicit == "cuda" and torch.cuda.is_available():
                return "cuda"
        except ImportError as exc:
            logger.debug(
                "torch unavailable for requested device '%s' (%s); falling back to CPU",
                explicit,
                type(exc).__name__,
            )
        return "cpu"

    if os.environ.get("CONTEXT_SCOUT_USE_LIVE_MLX", "").strip() == "1":
        # Avoid concurrent torch.mps + libmlx Metal command-buffer races on Apple Silicon.
        return "cpu"

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
    except ImportError as exc:
        logger.debug(
            "torch unavailable for device probe (%s); falling back to CPU",
            type(exc).__name__,
        )
    return "cpu"


class LocalSemanticEmbedder:
    """Local SentenceTransformers EmbeddingProviderSPI (`google/embeddinggemma-2` default)."""

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        settings: ScoutSettings | None = None,
        allow_deterministic_fallback: bool = False,
        *,
        dimensions: int | None = None,
        policy: ModelGovernancePolicy | None = None,
    ) -> None:
        self.settings = settings or ScoutSettings()
        gov = policy or ModelGovernancePolicy.default()
        raw_id = model_id or self.settings.embedding_model_id or DEFAULT_GOOGLE_EMBEDDING_MODEL
        self._model_id = gov.validate_local_model_id(raw_id)
        spec, resolved_dim = resolve_embedding_model_spec(
            self._model_id,
            requested_dimensions=dimensions,
        )
        self._spec = spec
        self._dimensions = resolved_dim
        self.device = device or resolve_torch_device()
        self.allow_deterministic_fallback = allow_deterministic_fallback
        self._model: SentenceTransformer | None = None
        self._lock: threading.Lock = _TORCH_EMBED_LOCK

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def provider_name(self) -> str:
        return "sentence-transformers"

    @property
    def supports_matryoshka(self) -> bool:
        return self._spec.supports_matryoshka

    def _load_model(self) -> SentenceTransformer:
        """Lazy-loads SentenceTransformer model onto designated device under lock."""
        with self._lock:
            if self._model is not None:
                return self._model
            cache_key = (self._model_id, self.device)
            cached = _EMBEDDER_MODEL_CACHE.get(cache_key)
            if cached is not None:
                self._model = cached
                return cached

            try:
                import torch
                from sentence_transformers import SentenceTransformer

                torch_dtype = torch.bfloat16 if self.device in ("mps", "cuda") else torch.float32
                logger.info(
                    "Loading semantic embedding model '%s' (%dd) on device '%s'...",
                    self._model_id,
                    self._dimensions,
                    self.device,
                )
                model_kwargs: dict[str, Any] = {
                    "torch_dtype": torch_dtype,
                }
                try:
                    self._model = SentenceTransformer(
                        self._model_id,
                        device=self.device,
                        model_kwargs=model_kwargs,
                        trust_remote_code=False,
                        local_files_only=True,
                    )
                except Exception as exc:
                    logger.debug(
                        "Local SentenceTransformer cache miss for %s (%s): %s; trying remote",
                        self._model_id,
                        type(exc).__name__,
                        exc,
                    )
                    self._model = SentenceTransformer(
                        self._model_id,
                        device=self.device,
                        model_kwargs=model_kwargs,
                        trust_remote_code=False,
                    )
                if hasattr(self._model, "max_seq_length"):
                    self._model.max_seq_length = 2048
                _EMBEDDER_MODEL_CACHE[cache_key] = self._model
                return self._model
            except Exception as err:
                raise RuntimeError(
                    f"Unable to initialize embedding model '{self._model_id}' "
                    f"on {self.device}: {err}"
                ) from err

    def embed_text(self, text: str, is_query: bool = False) -> list[float]:
        """Encodes a single text string into an L2-normalized float vector."""
        return self.embed_batch([text], is_query=is_query)[0]

    def embed_query(self, text: str) -> list[float]:
        """Encodes a search query string into an L2-normalized float vector."""
        return self.embed_text(text, is_query=True)

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        """Encodes document chunks/symbols into L2-normalized float vectors."""
        return self.embed_batch(texts, is_query=False, batch_size=batch_size)

    def embed_batch(
        self,
        texts: list[str],
        is_query: bool = False,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Encodes a batch of texts into L2-normalized float vectors."""
        if not texts:
            return []

        cleaned = [t.strip()[:6000] or " " for t in texts]
        results: list[list[float] | None] = [None] * len(cleaned)
        missing_indices: list[int] = []
        missing_texts: list[str] = []

        with _TEXT_VECTOR_CACHE_LOCK:
            for idx, txt in enumerate(cleaned):
                cache_key = (self._model_id, self._dimensions, is_query, txt)
                cached_vec = _TEXT_VECTOR_CACHE.get(cache_key)
                if cached_vec is not None:
                    results[idx] = list(cached_vec)
                else:
                    missing_indices.append(idx)
                    missing_texts.append(txt)

        if not missing_texts:
            return [r for r in results if r is not None]

        model = self._load_model()
        import torch

        encode_inputs = list(missing_texts)
        has_prompts = bool(hasattr(model, "prompts") and model.prompts)
        if is_query and not has_prompts and self._spec.query_instruction_prefix:
            encode_inputs = [f"{self._spec.query_instruction_prefix}{t}" for t in missing_texts]

        with self._lock:
            if self.device == "mps" and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            kwargs: dict[str, Any] = {
                "batch_size": batch_size,
                "normalize_embeddings": True,
                "show_progress_bar": False,
            }
            if has_prompts and model.prompts:
                if is_query and "query" in model.prompts:
                    kwargs["prompt_name"] = "query"
                elif not is_query and "document" in model.prompts:
                    kwargs["prompt_name"] = "document"

            try:
                raw_embeddings = model.encode(encode_inputs, **kwargs)
            except Exception as err:
                logger.warning(
                    "Encode on device '%s' failed (%s); completing batch on CPU",
                    self.device,
                    err,
                )
                if self.device == "mps" and torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                if hasattr(model, "to"):
                    model = model.to("cpu")
                    self._model = model
                self.device = "cpu"
                cpu_kwargs = dict(kwargs)
                cpu_kwargs["device"] = "cpu"
                cpu_kwargs["batch_size"] = min(batch_size, 8)
                raw_embeddings = model.encode(encode_inputs, **cpu_kwargs)
            finally:
                if self.device == "mps" and torch.backends.mps.is_available():
                    torch.mps.empty_cache()

        with _TEXT_VECTOR_CACHE_LOCK:
            for m_idx, raw_vec in zip(missing_indices, raw_embeddings, strict=True):
                normalized = normalize_embedding(
                    [float(v) for v in raw_vec],
                    target_dim=self._dimensions,
                    supports_matryoshka=self._spec.supports_matryoshka,
                )
                results[m_idx] = normalized
                cache_key = (self._model_id, self._dimensions, is_query, cleaned[m_idx])
                if len(_TEXT_VECTOR_CACHE) < 16_384:
                    _TEXT_VECTOR_CACHE[cache_key] = list(normalized)

        return [r for r in results if r is not None]

    @staticmethod
    def cosine_similarity(u: list[float], v: list[float]) -> float:
        """Computes cosine similarity between two float vectors."""
        dot = sum(a * b for a, b in zip(u, v, strict=False))
        norm_u = math.sqrt(sum(a * a for a in u))
        norm_v = math.sqrt(sum(b * b for b in v))
        if norm_u == 0.0 or norm_v == 0.0:
            return 0.0
        return dot / (norm_u * norm_v)


SentenceTransformersEmbeddingProvider = LocalSemanticEmbedder


class OpenAICompatibleEmbeddingProvider:
    """EmbeddingProviderSPI for self-hosted `vLLM --task embed` and Hugging Face `TEI`."""

    def __init__(
        self,
        model_id: str = DEFAULT_GOOGLE_EMBEDDING_MODEL,
        *,
        endpoint_url: str = "http://127.0.0.1:8000/v1",
        dimensions: int | None = None,
        api_key: str | None = None,
        provider_mode: str = "vllm",
        http_client: httpx.Client | None = None,
        policy: ModelGovernancePolicy | None = None,
    ) -> None:
        gov = policy or ModelGovernancePolicy.default()
        self._model_id = gov.validate_local_model_id(model_id)
        spec, resolved_dim = resolve_embedding_model_spec(
            self._model_id,
            requested_dimensions=dimensions,
        )
        self._spec = spec
        self._dimensions = resolved_dim
        cleaned_mode = provider_mode.strip().lower()
        if cleaned_mode not in {"vllm", "tei", "openai_compatible"}:
            raise ValueError(f"Unsupported embedding provider_mode: {provider_mode!r}")
        self._provider_mode = cleaned_mode
        if not endpoint_url or not endpoint_url.strip():
            raise ValueError(
                "endpoint_url must be non-empty for OpenAICompatibleEmbeddingProvider."
            )
        self._endpoint_url = endpoint_url.strip().rstrip("/")
        self._api_key = api_key.strip() if api_key and api_key.strip() else None
        self._http_client = http_client or httpx.Client(timeout=30.0)

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def provider_name(self) -> str:
        return self._provider_mode

    @property
    def supports_matryoshka(self) -> bool:
        return self._spec.supports_matryoshka

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def embed_text(self, text: str, is_query: bool = False) -> list[float]:
        return self.embed_batch([text], is_query=is_query)[0]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text, is_query=True)

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return self.embed_batch(texts, is_query=False, batch_size=batch_size)

    def embed_batch(
        self,
        texts: list[str],
        is_query: bool = False,
        batch_size: int = 32,
    ) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [t.strip()[:6000] or " " for t in texts]
        if is_query and self._spec.query_instruction_prefix:
            cleaned = [f"{self._spec.query_instruction_prefix}{t}" for t in cleaned]

        all_vectors: list[list[float]] = []
        step = max(1, batch_size)
        for start in range(0, len(cleaned), step):
            chunk = cleaned[start : start + step]
            if self._provider_mode == "tei" and not self._endpoint_url.endswith(
                ("/v1", "/embeddings")
            ):
                url = (
                    self._endpoint_url
                    if self._endpoint_url.endswith("/embed")
                    else f"{self._endpoint_url}/embed"
                )
                payload: dict[str, Any] = {
                    "inputs": chunk,
                    "normalize": True,
                    "truncate": True,
                }
                if self._spec.supports_matryoshka:
                    payload["dimensions"] = self._dimensions
            else:
                url = (
                    self._endpoint_url
                    if self._endpoint_url.endswith("/embeddings")
                    else f"{self._endpoint_url}/embeddings"
                )
                payload = {
                    "model": self._model_id,
                    "input": chunk,
                    "encoding_format": "float",
                }
                if self._spec.supports_matryoshka:
                    payload["dimensions"] = self._dimensions

            resp = self._http_client.post(url, json=payload, headers=self._headers())
            resp.raise_for_status()
            body = resp.json()
            raw_list: list[Sequence[float]] = []
            if isinstance(body, dict) and isinstance(body.get("data"), list):
                ordered = sorted(
                    body["data"],
                    key=lambda item: int(item.get("index", 0)) if isinstance(item, dict) else 0,
                )
                for item in ordered:
                    if isinstance(item, dict) and isinstance(item.get("embedding"), list):
                        raw_list.append(item["embedding"])
            elif isinstance(body, list):
                for item in body:
                    if isinstance(item, list):
                        raw_list.append(item)

            if len(raw_list) != len(chunk):
                raise RuntimeError(
                    f"Embedding endpoint {url} returned {len(raw_list)} "
                    f"vectors for batch of {len(chunk)}."
                )
            for raw_vec in raw_list:
                all_vectors.append(
                    normalize_embedding(
                        raw_vec,
                        target_dim=self._dimensions,
                        supports_matryoshka=self._spec.supports_matryoshka,
                    )
                )
        return all_vectors


class VertexAIEmbeddingProvider:
    """Optional keyless GCP Vertex AI EmbeddingProviderSPI (`vertexai=True` via ADC).

    Strictly enforces:
    1. `ModelGovernancePolicy` (`allow_external_cloud_egress=True` required; blocks retired
       `text-embedding-004`).
    2. GDPR EU data residency (`validate_vertex_location`, default `europe-west3`).
    3. Keyless Application Default Credentials (`vertexai=True`, zero static API keys).
    """

    def __init__(
        self,
        model_id: str = "gemini-embedding-001",
        *,
        project: str | None = None,
        location: str = "europe-west3",
        dimensions: int = EXPECTED_EMBEDDING_DIM,
        allow_non_eu_regions: bool = False,
        allow_uk_ch_adequacy_regions: bool = False,
        settings: ScoutSettings | None = None,
        policy: ModelGovernancePolicy | None = None,
        genai_client: Any | None = None,
    ) -> None:
        cfg = settings or ScoutSettings()
        allow_egress = bool(
            policy.allow_external_cloud_egress
            if policy is not None
            else cfg.allow_external_cloud_egress
        )
        gov = policy or ModelGovernancePolicy.from_registry_file(
            allow_external_cloud_egress=allow_egress
        )
        cleaned_model = model_id.strip()
        if gov.is_blocked_model(cleaned_model):
            raise ValueError(f"Prohibited or retired model ID '{model_id}'.")
        if not gov.allow_external_cloud_egress:
            raise PermissionError(
                f"Vertex AI embedding provider for '{model_id}' requires external cloud egress, "
                "which is blocked by ModelGovernancePolicy (allow_external_cloud_egress=False)."
            )
        gov.validate_model_id(
            cleaned_model,
            allow_cloud_handoff=True,
            enforce_egress_gate=True,
        )
        self._location = validate_vertex_location(
            location or cfg.gcp_location,
            allow_non_eu=allow_non_eu_regions or cfg.allow_non_eu_regions,
            allow_uk_ch_adequacy_regions=(
                allow_uk_ch_adequacy_regions or cfg.allow_uk_ch_adequacy_regions
            ),
        )
        self._project = (project or cfg.gcp_project or "").strip() or None
        spec, resolved_dim = resolve_embedding_model_spec(
            cleaned_model,
            requested_dimensions=dimensions,
        )
        self._model_id = cleaned_model
        self._spec = spec
        self._dimensions = resolved_dim
        self._client = genai_client

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def provider_name(self) -> str:
        return "vertex_ai"

    @property
    def supports_matryoshka(self) -> bool:
        return self._spec.supports_matryoshka

    @property
    def location(self) -> str:
        return self._location

    def _get_client(self) -> Any:
        if self._client is not None:
            return self._client
        from google import genai

        self._client = genai.Client(
            vertexai=True,
            project=self._project,
            location=self._location,
        )
        return self._client

    def embed_text(self, text: str, is_query: bool = False) -> list[float]:
        return self.embed_batch([text], is_query=is_query)[0]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text, is_query=True)

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        return self.embed_batch(texts, is_query=False, batch_size=batch_size)

    def embed_batch(
        self,
        texts: list[str],
        is_query: bool = False,
        batch_size: int = 32,
    ) -> list[list[float]]:
        if not texts:
            return []
        cleaned = [t.strip()[:6000] or " " for t in texts]
        client = self._get_client()
        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        all_vectors: list[list[float]] = []
        step = max(1, batch_size)
        for start in range(0, len(cleaned), step):
            chunk = cleaned[start : start + step]
            resp = client.models.embed_content(
                model=self._model_id,
                contents=chunk,
                config={
                    "task_type": task_type,
                    "output_dimensionality": self._dimensions,
                },
            )
            embeddings = getattr(resp, "embeddings", None) or []
            if len(embeddings) != len(chunk):
                raise RuntimeError(
                    f"Vertex AI embed_content returned {len(embeddings)} "
                    f"embeddings for batch of {len(chunk)}."
                )
            for emb_obj in embeddings:
                vals = getattr(emb_obj, "values", None)
                if vals is None and isinstance(emb_obj, dict):
                    vals = emb_obj.get("values")
                if not isinstance(vals, Sequence):
                    raise ValueError("Vertex AI embedding response missing numeric 'values'.")
                all_vectors.append(
                    normalize_embedding(
                        [float(x) for x in vals],
                        target_dim=self._dimensions,
                        supports_matryoshka=self._spec.supports_matryoshka,
                    )
                )
        return all_vectors


def get_embedding_provider(
    settings: ScoutSettings | None = None,
    *,
    provider_backend: str | None = None,
    model_id: str | None = None,
    dimensions: int | None = None,
    device: str | None = None,
    endpoint_url: str | None = None,
    http_client: httpx.Client | None = None,
    genai_client: Any | None = None,
    policy: ModelGovernancePolicy | None = None,
) -> EmbeddingProviderSPI:
    """Resolves the configured EmbeddingProviderSPI (`google/embeddinggemma-2` local default)."""
    cfg = settings or ScoutSettings()
    backend = (
        (provider_backend or cfg.embedding_provider or "sentence-transformers").strip().lower()
    )
    resolved_model = model_id or cfg.embedding_model_id or DEFAULT_GOOGLE_EMBEDDING_MODEL
    explicit_cfg_dim = (
        settings is not None and "embedding_dimensions" in cfg.model_fields_set
    ) or ("SCOUT_EMBEDDING_DIMENSIONS" in os.environ)
    if dimensions is not None:
        resolved_dim = dimensions
    elif explicit_cfg_dim:
        resolved_dim = cfg.embedding_dimensions
    else:
        _, resolved_dim = resolve_embedding_model_spec(resolved_model, requested_dimensions=None)

    if backend in {"sentence-transformers", "local", "st"}:
        resolved_device = device or resolve_torch_device()
        cache_key = (resolved_model, resolved_dim, resolved_device)
        if policy is None:
            with _PROVIDER_CACHE_LOCK:
                cached = _SHARED_LOCAL_PROVIDERS.get(cache_key)
                if cached is not None:
                    return cached
                instance = LocalSemanticEmbedder(
                    model_id=resolved_model,
                    device=resolved_device,
                    settings=cfg,
                    dimensions=resolved_dim,
                )
                _SHARED_LOCAL_PROVIDERS[cache_key] = instance
                return instance
        return LocalSemanticEmbedder(
            model_id=resolved_model,
            device=resolved_device,
            settings=cfg,
            dimensions=resolved_dim,
            policy=policy,
        )

    if backend in {"vllm", "tei", "openai_compatible"}:
        target_endpoint = endpoint_url or cfg.embedding_endpoint_url or cfg.local_mlx_endpoint
        return OpenAICompatibleEmbeddingProvider(
            model_id=resolved_model,
            endpoint_url=target_endpoint,
            dimensions=resolved_dim,
            api_key=cfg.embedding_api_key,
            provider_mode=backend,
            http_client=http_client,
            policy=policy,
        )

    if backend in {"vertex_ai", "vertex", "gcp_vertex"}:
        return VertexAIEmbeddingProvider(
            model_id=resolved_model,
            project=cfg.gcp_project,
            location=cfg.gcp_location,
            dimensions=resolved_dim,
            allow_non_eu_regions=cfg.allow_non_eu_regions,
            allow_uk_ch_adequacy_regions=cfg.allow_uk_ch_adequacy_regions,
            settings=cfg,
            policy=policy,
            genai_client=genai_client,
        )

    raise ValueError(f"Unsupported embedding provider backend: {backend!r}")
