"""Local semantic embedding engine enforcing 100% Google model policy and Apple Silicon MPS."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, Final

from context_scout.config import ScoutSettings, validate_google_model_id

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_GOOGLE_EMBEDDING_MODEL: Final[str] = "google/embeddinggemma-2"
EXPECTED_EMBEDDING_DIM: Final[int] = 768


def resolve_torch_device() -> str:
    """Selects Apple Silicon MPS acceleration if available, falling back to CPU."""
    try:
        import torch

        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


class LocalSemanticEmbedder:
    """Generates L2-normalized 768-dimensional semantic embeddings locally."""

    def __init__(
        self,
        model_id: str | None = None,
        device: str | None = None,
        settings: ScoutSettings | None = None,
        allow_deterministic_fallback: bool = False,
    ) -> None:
        self.settings = settings or ScoutSettings()
        raw_id = model_id or self.settings.embedding_model_id or DEFAULT_GOOGLE_EMBEDDING_MODEL
        self.model_id = validate_google_model_id(raw_id)
        self.device = device or resolve_torch_device()
        self.allow_deterministic_fallback = allow_deterministic_fallback
        self._model: SentenceTransformer | None = None

    def _load_model(self) -> SentenceTransformer:
        """Lazy-loads SentenceTransformer model onto designated device."""
        if self._model is not None:
            return self._model

        try:
            import torch
            from sentence_transformers import SentenceTransformer

            # Use bfloat16 on MPS/CUDA if supported, otherwise float32
            torch_dtype = (
                torch.bfloat16
                if self.device in ("mps", "cuda")
                else torch.float32
            )
            logger.info(
                "Loading semantic embedding model '%s' on device '%s'...",
                self.model_id,
                self.device,
            )
            try:
                self._model = SentenceTransformer(
                    self.model_id,
                    device=self.device,
                    model_kwargs={"torch_dtype": torch_dtype},
                    local_files_only=True,
                )
            except Exception:
                self._model = SentenceTransformer(
                    self.model_id,
                    device=self.device,
                    model_kwargs={"torch_dtype": torch_dtype},
                )
            if hasattr(self._model, "max_seq_length"):
                self._model.max_seq_length = 2048
            return self._model
        except Exception as err:
            if self.allow_deterministic_fallback:
                logger.warning(
                    "Failed to load '%s' on %s (%s). Falling back to deterministic embedding.",
                    self.model_id,
                    self.device,
                    err,
                )
                raise
            raise RuntimeError(
                f"Unable to initialize Google embedding model '{self.model_id}' "
                f"on {self.device}: {err}"
            ) from err

    def embed_text(self, text: str, is_query: bool = False) -> list[float]:
        """Encodes a single text string into an L2-normalized 768-dim float vector."""
        return self.embed_batch([text], is_query=is_query)[0]

    def embed_batch(
        self,
        texts: list[str],
        is_query: bool = False,
        batch_size: int = 32,
    ) -> list[list[float]]:
        """Encodes a batch of texts into L2-normalized 768-dim float vectors."""
        if not texts:
            return []

        cleaned = [t.strip()[:6000] or " " for t in texts]
        model = self._load_model()

        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        kwargs: dict[str, Any] = {
            "batch_size": batch_size,
            "normalize_embeddings": True,
            "show_progress_bar": False,
        }
        if hasattr(model, "prompts") and model.prompts:
            if is_query and "query" in model.prompts:
                kwargs["prompt_name"] = "query"
            elif not is_query and "document" in model.prompts:
                kwargs["prompt_name"] = "document"

        try:
            raw_embeddings = model.encode(cleaned, **kwargs)
        except Exception as err:
            logger.warning(
                "Encode on device '%s' failed (%s); completing batch on CPU",
                self.device,
                err,
            )
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            cpu_kwargs = dict(kwargs)
            cpu_kwargs["device"] = "cpu"
            cpu_kwargs["batch_size"] = min(batch_size, 8)
            raw_embeddings = model.encode(cleaned, **cpu_kwargs)
        finally:
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

        vectors: list[list[float]] = []
        for vec in raw_embeddings:
            float_list = [float(v) for v in vec]
            if len(float_list) > EXPECTED_EMBEDDING_DIM:
                float_list = float_list[:EXPECTED_EMBEDDING_DIM]
            elif len(float_list) < EXPECTED_EMBEDDING_DIM:
                float_list.extend([0.0] * (EXPECTED_EMBEDDING_DIM - len(float_list)))

            # Guarantee exact L2 unit normalization
            norm = math.sqrt(sum(x * x for x in float_list))
            if norm > 0.0:
                normalized = [round(x / norm, 6) for x in float_list]
            else:
                default_val = round(1.0 / math.sqrt(EXPECTED_EMBEDDING_DIM), 6)
                normalized = [default_val] * EXPECTED_EMBEDDING_DIM
            vectors.append(normalized)

        return vectors

    @staticmethod
    def cosine_similarity(u: list[float], v: list[float]) -> float:
        """Computes cosine similarity between two float vectors."""
        dot = sum(a * b for a, b in zip(u, v, strict=False))
        norm_u = math.sqrt(sum(a * a for a in u))
        norm_v = math.sqrt(sum(b * b for b in v))
        if norm_u == 0.0 or norm_v == 0.0:
            return 0.0
        return dot / (norm_u * norm_v)
