"""PostgreSQL RLS repository mixin: scout sessions and model artifact lifecycle."""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from typing import cast
from uuid import UUID

from psycopg import Connection

from context_scout.db.repos.types_and_helpers import (
    _as_opt_dt,
)
from context_scout.db.schema import (
    ModelArtifactKind,
    ModelArtifactRecord,
    ModelArtifactStatus,
    ScoutSessionRecord,
)

logger = logging.getLogger(__name__)


class _PgSessionsAndModelsMixin:
    @staticmethod
    def _row_to_scout_session(row: tuple[object, ...]) -> ScoutSessionRecord:
        recent_raw = row[6] if isinstance(row[6], list) else json.loads(str(row[6] or "[]"))
        blobs_raw = row[7] if isinstance(row[7], dict) else json.loads(str(row[7] or "{}"))
        syms_raw = row[8] if isinstance(row[8], list) else json.loads(str(row[8] or "[]"))
        scopes_raw = row[4]
        scopes = [str(s) for s in scopes_raw] if isinstance(scopes_raw, (list, tuple)) else []
        return ScoutSessionRecord(
            tenant_id=UUID(str(row[0])),
            session_id=str(row[1]),
            domain_id=str(row[2]),
            principal_id=str(row[3]),
            allowed_scopes=scopes,
            turn_count=int(str(row[5] or 0)),
            recent_turns_json=[dict(item) for item in recent_raw if isinstance(item, dict)],
            pinned_file_blobs_json={
                str(k): {str(sk): str(sv) for sk, sv in v.items()}
                for k, v in blobs_raw.items()
                if isinstance(v, dict)
            },
            pinned_symbols_json=[str(s) for s in syms_raw if str(s)],
            compacted_summary=str(row[9] or ""),
            total_tokens_compacted=int(str(row[10] or 0)),
            created_at=_as_opt_dt(row[11]),
            updated_at=_as_opt_dt(row[12]),
        )

    def get_scout_session(
        self,
        conn: Connection[tuple[object, ...]],
        session_id: str,
        *,
        principal_id: str | None = None,
    ) -> ScoutSessionRecord | None:
        if principal_id is not None:
            row = conn.execute(
                """
                SELECT tenant_id, session_id, domain_id, principal_id, allowed_scopes,
                       turn_count, recent_turns_json, pinned_file_blobs_json,
                       pinned_symbols_json, compacted_summary, total_tokens_compacted,
                       created_at, updated_at
                FROM scout_sessions
                WHERE session_id = %s AND principal_id = %s;
                """,
                (session_id, principal_id),
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT tenant_id, session_id, domain_id, principal_id, allowed_scopes,
                       turn_count, recent_turns_json, pinned_file_blobs_json,
                       pinned_symbols_json, compacted_summary, total_tokens_compacted,
                       created_at, updated_at
                FROM scout_sessions
                WHERE session_id = %s;
                """,
                (session_id,),
            ).fetchone()
        return self._row_to_scout_session(row) if row is not None else None

    def upsert_scout_session(
        self,
        conn: Connection[tuple[object, ...]],
        record: ScoutSessionRecord,
    ) -> ScoutSessionRecord:
        existing_owner_row = conn.execute(
            "SELECT principal_id FROM scout_sessions WHERE session_id = %s;",
            (record.session_id,),
        ).fetchone()
        if existing_owner_row is not None and str(existing_owner_row[0]) != record.principal_id:
            raise PermissionError(
                f"Session '{record.session_id}' belongs to principal "
                f"'{existing_owner_row[0]}', not '{record.principal_id}'."
            )
        row = conn.execute(
            """
            INSERT INTO scout_sessions (
                tenant_id, session_id, domain_id, principal_id, allowed_scopes,
                turn_count, compacted_summary, recent_turns_json,
                pinned_file_blobs_json, pinned_symbols_json,
                total_tokens_compacted, created_at, updated_at
            )
            VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, %s,
                COALESCE(%s, NOW()), COALESCE(%s, NOW())
            )
            ON CONFLICT (tenant_id, session_id) DO UPDATE SET
                domain_id = EXCLUDED.domain_id,
                principal_id = EXCLUDED.principal_id,
                allowed_scopes = EXCLUDED.allowed_scopes,
                turn_count = EXCLUDED.turn_count,
                compacted_summary = EXCLUDED.compacted_summary,
                recent_turns_json = EXCLUDED.recent_turns_json,
                pinned_file_blobs_json = EXCLUDED.pinned_file_blobs_json,
                pinned_symbols_json = EXCLUDED.pinned_symbols_json,
                total_tokens_compacted = EXCLUDED.total_tokens_compacted,
                updated_at = NOW()
            WHERE scout_sessions.principal_id = EXCLUDED.principal_id
            RETURNING tenant_id, session_id, domain_id, principal_id, allowed_scopes,
                      turn_count, recent_turns_json, pinned_file_blobs_json,
                      pinned_symbols_json, compacted_summary, total_tokens_compacted,
                      created_at, updated_at;
            """,
            (
                record.tenant_id,
                record.session_id,
                record.domain_id,
                record.principal_id,
                list(record.allowed_scopes),
                record.turn_count,
                record.compacted_summary,
                json.dumps(record.recent_turns_json),
                json.dumps(record.pinned_file_blobs_json),
                json.dumps(record.pinned_symbols_json),
                record.total_tokens_compacted,
                record.created_at,
                record.updated_at,
            ),
        ).fetchone()
        if row is None:
            return record
        return self._row_to_scout_session(row)

    def delete_scout_session(
        self,
        conn: Connection[tuple[object, ...]],
        session_id: str,
    ) -> int:
        cur = conn.execute("DELETE FROM scout_sessions WHERE session_id = %s;", (session_id,))
        return max(cur.rowcount or 0, 0)

    @staticmethod
    def _validate_model_artifact_record(record: ModelArtifactRecord) -> None:
        if record.lora_rank not in {8, 16, 32}:
            raise ValueError(
                f"model_artifacts lora_rank must be in {{8, 16, 32}}, got {record.lora_rank}"
            )
        if record.lora_alpha <= 0:
            raise ValueError("model_artifacts lora_alpha must be > 0")
        if not record.use_rslora:
            raise ValueError("model_artifacts requires use_rslora=True (vLLM multi-LoRA safety)")
        if record.use_dora:
            raise ValueError("model_artifacts prohibits use_dora=True (vLLM multi-LoRA safety)")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", record.base_model_revision_sha256.strip()):
            raise ValueError("base_model_revision_sha256 must be a 64-char SHA-256 hex digest")
        if not re.fullmatch(r"[0-9a-fA-F]{64}", record.weights_sha256.strip()):
            raise ValueError("weights_sha256 must be a 64-char SHA-256 hex digest")
        if not [s for s in record.training_scope_ceiling if s.strip()]:
            raise ValueError("training_scope_ceiling must be non-empty (OWASP ASI06)")
        if record.canary_leak_count != 0:
            raise ValueError("canary_leak_count must equal 0 (OWASP ASI06)")
        if record.status == "revoked" and record.is_active:
            raise PermissionError("Revoked model artifact cannot be registered with is_active=True")

    @staticmethod
    def _row_to_model_artifact(row: tuple[object, ...]) -> ModelArtifactRecord:
        raw_meta = row[20] if isinstance(row[20], dict) else json.loads(str(row[20] or "{}"))
        return ModelArtifactRecord(
            tenant_id=UUID(str(row[0])),
            domain_id=str(row[1]),
            artifact_id=str(row[2]),
            artifact_kind=cast(ModelArtifactKind, str(row[3])),
            base_model_id=str(row[4]),
            base_model_revision_sha256=str(row[5]),
            storage_uri=str(row[6]),
            weights_object_key=str(row[7]),
            config_object_key=str(row[8]),
            signature_object_key=str(row[9] or ""),
            weights_sha256=str(row[10]),
            cosign_signature_b64=str(row[11] or ""),
            cosign_key_id=str(row[12] or "default"),
            lora_rank=int(str(row[13])),
            lora_alpha=int(str(row[14])),
            use_rslora=bool(row[15]),
            use_dora=bool(row[16]),
            training_scope_ceiling=[str(s) for s in cast(Sequence[object], row[17] or [])],
            allowed_scopes=[str(s) for s in cast(Sequence[object], row[18] or [])],
            canary_leak_count=int(str(row[19] or 0)),
            metadata_json={str(k): v for k, v in raw_meta.items()},
            status=cast(ModelArtifactStatus, str(row[21])),
            is_active=bool(row[22]),
            created_at=_as_opt_dt(row[23]),
            activated_at=_as_opt_dt(row[24]),
            updated_at=_as_opt_dt(row[25]),
        )

    def register_model_artifact(
        self,
        conn: Connection[tuple[object, ...]],
        record: ModelArtifactRecord,
    ) -> ModelArtifactRecord:
        """Registers or updates a governed model artifact under Forced RLS."""
        self._validate_model_artifact_record(record)
        conn.execute(
            "SELECT pg_advisory_xact_lock("
            "hashtextextended("
            "COALESCE(current_setting('app.tenant_id', true), '') || ':model_artifacts:' "
            "|| %s || ':' || %s, 0)"
            ");",
            (record.domain_id, record.artifact_kind),
        )
        existing_status = conn.execute(
            """
            SELECT status
            FROM model_artifacts
            WHERE domain_id = %s AND artifact_id = %s
            LIMIT 1;
            """,
            (record.domain_id, record.artifact_id),
        ).fetchone()
        if existing_status is not None and str(existing_status[0]) == "revoked":
            raise PermissionError(
                f"Revoked model artifact '{record.artifact_id}' cannot be modified or re-activated."
            )
        if record.is_active:
            conn.execute(
                """
                UPDATE model_artifacts
                SET is_active = FALSE,
                    status = CASE WHEN status = 'active' THEN 'retired' ELSE status END,
                    updated_at = NOW()
                WHERE domain_id = %s
                  AND artifact_kind = %s
                  AND artifact_id <> %s
                  AND is_active = TRUE;
                """,
                (record.domain_id, record.artifact_kind, record.artifact_id),
            )
        effective_status: str = "active" if record.is_active else record.status
        row = conn.execute(
            """
            INSERT INTO model_artifacts (
                tenant_id, domain_id, artifact_id, artifact_kind,
                base_model_id, base_model_revision_sha256,
                storage_uri, weights_object_key, config_object_key, signature_object_key,
                weights_sha256, cosign_signature_b64, cosign_key_id,
                lora_rank, lora_alpha, use_rslora, use_dora,
                training_scope_ceiling, allowed_scopes, canary_leak_count,
                metadata_json, status, is_active, activated_at, updated_at
            )
            VALUES (
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s, %s,
                %s::jsonb, %s, %s,
                CASE WHEN %s THEN COALESCE(%s, NOW()) ELSE %s END,
                NOW()
            )
            ON CONFLICT (tenant_id, domain_id, artifact_id) DO UPDATE SET
                artifact_kind = EXCLUDED.artifact_kind,
                base_model_id = EXCLUDED.base_model_id,
                base_model_revision_sha256 = EXCLUDED.base_model_revision_sha256,
                storage_uri = EXCLUDED.storage_uri,
                weights_object_key = EXCLUDED.weights_object_key,
                config_object_key = EXCLUDED.config_object_key,
                signature_object_key = EXCLUDED.signature_object_key,
                weights_sha256 = EXCLUDED.weights_sha256,
                cosign_signature_b64 = EXCLUDED.cosign_signature_b64,
                cosign_key_id = EXCLUDED.cosign_key_id,
                lora_rank = EXCLUDED.lora_rank,
                lora_alpha = EXCLUDED.lora_alpha,
                use_rslora = EXCLUDED.use_rslora,
                use_dora = EXCLUDED.use_dora,
                training_scope_ceiling = EXCLUDED.training_scope_ceiling,
                allowed_scopes = EXCLUDED.allowed_scopes,
                canary_leak_count = EXCLUDED.canary_leak_count,
                metadata_json = EXCLUDED.metadata_json,
                status = EXCLUDED.status,
                is_active = EXCLUDED.is_active,
                activated_at = CASE
                    WHEN EXCLUDED.is_active THEN COALESCE(model_artifacts.activated_at, NOW())
                    ELSE model_artifacts.activated_at
                END,
                updated_at = NOW()
            RETURNING tenant_id, domain_id, artifact_id, artifact_kind,
                      base_model_id, base_model_revision_sha256,
                      storage_uri, weights_object_key, config_object_key, signature_object_key,
                      weights_sha256, cosign_signature_b64, cosign_key_id,
                      lora_rank, lora_alpha, use_rslora, use_dora,
                      training_scope_ceiling, allowed_scopes, canary_leak_count,
                      metadata_json, status, is_active, created_at, activated_at, updated_at;
            """,
            (
                record.tenant_id,
                record.domain_id,
                record.artifact_id,
                record.artifact_kind,
                record.base_model_id,
                record.base_model_revision_sha256,
                record.storage_uri,
                record.weights_object_key,
                record.config_object_key,
                record.signature_object_key,
                record.weights_sha256,
                record.cosign_signature_b64,
                record.cosign_key_id,
                record.lora_rank,
                record.lora_alpha,
                record.use_rslora,
                record.use_dora,
                list(record.training_scope_ceiling),
                list(record.allowed_scopes),
                record.canary_leak_count,
                json.dumps(record.metadata_json),
                effective_status,
                record.is_active,
                record.is_active,
                record.activated_at,
                record.activated_at,
            ),
        ).fetchone()
        if row is None:
            return record
        return self._row_to_model_artifact(row)

    def activate_model_artifact(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        artifact_id: str,
        *,
        artifact_kind: ModelArtifactKind = "lora_adapter",
    ) -> ModelArtifactRecord:
        """Atomically swaps the active model_artifact pointer for (tenant_id, domain_id, kind)."""
        conn.execute(
            "SELECT pg_advisory_xact_lock("
            "hashtextextended("
            "COALESCE(current_setting('app.tenant_id', true), '') || ':model_artifacts:' "
            "|| %s || ':' || %s, 0)"
            ");",
            (domain_id, artifact_kind),
        )
        existing = conn.execute(
            """
            SELECT status
            FROM model_artifacts
            WHERE domain_id = %s AND artifact_id = %s AND artifact_kind = %s
            LIMIT 1;
            """,
            (domain_id, artifact_id, artifact_kind),
        ).fetchone()
        if existing is None:
            raise KeyError(
                f"Model artifact '{artifact_id}' ({artifact_kind}) not registered for '{domain_id}'"
            )
        if str(existing[0]) == "revoked":
            raise PermissionError(f"Revoked model artifact '{artifact_id}' cannot be re-activated.")

        conn.execute(
            """
            UPDATE model_artifacts
            SET is_active = FALSE,
                status = CASE WHEN status = 'active' THEN 'retired' ELSE status END,
                updated_at = NOW()
            WHERE domain_id = %s
              AND artifact_kind = %s
              AND artifact_id <> %s
              AND is_active = TRUE;
            """,
            (domain_id, artifact_kind, artifact_id),
        )
        row = conn.execute(
            """
            UPDATE model_artifacts
            SET is_active = TRUE,
                status = 'active',
                activated_at = COALESCE(activated_at, NOW()),
                updated_at = NOW()
            WHERE domain_id = %s AND artifact_id = %s AND artifact_kind = %s
            RETURNING tenant_id, domain_id, artifact_id, artifact_kind,
                      base_model_id, base_model_revision_sha256,
                      storage_uri, weights_object_key, config_object_key, signature_object_key,
                      weights_sha256, cosign_signature_b64, cosign_key_id,
                      lora_rank, lora_alpha, use_rslora, use_dora,
                      training_scope_ceiling, allowed_scopes, canary_leak_count,
                      metadata_json, status, is_active, created_at, activated_at, updated_at;
            """,
            (domain_id, artifact_id, artifact_kind),
        ).fetchone()
        if row is None:
            raise KeyError(
                f"Model artifact '{artifact_id}' ({artifact_kind}) not found during activation"
            )
        return self._row_to_model_artifact(row)

    def retire_model_artifact(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        artifact_id: str,
        *,
        revoke: bool = False,
    ) -> bool:
        target_status = "revoked" if revoke else "retired"
        cur = conn.execute(
            """
            UPDATE model_artifacts
            SET is_active = FALSE,
                status = %s,
                updated_at = NOW()
            WHERE domain_id = %s AND artifact_id = %s;
            """,
            (target_status, domain_id, artifact_id),
        )
        return bool((cur.rowcount or 0) > 0)

    def get_active_model_artifact(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        artifact_kind: ModelArtifactKind = "lora_adapter",
    ) -> ModelArtifactRecord | None:
        row = conn.execute(
            """
            SELECT tenant_id, domain_id, artifact_id, artifact_kind,
                   base_model_id, base_model_revision_sha256,
                   storage_uri, weights_object_key, config_object_key, signature_object_key,
                   weights_sha256, cosign_signature_b64, cosign_key_id,
                   lora_rank, lora_alpha, use_rslora, use_dora,
                   training_scope_ceiling, allowed_scopes, canary_leak_count,
                   metadata_json, status, is_active, created_at, activated_at, updated_at
            FROM model_artifacts
            WHERE domain_id = %s AND artifact_kind = %s AND is_active = TRUE
            LIMIT 1;
            """,
            (domain_id, artifact_kind),
        ).fetchone()
        return self._row_to_model_artifact(row) if row is not None else None

    def list_active_model_artifacts(
        self,
        conn: Connection[tuple[object, ...]],
        *,
        artifact_kind: ModelArtifactKind = "lora_adapter",
    ) -> list[ModelArtifactRecord]:
        rows = conn.execute(
            """
            SELECT tenant_id, domain_id, artifact_id, artifact_kind,
                   base_model_id, base_model_revision_sha256,
                   storage_uri, weights_object_key, config_object_key, signature_object_key,
                   weights_sha256, cosign_signature_b64, cosign_key_id,
                   lora_rank, lora_alpha, use_rslora, use_dora,
                   training_scope_ceiling, allowed_scopes, canary_leak_count,
                   metadata_json, status, is_active, created_at, activated_at, updated_at
            FROM model_artifacts
            WHERE artifact_kind = %s AND is_active = TRUE
            ORDER BY domain_id, artifact_id;
            """,
            (artifact_kind,),
        ).fetchall()
        return [self._row_to_model_artifact(r) for r in rows]

    def list_model_artifacts(
        self,
        conn: Connection[tuple[object, ...]],
        domain_id: str,
        *,
        artifact_kind: ModelArtifactKind | None = None,
    ) -> list[ModelArtifactRecord]:
        if artifact_kind is not None:
            rows = conn.execute(
                """
                SELECT tenant_id, domain_id, artifact_id, artifact_kind,
                       base_model_id, base_model_revision_sha256,
                       storage_uri, weights_object_key, config_object_key, signature_object_key,
                       weights_sha256, cosign_signature_b64, cosign_key_id,
                       lora_rank, lora_alpha, use_rslora, use_dora,
                       training_scope_ceiling, allowed_scopes, canary_leak_count,
                       metadata_json, status, is_active, created_at, activated_at, updated_at
                FROM model_artifacts
                WHERE domain_id = %s AND artifact_kind = %s
                ORDER BY created_at DESC, artifact_id ASC;
                """,
                (domain_id, artifact_kind),
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT tenant_id, domain_id, artifact_id, artifact_kind,
                       base_model_id, base_model_revision_sha256,
                       storage_uri, weights_object_key, config_object_key, signature_object_key,
                       weights_sha256, cosign_signature_b64, cosign_key_id,
                       lora_rank, lora_alpha, use_rslora, use_dora,
                       training_scope_ceiling, allowed_scopes, canary_leak_count,
                       metadata_json, status, is_active, created_at, activated_at, updated_at
                FROM model_artifacts
                WHERE domain_id = %s
                ORDER BY created_at DESC, artifact_id ASC;
                """,
                (domain_id,),
            ).fetchall()
        return [self._row_to_model_artifact(r) for r in rows]
