"""Background indexing worker with signal handling and event dispatching (F07)."""

from __future__ import annotations

import logging
import signal
import threading
import time
import traceback
from collections.abc import Callable
from typing import Any
from uuid import UUID, uuid4

from context_scout.queue.task_queue import QueueTask, TaskQueueProtocol

logger = logging.getLogger(__name__)


class IndexingWorker:
    """Production background worker claiming tasks via SKIP LOCKED and dispatching events."""

    def __init__(
        self,
        task_queue: TaskQueueProtocol,
        worker_id: str | None = None,
        poll_interval_seconds: float = 1.0,
        max_idle_backoff_seconds: float = 5.0,
        stale_reclaim_interval_seconds: float = 60.0,
    ) -> None:
        self.task_queue = task_queue
        self.worker_id = worker_id or f"scout-worker-{uuid4().hex[:8]}"
        self.poll_interval = poll_interval_seconds
        self.max_idle_backoff = max_idle_backoff_seconds
        self.stale_reclaim_interval = stale_reclaim_interval_seconds

        self._stop_event = threading.Event()
        self._current_task_id: UUID | None = None
        self._handlers: dict[str, Callable[[QueueTask], None]] = {}
        self._register_default_handlers()

    def register_handler(self, event_type: str, handler: Callable[[QueueTask], None]) -> None:
        """Registers or overrides an event dispatcher handler."""
        self._handlers[event_type] = handler

    def stop(self) -> None:
        """Signals the worker loop to stop gracefully after completing current in-flight task."""
        logger.info("Worker %s received stop signal. Stopping gracefully...", self.worker_id)
        self._stop_event.set()

    def is_running(self) -> bool:
        return not self._stop_event.is_set()

    def install_signal_handlers(self) -> None:
        """Installs SIGINT and SIGTERM handlers to trigger graceful stop."""

        def _handle_signal(signum: int, _frame: Any) -> None:
            sig_name = signal.Signals(signum).name
            logger.info("Signal %s received by worker %s", sig_name, self.worker_id)
            self.stop()

        try:
            signal.signal(signal.SIGINT, _handle_signal)
            signal.signal(signal.SIGTERM, _handle_signal)
        except (ValueError, AttributeError) as exc:
            # Happens in secondary threads during testing
            logger.debug(
                "Skipping signal handler registration outside main thread (%s): %s",
                type(exc).__name__,
                exc,
            )

    def run_once(self) -> bool:
        """Attempts to claim and execute a single task. Returns True if task processed."""
        task: QueueTask | None = None
        try:
            task = self.task_queue.fetch_next_task(self.worker_id)
            if not task:
                return False

            self._current_task_id = task.task_id
            logger.info(
                "Worker %s claimed task %s [%s] (attempt %d/%d)",
                self.worker_id,
                task.task_id,
                task.event_type,
                task.attempts,
                task.max_attempts,
            )

            handler = self._handlers.get(task.event_type)
            if not handler:
                raise ValueError(f"No registered handler for event_type '{task.event_type}'")

            handler(task)
            self.task_queue.complete_task(task.task_id)
            logger.info("Worker %s successfully completed task %s", self.worker_id, task.task_id)
            return True

        except Exception as exc:
            if task is not None:
                err_msg = str(exc)
                tb = traceback.format_exc()
                logger.error(
                    "Worker %s failed executing task %s: %s\n%s",
                    self.worker_id,
                    task.task_id,
                    err_msg,
                    tb,
                )
                self.task_queue.fail_task(task.task_id, error_message=err_msg, stack_trace=tb)
            return True
        finally:
            self._current_task_id = None

    def run(self) -> None:
        """Main worker execution loop."""
        logger.info("Starting IndexingWorker %s", self.worker_id)
        idle_cycles = 0
        last_stale_reclaim = time.monotonic()

        while not self._stop_event.is_set():
            now = time.monotonic()
            if now - last_stale_reclaim >= self.stale_reclaim_interval:
                try:
                    reclaimed = self.task_queue.reclaim_stale_tasks(stale_threshold_seconds=300)
                    if reclaimed > 0:
                        logger.warning(
                            "Worker %s reclaimed %d stale tasks", self.worker_id, reclaimed
                        )
                except Exception as exc:
                    logger.error("Error during stale task reclamation: %s", exc)
                last_stale_reclaim = now

            processed = self.run_once()
            if processed:
                idle_cycles = 0
            else:
                idle_cycles += 1
                backoff = min(
                    self.max_idle_backoff, self.poll_interval * (1.2 ** min(idle_cycles, 10))
                )
                self._stop_event.wait(timeout=backoff)

        logger.info("IndexingWorker %s stopped cleanly", self.worker_id)

    def _register_default_handlers(self) -> None:
        self._handlers["gitlab_push"] = self._handle_gitlab_push
        self._handlers["github_push"] = self._handle_github_push
        self._handlers["confluence_update"] = self._handle_confluence_update
        self._handlers["jira_update"] = self._handle_jira_update
        self._handlers["cross_repo_relink"] = self._handle_cross_repo_relink

    def _handle_gitlab_push(self, task: QueueTask) -> None:
        logger.info(
            "Executing gitlab_push handler for task %s: %s",
            task.task_id,
            task.payload.get("project"),
        )

    def _handle_github_push(self, task: QueueTask) -> None:
        logger.info(
            "Executing github_push handler for task %s: %s",
            task.task_id,
            task.payload.get("repository"),
        )

    def _handle_confluence_update(self, task: QueueTask) -> None:
        logger.info(
            "Executing confluence_update handler for task %s: %s",
            task.task_id,
            task.payload.get("page_id"),
        )

    def _handle_jira_update(self, task: QueueTask) -> None:
        logger.info(
            "Executing jira_update handler for task %s: %s",
            task.task_id,
            task.payload.get("issue_key"),
        )

    def _handle_cross_repo_relink(self, task: QueueTask) -> None:
        logger.info(
            "Executing cross_repo_relink handler for task %s, domain: %s",
            task.task_id,
            task.domain_id,
        )
