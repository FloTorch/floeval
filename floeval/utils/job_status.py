"""Helpers for the optional progress file (logger name floeval.job_status).

floeval-worker can attach a file handler when WORKER_STATUS_LOG_PATH is set.
If nothing is attached, log_job_status / log_job_status_error do nothing.
"""

from __future__ import annotations

import logging
from typing import Any

JOB_STATUS_LOGGER_NAME = "floeval.job_status"


def log_job_status(msg: str, *args: Any, extra: dict[str, Any] | None = None) -> None:
    lg = logging.getLogger(JOB_STATUS_LOGGER_NAME)
    if not lg.handlers:
        return
    lg.info(msg, *args, extra=dict(extra) if extra else {})


def log_job_status_error(msg: str, *args: Any, extra: dict[str, Any] | None = None) -> None:
    lg = logging.getLogger(JOB_STATUS_LOGGER_NAME)
    if not lg.handlers:
        return
    lg.error(msg, *args, extra=dict(extra) if extra else {})
