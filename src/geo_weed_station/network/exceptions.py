"""Network and communication exceptions for drone onboard REST API."""

from __future__ import annotations


class DroneNetworkError(Exception):
    """Base exception for all drone network operations."""


class DroneTimeoutError(DroneNetworkError):
    """Raised when an HTTP request exceeds configured timeout threshold (<= 2.0s)."""


class DroneConnectionError(DroneNetworkError):
    """Raised when connection to drone host fails or drops during communication."""


class DroneApiError(DroneNetworkError):
    """Raised when drone returns an HTTP error code (400, 404, 409, 500)."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"Drone API Error {status_code}: {detail}")


class DroneConflictError(DroneApiError):
    """Raised on HTTP 409 Conflict (e.g. starting mission when already active)."""


class DroneNotFoundError(DroneApiError):
    """Raised on HTTP 404 Not Found (e.g. unknown run_id or missing file)."""


class DownloadCancelled(DroneNetworkError):
    """Raised when streaming download is cancelled via cancellation callback."""
