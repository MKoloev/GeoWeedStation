"""Network client and download management package for GeoWeedStation."""

from geo_weed_station.network.backoff import ExponentialBackoff
from geo_weed_station.network.download_manager import DownloadManager
from geo_weed_station.network.drone_client import DroneClient
from geo_weed_station.network.exceptions import (
    DownloadCancelled,
    DroneApiError,
    DroneConflictError,
    DroneConnectionError,
    DroneNetworkError,
    DroneNotFoundError,
    DroneTimeoutError,
)

__all__ = [
    # exceptions
    "DroneNetworkError",
    "DroneTimeoutError",
    "DroneConnectionError",
    "DroneApiError",
    "DroneConflictError",
    "DroneNotFoundError",
    "DownloadCancelled",
    # client
    "DroneClient",
    "ExponentialBackoff",
    "DownloadManager",
]
