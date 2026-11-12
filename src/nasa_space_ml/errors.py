"""Domain exceptions.

Every error the application can raise is one of these, so the API layer can map
exception type -> HTTP status in exactly one place (see `nasa_space_ml.api.problem`).
An unexpected exception is still caught and reported as a generic 500; nothing here
is allowed to carry an internal message into an HTTP response body.
"""

from __future__ import annotations


class NasaSpaceMlError(Exception):
    """Base class for every error this package raises deliberately."""

    code = "internal_error"
    http_status = 500

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigError(NasaSpaceMlError):
    """settings.yaml is missing, unreadable, or missing a required key."""

    code = "config_error"
    http_status = 500


class DataError(NasaSpaceMlError):
    """A NASA source returned something we could not use, or a cache artifact is bad."""

    code = "data_error"
    http_status = 502


class UpstreamNASAError(DataError):
    """The remote endpoint answered with an error status or an unexpected shape."""

    code = "upstream_nasa_error"
    http_status = 502


class ModelNotTrainedError(NasaSpaceMlError):
    """The service is running but no trained artifact exists on disk yet."""

    code = "model_not_trained"
    http_status = 503


class InvalidFeaturesError(NasaSpaceMlError):
    """A request row cannot be turned into the model's feature vector."""

    code = "invalid_features"
    http_status = 422
