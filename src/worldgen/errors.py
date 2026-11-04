"""Exception hierarchy. Every error raised deliberately by the compiler derives from WorldGenError."""


class WorldGenError(Exception):
    """Base class for compiler errors."""


class ConfigError(WorldGenError):
    """Invalid or inconsistent configuration."""


class SourceError(WorldGenError):
    """A data source could not be acquired or parsed."""


class OfflineCacheMiss(SourceError):
    """Offline mode was requested but the needed data is not cached."""


class GeometryError(WorldGenError):
    """Geometry could not be generated."""


class ValidationFailed(WorldGenError):
    """Validation found errors and the configuration asks to fail on them."""
