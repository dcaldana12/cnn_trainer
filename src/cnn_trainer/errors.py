"""Exception hierarchy used across the package."""


class CnnTrainerError(Exception):
    """Base class for all errors raised by cnn-trainer."""


class ConfigError(CnnTrainerError):
    """The pipeline YAML file is invalid or incomplete."""


class DatasetError(CnnTrainerError):
    """The source dataset cannot be read or is inconsistent."""


class BlockError(CnnTrainerError):
    """A pipeline block failed."""
