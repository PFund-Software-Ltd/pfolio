from __future__ import annotations
from dataclasses import dataclass


_config: PFolioConfig | None = None


def get_config() -> PFolioConfig:
    """Lazy singleton - only creates config when first called.
    Also loads the .env file.
    """
    global _config
    if _config is None:
        _config = PFolioConfig()
    return _config


def configure(
    debug: bool | None = None
):
    config = get_config()
    if debug is not None:
        config.debug = debug


@dataclass
class PFolioConfig:
    debug: bool = False
