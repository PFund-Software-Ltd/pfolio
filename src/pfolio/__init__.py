from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    import pfund_plot as plot

from importlib.metadata import version
from pfolio.config import Config



__version__ = version('pfolio')
config = Config()
def configure(
    lazy: bool = None,
    debug: bool = None
):
    if lazy is not None:
        config.set('lazy', lazy)
    if debug is not None:
        config.set('debug', debug)


def __getattr__(name: str):
    if name == 'plot':
        import pfund_plot as plot
        return plot
        

__all__ = (
    '__version__',
    'config',
    'configure',
    'plot'
)
def __dir__():
    return sorted(__all__)
