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
        
        
__all__ = (
    '__version__',
    'config',
    'configure'
)