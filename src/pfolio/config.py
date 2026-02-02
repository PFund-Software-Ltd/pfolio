from dataclasses import dataclass


@dataclass
class Config:
    _lazy: bool = True
    _debug: bool = False

    @property
    def lazy(self):
        return self._lazy

    @property
    def debug(self):
        return self._debug

    def set(self, key, value):
        private_key = f'_{key}'
        assert hasattr(self, private_key), f'Config has no attribute {key}'
        setattr(self, private_key, value)
        