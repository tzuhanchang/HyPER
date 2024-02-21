from types import ModuleType
from importlib import import_module

import HyPER.data
import HyPER.graphs
import HyPER.models
import HyPER.utils
import HyPER.evaluation


# python/util/lazy_loader.py
class LazyLoader(ModuleType):
    def __init__(self, local_name, parent_module_globals, name):
        self._local_name = local_name
        self._parent_module_globals = parent_module_globals
        super().__init__(name)

    def _load(self):
        module = import_module(self.__name__)
        self._parent_module_globals[self._local_name] = module
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item):
        module = self._load()
        return getattr(module, item)

    def __dir__(self):
        module = self._load()
        return dir(module)


data = LazyLoader('data', globals(), 'HyPER.data')
graphs = LazyLoader('graphs', globals(), 'HyPER.graphs')
models = LazyLoader('models', globals(), 'HyPER.models')
utils = LazyLoader('utils', globals(), 'HyPER.utils')
evaluation = LazyLoader('evaluation', globals(), 'HyPER.evaluation')

__version__ = 'v1'


__all__ = [
    'HyPER',
    '__version__',
]
