"""Read a PyTorch .pth checkpoint without PyTorch.

A modern .pth is a zip archive: `<root>/data.pkl` holds the pickled object
tree and `<root>/data/<key>` holds the raw bytes of each tensor storage.
The pickle refers to storages through `persistent_load` and rebuilds tensors
with `torch._utils._rebuild_tensor_v2`. We stand in for both with numpy.
"""
import pickle
import zipfile
from collections import OrderedDict

import numpy as np

_DTYPES = {
    'FloatStorage': np.float32, 'HalfStorage': np.float16,
    'DoubleStorage': np.float64, 'LongStorage': np.int64,
    'IntStorage': np.int32, 'ByteStorage': np.uint8, 'BoolStorage': np.bool_,
}


class _StorageType:
    def __init__(self, name):
        self.dtype = _DTYPES.get(name, np.float32)


def _rebuild_tensor_v2(storage, offset, size, stride, *_):
    item = storage.itemsize
    if not size:
        return storage[offset:offset + 1].reshape(())
    return np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size),
        strides=tuple(s * item for s in stride)).copy()


def _rebuild_parameter(data, *_):
    return data


class _Unpickler(pickle.Unpickler):
    def __init__(self, fh, zf, root):
        super().__init__(fh)
        self.zf, self.root, self.cache = zf, root, {}

    def find_class(self, module, name):
        if module == 'collections' and name == 'OrderedDict':
            return OrderedDict
        if module == 'torch._utils' and name == '_rebuild_tensor_v2':
            return _rebuild_tensor_v2
        if module == 'torch._utils' and name == '_rebuild_parameter':
            return _rebuild_parameter
        if module == 'torch' and name.endswith('Storage'):
            return _StorageType(name)
        raise pickle.UnpicklingError(f'refusing to load {module}.{name}')

    def persistent_load(self, pid):
        # ('storage', storage_type, key, location, numel)
        _, stype, key, _loc, _numel = pid
        if key not in self.cache:
            raw = self.zf.read(f'{self.root}/data/{key}')
            self.cache[key] = np.frombuffer(raw, dtype=stype.dtype)
        return self.cache[key]


def load_pth(path):
    """Return the checkpoint's object tree with numpy arrays for tensors.

    Only the classes a plain state_dict needs are allowed, so a malicious
    checkpoint cannot execute code the way torch.load(weights_only=False) can.
    """
    with zipfile.ZipFile(path) as zf:
        pkl = next(n for n in zf.namelist() if n.endswith('data.pkl'))
        root = pkl.rsplit('/', 1)[0]
        with zf.open(pkl) as fh:
            return _Unpickler(fh, zf, root).load()
