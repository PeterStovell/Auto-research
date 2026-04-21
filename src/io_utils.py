import fsspec
import pickle
import hashlib
import json
from omegaconf import OmegaConf
from typing import Any
import os
from glob import glob


def next_path(root):
    pattern = 'version_{}'
    paths = glob(os.path.join(root, pattern.format('*')))
    numbers = [-1] + [int(path.split('_')[-1]) for path in paths]
    next_number = max(numbers) + 1
    return os.path.join(root, pattern.format(next_number))


def exists(path: str) -> bool:
    return fsspec.open(path).fs.exists(path)


def conf_hash(cfg: Any) -> str:
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    encoded = json.dumps(cfg_dict, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(encoded.encode('utf-8')).hexdigest()


def conf_read(path: str) -> Any:
    with fsspec.open(path, 'r') as f:
        return OmegaConf.load(f)


def conf_write(cfg: Any, path: str):
    with fsspec.open(path, 'w') as f:
        f.write(OmegaConf.to_yaml(cfg))


def pickle_write(obj: Any, path: str):
    with fsspec.open(path, 'wb') as f:
        # noinspection PyTypeChecker
        pickle.dump(obj, f)


def pickle_read(path: str) -> Any:
    with fsspec.open(path, 'rb') as f:
        # noinspection PyTypeChecker
        return pickle.load(f)


def json_write(obj: Any, path: str):
    with fsspec.open(path, 'w') as f:
        # noinspection PyTypeChecker
        json.dump(obj, f)


def json_read(path: str) -> Any:
    with fsspec.open(path) as f:
        return json.load(f)
