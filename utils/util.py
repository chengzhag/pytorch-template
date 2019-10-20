import json
import pandas as pd
from pathlib import Path
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

class MetricTracker:
    def __init__(self):
        self._data = pd.DataFrame()
        self.reset()
        
    def reset(self):
        self._data = pd.DataFrame()

    def update(self, tag_scalar_dict):
        self._data = self._data.append(tag_scalar_dict, ignore_index=True)

    def to_dict(self):
        return {col:self._data[col].values for col in self._data.columns}

    def avg(self, key):
        return self._data.__getattr__(key).average
    
    def result(self):
        return dict(self._data.mean())
