"""Minimal readers for ascii OpenFOAM fields of the axisymmetric bubble cases."""
import json
import os
import re

import numpy as np


def _internal(path):
    """internalField of an ascii OpenFOAM volField: ndarray (n,) or (n,3), or a uniform value."""
    s = open(path).read()
    i = s.index('internalField')
    line = s[i:s.index('\n', i)]
    if 'nonuniform' not in line:
        return float(re.search(r'uniform\s+([^;]+);', line).group(1))
    rest = s[s.index('\n', i) + 1:]
    m = re.match(r'\s*(\d+)\s*\(', rest)
    n = int(m.group(1))
    j = m.end()
    body = rest[j:rest.index('\n)', j)]
    if 'vector' in line:
        return np.array(body.replace('(', ' ').replace(')', ' ').split(), float).reshape(n, 3)
    return np.array(body.split(), float)


class Case:
    def __init__(self, d):
        self.d = d
        self.meta = json.load(open(os.path.join(d, 'case.json')))
        C = _internal(os.path.join(d, 'constant/polyMesh/C'))
        self.r = np.hypot(C[:, 0], C[:, 1]) * 1e3   # mm
        self.z = C[:, 2] * 1e3
        self.n = len(self.r)

    def times(self):
        out = []
        for e in os.listdir(self.d):
            try:
                t = float(e)
            except ValueError:
                continue
            if t > 0 and os.path.exists(os.path.join(self.d, e, 'alpha.ethanol')):
                out.append((t, e))
        return sorted(out)

    def alpha(self, tname):
        a = _internal(os.path.join(self.d, tname, 'alpha.ethanol'))
        if np.ndim(a) == 0:
            a = np.full(self.n, float(a))
        return a

    def gas_volume(self, a, dx=None):
        """Volume (mm3) of gas per cell estimate needs cell volumes: use V file if present."""
        Vp = os.path.join(self.d, 'constant/polyMesh/V')
        if not hasattr(self, '_V'):
            self._V = None
        return None
