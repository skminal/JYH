"""Read Gerris text snapshots / monitors of the needle-in-methanol case."""
import glob
import json
import os
import re

import numpy as np

COLS = ['x', 'y', 'z', 'T', 'Th', 'Phi', 'Ex', 'Ey', 'U', 'V', 'Kg', 'Nc',
        'Rhoe', 'P', 'Lv']


class Case:
    def __init__(self, rundir):
        self.dir = rundir
        meta = json.load(open(os.path.join(rundir, 'case.json')))
        self.p = meta['params']
        self.d = meta['derived']
        self.meta = meta
        self.Lb = self.p['Lb']                    # mm per unit length
        self.t0 = self.d['t0']                    # s per unit time
        self.XT = self.d['XT']
        self.Eunit = self.d['Phi0'] / self.d['L']  # V/m per unit field
        self.Punit = self.p['sigma'] * self.d['L']**2 / self.t0  # W per unit power
        self.Vunit = (self.Lb)**3                 # mm3 per unit volume
        self.dT = self.d['dT']

    # ---- snapshots -----------------------------------------------------
    def snapshots(self):
        fs = sorted(glob.glob(os.path.join(self.dir, 'snap-*.txt')))
        out = []
        for f in fs:
            t = float(re.search(r'snap-([0-9.]+)\.txt', f).group(1))
            out.append((t * self.t0 * 1e3, f))      # time in ms
        return out

    def load(self, fname):
        a = np.loadtxt(fname, comments='#')
        s = {c: a[:, i] for i, c in enumerate(COLS)}
        s['zmm'] = (s['x'] - self.XT) * self.Lb
        s['rmm'] = s['y'] * self.Lb
        s['hmm'] = self.Lb / 2.0**s['Lv']
        s['E'] = np.hypot(s['Ex'], s['Ey']) * self.Eunit     # V/m
        return s

    # ---- monitors ------------------------------------------------------
    def _mon(self, name, col=-1):
        t, v = [], []
        # hist_* holds the monitor output of the segments before a resume
        for fn in ('hist_' + name, name):
            fn = os.path.join(self.dir, fn)
            if not os.path.exists(fn):
                continue
            with open(fn) as f:
                for line in f:
                    w = line.split()
                    try:
                        i = w.index('time:')
                        t.append(float(w[i + 1]))
                        v.append(float(w[col]))
                    except (ValueError, IndexError):
                        pass
        t = np.array(t) * self.t0 * 1e3
        v = np.array(v)
        # a restarted run rewrites earlier times: keep the last occurrence
        _, idx = np.unique(t[::-1], return_index=True)
        idx = len(t) - 1 - idx
        return t[idx], v[idx]

    def monitors(self):
        m = {}
        twopi = 2 * np.pi          # sums are per radian
        m['t'], m['theta_max'] = self._mon('mon_Th')
        _, m['E_max'] = self._mon('mon_E')
        m['E_max'] = m['E_max'] * self.Eunit
        _, v = self._mon('mon_Vgas')
        m['Vgas'] = v * twopi * self.Vunit            # mm3
        _, v = self._mon('mon_Pliq')
        m['Pliq'] = v * twopi * self.Punit            # W
        _, v = self._mon('mon_Pgas')
        m['Pgas'] = v * twopi * self.Punit            # W
        _, v = self._mon('mon_Sth')
        m['Qth'] = v * twopi * self.Vunit / self.t0   # mm3/s (thermal phase change)
        _, v = self._mon('mon_Sd')
        m['Qd'] = v * twopi * self.Vunit / self.t0    # mm3/s (discharge gas)
        _, v = self._mon('mon_Nc')
        m['Vnc'] = v * twopi * self.Vunit             # mm3 non-condensable
        _, v = self._mon('mon_Vel')
        m['Umax'] = v * self.d['U0']
        n = min(len(x) for x in m.values())
        return {k: x[:n] for k, x in m.items()}


def needle_outline(case, zmin=-3.2, n=40):
    """(r, z) polygon of the needle in mm."""
    p = case.p
    rt, rs, b = p['r_tip'], p['r_shaft'], np.radians(p['cone_deg'])
    zc = -rt
    zt = zc + rt * np.sin(b)
    ang = np.linspace(np.pi / 2, b, n)        # from apex to tangent point
    r = list(rt * np.cos(ang))
    z = list(zc + rt * np.sin(ang))
    zs = zt - (rs - rt * np.cos(b)) / np.tan(b)
    r += [rs, rs]
    z += [zs, zmin]
    return np.array(r), np.array(z)


def raster(s, key, rmax, zmin, zmax, n_r, fill=np.nan):
    """Paint the quadtree cells of field `key` on a uniform (z, r) image."""
    dr = rmax / n_r
    n_z = int(round((zmax - zmin) / dr))
    img = np.full((n_z, n_r), fill, dtype=float)
    r, z, h = s['rmm'], s['zmm'], s['hmm']
    v = s[key] if isinstance(key, str) else key
    sel = (r - h / 2 < rmax) & (z + h / 2 > zmin) & (z - h / 2 < zmax)
    # cut cells store the centre of mass, snap to the cell centre
    for ri, zi, hi, vi in zip(r[sel], z[sel], h[sel], v[sel]):
        rc = (np.floor(ri / hi) + 0.5) * hi
        z0c = zi  # z origin is not aligned with the grid; use centre of mass
        i0 = int(np.floor((rc - hi / 2) / dr + 1e-6))
        i1 = int(np.ceil((rc + hi / 2) / dr - 1e-6))
        j0 = int(np.floor((z0c - hi / 2 - zmin) / dr + 0.5))
        j1 = int(np.floor((z0c + hi / 2 - zmin) / dr + 0.5))
        img[max(j0, 0):max(j1, 0), max(i0, 0):i1] = vi
    return img
