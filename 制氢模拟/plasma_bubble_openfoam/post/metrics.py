#!/usr/bin/env python3
"""Bubble metrics per written time: attached height/depth/width, detached bubbles.

usage: metrics.py CASEDIR [CASEDIR ...]   (prints a table, writes metrics.json in each case)
"""
import json
import os
import sys

import numpy as np
from scipy import ndimage
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ofio  # noqa: E402

DR = 0.02   # raster spacing, mm


def raster(c, a, rmax=5.0, zmin=-3.0, zmax=20.0):
    rr = np.arange(DR / 2, rmax, DR)
    zz = np.arange(zmin + DR / 2, zmax, DR)
    R, Z = np.meshgrid(rr, zz)
    # mirror across the axis so that the first raster column (r < first cell centre) is covered
    sel = (c.r < rmax + 0.5)
    g = griddata((np.r_[c.r[sel], -c.r[sel]], np.r_[c.z[sel], c.z[sel]]), np.r_[a[sel], a[sel]], (R, Z), method='linear')
    # cells inside the capillary wall are not in the mesh: mark them as liquid
    wall = (R > 0.15) & (R < 0.25) & (Z < 0)
    g[wall] = 1.0
    g = np.where(np.isnan(g), 1.0, g)
    return rr, zz, g


def analyse(c, tname):
    a = c.alpha(tname)
    rr, zz, g = raster(c, a)
    gas = g < 0.5
    lab, n = ndimage.label(gas)
    # the attached bubble is the component containing the orifice region
    i0 = np.searchsorted(zz, 0.05)
    cand = lab[i0, :5]
    attached = int(cand[cand > 0][0]) if (cand > 0).any() else 0
    dV = 2 * np.pi * rr[None, :] * DR * DR      # ring volume per raster cell, mm3
    res = dict(n_free=0, V_free=0.0, z_free=[], R_free=[])
    if attached > 0:
        m = lab == attached
        above = m & (zz[:, None] > -2.9)
        rows = np.where(above.any(1))[0]
        res['top'] = float(zz[rows.max()])
        outside = m & (rr[None, :] > 0.3)
        res['bottom'] = float(zz[np.where(outside.any(1))[0].min()]) if outside.any() else 0.0
        res['width'] = float(2 * rr[np.where(m.any(0))[0].max()])
        res['V_att'] = float((dV * (m & (zz[:, None] > 0))).sum() + (dV * (m & (zz[:, None] <= 0) & (rr[None, :] > 0.25))).sum())
        # neck: minimum radius above the widest point
        rad = np.array([rr[np.where(m[k])[0].max()] if m[k].any() else 0 for k in range(len(zz))])
        kw = int(np.argmax(rad))
        up = np.arange(kw, rows.max() + 1)
        if len(up) > 3:
            k = up[int(np.argmin(rad[up][:-2]))]
            res['neck_r'] = float(rad[k])
            res['neck_z'] = float(zz[k])
    for k in range(1, n + 1):
        if k == attached:
            continue
        m = lab == k
        v = float((dV * m).sum())
        if v < 0.01:
            continue
        res['n_free'] += 1
        res['V_free'] += v
        zc = float((dV * m * zz[:, None]).sum() / (dV * m).sum())
        res['z_free'].append(zc)
        res['R_free'].append(float((3 * v / (4 * np.pi))**(1 / 3)))
    return res


def main():
    for d in sys.argv[1:]:
        c = ofio.Case(d)
        out = []
        old = {}
        p = os.path.join(d, 'metrics.json')
        if os.path.exists(p):
            old = {x['tname']: x for x in json.load(open(p))}
        for t, tn in c.times():
            if tn in old:
                out.append(old[tn])
                continue
            r = analyse(c, tn)
            r['t_ms'] = t * 1e3
            r['tname'] = tn
            out.append(r)
        json.dump(out, open(p, 'w'), indent=0)
        print('==', os.path.basename(d))
        for r in out[::max(1, len(out) // 12)] + out[-1:]:
            print('  t=%6.2f ms top %5.2f bot %5.2f w %4.2f V_att %5.2f neck r %4.2f@z %4.2f | free %d V %.2f z %s' % (
                r['t_ms'], r.get('top', 0), r.get('bottom', 0), r.get('width', 0), r.get('V_att', 0),
                r.get('neck_r', 0), r.get('neck_z', 0), r['n_free'], r['V_free'],
                ','.join('%.1f' % z for z in r['z_free'])))


if __name__ == '__main__':
    main()
