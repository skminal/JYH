#!/usr/bin/env python3
"""Multi-panel GIF in the group's template style.

usage: render_gif.py OUT.gif CASE:LABEL [CASE:LABEL ...] [--test T_MS] [--tmax T_MS]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from PIL import Image
from scipy.interpolate import griddata

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ofio  # noqa: E402

plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.unicode_minus': True})
GAS = LinearSegmentedColormap.from_list('gas', ['#ffffff', '#9ecae1', '#3a7fc1', '#0b3d7a'])
XMAX, ZMIN, ZMAX = 5.0, -3.0, 20.0
DR = 0.025


class Field:
    def __init__(self, case_dir):
        self.c = ofio.Case(case_dir)
        self.times = self.c.times()
        self.cache = {}
        rr = np.arange(-XMAX + DR / 2, XMAX, DR)
        zz = np.arange(ZMIN + DR / 2, ZMAX, DR)
        self.R, self.Z = np.meshgrid(rr, zz)
        self.sel = self.c.r < XMAX + 0.3

    def at(self, t_ms):
        ts = np.array([t for t, _ in self.times]) * 1e3
        k = int(np.argmin(np.abs(ts - t_ms)))
        tn = self.times[k][1]
        if tn not in self.cache:
            a = self.c.alpha(tn)[self.sel]
            r, z = self.c.r[self.sel], self.c.z[self.sel]
            g = griddata((np.r_[r, -r], np.r_[z, z]), np.r_[a, a], (self.R, self.Z), method='linear')
            g = np.where(np.isnan(g), 1.0, g)
            g[(np.abs(self.R) > 0.15) & (np.abs(self.R) < 0.25) & (self.Z < 0)] = 1.0
            self.cache = {tn: g}
        return ts[k], self.cache[tn]


def draw_panel(ax, g, t_ms, label):
    gas = np.clip(1.0 - g, 0, 1)
    ax.imshow(np.ma.masked_less(gas, 0.02), origin='lower', extent=[-XMAX, XMAX, ZMIN, ZMAX],
              cmap=GAS, vmin=0, vmax=1, interpolation='bilinear', aspect='equal', zorder=1)
    rr = np.linspace(-XMAX + DR / 2, XMAX - DR / 2, g.shape[1])
    zz = np.linspace(ZMIN + DR / 2, ZMAX - DR / 2, g.shape[0])
    ax.contour(rr, zz, g, levels=[0.5], colors='#0b2a55', linewidths=0.9, zorder=2)
    # capillary: silica wall (grey) and tungsten needle (dark)
    for s in (-1, 1):
        x0 = 0.15 if s > 0 else -0.25
        ax.add_patch(Rectangle((x0, ZMIN), 0.10, -ZMIN, fc='#c9ccd1', ec='#7d828a', lw=0.5, zorder=3))
    ax.add_patch(Rectangle((-0.04, ZMIN), 0.08, -ZMIN, fc='#1d2733', ec='none', zorder=3))
    ax.axhline(0.0, color='#d9541a', lw=1.0, ls='--', zorder=4)
    ax.set_xlim(-XMAX, XMAX)
    ax.set_ylim(ZMIN, ZMAX)
    ax.set_xticks([-4, -2, 0, 2, 4])
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.tick_params(labelsize=9, length=3)
    ax.set_xlabel('x  [mm]', fontsize=10)
    ax.set_ylabel('z  [mm]', fontsize=10)
    ax.set_title('t = %.3f ms' % t_ms, fontsize=11, loc='left', pad=4)
    for sp in ax.spines.values():
        sp.set_linewidth(1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('cases', nargs='+', help='CASE_DIR:LABEL')
    ap.add_argument('--test', type=float, default=None)
    ap.add_argument('--tmax', type=float, default=None)
    ap.add_argument('--dt', type=float, default=0.5, help='frame spacing, ms')
    ap.add_argument('--ms_per_frame', type=int, default=70)
    a = ap.parse_args()

    panels = []
    for s in a.cases:
        d, lab = s.split(':', 1)
        panels.append((Field(d), lab))
    tend = min(p.times[-1][0] for p, _ in panels) * 1e3
    if a.tmax:
        tend = min(tend, a.tmax)
    times = [a.test] if a.test is not None else list(np.arange(a.dt, tend + 1e-9, a.dt))

    n = len(panels)
    frames = []
    for k, t in enumerate(times):
        fig = plt.figure(figsize=(4.0 * n, 8.2), dpi=100, facecolor='white')
        for j, (fld, lab) in enumerate(panels):
            left = 0.06 / n * 4 * 0.25 + j / n
            ax = fig.add_axes([j / n + 0.075 / n * 4 * 0.5, 0.075, 0.70 / n * 1.0, 0.80])
            tt, g = fld.at(t)
            draw_panel(ax, g, tt, lab)
            fig.text(j / n + 0.01, 0.955, lab, fontsize=20, fontweight='bold', ha='left', va='center')
        fig.text(1 - 0.01, 0.955, 't = %.1f ms' % t, fontsize=20, fontweight='bold', color='#b00000',
                 ha='right', va='center')
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        if a.test is not None:
            Image.fromarray(buf).save(a.out)
            print('wrote', a.out)
            return
        frames.append(Image.fromarray(buf))
        if k % 25 == 0:
            print('frame %d/%d  t=%.1f ms' % (k, len(times), t), flush=True)

    pal = frames[len(frames) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    dur = [a.ms_per_frame] * len(q)
    dur[-1] = 2000
    q[0].save(a.out, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=False)
    frames[-1].save(a.out.replace('.gif', '_last.png'))
    print('wrote', a.out, len(q), 'frames', '%.1f MB' % (os.path.getsize(a.out) / 1e6))


if __name__ == '__main__':
    main()
