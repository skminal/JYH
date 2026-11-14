#!/usr/bin/env python3
"""Minimal, Origin-style animation: discharge off vs discharge on at 2.6 kV.

Only the fields, the axes and the time stamp. All text is set in a Times
face (Times New Roman if installed, otherwise the metric-compatible
Liberation Serif) with proper sub/superscripts through mathtext.

usage: render_gif_min.py OUT.gif [--test T_MS]
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import LogLocator, MultipleLocator, NullFormatter
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfsio  # noqa: E402

W = os.environ.get('RUNS', '/tmp/claude-0/-home-user-JYH/c62af2be-1b9c-5e55-83fe-02b1a29689e9/scratchpad')

have = {f.name for f in font_manager.fontManager.ttflist}
TIMES = 'Times New Roman' if 'Times New Roman' in have else 'Liberation Serif'
plt.rcParams.update({
    'font.family': TIMES,
    'font.size': 13,
    'mathtext.fontset': 'custom',
    'mathtext.rm': TIMES,
    'mathtext.it': TIMES + ':italic',
    'mathtext.bf': TIMES + ':bold',
    'axes.unicode_minus': True,
    'axes.linewidth': 1.0,
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.top': True, 'ytick.right': True,
    'xtick.major.size': 4.5, 'ytick.major.size': 4.5,
    'xtick.minor.size': 2.5, 'ytick.minor.size': 2.5,
    'xtick.major.width': 0.9, 'ytick.major.width': 0.9,
    'legend.frameon': False,
})

INK = '#000000'
C_OFF, C_ON = '#1f5fbf', '#d9480f'
GAS = '#16233f'
CM_T = LinearSegmentedColormap.from_list('t', ['#f3f6fb', '#fde6c8', '#f9b877', '#ec7a3c', '#c2372c'])
CM_P = LinearSegmentedColormap.from_list('p', ['#16233f', '#4a2a8a', '#b04fd0', '#ffd9ff'])


def fill_nan(a, fill):
    bad = np.isnan(a)
    if not bad.any():
        return a
    if bad.all():
        return np.full_like(a, fill)
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def view(ax, case, s, rmax, zmin, zmax, nr, label):
    ext = [-rmax, rmax, zmin, zmax]
    T = fill_nan(gfsio.raster(s, 'T', rmax, zmin, zmax, nr, fill=np.nan), 1.0)
    th = gfsio.raster(s, 'Th', rmax, zmin, zmax, nr, fill=np.nan)
    kg_si = s['Kg'] * 8.8541878128e-12 / case.t0
    pd = gfsio.raster(s, kg_si * s['E']**2, rmax, zmin, zmax, nr, fill=0.0)
    th = fill_nan(np.where(T > 0.5, th, np.nan), 0.0)
    sg = max(nr * case.Lb / 2**case.p['lmax'] / rmax * 0.45, 0.6)
    mir = lambda a: np.concatenate([a[:, ::-1], a], axis=1)
    Ts = mir(ndimage.gaussian_filter(T, sg, mode='nearest'))
    ths = mir(ndimage.gaussian_filter(th, sg, mode='nearest'))
    pds = mir(ndimage.gaussian_filter(pd, sg, mode='nearest'))
    nz = Ts.shape[0]
    zz = zmin + (np.arange(nz) + 0.5) * (zmax - zmin) / nz
    rr = -rmax + (np.arange(2 * nr) + 0.5) * rmax / nr
    im = ax.imshow(ths, origin='lower', extent=ext, cmap=CM_T, vmin=0, vmax=1.2,
                   interpolation='bicubic', aspect='equal', zorder=1)
    if (Ts < 0.5).any():
        ax.contourf(rr, zz, Ts, levels=[-1, 0.5], colors=[GAS], zorder=2)
        lp = np.where(Ts < 0.5, np.log10(np.maximum(pds, 1.0)), np.nan)
        if np.nanmax(lp) > 7.0:
            ax.contourf(rr, zz, lp, levels=np.linspace(7.0, 11.0, 9), cmap=CM_P, extend='max', zorder=3)
        ax.contour(rr, zz, Ts, levels=[0.5], colors='white', linewidths=1.0, zorder=4)
    r, z = gfsio.needle_outline(case, zmin=zmin - 0.1)
    ax.fill(np.concatenate([-r[::-1], r]), np.concatenate([z[::-1], z]), color='#8f8d87',
            ec='#3d3c39', lw=0.8, zorder=5)
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(zmin, zmax)
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which='both', color=INK, labelsize=12, zorder=6)
    ax.set_xlabel(r'$x$ (mm)', fontsize=14)
    ax.set_title(label, fontsize=15, loc='left', pad=7)
    for sp in ax.spines.values():
        sp.set_zorder(7)
    return im


def nearest(snaps, t):
    ts = np.array([x[0] for x in snaps])
    return snaps[int(np.argmin(np.abs(ts - t)))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--off', default='run_off')
    ap.add_argument('--on', default='run_corona')
    ap.add_argument('--rmax', type=float, default=0.3)
    ap.add_argument('--zmin', type=float, default=-0.3)
    ap.add_argument('--zmax', type=float, default=0.7)
    ap.add_argument('--nr', type=int, default=96)
    ap.add_argument('--dt_heat', type=float, default=1.0)
    ap.add_argument('--dt_fast', type=float, default=0.05)
    ap.add_argument('--tmax', type=float, default=36.40)
    ap.add_argument('--test', type=float, default=None)
    a = ap.parse_args()

    cases = [gfsio.Case(os.path.join(W, a.off)), gfsio.Case(os.path.join(W, a.on))]
    snaps = [c.snapshots() for c in cases]
    mons = [c.monitors() for c in cases]
    cols = [C_OFF, C_ON]
    kp = cases[1].p.get('Kp_over_Kl', 0.1)
    labels = ['(a) Discharge off',
              r'(b) Discharge on, $K_{\mathrm{p}} = %g\,\sigma_{\mathrm{e}}$' % kp]
    tend = min(min(s[-1][0] for s in snaps), a.tmax)
    tnuc = min(m['t'][np.argmax(m['Vgas'] > 0)] for m in mons)
    times = list(np.arange(0.0, tnuc - 0.3, a.dt_heat)) + list(np.arange(tnuc - 0.3, tend + 1e-9, a.dt_fast))
    if a.test is not None:
        times = [a.test]

    frames = []
    for k, t in enumerate(times):
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor='white')
        gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1.25], left=0.06, right=0.975,
                              top=0.90, bottom=0.095, wspace=0.30, hspace=0.30)
        im = None
        for j in range(2):
            ax = fig.add_subplot(gs[:, j])
            im = view(ax, cases[j], cases[j].load(nearest(snaps[j], t)[1]),
                      a.rmax, a.zmin, a.zmax, a.nr, labels[j])
            if j == 0:
                ax.set_ylabel(r'$z$ (mm)', fontsize=14)
                cax = ax.inset_axes([0.07, 0.72, 0.035, 0.22])
                cb = fig.colorbar(im, cax=cax)
                cb.set_ticks([0, 0.5, 1.0])
                cb.ax.tick_params(labelsize=11, length=3, direction='in')
                cb.ax.set_title(r'$\theta$', fontsize=14, pad=4)
                cb.outline.set_linewidth(0.7)
        fig.text(0.975, 0.955, r'$t$ = %.3f ms' % t, fontsize=18, ha='right', va='center')

        fast = t >= tnuc - 0.3
        xl = (tnuc - 0.5, max(tend, tnuc + 0.5)) if fast else (0, tend)

        def trace(ax, ylab, ylog=False, last=False):
            ax.set_xlim(*xl)
            ax.axvline(t, color=INK, lw=0.9, zorder=5)
            ax.set_ylabel(ylab, fontsize=14)
            ax.tick_params(which='both', labelsize=12)
            ax.xaxis.set_major_locator(MultipleLocator(0.5 if fast else 10))
            ax.xaxis.set_minor_locator(MultipleLocator(0.25 if fast else 5))
            if ylog:
                ax.set_yscale('log')
                ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,), numticks=12))
                ax.yaxis.set_minor_locator(LogLocator(base=10, subs=(1.0,), numticks=12))
                ax.yaxis.set_minor_formatter(NullFormatter())
            if last:
                ax.set_xlabel(r'$t$ (ms)', fontsize=14)
            else:
                ax.tick_params(labelbottom=False)

        ax1 = fig.add_subplot(gs[0, 2])
        trace(ax1, r'$\theta_{\mathrm{max}}$')
        for m, c_ in zip(mons, cols):
            ax1.plot(m['t'], m['theta_max'], color=c_, lw=1.6)
        ax1.axhline(1.0, color=INK, lw=0.8, ls='--')
        ax1.text(xl[0] + 0.03 * (xl[1] - xl[0]), 0.80, r'$\theta = 1\ (T_{\mathrm{sat}})$', fontsize=12)
        ax1.set_ylim(0, 2.0)
        ax1.yaxis.set_major_locator(MultipleLocator(0.5))
        ax1.yaxis.set_minor_locator(MultipleLocator(0.25))

        ax2 = fig.add_subplot(gs[1, 2])
        trace(ax2, r'$V_{\mathrm{gas}}$ (mm$^{3}$)', ylog=True)
        ax2.plot(mons[0]['t'], np.maximum(mons[0]['Vgas'], 1e-12), color=C_OFF, lw=1.6, label='Discharge off')
        ax2.plot(mons[1]['t'], np.maximum(mons[1]['Vgas'], 1e-12), color=C_ON, lw=1.6, label='Discharge on')
        ax2.plot(mons[1]['t'], np.maximum(mons[1]['Vnc'], 1e-12), color=C_ON, lw=1.2, ls='--',
                 label='Non-condensable')
        ax2.set_ylim(1e-7, 1e3)
        ax2.yaxis.set_major_locator(LogLocator(base=100, numticks=8))
        ax2.legend(fontsize=11, loc='upper left', ncol=2, handlelength=1.4, columnspacing=0.9,
                   handletextpad=0.4, borderaxespad=0.3)

        ax3 = fig.add_subplot(gs[2, 2])
        trace(ax3, r'$P$ (W)', ylog=True, last=True)
        ax3.plot(mons[1]['t'], np.maximum(mons[1]['Pliq'], 1e-12), color=INK, lw=1.4, label='Joule heating, liquid')
        ax3.plot(mons[1]['t'], np.maximum(mons[1]['Pgas'], 1e-12), color=C_ON, lw=1.6, label='Discharge, gas')
        ax3.set_ylim(1e-6, 100)
        ax3.yaxis.set_major_locator(LogLocator(base=100, numticks=8))
        ax3.legend(fontsize=11, loc='upper left', ncol=2, handlelength=1.4, columnspacing=0.9,
                   handletextpad=0.4, borderaxespad=0.3)

        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        if a.test is not None:
            Image.fromarray(buf).save(a.out)
            print('wrote', a.out, 'font:', TIMES)
            return
        frames.append(Image.fromarray(buf))
        if k % 20 == 0:
            print('frame %d/%d t=%.3f' % (k, len(times), t), flush=True)

    pal = frames[-1].quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    nheat = int(np.sum(np.array(times) < tnuc - 0.3))
    dur = [100] * nheat + [90] * (len(q) - nheat)
    dur[-1] = 2500
    # the last frame is shown first so that a static view of the file is informative
    q = [q[-1]] + q
    dur = [1800] + dur
    q[0].save(a.out, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=False)
    frames[-1].save(a.out.replace('.gif', '_last.png'))
    print('wrote', a.out, len(q), 'frames', '%.1f MB' % (os.path.getsize(a.out) / 1e6), 'font:', TIMES)


if __name__ == '__main__':
    main()
