#!/usr/bin/env python3
"""Render the comparison animation: discharge OFF vs discharge ON at 2.6 kV.

usage: render_gif.py OUT.gif [--frames N] [--test T_MS]
Each frame: two near-tip views (liquid temperature, gas phase, discharge power
density, interface) and three time traces with a moving cursor.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import FuncFormatter, LogLocator
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfsio  # noqa: E402

FONT = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
if os.path.exists(FONT):
    font_manager.fontManager.addfont(FONT)
    plt.rcParams['font.family'] = ['DejaVu Sans', font_manager.FontProperties(fname=FONT).get_name()]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'

W = os.environ.get('RUNS', '/tmp/claude-0/-home-user-JYH/c62af2be-1b9c-5e55-83fe-02b1a29689e9/scratchpad')

INK, MUTE, GRID, PAPER = '#1a1a19', '#6b6a66', '#e4e2dc', '#fbfaf7'
C_OFF, C_ON, C_C3 = '#2a78d6', '#eb6c15', '#4a42c6'
GAS = '#16233f'
# liquid temperature: pale -> warm (single hue ramp, light = cold)
CM_T = LinearSegmentedColormap.from_list('t', ['#f3f6fb', '#fde6c8', '#f9b877', '#ec7a3c', '#c2372c'])
# discharge power density inside the gas: dark -> violet -> white
CM_P = LinearSegmentedColormap.from_list('p', ['#16233f', '#4a2a8a', '#b04fd0', '#ffd9ff'])


def _fill_nan(a, fill):
    """Replace NaN (solid) by the nearest valid value so that smoothing works."""
    from scipy import ndimage
    bad = np.isnan(a)
    if not bad.any():
        return a
    if bad.all():
        return np.full_like(a, fill)
    idx = ndimage.distance_transform_edt(bad, return_distances=False, return_indices=True)
    return a[tuple(idx)]


def draw_view(ax, case, s, rmax, zmin, zmax, nr, title, color):
    from scipy import ndimage
    ext = [-rmax, rmax, zmin, zmax]
    T = gfsio.raster(s, 'T', rmax, zmin, zmax, nr, fill=np.nan)
    th = gfsio.raster(s, 'Th', rmax, zmin, zmax, nr, fill=np.nan)
    # discharge power density K_g E^2  [W/m3]
    kg_si = s['Kg'] * 8.8541878128e-12 / case.t0
    pd = gfsio.raster(s, kg_si * s['E']**2, rmax, zmin, zmax, nr, fill=0.0)
    T = _fill_nan(T, 1.0)
    # liquid temperature: extend the liquid values into gas / solid before smoothing
    th = _fill_nan(np.where(T > 0.5, th, np.nan), 0.0)
    sg = max(nr * case.Lb / 2**case.p['lmax'] / rmax * 0.45, 0.6)   # ~ half a cell
    Ts = ndimage.gaussian_filter(T, sg, mode='nearest')
    ths = ndimage.gaussian_filter(th, sg, mode='nearest')
    pds = ndimage.gaussian_filter(pd, sg, mode='nearest')
    mir = lambda a: np.concatenate([a[:, ::-1], a], axis=1)
    Ts, ths, pds = mir(Ts), mir(ths), mir(pds)
    nz = Ts.shape[0]
    zz = zmin + (np.arange(nz) + 0.5) * (zmax - zmin) / nz
    rr = -rmax + (np.arange(2 * nr) + 0.5) * rmax / nr
    im = ax.imshow(ths, origin='lower', extent=ext, cmap=CM_T, vmin=0, vmax=1.2,
                   interpolation='bicubic', aspect='equal', zorder=1)
    if (Ts < 0.5).any():
        ax.contourf(rr, zz, Ts, levels=[-1, 0.5], colors=[GAS], zorder=2)
        lp = np.log10(np.maximum(pds, 1.0))
        lp = np.where(Ts < 0.5, lp, np.nan)
        if np.nanmax(lp) > 7.0:
            ax.contourf(rr, zz, lp, levels=np.linspace(7.0, 11.0, 9), cmap=CM_P, extend='max', zorder=3)
        ax.contour(rr, zz, Ts, levels=[0.5], colors='white', linewidths=1.0, zorder=4)
    r, z = gfsio.needle_outline(case, zmin=zmin - 0.1)
    ax.fill(np.concatenate([-r[::-1], r]), np.concatenate([z[::-1], z]), color='#8f8d87',
            ec='#3d3c39', lw=0.8, zorder=5)
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(zmin, zmax)
    ax.set_title(title, fontsize=11.5, color=color, loc='left', pad=6, fontweight='bold')
    ax.set_xlabel('x  [mm]', fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=MUTE, length=3)
    for sp in ax.spines.values():
        sp.set_color('#3d3c39')
        sp.set_linewidth(0.8)
    return im


def nearest(snaps, t):
    ts = np.array([x[0] for x in snaps])
    i = int(np.argmin(np.abs(ts - t)))
    return snaps[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--runs', default='run_off,run_corona,run_glow')
    ap.add_argument('--labels', default='A  放电关闭（只有焦耳热）|B  弱放电（电晕型）|C  强放电（辉光型）')
    ap.add_argument('--short', default='A 放电关闭|B 弱放电|C 强放电')
    ap.add_argument('--rmax', type=float, default=0.25)
    ap.add_argument('--zmin', type=float, default=-0.3)
    ap.add_argument('--zmax', type=float, default=0.7)
    ap.add_argument('--nr', type=int, default=80)
    ap.add_argument('--dt_heat', type=float, default=1.0, help='frame spacing before nucleation, ms')
    ap.add_argument('--dt_fast', type=float, default=0.05, help='frame spacing after nucleation, ms')
    ap.add_argument('--tmax', type=float, default=None)
    ap.add_argument('--test', type=float, default=None, help='render one PNG at this time')
    a = ap.parse_args()

    names = a.runs.split(',')
    labels = a.labels.split('|')
    short = a.short.split('|')
    cols = [C_OFF, C_ON, C_C3][:len(names)]
    cases = [gfsio.Case(os.path.join(W, n)) for n in names]
    snaps = [c.snapshots() for c in cases]
    mons = [c.monitors() for c in cases]
    tend = min(s[-1][0] for s in snaps)
    if a.tmax:
        tend = min(tend, a.tmax)
    tnuc = min([m['t'][np.argmax(m['Vgas'] > 0)] if (m['Vgas'] > 0).any() else tend for m in mons])
    times = list(np.arange(0.0, tnuc - 0.3, a.dt_heat)) + list(np.arange(tnuc - 0.3, tend + 1e-9, a.dt_fast))
    if a.test is not None:
        times = [a.test]
    nv = len(names)

    frames = []
    for k, t in enumerate(times):
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=PAPER)
        gs = fig.add_gridspec(3, nv + 1, width_ratios=[1] * nv + [1.45], left=0.05, right=0.975,
                              top=0.815, bottom=0.125, wspace=0.26, hspace=0.42)
        im = None
        for j in range(nv):
            ax = fig.add_subplot(gs[:, j])
            _, fn = nearest(snaps[j], t)
            im = draw_view(ax, cases[j], cases[j].load(fn), a.rmax, a.zmin, a.zmax, a.nr, labels[j], cols[j])
            if j == 0:
                ax.set_ylabel('z  [mm]', fontsize=9, color=INK)
        cax = fig.add_axes([0.066, 0.60, 0.009, 0.15])
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([0, 0.5, 1.0])
        cb.ax.tick_params(labelsize=7, colors=INK, length=2)
        cb.ax.set_title('θ', fontsize=9, color=INK, pad=3)
        cb.outline.set_linewidth(0.5)

        fig.text(0.05, 0.945, '甲醇中针电极放电：针尖气相从哪里来？', fontsize=16, color=INK,
                 fontweight='bold', va='center')
        fig.text(0.05, 0.903, '针–环电极 · DC 2.6 kV · 甲醇 25 °C · σe = 44 µS/m · 无通气 · 轴对称 VOF（Gerris）',
                 fontsize=10, color=MUTE, va='center')
        stage = '阶段 I：焦耳加热，液体单相' if t < tnuc else '阶段 II：针尖成核之后'
        fig.text(0.975, 0.945, 't = %7.3f ms' % t, fontsize=16, color=INK, ha='right',
                 va='center', family='DejaVu Sans Mono')
        fig.text(0.975, 0.903, stage, fontsize=10, color=MUTE, ha='right', va='center')

        fast = t >= tnuc - 0.3

        def trace(ax, ylab, ylog=False):
            ax.set_facecolor(PAPER)
            ax.grid(color=GRID, lw=0.6)
            ax.tick_params(labelsize=8, colors=MUTE, length=3)
            ax.set_ylabel(ylab, fontsize=9, color=INK)
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            for sp in ('left', 'bottom'):
                ax.spines[sp].set_color(MUTE)
            ax.axvline(t, color=INK, lw=0.9, zorder=5)
            # after nucleation the traces zoom on the two-phase stage
            ax.set_xlim((tnuc - 0.5, max(tend, tnuc + 0.5)) if fast else (0, tend))
            if ylog:
                ax.set_yscale('log')
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
                ax.yaxis.set_major_formatter(FuncFormatter(
                    lambda v, _: '%g' % v if 1e-3 <= v < 1e4 else '1e%d' % int(np.round(np.log10(v)))))
                ax.minorticks_off()

        ax1 = fig.add_subplot(gs[0, nv])
        trace(ax1, '针尖液体最高 θ')
        for m, c_ in zip(mons, cols):
            ax1.plot(m['t'], m['theta_max'], color=c_, lw=1.5)
        ax1.axhline(1.0, color=INK, lw=0.8, ls='--')
        x0, x1 = ax1.get_xlim()
        ax1.text(x0 + 0.02 * (x1 - x0), 1.04, '沸点 θ = 1', fontsize=8, color=INK, va='bottom')
        ax1.set_ylim(0, 1.8)

        ax2 = fig.add_subplot(gs[1, nv])
        trace(ax2, '气相体积  [mm³]', ylog=True)
        for m, c_, sh in zip(mons, cols, short):
            ax2.plot(m['t'], np.maximum(m['Vgas'], 1e-9), color=c_, lw=1.5, label=sh)
        for m, c_ in zip(mons[1:], cols[1:]):
            ax2.plot(m['t'], np.maximum(m['Vnc'], 1e-9), color=c_, lw=1.1, ls='--')
        ax2.plot([], [], color=MUTE, lw=1.1, ls='--', label='其中不可凝气')
        vmax = max([m['Vgas'].max() for m in mons] + [1e-5])
        ax2.set_ylim(1e-7, vmax * 300)
        ax2.legend(frameon=False, fontsize=7.5, loc='upper left', labelcolor=INK, ncol=2,
                   handlelength=1.6, columnspacing=1.0, borderaxespad=0.2)

        ax3 = fig.add_subplot(gs[2, nv])
        trace(ax3, '功率  [W]', ylog=True)
        ax3.plot(mons[0]['t'], np.maximum(mons[0]['Pliq'], 1e-7), color=INK, lw=1.3, label='液相焦耳热（全针）')
        for m, c_, sh in zip(mons[1:], cols[1:], short[1:]):
            ax3.plot(m['t'], np.maximum(m['Pgas'], 1e-7), color=c_, lw=1.5, label=sh + ' 气相放电功率')
        ax3.set_ylim(1e-6, 10)
        ax3.set_xlabel('t  [ms]', fontsize=9, color=INK)
        ax3.legend(frameon=False, fontsize=7.5, loc='center left', labelcolor=INK, borderaxespad=0.2)

        fig.text(0.05, 0.03, '底色：液体无量纲温升 θ=(T−T0)/(Tsat−T0)；深蓝：气相；紫–粉：气相内放电功率密度（10⁷–10¹¹ W/m³）；白线：气液界面；灰：钨针',
                 fontsize=8, color=MUTE)
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        if a.test is not None:
            Image.fromarray(buf).save(a.out)
            print('wrote', a.out)
            return
        frames.append(Image.fromarray(buf))
        if k % 20 == 0:
            print('frame %d/%d t=%.3f' % (k, len(times), t), flush=True)

    # one shared palette built from a late frame, longer hold on the last frame
    pal = frames[-1].quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    nheat = int(np.sum(np.array(times) < tnuc - 0.3))
    dur = [100] * nheat + [90] * (len(q) - nheat)
    dur[-1] = 2500
    q[0].save(a.out, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=False)
    key = [len(q) - 1]
    frames[-1].save(a.out.replace('.gif', '_last.png'))
    print('wrote', a.out, len(q), 'frames', '%.1f MB' % (os.path.getsize(a.out) / 1e6))


if __name__ == '__main__':
    main()
