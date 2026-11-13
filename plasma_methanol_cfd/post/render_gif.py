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
C_OFF, C_ON, C_NC = '#2a78d6', '#eb6c15', '#1baf7c'
GAS = '#16233f'
# liquid temperature: pale -> warm (single hue ramp, light = cold)
CM_T = LinearSegmentedColormap.from_list('t', ['#f3f6fb', '#fde6c8', '#f9b877', '#ec7a3c', '#c2372c'])
# discharge power density inside the gas: dark -> violet -> white
CM_P = LinearSegmentedColormap.from_list('p', ['#16233f', '#4a2a8a', '#b04fd0', '#ffd9ff'])


def draw_view(ax, case, s, rmax, zmin, zmax, nr, title, color):
    ext = [-rmax, rmax, zmin, zmax]
    T = gfsio.raster(s, 'T', rmax, zmin, zmax, nr, fill=np.nan)
    th = gfsio.raster(s, 'Th', rmax, zmin, zmax, nr, fill=np.nan)
    # discharge power density K_g E^2  [W/m3]
    kg_si = s['Kg'] * 8.8541878128e-12 / case.t0
    pd = gfsio.raster(s, kg_si * s['E']**2, rmax, zmin, zmax, nr, fill=0.0)
    mir = lambda a: np.concatenate([a[:, ::-1], a], axis=1)
    T, th, pd = mir(T), mir(th), mir(pd)
    liquid = np.where(T > 0.5, th, np.nan)
    im = ax.imshow(liquid, origin='lower', extent=ext, cmap=CM_T, vmin=0, vmax=1.1,
                   interpolation='nearest', aspect='equal', zorder=1)
    gas = np.where(T <= 0.5, 1.0, np.nan)
    ax.imshow(gas, origin='lower', extent=ext, cmap=LinearSegmentedColormap.from_list('g', [GAS, GAS]),
              interpolation='nearest', aspect='equal', zorder=2)
    glow = np.where((T <= 0.5) & (pd > 1e7), pd, np.nan)
    ax.imshow(glow, origin='lower', extent=ext, cmap=CM_P, norm=LogNorm(1e7, 1e11),
              interpolation='nearest', aspect='equal', zorder=3)
    Tc = np.nan_to_num(T, nan=1.0)
    if (Tc < 0.5).any():
        zz = np.linspace(zmin, zmax, T.shape[0], endpoint=False) + (zmax - zmin) / T.shape[0] / 2
        rr = np.linspace(-rmax, rmax, T.shape[1], endpoint=False) + rmax / T.shape[1]
        ax.contour(rr, zz, Tc, levels=[0.5], colors='white', linewidths=0.9, zorder=4)
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
    ap.add_argument('--on', default='run_glow')
    ap.add_argument('--off', default='run_off')
    ap.add_argument('--rmax', type=float, default=0.6)
    ap.add_argument('--zmin', type=float, default=-0.7)
    ap.add_argument('--zmax', type=float, default=1.3)
    ap.add_argument('--nr', type=int, default=150)
    ap.add_argument('--dt_heat', type=float, default=1.0, help='frame spacing before nucleation, ms')
    ap.add_argument('--dt_fast', type=float, default=0.05, help='frame spacing after nucleation, ms')
    ap.add_argument('--tmax', type=float, default=None)
    ap.add_argument('--test', type=float, default=None, help='render one PNG at this time')
    ap.add_argument('--label_on', default='放电开启')
    a = ap.parse_args()

    c_on = gfsio.Case(os.path.join(W, a.on))
    c_off = gfsio.Case(os.path.join(W, a.off))
    s_on, s_off = c_on.snapshots(), c_off.snapshots()
    m_on, m_off = c_on.monitors(), c_off.monitors()
    tend = min(s_on[-1][0], s_off[-1][0])
    if a.tmax:
        tend = min(tend, a.tmax)
    inuc = np.where((m_on['Vgas'] > 0) | (m_off['Vgas'][:len(m_on['Vgas'])] > 0)
                    if len(m_off['Vgas']) >= len(m_on['Vgas']) else (m_on['Vgas'] > 0))[0]
    tnuc = m_on['t'][inuc[0]] if len(inuc) else tend
    times = list(np.arange(0.0, tnuc - 0.3, a.dt_heat)) + list(np.arange(tnuc - 0.3, tend + 1e-9, a.dt_fast))
    if a.test is not None:
        times = [a.test]

    frames = []
    for k, t in enumerate(times):
        fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=PAPER)
        gs = fig.add_gridspec(3, 3, width_ratios=[1, 1, 1.05], left=0.055, right=0.975,
                              top=0.815, bottom=0.125, wspace=0.22, hspace=0.42)
        axA = fig.add_subplot(gs[:, 0])
        axB = fig.add_subplot(gs[:, 1])
        ta, fa = nearest(s_off, t)
        tb, fb = nearest(s_on, t)
        im = draw_view(axA, c_off, c_off.load(fa), a.rmax, a.zmin, a.zmax, a.nr,
                       'A  只有焦耳热（放电关闭）', C_OFF)
        draw_view(axB, c_on, c_on.load(fb), a.rmax, a.zmin, a.zmax, a.nr,
                  'B  焦耳热 + ' + a.label_on, C_ON)
        axA.set_ylabel('z  [mm]', fontsize=9, color=INK)
        cax = fig.add_axes([0.072, 0.16, 0.011, 0.17])
        cb = fig.colorbar(im, cax=cax)
        cb.set_ticks([0, 0.5, 1.0])
        cb.ax.tick_params(labelsize=7, colors=INK, length=2)
        cb.ax.set_title('θ', fontsize=9, color=INK, pad=3)
        cb.outline.set_linewidth(0.5)

        fig.text(0.055, 0.945, '甲醇中针电极放电：针尖气相从哪里来？', fontsize=16, color=INK,
                 fontweight='bold', va='center')
        fig.text(0.055, 0.903, '针–环电极 · DC 2.6 kV · 甲醇 25 °C · σe = 44 µS/m · 无通气 · 轴对称 VOF',
                 fontsize=10, color=MUTE, va='center')
        stage = '阶段 I：焦耳加热（单相）' if t < tnuc else '阶段 II：成核后'
        fig.text(0.975, 0.945, 't = %7.3f ms' % t, fontsize=16, color=INK, ha='right',
                 va='center', family='DejaVu Sans Mono')
        fig.text(0.975, 0.903, stage, fontsize=10, color=MUTE, ha='right', va='center')

        # ---- traces
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
            ax.set_xlim((0, tend) if t < tnuc - 0.3 else (tnuc - 0.5, max(tend, tnuc + 0.5)))
            if ylog:
                ax.set_yscale('log')
                ax.yaxis.set_major_locator(LogLocator(base=10, numticks=6))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: '%g' % v if 1e-3 <= v < 1e4 else '1e%d' % int(np.round(np.log10(v)))))
                ax.minorticks_off()

        ax1 = fig.add_subplot(gs[0, 2])
        trace(ax1, '针尖液体最高 θ')
        ax1.plot(m_off['t'], m_off['theta_max'], color=C_OFF, lw=1.6)
        ax1.plot(m_on['t'], m_on['theta_max'], color=C_ON, lw=1.6)
        ax1.axhline(1.0, color=INK, lw=0.8, ls='--')
        ax1.text(ax1.get_xlim()[0] + 0.02 * (ax1.get_xlim()[1] - ax1.get_xlim()[0]), 1.03, '沸点 θ = 1', fontsize=8, color=INK, va='bottom')
        ax1.set_ylim(0, 1.6)

        ax2 = fig.add_subplot(gs[1, 2])
        trace(ax2, '气相体积  [mm³]', ylog=True)
        ax2.plot(m_off['t'], np.maximum(m_off['Vgas'], 1e-9), color=C_OFF, lw=1.6, label='A 放电关闭')
        ax2.plot(m_on['t'], np.maximum(m_on['Vgas'], 1e-9), color=C_ON, lw=1.6, label='B 放电开启')
        ax2.plot(m_on['t'], np.maximum(m_on['Vnc'], 1e-9), color=C_NC, lw=1.3, ls='--', label='B 其中不可凝气')
        vmax = max(m_on['Vgas'].max(), m_off['Vgas'].max(), 1e-5)
        ax2.set_ylim(1e-7, vmax * 5)
        ax2.legend(frameon=False, fontsize=7.5, loc='upper left', labelcolor=INK)

        ax3 = fig.add_subplot(gs[2, 2])
        trace(ax3, '功率  [W]', ylog=True)
        ax3.plot(m_on['t'], np.maximum(m_on['Pliq'], 1e-7), color=INK, lw=1.4, label='液相焦耳热（全针）')
        ax3.plot(m_on['t'], np.maximum(m_on['Pgas'], 1e-7), color=C_ON, lw=1.6, label='B 气相放电功率')
        ax3.set_ylim(1e-5, 10)
        ax3.set_xlabel('t  [ms]', fontsize=9, color=INK)
        ax3.legend(frameon=False, fontsize=7.5, loc='lower left', labelcolor=INK)

        fig.text(0.055, 0.03, '底色：液体无量纲温升 θ=(T−T0)/(Tsat−T0)；深蓝：气相；紫–白：气相内放电功率密度；白线：气液界面；灰：钨针',
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

    # shared palette, longer hold on the last frame
    pal = frames[len(frames) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    q = [f.quantize(palette=pal, dither=Image.Dither.NONE) for f in frames]
    nheat = int(np.sum(np.array(times) < tnuc - 0.3))
    dur = [90] * nheat + [70] * (len(q) - nheat)
    dur[-1] = 2500
    q[0].save(a.out, save_all=True, append_images=q[1:], duration=dur, loop=0, optimize=False)
    print('wrote', a.out, len(q), 'frames', os.path.getsize(a.out) / 1e6, 'MB')


if __name__ == '__main__':
    main()
