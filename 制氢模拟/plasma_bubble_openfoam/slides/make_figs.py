#!/usr/bin/env python3
"""Figures for the OpenFOAM reproduction deck: equation images, video frames,
geometry/mesh sketch and simulation key frames.  Output: slides/fig/."""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'fig')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(HERE, '..', 'post'))
import render_gif as rg  # noqa: E402

SCR = '/tmp/claude-0/-home-user-JYH/c62af2be-1b9c-5e55-83fe-02b1a29689e9/scratchpad'
FONT = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
font_manager.fontManager.addfont(FONT)
ZH = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({'font.family': ['DejaVu Sans', ZH], 'mathtext.fontset': 'stix',
                     'axes.unicode_minus': False})
INK = '#1a1a19'


# --------------------------------------------------------------- equations
def eq(name, lines, size=21, w=10.0, lh=0.62):
    """Render a list of mathtext lines (left aligned) to a transparent PNG."""
    h = lh * len(lines) + 0.15
    fig = plt.figure(figsize=(w, h), dpi=220)
    for k, s in enumerate(lines):
        fig.text(0.005, 1 - (k + 0.55) * lh / h, s, fontsize=size, color=INK, va='center', ha='left')
    fig.savefig(os.path.join(OUT, name + '.png'), transparent=True, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)


eq('eq_vof', [
    r'$\nabla\cdot\mathbf{U}=0$',
    r'$\dfrac{\partial\alpha}{\partial t}+\nabla\cdot(\alpha\mathbf{U})'
    r'+\nabla\cdot\left[\alpha(1-\alpha)\,\mathbf{U}_r\right]=0,\quad'
    r'\mathbf{U}_r=c_\alpha|\mathbf{U}|\,\dfrac{\nabla\alpha}{|\nabla\alpha|}$',
    r'$\rho=\alpha\rho_l+(1-\alpha)\rho_g,\qquad \mu=\alpha\mu_l+(1-\alpha)\mu_g$',
], lh=0.78)

eq('eq_mom', [
    r'$\dfrac{\partial(\rho\mathbf{U})}{\partial t}+\nabla\cdot(\rho\mathbf{U}\mathbf{U})'
    r'=-\nabla p_{rgh}-(\mathbf{g}\cdot\mathbf{x})\nabla\rho'
    r'+\nabla\cdot\left[\mu\left(\nabla\mathbf{U}+\nabla\mathbf{U}^{T}\right)\right]'
    r'+\mathbf{f}_\sigma+\mathbf{f}_{EHD}+\mathbf{f}_{pl}$',
    r'$p_{rgh}=p-\rho\,\mathbf{g}\cdot\mathbf{x}$',
], lh=0.8)

eq('eq_sigma', [
    r'$\mathbf{f}_\sigma=\sigma\,\kappa\,\nabla\alpha,\qquad'
    r'\kappa=-\nabla\cdot\hat{\mathbf{n}},\qquad \hat{\mathbf{n}}=\dfrac{\nabla\alpha}{|\nabla\alpha|+\delta_N}$',
    r'$[p]_{g}-[p]_{l}=\sigma\kappa\ \ \Rightarrow\ \ \Delta p=\dfrac{2\sigma}{R}'
    r'\approx 29\ \mathrm{Pa}\ \ (R=1.5\ \mathrm{mm})$',
    r'$\hat{\mathbf{n}}\cdot\hat{\mathbf{n}}_w=\cos\theta_w\ \ \mathrm{(capillary\ wall)},\qquad \theta_w=45^\circ$',
    r'$\Delta t\leq\sqrt{\dfrac{(\rho_l+\rho_g)\,\Delta x^{3}}{4\pi\sigma}}\approx 1.9\times10^{-5}\ \mathrm{s}\ \ (\Delta x=50\ \mu\mathrm{m})$',
], lh=0.86)

eq('eq_face', [
    r'$\phi^{*}=\left(\mathbf{H}/a\right)_f\cdot\mathbf{S}_f'
    r'+\left(\dfrac{1}{a}\right)_f\left[\sigma\kappa_f\,(\nabla\alpha)_f^{\perp}'
    r'-(\mathbf{g}\cdot\mathbf{x})_f(\nabla\rho)_f^{\perp}+f_{EHD,f}+f_{pl,f}\right]|\mathbf{S}_f|$',
    r'$\nabla\cdot\left[\left(\dfrac{1}{a}\right)_f\nabla p_{rgh}\right]=\nabla\cdot\phi^{*}$',
], size=19, lh=0.82)

eq('eq_ehd', [
    r'$\mathbf{f}_{EHD}=-F_0\,S(d)\,\tanh\!\left(\dfrac{d}{d_0}\right)(1-\alpha)\,\hat{\mathbf{d}}\,\chi_c,\qquad'
    r'S(d)=\dfrac{1}{2}\left[1-\tanh\dfrac{d-R_E}{w_E}\right]$',
    r'$\mathbf{f}_{pl}=\left[P_{up}\max(\cos\vartheta,0)^{2}+P_{fl}\,\xi(\vartheta,t)\right]\nabla\alpha\;\chi_c$',
    r'$\xi=\dfrac{1}{\sqrt{N}}\sum_{m=4}^{24}\left(a_m\cos m\vartheta+b_m\sin m\vartheta\right),\qquad'
    r'\mathrm{d}a_m=-\dfrac{a_m}{\tau}\mathrm{d}t+\sqrt{\dfrac{2}{\tau}}\,\mathrm{d}W$',
], size=19, lh=0.88)

eq('eq_balance', [
    r'$F_{EHD}\approx F_\rho+F_\sigma,\qquad F_\rho=\rho_l g V,\qquad F_\sigma\sim\pi D_n\sigma$',
    r'$\kappa_E=F_0/(\rho_l g)=40,\qquad Bo=\rho_l g D^{2}/\sigma\approx 3.2\ \ (D=3\ \mathrm{mm})$',
], size=19, lh=0.8)


# --------------------------------------------------------------- video frames
def video_strip():
    import glob
    fs = sorted(glob.glob(os.path.join(SCR, 'vid/full/f_*.png')))
    PX, Y0, XC = 148.0, 782.6, 547
    picks = [(40, 'remnant'), (300, 'growth'), (500, 'necking'), (553, 'pinch-off')]
    fig, ax = plt.subplots(2, 4, figsize=(8.0, 4.7), dpi=200)
    for row, (c0, lab) in enumerate(((0, '+2.0 kV'), (1078, '−2.0 kV'))):
        for k, (i, nm) in enumerate(picks):
            a = np.asarray(Image.open(fs[i]).convert('L'))[:, c0:c0 + 1078]
            ztop = Y0 / PX                      # the frame starts 5.29 mm above the orifice
            crop = a[0:int(Y0 + 1.8 * PX), int(XC - 2.6 * PX):int(XC + 2.6 * PX)]
            ext = [-2.6, 2.6, -1.8, ztop]
            ax[row, k].imshow(crop, cmap='gray', extent=ext, vmin=0, vmax=255)
            ax[row, k].axhline(0, color='#d9541a', lw=0.8, ls='--')
            ax[row, k].set_xticks([-2, 0, 2]); ax[row, k].set_yticks([0, 2, 4])
            ax[row, k].tick_params(labelsize=7, length=2)
            if row == 0:
                ax[row, k].set_title('%s\n%.1f ms' % (nm, (i - 38) * 0.1), fontsize=8.5)
            if k == 0:
                ax[row, k].set_ylabel('%s\nz [mm]' % lab, fontsize=9)
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, 'video_frames.png'))
    plt.close(fig)


# --------------------------------------------------------------- geometry & mesh
def geometry():
    fig, (ax, az) = plt.subplots(1, 2, figsize=(7.6, 5.4), dpi=200, gridspec_kw=dict(width_ratios=[1, 1.6]))
    # full domain
    ax.add_patch(Rectangle((0, -3), 5, 23, fc='#f3f6fb', ec=INK, lw=1.0))
    ax.add_patch(Rectangle((0, -2), 2.5, 9, fc='#dbe7f6', ec='#3a7fc1', lw=0.8, ls='--'))
    ax.add_patch(Rectangle((0.15, -3), 0.10, 3, fc='#8f8d87', ec=INK, lw=0.5))
    th = np.linspace(-np.pi / 2, np.pi / 2, 80)
    ax.fill(1.54 * np.cos(th), 0.1 + 1.54 * np.sin(th), color='#0b3d7a', alpha=0.85)
    ax.plot([0, 0], [-3, 20], color='#b00000', lw=1.4, ls='-.')
    ax.text(2.5, 20.5, 'top: p_rgh total = 0', fontsize=7.5, ha='center')
    ax.text(5.25, 9, 'outer: slip', fontsize=7.5, rotation=90, va='center')
    ax.text(2.5, -3.9, 'bottom: wall', fontsize=7.5, ha='center')
    ax.text(1.25, 5.6, '50 µm', fontsize=8, ha='center', color='#1f4e8c')
    ax.text(2.6, 13.5, 'graded\nto ~0.2 mm', fontsize=7.5, ha='center', color='#6b6a66')
    ax.text(0.2, 18.5, 'axis\n(wedge 5°)', fontsize=7.5, color='#b00000')
    ax.add_patch(Rectangle((0, -3), 2.6, 6.0, fc='none', ec='#eb6c15', lw=1.0))
    ax.set_xlim(-0.3, 5.8); ax.set_ylim(-4.4, 21.2)
    ax.set_aspect('equal')
    ax.set_xlabel('r [mm]', fontsize=9); ax.set_ylabel('z [mm]', fontsize=9)
    ax.tick_params(labelsize=7.5)
    # zoom on the orifice with every 2nd grid line
    for r in np.arange(0, 2.61, 0.1):
        az.plot([r, r], [-3, 3], color='#c8d3e2', lw=0.3, zorder=0)
    for z in np.arange(-3, 3.01, 0.1):
        az.plot([0, 2.6], [z, z], color='#c8d3e2', lw=0.3, zorder=0)
    az.fill(1.54 * np.cos(th), 0.1 + 1.54 * np.sin(th), color='#0b3d7a', alpha=0.80, zorder=1)
    az.add_patch(Rectangle((0.15, -3), 0.10, 3, fc='#8f8d87', ec=INK, lw=0.6, zorder=2))
    az.add_patch(Rectangle((0, -3), 0.15, 0.12, fc='#eb6c15', ec='none', zorder=2))
    az.plot([0], [0], marker='*', color='#ffd200', ms=11, mec=INK, mew=0.5, zorder=3)
    kw = dict(fontsize=8, color=INK, zorder=4)
    az.annotate('He inlet, U = 0.707 m/s\n(3 mL/min)', xy=(0.08, -2.94), xytext=(0.7, -2.6),
                arrowprops=dict(arrowstyle='->', lw=0.6), **kw)
    az.annotate('silica capillary\nID 0.3 / OD 0.5 mm\nnoSlip, θw = 45°', xy=(0.25, -1.6), xytext=(0.7, -1.7),
                arrowprops=dict(arrowstyle='->', lw=0.6), **kw)
    az.annotate('discharge point x0\n(needle tip, z = 0)', xy=(0.02, 0.02), xytext=(1.7, 1.9),
                arrowprops=dict(arrowstyle='->', lw=0.6), **kw)
    az.text(0.55, 0.6, 'initial\nremnant\nR = 1.54 mm', fontsize=8, color='white', zorder=4)
    az.axhline(0, color='#d9541a', lw=0.8, ls='--', zorder=3)
    az.set_xlim(0, 2.6); az.set_ylim(-3, 3)
    az.set_aspect('equal')
    az.set_xlabel('r [mm]  (grid lines every 2 cells)', fontsize=9)
    az.tick_params(labelsize=7.5)
    for sp in az.spines.values():
        sp.set_color('#eb6c15')
    fig.tight_layout(pad=0.4)
    fig.savefig(os.path.join(OUT, 'geometry.png'))
    plt.close(fig)


# --------------------------------------------------------------- simulation key frames
def key_frames():
    runs = [('fin40_pos', '+2.0 kV'), ('fin40_neg8', '−2.0 kV'), ('pc50_off', '0 kV')]
    times = [0.5, 8.0, 'pre', 'post', 30.0]
    meta = {r: json.load(open(os.path.join(HERE, '..', 'runs', r, 'metrics.json'))) for r, _ in runs}
    fig, ax = plt.subplots(3, 5, figsize=(8.4, 6.4), dpi=200)
    for row, (r, lab) in enumerate(runs):
        fld = rg.Field(os.path.join(SCR, r))
        m = meta[r]
        k1 = next(i for i, x in enumerate(m) if x['n_free'] > 0)
        tpre, tpost = m[k1 - 1]['t_ms'], m[k1]['t_ms'] + 0.5
        for col, t in enumerate(times):
            tt = {'pre': tpre, 'post': tpost}.get(t, t)
            tsh, g = fld.at(tt)
            a = ax[row, col]
            gas = np.clip(1 - g, 0, 1)
            a.imshow(np.ma.masked_less(gas, 0.02), origin='lower', extent=[-5, 5, -3, 20], cmap=rg.GAS,
                     vmin=0, vmax=1, interpolation='bilinear', aspect='equal')
            rr = np.linspace(-5 + rg.DR / 2, 5 - rg.DR / 2, g.shape[1])
            zz = np.linspace(-3 + rg.DR / 2, 20 - rg.DR / 2, g.shape[0])
            a.contour(rr, zz, g, levels=[0.5], colors='#0b2a55', linewidths=0.6)
            for x0 in (0.15, -0.25):
                a.add_patch(Rectangle((x0, -3), 0.1, 3, fc='#c9ccd1', ec='#7d828a', lw=0.3))
            a.axhline(0, color='#d9541a', lw=0.6, ls='--')
            a.set_xlim(-2.8, 2.8); a.set_ylim(-2.5, 9.5)
            a.set_xticks([-2, 0, 2]); a.set_yticks([0, 4, 8])
            a.tick_params(labelsize=6.5, length=2)
            a.set_title('%.1f ms' % tsh, fontsize=8)
            if col == 0:
                a.set_ylabel('%s\nz [mm]' % lab, fontsize=9)
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, 'key_frames.png'))
    plt.close(fig)


if __name__ == '__main__':
    if '--eq-only' not in sys.argv:
        video_strip()
        geometry()
        if '--no-frames' not in sys.argv:
            key_frames()
    print(sorted(os.listdir(OUT)))
