#!/usr/bin/env python3
"""Illustrations and equation images for the beginner's guide (guide/fig/)."""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle, FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'fig')
os.makedirs(OUT, exist_ok=True)
FONT = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
font_manager.fontManager.addfont(FONT)
ZH = font_manager.FontProperties(fname=FONT).get_name()
plt.rcParams.update({'font.family': [ZH, 'DejaVu Sans'], 'mathtext.fontset': 'stix', 'axes.unicode_minus': False})
INK, GAS, LIQ, ACC = '#1a1a19', '#0b3d7a', '#dbe9f7', '#eb6c15'


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=200, bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)


def eq(name, s, size=22):
    fig = plt.figure(figsize=(8, 0.9))
    fig.text(0.0, 0.5, s, fontsize=size, color=INK, va='center', ha='left')
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=220, transparent=True, bbox_inches='tight', pad_inches=0.03)
    plt.close(fig)


# ---- 1. VOF: what alpha means -------------------------------------------------
def vof_cells():
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 4.0), gridspec_kw=dict(width_ratios=[1.15, 1]))
    th = np.linspace(0, 2 * np.pi, 400)
    R, cx, cy = 2.3, 4.0, 3.6
    n = 8
    for i in range(n):
        for j in range(n):
            xs = np.linspace(i, i + 1, 30); ys = np.linspace(j, j + 1, 30)
            X, Y = np.meshgrid(xs, ys)
            frac_gas = np.mean((X - cx)**2 + (Y - cy)**2 < R**2)
            alpha = 1 - frac_gas
            a.add_patch(Rectangle((i, j), 1, 1, fc=plt.cm.Blues(0.15 + 0.75 * (1 - alpha)), ec='white', lw=1.2))
            col = 'white' if alpha < 0.45 else INK
            a.text(i + 0.5, j + 0.5, '%.1f' % alpha if 0.05 < alpha < 0.95 else ('%d' % round(alpha)),
                   ha='center', va='center', fontsize=8.5, color=col)
    a.plot(cx + R * np.cos(th), cy + R * np.sin(th), color=ACC, lw=2)
    a.set_xlim(0, n); a.set_ylim(0, n); a.set_aspect('equal'); a.axis('off')
    a.set_title('每个格子只存一个数 α（液体所占比例）', fontsize=11)
    b.axis('off')
    items = [('α = 1', '格子里全是乙醇（液体）', plt.cm.Blues(0.15)),
             ('α = 0', '格子里全是氦气（气体）', plt.cm.Blues(0.9)),
             ('0 < α < 1', '界面穿过这个格子', plt.cm.Blues(0.5)),
             ('α = 0.5 等值线', '画图时当作气液界面', None)]
    for k, (t1, t2, c) in enumerate(items):
        y = 0.85 - k * 0.22
        if c is not None:
            b.add_patch(Rectangle((0.02, y - 0.07), 0.12, 0.14, fc=c, ec='#999', transform=b.transAxes))
        else:
            b.plot([0.02, 0.14], [y, y], color=ACC, lw=2.5, transform=b.transAxes)
        b.text(0.18, y + 0.03, t1, fontsize=12, fontweight='bold', transform=b.transAxes, va='center')
        b.text(0.18, y - 0.05, t2, fontsize=10.5, transform=b.transAxes, va='center', color='#444')
    save(fig, 'vof_cells.png')


# ---- 2. one time step ------------------------------------------------------------
def time_loop():
    fig, ax = plt.subplots(figsize=(9.5, 3.3))
    ax.axis('off'); ax.set_xlim(0, 10); ax.set_ylim(-0.45, 2.7)
    boxes = [(0.1, '① 已知 t 时刻\nα, U, p'), (2.05, '② 输运 α\n（界面跟着流走）'),
             (4.0, '③ 算各种力\n表面张力、浮力、\n等离子体力'), (5.95, '④ 预测速度 U*\n再解压力方程\n修正成 ∇·U=0'),
             (7.9, '⑤ 得到 t+Δt\n时刻的 α, U, p')]
    for x, t in boxes:
        ax.add_patch(FancyBboxPatch((x, 1.0), 1.85, 1.55, boxstyle='round,pad=0.05', fc='#f3f2ee', ec='#9a978f'))
        ax.text(x + 0.92, 1.78, t, ha='center', va='center', fontsize=10)
    for k in range(4):
        x = boxes[k][0] + 1.9
        ax.add_patch(FancyArrowPatch((x, 1.78), (x + 0.15, 1.78), arrowstyle='-|>', mutation_scale=14, color=INK))
    ax.add_patch(FancyArrowPatch((8.8, 0.95), (1.0, 0.95), connectionstyle='arc3,rad=-0.25', arrowstyle='-|>',
                                 mutation_scale=14, color=ACC, lw=1.5))
    ax.text(4.9, -0.35, '重复约 6 000–9 000 次（每步 Δt ≈ 10 µs），得到 0–60 ms 的全过程', ha='center', fontsize=10, color=ACC)
    save(fig, 'time_loop.png')


# ---- 3. surface tension ----------------------------------------------------------
def surface_tension():
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(11, 3.9))
    # (a) molecules
    rng = np.random.default_rng(3)
    for k in range(220):
        x, y = rng.uniform(0, 4), rng.uniform(0, 2.2)
        a.add_patch(Circle((x, y), 0.09, fc='#7fb0dd', ec='none'))
    for x, y in ((1.2, 1.0), (2.8, 2.17)):
        a.add_patch(Circle((x, y), 0.11, fc=ACC, ec='none'))
    for ang in np.linspace(0, 2 * np.pi, 8, endpoint=False):
        a.add_patch(FancyArrowPatch((1.2, 1.0), (1.2 + 0.45 * np.cos(ang), 1.0 + 0.45 * np.sin(ang)), arrowstyle='-|>', mutation_scale=8, color=INK, lw=0.8))
    for ang in np.linspace(np.pi, 2 * np.pi, 5):
        a.add_patch(FancyArrowPatch((2.8, 2.17), (2.8 + 0.45 * np.cos(ang), 2.17 + 0.45 * np.sin(ang)), arrowstyle='-|>', mutation_scale=8, color=INK, lw=0.8))
    a.axhline(2.25, color=ACC, lw=1.5)
    a.text(2.0, 2.55, '气体', ha='center', fontsize=10)
    bb = dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.9)
    a.text(1.2, 0.22, '内部分子：四面受拉，合力为零', ha='center', fontsize=9, bbox=bb)
    a.text(2.9, 1.5, '表面分子：只被往里拉', ha='center', fontsize=9, bbox=bb)
    a.set_xlim(0, 4); a.set_ylim(0, 2.9); a.set_aspect('equal'); a.axis('off')
    a.set_title('(a) 表面张力从哪来', fontsize=11)
    # (b) laplace
    th = np.linspace(0, 2 * np.pi, 200)
    b.fill(np.cos(th), np.sin(th), color=GAS, alpha=0.9)
    b.add_patch(Rectangle((-1.8, -1.8), 3.6, 3.6, fc=LIQ, ec='none', zorder=0))
    for ang in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        b.add_patch(FancyArrowPatch((1.32 * np.cos(ang), 1.32 * np.sin(ang)), (1.02 * np.cos(ang), 1.02 * np.sin(ang)), arrowstyle='-|>', mutation_scale=9, color=ACC))
    b.text(0, 0.1, '气内压力\n$p_g$ 更高', color='white', ha='center', va='center', fontsize=10)
    b.text(0, -1.6, '液体 $p_l$', ha='center', fontsize=10)
    b.text(0, 1.5, '界面像一张绷紧的膜往里压', ha='center', fontsize=9)
    b.set_xlim(-1.8, 1.8); b.set_ylim(-1.8, 1.8); b.set_aspect('equal'); b.axis('off')
    b.set_title('(b) 弯曲界面两侧有压差', fontsize=11)
    # (c) CSF
    xs = np.linspace(-1.5, 1.5, 7)
    for i in range(6):
        for j in range(6):
            x0, y0 = -1.5 + i * 0.5, -1.5 + j * 0.5
            xc, yc = x0 + 0.25, y0 + 0.25
            r = np.hypot(xc + 1.9, yc + 1.9)
            alpha = np.clip((r - 2.1) / 0.5 + 0.5, 0, 1)
            c.add_patch(Rectangle((x0, y0), 0.5, 0.5, fc=plt.cm.Blues(0.15 + 0.75 * (1 - alpha)), ec='white'))
            if 0.05 < alpha < 0.95:
                gx, gy = (xc + 1.9) / r, (yc + 1.9) / r
                c.add_patch(FancyArrowPatch((xc, yc), (xc - 0.32 * gx, yc - 0.32 * gy), arrowstyle='-|>', mutation_scale=8, color=ACC, lw=1.2))
    th = np.linspace(0, np.pi / 2, 50)
    c.plot(-1.9 + 2.1 * np.cos(th), -1.9 + 2.1 * np.sin(th), color=INK, lw=1, ls='--')
    c.text(-1.4, -1.3, '气', color='white', fontsize=11)
    c.text(1.0, 1.1, '液', color=INK, fontsize=11)
    c.set_xlim(-1.5, 1.5); c.set_ylim(-1.5, 1.5); c.set_aspect('equal'); c.axis('off')
    c.set_title('(c) CSF：把界面力摊到界面格子里', fontsize=11)
    fig.tight_layout()
    save(fig, 'surface_tension.png')


# ---- 4. the three plasma forces ----------------------------------------------------
def plasma_forces():
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    # vase-like bubble
    z = np.linspace(-1.1, 4.0, 300)
    r = 1.2 * np.exp(-((z - 0.2) / 1.25)**2) + 0.78 * np.exp(-((z - 3.1) / 0.75)**2) + 0.25
    r = np.where(z < -0.95, 0.25 + (r - 0.25) * (z + 1.1) / 0.15, r)
    ax.fill(np.r_[r, -r[::-1]], np.r_[z, z[::-1]], color=GAS, alpha=0.9)
    ax.add_patch(Rectangle((-0.25, -2.4), 0.5, 2.4, fc='#9a9893', ec=INK, lw=0.6))
    ax.plot(0, 0, marker='*', ms=16, color='#ffd200', mec=INK)
    # EHD: inward arrows in lower part
    for ang in np.linspace(-150, -30, 5) * np.pi / 180:
        x0, y0 = 1.25 * np.cos(ang), 0.2 + 1.25 * np.sin(ang)
        ax.add_patch(FancyArrowPatch((x0, y0), (0.45 * x0, 0.45 * (y0 - 0.2) + 0.2), arrowstyle='-|>', mutation_scale=12, color='#ffd200', lw=1.6))
    for ang in np.linspace(20, 160, 4) * np.pi / 180:
        x0, y0 = 1.0 * np.cos(ang), 0.2 + 1.0 * np.sin(ang)
        ax.add_patch(FancyArrowPatch((x0, y0), (0.5 * x0, 0.5 * (y0 - 0.2) + 0.2), arrowstyle='-|>', mutation_scale=12, color='#ffd200', lw=1.6))
    # top push
    for x in (-0.5, 0, 0.5):
        ax.add_patch(FancyArrowPatch((x, 3.6), (x, 4.4), arrowstyle='-|>', mutation_scale=14, color=ACC, lw=2))
    # fluctuation squiggles
    tt = np.linspace(0, 1, 60)
    for zz in (0.9, 2.2):
        rr = np.interp(zz, z, r)
        ax.plot(rr + 0.08 * np.sin(tt * 18), zz + tt * 0.5 - 0.25, color='#1baf7c', lw=1.5)
        ax.plot(-rr - 0.08 * np.sin(tt * 18), zz + tt * 0.5 - 0.25, color='#1baf7c', lw=1.5)
    ax.axhline(0, color=INK, lw=0.6, ls=':')
    ax.text(1.75, 0.0, '① EHD 拉力（黄）：把气体拉向针尖，\n    相当于把气泡“拴”在管口', fontsize=9.5, va='center')
    ax.text(1.75, 4.0, '② 顶部推力（橙）：离子风把\n    气泡顶部往上推、把气泡拉长', fontsize=9.5, va='center')
    ax.text(1.75, 2.1, '③ 随机抖动（绿）：界面上忽正\n    忽负的小压力，负极性更强', fontsize=9.5, va='center')
    ax.text(0, -2.75, '毛细管（氦气从里面流出）', ha='center', fontsize=9)
    ax.text(-2.1, 0.12, '针尖高度（管口）', fontsize=8.5, color=INK)
    ax.set_xlim(-2.2, 5.6); ax.set_ylim(-3.0, 4.8); ax.set_aspect('equal'); ax.axis('off')
    save(fig, 'plasma_forces.png')


# ---- equations, one per image ---------------------------------------------------
eq('eq_cont', r'$\nabla\cdot\mathbf{U}=0$')
eq('eq_alpha', r'$\dfrac{\partial\alpha}{\partial t}+\nabla\cdot(\alpha\mathbf{U})+\nabla\cdot\left[\alpha(1-\alpha)\mathbf{U}_r\right]=0$')
eq('eq_mix', r'$\rho=\alpha\rho_l+(1-\alpha)\rho_g,\qquad\mu=\alpha\mu_l+(1-\alpha)\mu_g$')
eq('eq_mom', r'$\dfrac{\partial(\rho\mathbf{U})}{\partial t}+\nabla\cdot(\rho\mathbf{U}\mathbf{U})'
   r'=-\nabla p_{rgh}-(\mathbf{g}\cdot\mathbf{x})\nabla\rho'
   r'+\nabla\cdot\left[\mu(\nabla\mathbf{U}+\nabla\mathbf{U}^{T})\right]'
   r'+\mathbf{f}_\sigma+\mathbf{f}_{EHD}+\mathbf{f}_{pl}$', size=20)
eq('eq_laplace', r'$p_g-p_l=\dfrac{2\sigma}{R}=\dfrac{2\times0.02214}{0.0015}\approx 29\ \mathrm{Pa}$')
eq('eq_csf', r'$\mathbf{f}_\sigma=\sigma\,\kappa\,\nabla\alpha,\qquad\kappa=-\nabla\cdot\left(\dfrac{\nabla\alpha}{|\nabla\alpha|}\right)$')
eq('eq_angle', r'$\hat{\mathbf{n}}\cdot\hat{\mathbf{n}}_{w}=\cos\theta_w,\qquad\theta_w=45^{\circ}$')
eq('eq_dt', r'$\Delta t\leq\sqrt{\dfrac{(\rho_l+\rho_g)\,\Delta x^{3}}{4\pi\sigma}}\approx1.9\times10^{-5}\ \mathrm{s}$')
eq('eq_bo', r'$Bo=\dfrac{\rho_l\,g\,D^{2}}{\sigma}=\dfrac{791\times9.81\times0.003^{2}}{0.02214}\approx3.2$')
eq('eq_ehd', r'$\mathbf{f}_{EHD}=-F_0\,S(d)\,(1-\alpha)\,\hat{\mathbf{d}},\qquad S(d)=\dfrac{1}{2}\left[1-\tanh\dfrac{d-R_E}{w_E}\right]$')
eq('eq_pl', r'$\mathbf{f}_{pl}=\left[P_{up}\max(\cos\vartheta,0)^{2}+P_{fl}\,\xi(\vartheta,t)\right]\nabla\alpha$')
eq('eq_xi', r'$\xi(\vartheta,t)=\dfrac{1}{\sqrt{N}}\sum_{m=4}^{24}\left[a_m(t)\cos m\vartheta+b_m(t)\sin m\vartheta\right]$')
eq('eq_uin', r'$U_{in}=\dfrac{Q}{\pi r_{in}^{2}}=\dfrac{5\times10^{-8}\ \mathrm{m^3/s}}{\pi\,(1.5\times10^{-4}\ \mathrm{m})^{2}}\approx0.707\ \mathrm{m/s}$')

if __name__ == '__main__':
    vof_cells(); time_loop(); surface_tension(); plasma_forces()
    print(sorted(os.listdir(OUT)))
