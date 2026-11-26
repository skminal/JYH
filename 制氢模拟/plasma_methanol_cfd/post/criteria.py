#!/usr/bin/env python3
"""Inception criteria at the needle tip: Joule boiling vs gas discharge.

Uses the Gerris runs for the tip field / heating history and closed-form
relations for the criteria. Writes figures and key_numbers.json to results/.
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfsio  # noqa: E402

for f in ('/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc',):
    if os.path.exists(f):
        font_manager.fontManager.addfont(f)
        plt.rcParams['font.family'] = ['DejaVu Sans', font_manager.FontProperties(fname=f).get_name()]
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'

W = os.environ.get('RUNS', '/tmp/claude-0/-home-user-JYH/c62af2be-1b9c-5e55-83fe-02b1a29689e9/scratchpad')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)

C_INK, C_MUTE, C_GRID = '#1a1a19', '#6b6a66', '#e4e2dc'
C_A, C_B, C_C, C_D = '#2a78d6', '#eb6c15', '#1baf7c', '#4a42c6'

# ---------------------------------------------------------------- inputs
adia = gfsio.Case(os.path.join(W, 'run_off'))       # heating history, adiabatic needle
sink = gfsio.Case(os.path.join(W, 'prod_sink'))     # heating history, heat-sink needle
p = adia.p
V = p['V']
k, dT, Kl = p['k_l'], adia.dT, p['K_l']
alpha = k / (p['rho_l'] * p['cp_l'])
key = {}

ma, ms = adia.monitors(), sink.monitors()
key['E_tip_MV_m'] = float(ma['E_max'][0] / 1e6)
key['I_needle_mA'] = float(ma['Pliq'][0] / V * 1e3)
key['P_joule_W'] = float(ma['Pliq'][0])
key['kohlrausch_theta_inf'] = Kl * V**2 / (8 * k * dT)
key['sigma_crit_uS_m'] = 8 * k * dT / V**2 * 1e6
key['V_crit_at_sigma_V'] = float(np.sqrt(8 * k * dT / Kl))


def t_cross(t, y, level):
    i = np.where(y >= level)[0]
    return float(t[i[0]]) if len(i) else None


pre = ma['Vgas'] <= 0
key['t_sat_adiabatic_ms'] = t_cross(ma['t'][pre], ma['theta_max'][pre], 1.0)
key['t_nuc_adiabatic_ms'] = t_cross(ma['t'], ma['Vgas'], 1e-12)
key['theta_sink_last'] = [float(ms['t'][-1]), float(ms['theta_max'][-1])]
key['t_sat_sink_ms'] = t_cross(ms['t'], ms['theta_max'], 1.0)

# ---------------------------------------------------------------- fig 1: heating
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
a = ax[0]
a.plot(ma['t'][pre], ma['theta_max'][pre], color=C_B, lw=2, label='针体绝热（上限）')
a.plot(ms['t'], ms['theta_max'], color=C_A, lw=2, label='针体等温散热（钨针，更接近实际）')
a.axhline(1.0, color=C_INK, lw=1, ls='--')
a.text(1.0, 1.02, '沸点 Tsat = 64.7 °C', color=C_INK, fontsize=9, va='bottom')
a.set_xlabel('t  [ms]')
a.set_ylabel('针尖液体最高无量纲温升  θ = (T−T0)/(Tsat−T0)')
a.set_title('2.6 kV 下针尖液体的焦耳升温（Gerris 计算）', fontsize=11, loc='left')
a.set_xlim(0, max(ms['t'][-1], ma['t'][pre][-1]) * 1.02)
a.set_ylim(0, 1.25)
a.grid(color=C_GRID, lw=0.6)
a.legend(frameon=False, fontsize=9, loc='center right')

a = ax[1]
sig = np.logspace(-7, -3.5, 300)
for Vv, col in ((1000, C_C), (1800, C_D), (2600, C_B)):
    a.loglog(sig * 1e6, sig * Vv**2 / (8 * k * dT), color=col, lw=2,
             label='%.1f kV' % (Vv / 1e3))
a.axhline(1.0, color=C_INK, lw=1, ls='--')
for s_, lab in ((7e-6, '7 µS/m'), (4.4e-5, '44 µS/m')):
    a.axvline(s_ * 1e6, color=C_MUTE, lw=1, ls=':')
    a.text(s_ * 1e6 * 1.08, 0.012, lab, color=C_MUTE, fontsize=8.5, rotation=90, va='bottom')
a.text(0.13, 1.25, '能沸腾', fontsize=9, color=C_INK)
a.text(0.13, 0.55, '到不了沸点', fontsize=9, color=C_INK)
a.set_xlabel('甲醇电导率  σe  [µS/m]')
a.set_ylabel('稳态上限  θ∞ = σe·V²/(8kΔT)')
a.set_title('焦耳沸腾判据（导热极限，与电极形状无关）', fontsize=11, loc='left')
a.set_xlim(0.1, 300)
a.set_ylim(0.01, 100)
a.grid(color=C_GRID, lw=0.6, which='both')
a.legend(frameon=False, fontsize=9, loc='lower right', title='外加电压')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_joule.png'))
plt.close(fig)

# ---------------------------------------------------------------- fig 2: discharge
# potential along the axis above the tip, single-phase solution
snaps = adia.snapshots()
s0 = adia.load(snaps[0][1])
on = (s0['rmm'] < 0.5 * s0['hmm'] + 1e-9) & (s0['zmm'] > 0) & (s0['zmm'] < 3.0)
o = np.argsort(s0['zmm'][on])
z_ax = s0['zmm'][on][o]                       # mm
phi_ax = s0['Phi'][on][o] * adia.d['Phi0']     # V
E_ax = s0['E'][on][o]
dV = V - phi_ax                                # voltage available across a gap of size z

TORR = 760.0


def paschen(d_mm, A, B, gamma=0.01):
    """Breakdown voltage of a gap d at 1 atm; A [1/(cm Torr)], B [V/(cm Torr)]."""
    pd = TORR * d_mm / 10.0
    x = A * pd / np.log(1 + 1 / gamma)
    out = np.full_like(pd, np.nan, dtype=float)
    ok = x > 1.0
    out[ok] = B * pd[ok] / np.log(x[ok])
    return out


d = np.logspace(-2.3, 0.5, 300)
gases = [  # Raizer, Gas Discharge Physics, table of A, B
    ('H₂', 5.0, 130.0, C_C),
    ('空气', 15.0, 365.0, C_A),
    ('H₂O 蒸气', 13.0, 290.0, C_D),
]
# alcohol vapour: fit through the measured Paschen minimum (ethanol, ~450 V at
# pd ~ 0.35 Torr cm; methanol similar, Sivos et al. EPJD 74 (2020) 64)
gam = 0.01
A_alc = np.e * np.log(1 + 1 / gam) / 0.35
B_alc = 450.0 / 0.35
gases.append(('醇蒸气（由 Paschen 最小值外推）', A_alc, B_alc, C_B))

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
a = ax[0]
a.loglog(z_ax, E_ax / 1e6, color=C_INK, lw=2)
a.axhspan(100, 1000, color='#f4d9c7', lw=0)
a.text(0.014, 130, '液相直接电离所需场强量级 ≥ 100 MV/m', fontsize=9, color=C_INK)
a.set_xlabel('距针尖距离 z  [mm]')
a.set_ylabel('轴线场强 |E|  [MV/m]')
a.set_title('2.6 kV 下针尖轴线场强（单相液体）', fontsize=11, loc='left')
a.set_xlim(0.012, 3)
a.set_ylim(0.1, 1000)
a.grid(color=C_GRID, lw=0.6, which='both')
a.annotate('针尖 %.0f MV/m' % (ma['E_max'][0] / 1e6), xy=(z_ax[0], E_ax[0] / 1e6),
           xytext=(0.03, 45), fontsize=9, color=C_INK,
           arrowprops=dict(arrowstyle='-', color=C_MUTE, lw=0.8))

a = ax[1]
a.loglog(z_ax, dV, color=C_INK, lw=2.4, label='针尖处尺寸为 d 的气核上的可用电压（下限）')
dc = {}
for name, A, B, col in gases:
    vb = paschen(d, A, B)
    a.loglog(d, vb, color=col, lw=1.8, label='击穿电压  ' + name)
    vb_i = np.interp(z_ax, d, np.nan_to_num(vb, nan=1e9))
    i = np.where(dV >= vb_i)[0]
    dc[name] = float(z_ax[i[0]]) if len(i) else None
a.set_xlabel('气核尺寸 d  [mm]')
a.set_ylabel('电压  [V]')
a.set_title('气核内放电判据（Townsend / Paschen，1 atm）', fontsize=11, loc='left')
a.set_xlim(0.012, 3)
a.set_ylim(100, 2e4)
a.grid(color=C_GRID, lw=0.6, which='both')
a.legend(frameon=False, fontsize=8, loc='upper left')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig_discharge.png'))
plt.close(fig)
key['d_crit_mm'] = {k_.replace('$', ''): v for k_, v in dc.items()}
key['dV_at_0p05_0p1_0p2_mm'] = [float(np.interp(x, z_ax, dV)) for x in (0.05, 0.1, 0.2)]
key['alcohol_fit_A_B'] = [A_alc, B_alc]

# ---------------------------------------------------------------- other numbers
Ja = p['rho_l'] * p['cp_l'] * dT / (p['rho_g'] * p['hfg'])
key['Ja_subcooling'] = Ja
key['t_collapse_us'] = {('%g' % R): float(np.pi * (R * 1e-3)**2 / (4 * Ja**2 * alpha) * 1e6)
                        for R in (0.05, 0.1, 0.5)}
key['gas_mm3_per_s_per_mW'] = 1e-3 / (p['rho_g'] * p['hfg']) * 1e9
key['tau_charge_us'] = p['eps_l'] * 8.854e-12 / Kl * 1e6
key['E_maxwell_kPa_tip'] = 0.5 * p['eps_l'] * 8.854e-12 * (ma['E_max'][0])**2 / 1e3
json.dump(key, open(os.path.join(OUT, 'key_numbers.json'), 'w'), indent=1, ensure_ascii=False)
print(json.dumps(key, indent=1, ensure_ascii=False))
