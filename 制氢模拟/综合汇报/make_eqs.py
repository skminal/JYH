#!/usr/bin/env python3
"""Equation images for the combined deck (methanol Gerris model + criteria)."""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fig')
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'mathtext.fontset': 'stix', 'font.family': 'DejaVu Sans'})
INK = '#1a1a19'


def eq(name, lines, size=20, lh=0.78):
    h = lh * len(lines) + 0.1
    fig = plt.figure(figsize=(10, h), dpi=220)
    for k, s in enumerate(lines):
        fig.text(0.005, 1 - (k + 0.55) * lh / h, s, fontsize=size, color=INK, va='center', ha='left')
    fig.savefig(os.path.join(OUT, name + '.png'), transparent=True, bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)


# fluid part (Gerris, VOF + CSF + Boussinesq + volume source from phase change / discharge)
eq('m_fluid', [
    r'$\nabla\cdot\mathbf{u}=S_{th}+S_{d}$',
    r'$\dfrac{\partial c}{\partial t}+\nabla\cdot(c\,\mathbf{u})=0\quad(c:\ \mathrm{liquid\ volume\ fraction})$',
    r'$\rho\left(\dfrac{\partial\mathbf{u}}{\partial t}+\mathbf{u}\cdot\nabla\mathbf{u}\right)'
    r'=-\nabla p+\nabla\cdot(2\mu\mathbf{D})+\sigma\kappa\,\mathbf{n}\,\delta_s'
    r'+\rho\,\mathbf{g}\left[1-\beta_T(T-T_0)\,c\right]$',
], lh=0.86)

# electric part: leaky dielectric / ohmic charge relaxation
eq('m_elec', [
    r'$\nabla\cdot(\varepsilon\nabla\phi)=-\rho_e,\qquad \mathbf{E}=-\nabla\phi$',
    r'$\dfrac{\partial\rho_e}{\partial t}+\nabla\cdot(\rho_e\mathbf{u})=\nabla\cdot(K\nabla\phi),'
    r'\qquad K=c\,K_l+(1-c)\,K_g$',
    r'$K_g=K_p\min\left[\left(|\mathbf{E}|/E_c\right)^{4},\,10\right],\qquad E_c=3\ \mathrm{MV/m}$',
], lh=0.86)

# energy, phase change, discharge gas
eq('m_heat', [
    r'$\rho_l c_p\left(\dfrac{\partial T}{\partial t}+\mathbf{u}\cdot\nabla T\right)'
    r'=k\nabla^{2}T+K_l|\mathbf{E}|^{2}-\rho_g h_{fg}S_{th}$',
    r'$S_{th}=\dfrac{\rho_l c_p (T_{sat}-T_0)}{\rho_g h_{fg}}\;c\,\dfrac{\theta-1}{\tau_r}'
    r'\quad(\mathrm{interface\ cells},\ \theta=\dfrac{T-T_0}{T_{sat}-T_0})$',
    r'$S_{d}=F_{gas}\,\dfrac{K_g|\mathbf{E}|^{2}}{\rho_g h_{fg}}\,(1-c),\qquad'
    r'\mathrm{CH_3OH}\rightarrow\mathrm{CO}+2\mathrm{H_2}\ \ (\eta=30\%)$',
], lh=0.92)

# inception criteria
eq('m_crit', [
    r'$\theta_\infty=\dfrac{\sigma_e V^{2}}{8k\,(T_{sat}-T_0)}\geq1$',
    r'$\Delta V(d)\geq V_b(pd)=\dfrac{B\,pd}{\ln\!\left[A\,pd/\ln(1+1/\gamma)\right]}$',
], size=20, lh=0.95)
print(sorted(os.listdir(OUT)))
