#!/usr/bin/env python3
"""Generate the Gerris case: needle electrode discharge in methanol.

Needle-ring configuration. Dimensions and voltage follow Wang et al.,
Matter 7 (2024) 3024-3035 where they can: tungsten needle with a 0.04 mm tip
radius (their needle is d = 0.08 mm), 0.5 mm shaft (their capillary OD), ring
electrode (ID 20 mm, wire d = 2 mm) 30 mm above the tip, DC 0-2.8 kV.
Differences, on purpose: the liquid is methanol and there is NO gas feed and
no glass capillary, so every bit of gas in the run has to be produced in
situ, which is the question asked.

Everything is written in units where the box edge Lb is the unit length
(this Gerris build's implicit charge relaxation is only stable for L = 1),
liquid density = 1 and surface tension = 1.

modes
  heat   single phase, electric field + Joule heating only (no nucleation)
  joule  + nucleation and evaporation/condensation, discharge switched OFF
  full   + gas discharge (field-dependent gas conductivity, gas source)
"""
import argparse
import json
import math

EPS0 = 8.8541878128e-12


def params(V=2600.0, K_l=4.4e-5, lmax=10):
    p = dict(
        # ---- geometry, mm -------------------------------------------------
        Lb=12.9,            # box edge (unit length); chosen so that no solid
                            # surface coincides with a grid line at any level
        nbox=3,
        z0=-3.17,           # bottom of domain
        r_tip=0.04,         # needle tip radius (paper: needle d = 0.08 mm)
        r_shaft=0.25,       # needle shaft radius (paper: capillary OD 0.5 mm)
        cone_deg=15.0,      # cone half angle
        z_ring=30.0,        # ring electrode height above the tip
        r_ring=11.0,        # ring centre-line radius (ID 20 mm, wire d=2 mm)
        a_ring=1.0,         # ring wire radius
        # ---- operating point ---------------------------------------------
        V=V,                # applied DC voltage, V
        T0=298.15,          # bulk liquid temperature, K
        p_amb=101325.0,
        # ---- methanol liquid at 25 C -------------------------------------
        rho_l=786.6,        # kg/m3
        mu_l=5.44e-4,       # Pa s
        sigma=0.02207,      # N/m
        eps_l=32.66,        # relative permittivity
        cp_l=2530.0,        # J/kg/K
        k_l=0.200,          # W/m/K
        Tsat=337.85,        # K (64.7 C at 1 atm)
        hfg=1.10e6,         # J/kg at Tsat
        betaT=1.2e-3,       # 1/K thermal expansion
        K_l=K_l,            # S/m electrical conductivity
        # ---- gas (methanol vapour / product gas at Tsat, 1 atm) ----------
        rho_g=1.154,        # kg/m3  (pM/RT, M = 32.04 g/mol)
        mu_g=1.1e-5,        # Pa s
        eps_g=1.0,
        D_g=2.0e-5,         # m2/s species mixing inside the bubble
        # ---- discharge closure (assumptions, see NOTES.md) ----------------
        E_c=3.0e6,          # V/m sustaining field of the gas
        Kp_over_Kl=0.1,     # plasma conductivity at E = E_c, relative to liquid
                            # (corona-like: the gas stays the main voltage drop)
        Sd_max=5000.0,      # 1/s cap on the local gas-source rate (numerical)
        nK=4.0,             # K_g = K_p (E/E_c)^nK, capped at Kcap*K_p
        Kcap=10.0,
        eta_dec=0.30,       # share of plasma power going into CH3OH -> CO + 2H2
        # ---- phase change -------------------------------------------------
        theta_nuc=1.10,     # nucleation at ~4 K superheat
        r_site=0.08,        # mm, thickness of the nucleation layer on the needle
        lmax=lmax,
    )
    return p


def derived(p):
    d = {}
    L = p['Lb'] * 1e-3
    d['L'] = L
    d['t0'] = math.sqrt(p['rho_l'] * L**3 / p['sigma'])          # s
    d['U0'] = L / d['t0']                                       # m/s
    d['p0'] = p['sigma'] / L                                    # Pa
    d['Phi0'] = math.sqrt(p['sigma'] * L / EPS0)                # V
    d['G'] = 9.81 * d['t0']**2 / L
    d['MUL'] = p['mu_l'] / (p['rho_l'] * d['U0'] * L)
    d['MUG'] = p['mu_g'] / (p['rho_l'] * d['U0'] * L)
    d['RHOG'] = p['rho_g'] / p['rho_l']
    d['VAPP'] = p['V'] / d['Phi0']
    d['KL'] = p['K_l'] * d['t0'] / EPS0
    d['KP'] = p['Kp_over_Kl'] * d['KL']
    d['EC'] = p['E_c'] * L / d['Phi0']
    dT = p['Tsat'] - p['T0']
    d['dT'] = dT
    d['ALPHAL'] = p['k_l'] / (p['rho_l'] * p['cp_l']) * d['t0'] / L**2
    d['GAM'] = d['p0'] / (p['rho_l'] * p['cp_l'] * dT)          # Joule -> theta
    d['GAMV'] = d['p0'] / (p['rho_g'] * p['hfg'])               # W/m3 -> 1/s volume
    d['BETAV'] = p['rho_l'] * p['cp_l'] * dT / (p['rho_g'] * p['hfg'])
    d['BT'] = p['betaT'] * dT
    d['DG'] = p['D_g'] * d['t0'] / L**2
    d['SDMAX'] = p['Sd_max'] * d['t0']
    # gas made per joule of plasma power: (1-eta) evaporates, eta decomposes
    # CH3OH(g) -> CO + 2H2, dH = 90.6 kJ/mol; evaporation 35.2 kJ/mol
    nc = 3.0 * p['eta_dec'] / 90.6
    vap = (1.0 - p['eta_dec']) / 35.2 - p['eta_dec'] / 90.6
    d['PHINC'] = nc / (nc + vap)                 # non-condensable mole fraction
    d['FGAS'] = (nc + vap) * 35.2                # gas yield relative to pure evaporation
    mm = 1.0 / p['Lb']
    d['XT'] = -0.5 + (0.0 - p['z0']) * mm
    d['RT'] = p['r_tip'] * mm
    d['RS'] = p['r_shaft'] * mm
    d['SB'] = math.sin(math.radians(p['cone_deg']))
    d['CB'] = math.cos(math.radians(p['cone_deg']))
    d['TB'] = math.tan(math.radians(p['cone_deg']))
    d['XR'] = d['XT'] + p['z_ring'] * mm
    d['YR'] = p['r_ring'] * mm
    d['AR'] = p['a_ring'] * mm
    d['MM'] = mm
    d['HMIN'] = 1.0 / 2**p['lmax']
    return d


TEMPLATE = r'''# Needle electrode in methanol, %(V)g V  -- mode: %(mode)s
# generated by make_case.py ; unit length = box edge = %(Lb)g mm,
# unit time = %(t0).6g s, unit potential = %(Phi0).6g V
GModule electrohydro

%(nbox)d %(nedge)d GfsElectroHydroAxi GfsBox GfsGEdge {} {
    Global {
        #define XT    (%(XT).10g)
        #define RT    (%(RT).10g)
        #define RS    (%(RS).10g)
        #define SB    (%(SB).10g)
        #define CB    (%(CB).10g)
        #define TB    (%(TB).10g)
        #define XR    (%(XR).10g)
        #define YR    (%(YR).10g)
        #define AR    (%(AR).10g)
        #define MM    (%(MM).10g)
        #define HMIN  (%(HMIN).10g)
        #define LMAX  %(lmax)d
        #define VAPP  (%(VAPP).10g)
        #define KL    (%(KL).10g)
        #define KP    (%(KP).10g)
        #define KCAP  (%(Kcap).10g)
        #define NK    (%(nK).10g)
        #define EC    (%(EC).10g)
        #define EPSL  (%(eps_l).10g)
        #define EPSG  (%(eps_g).10g)
        #define RHOG  (%(RHOG).10g)
        #define MUL   (%(MUL).10g)
        #define MUG   (%(MUG).10g)
        #define GRAV  (%(G).10g)
        #define BT    (%(BT).10g)
        #define ALPHAL (%(ALPHAL).10g)
        #define GAM   (%(GAM).10g)
        #define GAMV  (%(GAMV).10g)
        #define BETAV (%(BETAV).10g)
        #define PHINC (%(PHINC).10g)
        #define FGAS  (%(FGAS).10g)
        #define DG    (%(DG).10g)
        #define THNUC (%(theta_nuc).10g)
        #define SDMAX (%(SDMAX).10g)
        #define RSITE (%(r_site).10g*MM)

        static double c01 (double a) { return a < 0. ? 0. : a > 1. ? 1. : a; }

        /* needle: sphere-capped cone (half angle beta) on a cylindrical shaft.
           > 0 outside the needle; close to the wall it is ~ the wall distance */
        static double fneedle (double x, double y) {
            double xc = XT - RT;            /* centre of the tip sphere */
            double xt = xc + RT*SB;         /* tangent point sphere / cone */
            if (x >= xt) return sqrt ((x - xc)*(x - xc) + y*y) - RT;
            double rs = RT*CB + (xt - x)*TB;
            return y - (rs < RS ? rs : RS);
        }
        static double fring (double x, double y) {
            return sqrt ((x - XR)*(x - XR) + (y - YR)*(y - YR)) - AR;
        }
        /* > 0 in the fluid, < 0 inside the electrodes */
        static double fsolid (double x, double y) {
            return fmin (fneedle (x, y), fring (x, y));
        }
        /* electrode potential: ring grounded, needle at the applied voltage */
        static double velec (double x, double y) { return fring (x, y) < AR ? 0. : VAPP; }
        static double lvl (double x, double y) {
            double d = sqrt ((x - XT)*(x - XT) + y*y);
            double fn = fneedle (x, y);
            if (d < 0.30*MM) return LMAX;
            if (fn < 0.06*MM && x > XT - 0.6*MM) return LMAX;
            if (d < 1.0*MM) return LMAX - 1;
            if (fn < 0.15*MM) return LMAX - 1;
            if (d < 2.5*MM || fn < 0.4*MM) return 8;
            if (fring (x, y) < 1.0*AR) return 7;
            if (y < 3.*MM && x < XT + 12.*MM) return 6;
            return 5;
        }
        /* cells cut by an electrode keep their static level: refining a
           cut cell crashes this Gerris build in axisymmetric mode */
        static double lmx (double x, double y, double lm) {
            double l = lvl (x, y);
            if (fneedle (x, y) < 0.05*MM || fring (x, y) < 0.3*MM) return l;
            return lm > l ? lm : l;
        }
    }

    PhysicalParams { L = 1 }
    Time { end = %(tend).10g dtmax = %(dtmax).10g }
    ApproxProjectionParams { tolerance = 1e-4 nitermax = 100 }
    ProjectionParams { tolerance = 1e-4 nitermax = 100 }

    Refine lvl(x,y)
    Solid (fsolid(x,y))

    VariableTracerVOF T
    VariableFiltered Tf T 1
    VariableCurvature K T
    SourceTension T 1. K
    VariableVOFConcentration Hl T
    VariableTracer Rhoe
    VariableTracer Nc
    PhysicalParams { alpha = 1./(c01(Tf)*(1. - RHOG) + RHOG) }

    Init {} { T = 1. Hl = 0. Nc = 0. Kg = 0. Sth = 0. Sd = 0. Qj = 0. Nuc = 0. Th = 0. Lv = Level }
%(seed)s
%(perstep)s
    SourceViscosity (c01(Tf)*(MUL - MUG) + MUG)
    Source U (-GRAV + GRAV*BT*Th*c01(T))
%(ehd)s    SourceDiffusion Rhoe (c01(T)*KL + (1. - c01(T))*Kg) { tolerance = %(tolphi)g nitermax = %(nitphi)d nrelax = 4 erelax = 2 } Phi
    SourceDiffusion Hl ALPHAL
    Source Hl (Qj - Sth/BETAV)
    SourceDiffusion Nc (T < 0.5 ? DG : 0.)
    Source Nc (PHINC*Sd)
    Source P (Sth + Sd)

    SurfaceBc Phi Dirichlet { return velec(x,y); }
%(sink)s
    AdaptGradient { istep = 1 } { cmax = 1e-2 minlevel = lvl(x,y) maxlevel = lmx(x,y,LMAX) } T
    AdaptGradient { istep = 1 } { cmax = 0.05 minlevel = lvl(x,y) maxlevel = lmx(x,y,LMAX) } Th
    AdaptVorticity { istep = 1 } { cmax = %(cvort)g minlevel = lvl(x,y) maxlevel = lmx(x,y,LMAX - 1) }

    OutputTime { istep = %(ilog)d } log_time
    OutputTiming { istep = 200 } log_timing
    OutputDiffusionStats { istep = %(ilog)d } log_diff
    OutputBalance { istep = %(ilog)d } log_balance
    OutputProjectionStats { istep = %(ilog)d } log_proj
    OutputScalarStats { istep = %(imon)d } mon_Th { v = Th }
    OutputScalarStats { istep = %(imon)d } mon_E { v = sqrt(Ex*Ex + Ey*Ey) }
    OutputScalarStats { istep = %(imon)d } mon_Vel { v = Velocity }
    OutputScalarSum { istep = %(imon)d } mon_Vgas { v = (1. - T) }
    OutputScalarSum { istep = %(imon)d } mon_Pliq { v = c01(T)*KL*(Ex*Ex + Ey*Ey) }
    OutputScalarSum { istep = %(imon)d } mon_Pgas { v = (1. - c01(T))*Kg*(Ex*Ex + Ey*Ey) }
    OutputScalarSum { istep = %(imon)d } mon_Sth { v = Sth }
    OutputScalarSum { istep = %(imon)d } mon_Sd { v = Sd }
    OutputScalarSum { istep = %(imon)d } mon_Nc { v = Nc }
    OutputScalarSum { istep = %(imon)d } mon_Hl { v = Hl }
    OutputScalarSum { istep = %(imon)d } mon_Vol { v = 1. }
    OutputSimulation { step = %(dtout).10g } snap-%%09.6f.txt {
        format = text
        variables = T,Th,Phi,Ex,Ey,U,V,Kg,Nc,Rhoe,P,Lv
    }
    OutputSimulation { step = %(dtsave).10g } save-%%09.6f.gfs
} {
    perm = (c01(T)*(EPSL - EPSG) + EPSG)
    charge = Rhoe
}
%(boxes)s
'''

PERSTEP_HEAT = r'''    Init { istep = 1 } {
        Lv = Level
        Th = (T > 0.05 ? Hl/T : 0.)
        Qj = (T > 0.99 ? KL*(Ex*Ex + Ey*Ey)*GAM : 0.)
    }
'''

# Nucleation: liquid in the thin layer on the needle tip that exceeds the
# nucleation superheat is turned into a vapour film (smooth VOF profile).
# Phase change: mixed cells relax to saturation (theta = 1); superheat makes
# vapour, subcooling condenses the condensable (1 - Xnc) part of the gas.
PERSTEP_PC = r'''    Init { istep = 1 } {
        Lv = Level
        Th = (T > 0.05 ? Hl/T : 0.)
        Nuc = (T > 0.99 && Th > THNUC && fneedle(x,y) < RSITE ? 1. : 0.)
        Hl = (Nuc > 0.5 ? Hl*c01(0.5 + (fneedle(x,y) - 0.6*RSITE)/dL) : Hl)
        T = (Nuc > 0.5 ? c01(0.5 + (fneedle(x,y) - 0.6*RSITE)/dL) : T)
        Th = (T > 0.05 ? Hl/T : 0.)
        Qj = (T > 0.99 ? KL*(Ex*Ex + Ey*Ey)*GAM : 0.)
        Sth = (T > 0.02 && T < 0.98 ?
               (Th > 1. ? fmin (BETAV*T*(Th - 1.)*0.2/dt, 0.15/dt) :
                          fmax (BETAV*T*(Th - 1.)*0.2/dt*(1. - c01(Nc/fmax(1. - T, 0.1))),
                                -0.15*(1. - T)/dt)) : 0.)
%(discharge)s
    }
'''

DISCHARGE_ON = r'''        Kg = (T < 0.5 ? 0.7*Kg + 0.3*KP*fmin (pow (sqrt(Ex*Ex + Ey*Ey)/EC, NK), KCAP) : 0.)
        Sd = (T < 0.5 ? fmin (fmin (FGAS*Kg*(Ex*Ex + Ey*Ey)*GAMV, SDMAX), 0.15/dt) : 0.)'''
DISCHARGE_OFF = r'''        Kg = 0.
        Sd = 0.'''


def boxes(n):
    out = []
    for i in range(n):
        b = ['GfsBox {']
        if i == 0:
            b.append('    left = Boundary')
        if i == n - 1:
            b.append('    right = BoundaryOutflow')
        b.append('    top = Boundary')
        b.append('    bottom = Boundary')
        b.append('}')
        out.append('\n'.join(b))
    for i in range(n - 1):
        out.append('%d %d right' % (i + 1, i + 2))
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', default='full', choices=['heat', 'joule', 'full'])
    ap.add_argument('--V', type=float, default=2600.0)
    ap.add_argument('--Kl', type=float, default=4.4e-5)
    ap.add_argument('--lmax', type=int, default=10)
    ap.add_argument('--tend_ms', type=float, default=60.0)
    ap.add_argument('--dtout_ms', type=float, default=0.25)
    ap.add_argument('--dtmax_us', type=float, default=5.0)
    ap.add_argument('--out', default='case.gfs')
    ap.add_argument('--ehd', action='store_true',
                    help='add the Maxwell-stress body force (SourceElectric)')
    ap.add_argument('--sink', action='store_true',
                    help='needle is an isothermal heat sink (theta = 0 on its surface)')
    ap.add_argument('--tolphi', type=float, default=0.01)
    ap.add_argument('--nitphi', type=int, default=60)
    ap.add_argument('--dtsave_ms', type=float, default=2.0)
    ap.add_argument('--seed_mm', type=float, default=0.0,
                    help='radius of a pre-existing gas nucleus on the tip (0 = none)')
    a = ap.parse_args()

    p = params(V=a.V, K_l=a.Kl, lmax=a.lmax)
    d = derived(p)
    v = dict(p)
    v.update(d)
    v['mode'] = a.mode
    v['nedge'] = p['nbox'] - 1
    v['tend'] = a.tend_ms * 1e-3 / d['t0']
    v['dtmax'] = a.dtmax_us * 1e-6 / d['t0']
    v['dtout'] = a.dtout_ms * 1e-3 / d['t0']
    v['dtsave'] = a.dtsave_ms * 1e-3 / d['t0']
    v['nitphi'] = a.nitphi
    v['seed'] = ('    Init {} { T = c01(0.5 + (sqrt((x - XT)*(x - XT) + y*y) - %.10g*MM)/dL) }\n' % a.seed_mm) if a.seed_mm > 0 else ''
    v['ilog'] = 20
    v['imon'] = 10
    v['cvort'] = 5.0
    v['ehd'] = '    SourceElectric\n' if a.ehd else ''
    v['sink'] = '    SurfaceBc Hl Dirichlet 0\n' if a.sink else ''
    v['tolphi'] = a.tolphi
    if a.mode == 'heat':
        v['perstep'] = PERSTEP_HEAT
    else:
        v['perstep'] = PERSTEP_PC % dict(
            discharge=DISCHARGE_ON if a.mode == 'full' else DISCHARGE_OFF)
    v['boxes'] = boxes(p['nbox'])
    with open(a.out, 'w') as f:
        f.write(TEMPLATE % v)
    meta = dict(params=p, derived=d, mode=a.mode, ehd=a.ehd, sink=a.sink, seed_mm=a.seed_mm,
                tend_ms=a.tend_ms, dtout_ms=a.dtout_ms, dtmax_us=a.dtmax_us)
    with open(a.out.replace('.gfs', '.json'), 'w') as f:
        json.dump(meta, f, indent=1)
    print('wrote', a.out)
    for k in ('t0', 'U0', 'p0', 'Phi0', 'G', 'MUL', 'RHOG', 'VAPP', 'KL', 'KP',
              'EC', 'ALPHAL', 'GAM', 'GAMV', 'BETAV', 'PHINC', 'FGAS', 'XT', 'HMIN'):
        print('  %-7s %.6g' % (k, d[k]))


if __name__ == '__main__':
    main()
