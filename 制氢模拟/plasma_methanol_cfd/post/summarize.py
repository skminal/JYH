#!/usr/bin/env python3
"""Summarise the variant runs (discharge off / corona-like / glow-like)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gfsio  # noqa: E402

W = os.environ.get('RUNS', '/tmp/claude-0/-home-user-JYH/c62af2be-1b9c-5e55-83fe-02b1a29689e9/scratchpad')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
os.makedirs(OUT, exist_ok=True)

res = {}
for name in sys.argv[1:] or ['run_off', 'run_corona', 'run_glow']:
    c = gfsio.Case(os.path.join(W, name))
    m = c.monitors()
    t = m['t']
    # every variant ends in a numerical blow-up (velocity runaway); keep only
    # the part before the first time the peak velocity exceeds 30 m/s
    bad = np.where((m['Umax'] > 30.0) & (t > 30.0))[0]
    t_valid = float(t[bad[0]] - 0.05) if len(bad) else float(t[-1])
    keep = t <= t_valid
    m = {k_: v_[keep] for k_, v_ in m.items()}
    t = m['t']
    has = m['Vgas'] > 1e-9
    r = dict(Kp_over_Kl=c.p.get('Kp_over_Kl'), t_valid_ms=t_valid, blew_up=bool(len(bad)))
    if has.any():
        i0 = int(np.argmax(has))
        tn = float(t[i0])
        post = t >= tn
        r.update(
            t_nuc_ms=tn,
            theta_at_nuc=float(m['theta_max'][max(i0 - 1, 0)]),
            Vgas_max_mm3=float(m['Vgas'].max()),
            t_Vgas_max_ms=float(t[int(np.argmax(m['Vgas']))]),
            Vgas_end_mm3=float(m['Vgas'][-1]),
            Vgas_min_after_mm3=float(m['Vgas'][post][5:].min()) if post.sum() > 6 else None,
            Vgas_median_mm3=float(np.median(m['Vgas'][post])),
            Vnc_end_mm3=float(m['Vnc'][-1]),
            Req_max_um=float((3 * m['Vgas'].max() / (4 * np.pi))**(1 / 3) * 1e3),
            Pgas_mean_mW=float(np.mean(m['Pgas'][post]) * 1e3),
            Pgas_max_mW=float(m['Pgas'].max() * 1e3),
            Pliq_before_W=float(m['Pliq'][max(i0 - 1, 0)]),
            Pliq_min_after_W=float(m['Pliq'][post].min()),
            gas_time_fraction=float(np.mean(m['Vgas'][post] > 1e-9)),
            # integrated gas made by each channel after nucleation, mm3
            V_made_discharge_mm3=float(np.trapezoid(m['Qd'][post], t[post]) * 1e-3),
            V_evaporated_mm3=float(np.trapezoid(np.maximum(m['Qth'][post], 0), t[post]) * 1e-3),
            V_condensed_mm3=float(-np.trapezoid(np.minimum(m['Qth'][post], 0), t[post]) * 1e-3),
            Umax_m_s=float(m['Umax'][post].max()),
        )
    res[name] = r
json.dump(res, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)
print(json.dumps(res, indent=1))
