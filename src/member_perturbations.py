#!/usr/bin/env python3
"""
Write CSV tables of what makes each ensemble member different, using the same rules as
fcst.py (WeatherForecaster.__init__ / make_noise / autoregressive_rollout).

Members all start from the same HRRR analysis. They differ by:
  1. Diffusion noise: the sampler's starting noise is seeded by the member number.
     hour 0 anchor = N(0,1) with seed [member, 0]; each hour
     noise(h) = rho * noise(h-1) + sqrt(1 - rho^2) * N(0,1)[seed = (member, h)]
     (all predicted variables/levels, every grid point; same for a member ID in any run size)
  2. GFS boundary time shift: the GFS forcing read for hour h is taken from hour
     h + round((from_hour // 12) * phase_angle[member]), where phase angles are
     0, (0), +s, -s, +2s, -2s, ... with s = 1/half_count (depends on --num_members).
  3. Autoregressive restarts: hours 1-6 are predicted from hour 0, 7-12 from that member's own
     hour 6, ... so differences carry forward from each 6-h restart.

Outputs (in --out_dir):
  member_perturbations_N<n>_L<lead>.csv          one row per member and forecast hour
  member_perturbations_N<n>_L<lead>_summary.csv  one row per member, GFS shift per 6-h block

Usage:
  python src/member_perturbations.py --num_members 9 --lead_hours 48
"""

import argparse
import csv
import math
import os


def phase_angles(num_members: int) -> dict:
    """Same construction as fcst.py autoregressive_rollout."""
    half_count = (num_members // 2 - ((num_members + 1) % 2))
    step = 1.0 / half_count if half_count > 0 else 0.0
    seq = [0.0]
    if num_members % 2 == 0:
        seq.append(0.0)
    for i in range(half_count):
        seq.append(step * (i + 1))
        seq.append(-step * (i + 1))
    return {m: seq[m] for m in range(num_members)}


def rows_for(num_members: int, lead_hours: int, n_forcing: int, rho: float, rollout_hour: int = 6):
    angles = phase_angles(num_members)
    innov = math.sqrt(1.0 - rho ** 2)
    for m in range(num_members):
        for h in range(1, lead_hours + 1):
            from_hour = ((h - 1) // rollout_hour) * rollout_hour
            phase_width = from_hour // 12
            shift = round(phase_width * angles[m])            # Python round(), as in fcst.py
            idx = h - 1 + shift
            idx_c = min(max(idx, 0), n_forcing - 1)
            yield {
                "member": f"m{m:02d}",
                "forecast_hour": h,
                "predicted_from_hour": from_hour,
                "step_h": h - from_hour,
                "initial_state": "same HRRR analysis (all members)" if from_hour == 0
                                 else f"own f{from_hour:02d} forecast",
                "noise_anchor_seed": f"[{m}, 0]",
                "noise_innovation_seed": f"[{m}, {h}]",
                "noise_rho": rho,
                "noise_weight_previous": rho,
                "noise_weight_new": round(innov, 4),
                "gfs_phase_angle": round(angles[m], 4),
                "gfs_phase_width": phase_width,
                "gfs_shift_h": idx_c - (h - 1),
                "gfs_forcing_valid_hour": idx_c + 1,       # forcing index 0 = f01
                "gfs_shift_clipped": idx_c != idx,
            }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num_members", type=int, required=True, help="ensemble size used in fcst.py (--num_members)")
    ap.add_argument("--lead_hours", type=int, required=True)
    ap.add_argument("--n_forcing", type=int, default=None,
                    help="number of GFS forcing hours in the gfs_*.npz (default: lead_hours); shifts past it are clipped")
    ap.add_argument("--noise_rho", type=float, default=0.9, help="fcst.py --noise_rho (default 0.9)")
    ap.add_argument("--out_dir", default=".")
    a = ap.parse_args()
    n_forcing = a.n_forcing or a.lead_hours
    os.makedirs(a.out_dir, exist_ok=True)
    base = os.path.join(a.out_dir, f"member_perturbations_N{a.num_members}_L{a.lead_hours}")

    rows = list(rows_for(a.num_members, a.lead_hours, n_forcing, a.noise_rho))
    with open(base + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    # summary: one row per member, the GFS shift in each 6-h block
    blocks = [(b + 1, min(b + 6, a.lead_hours)) for b in range(0, a.lead_hours, 6)]
    angles = phase_angles(a.num_members)
    with open(base + "_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["member", "noise_seeds", "noise_rho", "gfs_phase_angle"]
                   + [f"gfs_shift_h_f{b0:02d}-f{b1:02d}" for b0, b1 in blocks])
        for m in range(a.num_members):
            shifts = []
            for b0, b1 in blocks:
                s = {r["gfs_shift_h"] for r in rows if r["member"] == f"m{m:02d}" and b0 <= r["forecast_hour"] <= b1}
                shifts.append("/".join(f"{x:+d}" for x in sorted(s)))
            w.writerow([f"m{m:02d}", f"[{m},0] then [{m},hour]", a.noise_rho, round(angles[m], 4)] + shifts)
    print(f"Wrote {base}.csv ({len(rows)} rows) and {base}_summary.csv")


if __name__ == "__main__":
    main()
