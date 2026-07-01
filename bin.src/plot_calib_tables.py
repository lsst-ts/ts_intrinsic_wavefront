#!/usr/bin/env python3
# This file is part of ts_intrinsic_wavefront.
#
# Developed for the LSST Telescope and Site Systems.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Plot the OCS and CCS components of generated calib tables to a PDF.

Reads the products of ``run_make_calib_tables.py`` (the single
``intrinsic_aberrations_OCS.parquet`` and the per-detector
``intrinsic_aberrations_CCS_det<NNN>.parquet``) and writes a multi-page PDF:

* a summary page of the per-detector CCS Z4 height piston (from
  ``provenance.yaml``), and
* one page per Noll index with the OCS (telescope-fixed) map next to the CCS
  (camera-fixed) map for a representative detector.

The smooth camera field ``C`` is shared across detectors; only the Z4 piston
differs, so one representative detector shows the CCS field shape while the
summary page shows the per-detector piston spread.  Pure
numpy/matplotlib/astropy — no LSST stack.

Example::

    plot_calib_tables.py \
        --tables-dir /sdf/group/rubin/repo/aos_imsim/gmegias/intrinsic_zernikes/v2 \
        --out intrinsic_zernikes_v2_maps.pdf
"""
import argparse
import re
from pathlib import Path

import numpy as np
import yaml
from astropy.table import Table


def _read_table(path):
    t = Table.read(str(path), format="parquet")
    x = t["x"].to("deg").value
    y = t["y"].to("deg").value
    js = sorted(int(m.group(1)) for c in t.colnames
                for m in [re.match(r"Z(\d+)$", c)] if m)
    return x, y, js, {j: t[f"Z{j}"].to("um").value for j in js}, dict(t.meta)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tables-dir", required=True,
                    help="dir with intrinsic_aberrations_OCS.parquet + CCS_det*.parquet")
    ap.add_argument("--out", required=True, help="output PDF path")
    ap.add_argument("--detector", type=int, default=None,
                    help="detector for the CCS maps (default: median detector id present)")
    ap.add_argument("--zernikes", type=int, nargs="+", default=None,
                    help="subset of Noll indices (default: all)")
    ap.add_argument("--fp-radius-deg", type=float, default=1.75)
    ap.add_argument("--pct", type=float, default=98.0)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    d = Path(args.tables_dir)
    xo, yo, js_o, ocs, ocs_meta = _read_table(d / "intrinsic_aberrations_OCS.parquet")

    ccs_paths = {}
    for p in sorted(d.glob("intrinsic_aberrations_CCS_det*.parquet")):
        m = re.match(r"^intrinsic_aberrations_CCS_det(\d+)\.parquet$", p.name)
        if m:
            ccs_paths[int(m.group(1))] = p
    if not ccs_paths:
        raise SystemExit(f"no per-detector CCS tables in {d}")
    det = args.detector
    if det is None:
        det = sorted(ccs_paths)[len(ccs_paths) // 2]
    if det not in ccs_paths:
        raise SystemExit(f"detector {det} has no CCS table (have {sorted(ccs_paths)[:5]}...)")
    xc, yc, js_c, ccs, ccs_meta = _read_table(ccs_paths[det])

    js = sorted(set(js_o) & set(js_c))
    if args.zernikes:
        js = [j for j in js if j in args.zernikes]
    ocs_only = set(int(j) for j in (ocs_meta.get("ocs_only") or []))
    R = args.fp_radius_deg

    prov = {}
    pfile = d / "provenance.yaml"
    if pfile.exists():
        prov = yaml.safe_load(open(pfile)) or {}

    def plot_map(ax, x, y, vals, title, vlim, cmap="RdBu_r"):
        # Per-sample scatter (no hexbin binning): the OCS/CCS tables already
        # carry the values on the maps' (thx, thy) grid points, so colour each
        # sample directly rather than re-aggregating into hex cells.
        v = np.asarray(vals, float)
        fin = np.isfinite(v)
        sc = ax.scatter(x[fin], y[fin], c=v[fin], s=6,
                        cmap=cmap, vmin=-vlim, vmax=vlim,
                        marker="o", linewidths=0.0)
        ax.add_patch(plt.Circle((0, 0), R, fill=False, ec="k", lw=0.6, alpha=0.4))
        ax.set_aspect("equal")
        ax.set_xlim(-R, R)
        ax.set_ylim(-R, R)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("x [deg]")
        ax.set_ylabel("y [deg]")
        return sc

    def vlim_for(*arrays):
        vv = np.concatenate([np.asarray(a, float)[np.isfinite(a)] for a in arrays])
        return max(float(np.nanpercentile(np.abs(vv), args.pct)), 1e-6) if vv.size else 1.0

    n_pages = 0
    with PdfPages(args.out) as pdf:
        # ---- summary / piston page ----
        fig = plt.figure(figsize=(11, 7.5), layout="constrained")
        fig.suptitle("IntrinsicZernikes calib tables — overview", fontsize=13)
        ax = fig.add_subplot(111)
        pistons = (prov.get("heights", {}) or {}).get("piston_z4_um", {}) or {}
        if pistons:
            ids = sorted(int(k) for k in pistons)
            vals = [pistons[str(i)] if str(i) in pistons else pistons.get(i, 0.0) for i in ids]
            ax.bar(ids, vals, width=1.0, color="indianred")
            ax.set_xlabel("detector id")
            ax.set_ylabel("CCS Z4 height piston [µm]")
            ax.set_title(f"Per-detector Z4 height piston "
                         f"({len(ids)} detectors; representative shown below = det {det})",
                         fontsize=10)
            ax.grid(axis="y", alpha=0.3)
        else:
            ax.axis("off")
            ax.text(0.5, 0.6, "no per-detector piston info in provenance.yaml",
                    ha="center", fontsize=11)
        meta_lines = [
            f"tables_dir: {d}",
            f"version: {prov.get('version', '?')}   generated_utc: {prov.get('generated_utc', '?')}",
            f"maps_source: {prov.get('maps_source', '?')}",
            f"git: {prov.get('git_describe', '?')}",
            f"detectors: {prov.get('n_detectors', len(ccs_paths))}   "
            f"Noll: {js}",
            f"heights: {prov.get('heights', {}).get('source', '?')} "
            f"(applied={prov.get('heights', {}).get('applied', '?')})",
        ]
        fig.text(0.01, -0.02, "\n".join(meta_lines), fontsize=7, va="top", family="monospace")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        n_pages += 1

        # ---- per-Zernike OCS + CCS ----
        for j in js:
            O = ocs[j]
            C = ccs[j]
            vlim = vlim_for(O, C)
            fig, axs = plt.subplots(1, 2, figsize=(11, 4.7), layout="constrained")
            tcf = plot_map(axs[0], xo, yo, O, f"Z{j}  OCS (telescope)", vlim)
            note = "   [OCS-only: C≈0]" if j in ocs_only else ""
            plot_map(axs[1], xc, yc, C, f"Z{j}  CCS det{det} (camera){note}", vlim)
            fig.colorbar(tcf, ax=axs, shrink=0.85, label="µm")
            fig.suptitle(f"IntrinsicZernikes calib — Z{j}", fontsize=12)
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            n_pages += 1

    print(f"wrote {args.out}  ({n_pages} pages; OCS + CCS det {det}; Noll {js})")


if __name__ == "__main__":
    main()
