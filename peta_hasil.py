"""Peta hasil optimisasi Kopdes untuk data nyata (Surabaya) - tidak mengubah kode A.

Tambahan dibanding visualize.py:
  * sumbu lintang/bujur (bukan km relatif)
  * MAP COLORING sungguhan: wilayah layanan yang bertetangga diberi warna
    berbeda (graf ketetanggaan + pewarnaan DSATUR), jumlah warna minimal
  * area belum terjangkau berwarna abu-abu; sel tanpa penduduk hampir putih
  * lokasi rekomendasi bernomor + file CSV berisi koordinat lat/lon

Pakai:
  py peta_hasil.py --pop population.csv --cand candidates.csv
  py peta_hasil.py --pop population.csv --cand candidates.csv --k 15 --radius 1.5 --seed 3
  py peta_hasil.py --pop population.csv --cand candidates.csv --boundary surabaya.geojson
    (--boundary: GeoJSON Polygon/MultiPolygon batas kota -> garis batas + latar laut/luar kota)
Keluaran (folder --out, default results/surabaya):
  peta_hasil_sebelum_sesudah.png, peta_rekomendasi.png, lokasi_rekomendasi.csv
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse, Patch
from matplotlib.path import Path
import matplotlib.patheffects as pe

from data import load_csv
from model import Problem, Weights
from optimizers import simulated_annealing

PALETTE = ["#4e79a7", "#59a14f", "#edc948", "#b07aa1", "#f28e2b", "#76b7b2",
           "#9c755f", "#bab0ac", "#86bcb6", "#d37295"]
UNCOVERED = "#c9c9c9"
NO_POP = "#f4f4f4"


# ----------------------------------------------------------------------------
# Koordinat: km (internal) <-> lintang/bujur  (kebalikan proyeksi di load_csv)
# ----------------------------------------------------------------------------
def load_boundary(path):
    """Baca GeoJSON (FeatureCollection/Feature/geometry). Hasil: list poligon,
    tiap poligon = (ring_luar, [lubang...]) dengan koordinat (lon, lat)."""
    gj = json.load(open(path, encoding="utf-8"))
    geoms = []
    if gj.get("type") == "FeatureCollection":
        geoms = [f["geometry"] for f in gj["features"]]
    elif gj.get("type") == "Feature":
        geoms = [gj["geometry"]]
    else:
        geoms = [gj]
    polys = []
    for g in geoms:
        if g["type"] == "Polygon":
            rings = [g["coordinates"]]
        elif g["type"] == "MultiPolygon":
            rings = g["coordinates"]
        else:
            continue   # Point/LineString diabaikan
        for poly in rings:
            polys.append((np.array(poly[0], float)[:, :2],
                          [np.array(h, float)[:, :2] for h in poly[1:]]))
    if not polys:
        raise ValueError("GeoJSON tidak berisi Polygon/MultiPolygon (mungkin hanya Point?)")
    return polys


def inside_boundary(polys, lonlat):
    """True bila titik (lon, lat) berada di dalam salah satu poligon (bukan di lubang)."""
    ok = np.zeros(len(lonlat), bool)
    for outer, holes in polys:
        m = Path(outer).contains_points(lonlat)
        for h in holes:
            m &= ~Path(h).contains_points(lonlat)
        ok |= m
    return ok


class Geo:
    def __init__(self, pop_csv, cand_csv, boundary=None):
        p, c = pd.read_csv(pop_csv), pd.read_csv(cand_csv)
        self.lat0 = float(np.mean(np.r_[p["lat"], c["lat"]]))
        self.lon0 = float(np.mean(np.r_[p["lon"], c["lon"]]))
        self.cand = c
        self.kx = 111.32 * np.cos(np.radians(self.lat0))   # km per derajat bujur
        self.ky = 110.57                                    # km per derajat lintang
        self.boundary = load_boundary(boundary) if boundary else None

    def to_lonlat(self, xy):
        xy = np.atleast_2d(xy)
        return np.c_[self.lon0 + xy[:, 0] / self.kx, self.lat0 + xy[:, 1] / self.ky]


# ----------------------------------------------------------------------------
# Map coloring
# ----------------------------------------------------------------------------
def label_regions(prob, sol):
    d = prob.D[sol]                              # (k, P)
    return d.argmin(axis=0), d.min(axis=0) <= prob.R


def _grid_spacing(xy):
    idx = np.random.default_rng(0).choice(len(xy), size=min(300, len(xy)), replace=False)
    d = Problem._cdist(xy[idx], xy)
    d[d < 1e-9] = np.inf
    return float(np.median(d.min(axis=1)))


def adjacency(prob, lab, cov):
    """Dua wilayah bertetangga bila ada titik penduduk terlayani dari Kopdes
    berbeda yang berdekatan (jarak <= 1.5 x jarak antar-sel grid)."""
    k = len(np.unique(lab[cov])) if cov.any() else 0
    n = int(lab.max()) + 1
    adj = {i: set() for i in range(n)}
    pts, l = prob.ds.pop_xy[cov], lab[cov]
    thr = 1.5 * _grid_spacing(prob.ds.pop_xy)
    for s in range(0, len(pts), 800):
        d = Problem._cdist(pts[s:s + 800], pts)
        i, j = np.nonzero(d <= thr)
        i = i + s
        diff = l[i] != l[j]
        for a, b in zip(l[i][diff], l[j][diff]):
            adj[int(a)].add(int(b)); adj[int(b)].add(int(a))
    return adj


def dsatur(adj, n):
    """Pewarnaan graf greedy DSATUR -> daftar indeks warna per wilayah."""
    col, sat = {}, {v: set() for v in range(n)}
    while len(col) < n:
        v = max((u for u in range(n) if u not in col),
                key=lambda u: (len(sat[u]), len(adj[u])))
        c = 0
        while c in sat[v]:
            c += 1
        col[v] = c
        for w in adj[v]:
            sat[w].add(c)
    return [col[i] for i in range(n)]


def color_regions(prob, sol):
    lab, cov = label_regions(prob, sol)
    adj = adjacency(prob, lab, cov)
    colors = dsatur(adj, len(sol))
    assert all(colors[a] != colors[b] for a in adj for b in adj[a]), "pewarnaan tidak valid"
    return lab, cov, adj, colors


# ----------------------------------------------------------------------------
# Gambar
# ----------------------------------------------------------------------------
def _setup(ax, geo, ds_lonlat, pad=0.012):
    pts = ds_lonlat
    if geo.boundary:
        pts = np.vstack([pts] + [o for o, _ in geo.boundary])
        ax.set_facecolor("#e3edf5")          # luar kota / laut
    ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
    ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
    ax.set_aspect(1 / np.cos(np.radians(geo.lat0)))
    ax.set_xlabel("Bujur"); ax.set_ylabel("Lintang")
    ax.grid(alpha=0.2)


def _square_size(ax, cell_lon, cell_lat):
    """Ukuran marker 's' (pt^2) agar persegi pas menutup satu sel grid."""
    ax.apply_aspect()
    x0, y0 = ax.transData.transform((0, 0))
    x1, y1 = ax.transData.transform((cell_lon, cell_lat))
    side_px = max(abs(x1 - x0), abs(y1 - y0))
    side_pt = side_px * 72.0 / ax.figure.dpi
    return side_pt ** 2


def draw_solution_geo(ax, prob, sol, title, geo):
    ds = prob.ds
    ll = geo.to_lonlat(ds.pop_xy)
    cl = geo.to_lonlat(prob.xy[sol])
    lab, cov, adj, colors = color_regions(prob, sol)
    has_pop = ds.pop_w > 0

    _setup(ax, geo, ll)          # batas sumbu & aspek DULU, baru hitung ukuran marker
    lon_u, lat_u = np.unique(np.round(ll[:, 0], 6)), np.unique(np.round(ll[:, 1], 6))
    cell_lon = float(np.median(np.diff(lon_u))); cell_lat = float(np.median(np.diff(lat_u)))
    s = _square_size(ax, cell_lon, cell_lat) * 1.05

    if geo.boundary:
        for outer, holes in geo.boundary:
            ax.fill(outer[:, 0], outer[:, 1], fc="white", ec="none", zorder=0.5)
            for h in holes:
                ax.fill(h[:, 0], h[:, 1], fc="#e3edf5", ec="none", zorder=0.6)
    m = ~has_pop
    ax.scatter(ll[m, 0], ll[m, 1], s=s, c=NO_POP, marker="s", linewidths=0, zorder=1)
    m = has_pop & ~cov
    ax.scatter(ll[m, 0], ll[m, 1], s=s, c=UNCOVERED, marker="s", linewidths=0, zorder=2)
    m = has_pop & cov
    ax.scatter(ll[m, 0], ll[m, 1], s=s, marker="s", linewidths=0, zorder=3, alpha=0.9,
               c=[PALETTE[colors[i] % len(PALETTE)] for i in lab[m]])

    if geo.boundary:
        for outer, holes in geo.boundary:
            ax.plot(outer[:, 0], outer[:, 1], color="#333", lw=1.1, zorder=3.5)
    for i, c in enumerate(sol):
        ax.add_patch(Ellipse(cl[i], 2 * prob.R / geo.kx, 2 * prob.R / geo.ky, fill=False,
                             ec="k", lw=0.8, ls="--", alpha=0.55, zorder=4))
    ax.scatter(cl[:, 0], cl[:, 1], marker="*", s=230, c="gold", ec="k", zorder=5)
    for i in range(len(sol)):
        ax.annotate(str(i + 1), cl[i], xytext=(0, 11), textcoords="offset points",
                    ha="center", fontsize=9, weight="bold", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

    rep = prob.report(sol)
    ax.set_title(f"{title}\nterlayani {rep['pop_served_pct']:.1f}% | "
                 f"belum terjangkau {rep['pop_unserved']:,.0f} jiwa | skor {prob.score(sol):.4f}",
                 fontsize=10)
    return len(set(colors))


def _legend(ax, n_colors):
    h = [Patch(fc=PALETTE[i], label=f"Wilayah layanan (warna {i + 1})") for i in range(n_colors)]
    h += [Patch(fc=UNCOVERED, label="Belum terjangkau (di luar radius)"),
          Patch(fc=NO_POP, ec="#ddd", label="Sel tanpa penduduk"),
          Line2D([], [], marker="*", ls="", mfc="gold", mec="k", ms=14, label="Kopdes terpilih (bernomor)")]
    ax.legend(handles=h, loc="lower left", fontsize=8, framealpha=0.95)


def plot_before_after_geo(prob, result, path, geo):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.6))
    fig.subplots_adjust(left=0.05, right=0.99, bottom=0.08, top=0.84, wspace=0.12)
    nb = draw_solution_geo(axes[0], prob, result.init_sol, "SEBELUM optimisasi (acak)", geo)
    na = draw_solution_geo(axes[1], prob, result.best_sol, f"SESUDAH ({result.name})", geo)
    _legend(axes[1], na)
    fig.suptitle(f"Penempatan Kopdes Merah Putih — Surabaya (k={prob.k}, radius {prob.R:g} km)", fontsize=13)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return nb, na


def plot_after_geo(prob, result, path, geo):
    fig, ax = plt.subplots(figsize=(9.5, 7.4))
    fig.subplots_adjust(left=0.1, right=0.98, bottom=0.08, top=0.9)
    n = draw_solution_geo(ax, prob, result.best_sol, f"Lokasi rekomendasi ({result.name})", geo)
    _legend(ax, n)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_recommendations(prob, sol, geo, path):
    lab, cov, adj, colors = color_regions(prob, sol)
    ll = geo.to_lonlat(prob.xy[sol])
    rows = []
    for i, c in enumerate(sol):
        cid = int(prob.cand_ids[c])
        served = float(prob.ds.pop_w[(lab == i) & cov].sum())
        rows.append(dict(no=i + 1, id_kandidat=cid, lat=round(ll[i, 1], 6), lon=round(ll[i, 0], 6),
                         land_area_m2=prob.ds.cand_land[cid], econ_score=prob.ds.cand_econ[cid],
                         road_dist_km=prob.ds.cand_road[cid], penduduk_terlayani=round(served),
                         warna_peta=colors[i] + 1))
        if geo.boundary:
            rows[-1]["di_dalam_batas_kota"] = bool(inside_boundary(geo.boundary, ll[i:i + 1])[0])
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pop", required=True); p.add_argument("--cand", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--radius", type=float, default=2.0)
    p.add_argument("--min-sep", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="results/surabaya")
    p.add_argument("--boundary", help="GeoJSON batas kota (Polygon/MultiPolygon), opsional")
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)

    ds = load_csv(a.pop, a.cand)
    geo = Geo(a.pop, a.cand, a.boundary)
    if geo.boundary:
        ll = geo.to_lonlat(ds.pop_xy)
        out_pop = ~inside_boundary(geo.boundary, ll)
        print(f"Cek batas: {out_pop.sum()} dari {len(ll)} titik penduduk "
              f"({ds.pop_w[out_pop].sum():,.0f} jiwa) berada di luar poligon batas kota")
        lc = geo.to_lonlat(ds.cand_xy)
        print(f"           {(~inside_boundary(geo.boundary, lc)).sum()} dari {len(lc)} kandidat di luar poligon")
    prob = Problem(ds, k=a.k, radius=a.radius, weights=Weights(), min_sep=a.min_sep)
    res = simulated_annealing(prob, np.random.default_rng(a.seed))
    nb, na = plot_before_after_geo(prob, res, f"{a.out}/peta_hasil_sebelum_sesudah.png", geo)
    plot_after_geo(prob, res, f"{a.out}/peta_rekomendasi.png", geo)
    df = save_recommendations(prob, res.best_sol, geo, f"{a.out}/lokasi_rekomendasi.csv")
    print(f"Skor {res.init_score:.4f} -> {res.best_score:.4f} | terlayani "
          f"{prob.report(res.init_sol)['pop_served_pct']:.1f}% -> {prob.report(res.best_sol)['pop_served_pct']:.1f}%")
    print(f"Map coloring: {nb} warna (sebelum), {na} warna (sesudah) untuk {a.k} wilayah")
    print(df.to_string(index=False))
    print(f"\nTersimpan di {a.out}/")


if __name__ == "__main__":
    main()
