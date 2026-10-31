"""
prepare_data.py  --  Pipeline data Surabaya (

Menghasilkan dua file yang dibaca kode Anggota A:
  population.csv : lat, lon, population
  candidates.csv : id, lat, lon, land_area_m2, econ_score, road_dist_km

Cara pakai (butuh internet, karena mengunduh dari OpenStreetMap):
  pip install osmnx geopandas pandas numpy shapely
  # opsional, kalau pakai raster WorldPop:  pip install rasterio

  # A) Penduduk dari raster WorldPop (disarankan)
  python prepare_data.py --worldpop idn_ppp_2020.tif

  # B) Tanpa raster: penduduk diperkirakan dari kepadatan bangunan OSM,
  #    lalu diskalakan ke total penduduk resmi (isi dari BPS)
  python prepare_data.py --total-pop 2900000

Lalu:  python main.py demo --pop population.csv --cand candidates.csv

CATATAN ASUMSI (tulis di laporan Bab 1 dan Bab 4):
  - Kandidat lokasi = titik grid (default 500 m) di dalam batas kota.
  - econ_score = proxy jumlah POI usaha OSM dalam radius tertentu (log, min-max 0..1).
  - land_area_m2 = SIMULASI (lognormal, seed tetap) karena data lahan nyata tidak
    tersedia terbuka. Ganti dengan data asli kalau ada.
"""
import argparse
import math
import sys

import numpy as np
import pandas as pd
import geopandas as gpd
import osmnx as ox

PLACE_DEFAULT = "Surabaya, East Java, Indonesia"
CRS_M = "EPSG:32749"   # UTM zona 49S (meter), cocok untuk Surabaya
CRS_LL = "EPSG:4326"   # lat/lon

ROAD_TYPES = ["motorway", "trunk", "primary", "secondary"]
POI_TAGS = {
    "shop": True,
    "office": True,
    "amenity": ["marketplace", "restaurant", "cafe", "fast_food",
                "food_court", "bank", "atm"],
}


# ---------------------------------------------------------------- helper umum
def union(gs):
    """Gabungkan geometri (kompatibel geopandas lama dan baru)."""
    return gs.union_all() if hasattr(gs, "union_all") else gs.unary_union


def min_dist(a_xy, b_xy, chunk=500):
    """Jarak minimum tiap titik di a ke himpunan titik b (meter)."""
    out = np.empty(len(a_xy))
    for i in range(0, len(a_xy), chunk):
        a = a_xy[i:i + chunk]
        d = np.sqrt(((a[:, None, :] - b_xy[None, :, :]) ** 2).sum(axis=2))
        out[i:i + chunk] = d.min(axis=1)
    return out


def count_within(a_xy, b_xy, radius, chunk=500):
    """Jumlah titik b dalam radius (meter) dari tiap titik a."""
    out = np.zeros(len(a_xy), dtype=int)
    if len(b_xy) == 0:
        return out
    for i in range(0, len(a_xy), chunk):
        a = a_xy[i:i + chunk]
        d2 = ((a[:, None, :] - b_xy[None, :, :]) ** 2).sum(axis=2)
        out[i:i + chunk] = (d2 <= radius ** 2).sum(axis=1)
    return out


def minmax(x):
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()
    return np.zeros_like(x) if rng == 0 else (x - x.min()) / rng


def to_xy(geoseries):
    return np.column_stack([geoseries.x.values, geoseries.y.values])


# ------------------------------------------------------------------- batas kota
def get_boundary(place):
    gdf = ox.geocode_to_gdf(place).to_crs(CRS_M)
    return union(gdf.geometry)


# --------------------------------------------------------------------- penduduk
def pop_from_worldpop(path, boundary_m, step):
    """Raster WorldPop -> titik penduduk teragregasi per blok ~step meter."""
    try:
        import rasterio
        from rasterio.mask import mask
    except ImportError:
        sys.exit("Butuh rasterio: pip install rasterio (atau jalankan tanpa --worldpop)")

    boundary_ll = gpd.GeoSeries([boundary_m], crs=CRS_M).to_crs(CRS_LL).iloc[0]
    with rasterio.open(path) as src:
        nodata = src.nodata if src.nodata is not None else -9999.0
        arr, tr = mask(src, [boundary_ll.__geo_interface__], crop=True, nodata=nodata)

    a = arr[0].astype("float64")
    a[(a == nodata) | np.isnan(a) | (a < 0)] = 0.0

    px_m = abs(tr.a) * 111_320 * math.cos(math.radians(7.27))  # ukuran piksel (m)
    f = max(1, int(round(step / px_m)))
    h, w = a.shape
    h2, w2 = (h // f) * f, (w // f) * f
    blocks = a[:h2, :w2].reshape(h2 // f, f, w2 // f, f).sum(axis=(1, 3))

    rows, cols = np.nonzero(blocks > 0)
    lon = tr.c + (cols * f + f / 2) * tr.a
    lat = tr.f + (rows * f + f / 2) * tr.e
    pts = gpd.GeoSeries(gpd.points_from_xy(lon, lat), crs=CRS_LL).to_crs(CRS_M)
    df = pd.DataFrame({"x": pts.x.values, "y": pts.y.values,
                       "population": blocks[rows, cols]})
    inside = gpd.GeoSeries(gpd.points_from_xy(df.x, df.y), crs=CRS_M).within(boundary_m)
    return df[inside.values].reset_index(drop=True)


def pop_from_buildings(place, boundary_m, step, total_pop):
    """Tanpa raster: sebar total_pop sebanding jumlah bangunan OSM per sel grid."""
    print("  mengunduh bangunan OSM (bisa beberapa menit)...")
    b = ox.features_from_place(place, tags={"building": True}).to_crs(CRS_M)
    c = b.geometry.representative_point()
    c = c[c.within(boundary_m)]
    cells = pd.DataFrame({"ix": np.floor(c.x.values / step).astype(int),
                          "iy": np.floor(c.y.values / step).astype(int)})
    cnt = cells.groupby(["ix", "iy"]).size().reset_index(name="n")
    cnt["population"] = cnt["n"] / cnt["n"].sum() * total_pop
    cnt["x"] = (cnt["ix"] + 0.5) * step
    cnt["y"] = (cnt["iy"] + 0.5) * step
    return cnt[["x", "y", "population"]]


# ---------------------------------------------------------------------- kandidat
def make_grid(boundary_m, step):
    minx, miny, maxx, maxy = boundary_m.bounds
    xs = np.arange(minx + step / 2, maxx, step)
    ys = np.arange(miny + step / 2, maxy, step)
    gx, gy = np.meshgrid(xs, ys)
    pts = gpd.GeoSeries(gpd.points_from_xy(gx.ravel(), gy.ravel()), crs=CRS_M)
    return pts[pts.within(boundary_m).values].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--place", default=PLACE_DEFAULT)
    ap.add_argument("--step", type=float, default=500, help="jarak grid kandidat (m)")
    ap.add_argument("--worldpop", help="path raster WorldPop (.tif)")
    ap.add_argument("--total-pop", type=float,
                    help="total penduduk kota (BPS). Wajib jika tanpa --worldpop; "
                         "jika dengan --worldpop, hasil raster diskalakan ke angka ini")
    ap.add_argument("--poi-radius", type=float, default=500, help="radius POI ekonomi (m)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", default=".")
    args = ap.parse_args()

    if not args.worldpop and not args.total_pop:
        sys.exit("Isi --worldpop <file.tif> ATAU --total-pop <angka BPS>.")

    print("[1/6] Batas wilayah...")
    boundary = get_boundary(args.place)

    print("[2/6] Data penduduk...")
    if args.worldpop:
        pop = pop_from_worldpop(args.worldpop, boundary, args.step)
        if args.total_pop:
            pop["population"] *= args.total_pop / pop["population"].sum()
    else:
        pop = pop_from_buildings(args.place, boundary, args.step, args.total_pop)
    pop_xy = pop[["x", "y"]].to_numpy()
    print(f"  {len(pop)} titik penduduk, total {pop['population'].sum():,.0f} jiwa")

    print("[3/6] Kandidat lokasi (grid)...")
    cand = make_grid(boundary, args.step)
    cxy = to_xy(cand)
    # buang sel tanpa penduduk di dekatnya (laut / lahan kosong luas)
    keep = min_dist(cxy, pop_xy) <= args.step
    cand, cxy = cand[keep].reset_index(drop=True), cxy[keep]
    print(f"  {len(cand)} kandidat setelah buang area tanpa penduduk")

    print("[4/6] Jarak ke jalan utama...")
    roads = ox.features_from_place(args.place, tags={"highway": ROAD_TYPES}).to_crs(CRS_M)
    roads = roads[roads.geom_type.isin(["LineString", "MultiLineString"])]
    if roads.empty:
        sys.exit("Tidak ada jalan utama ditemukan dari OSM.")
    road_dist_km = cand.distance(union(roads.geometry)).values / 1000.0

    print("[5/6] Potensi ekonomi (proxy POI)...")
    poi = ox.features_from_place(args.place, tags=POI_TAGS).to_crs(CRS_M)
    poi_xy = to_xy(poi.geometry.representative_point())
    n_poi = count_within(cxy, poi_xy, args.poi_radius)
    econ = minmax(np.log1p(n_poi))
    print(f"  {len(poi)} POI usaha ditemukan")

    print("[6/6] Luas lahan (SIMULASI, seed tetap)...")
    rng = np.random.default_rng(args.seed)
    land = rng.lognormal(mean=math.log(1500), sigma=0.8, size=len(cand))

    ll = cand.to_crs(CRS_LL)
    candidates = pd.DataFrame({
        "id": np.arange(len(cand)),
        "lat": ll.y.round(6), "lon": ll.x.round(6),
        "land_area_m2": land.round(0),
        "econ_score": econ.round(4),
        "road_dist_km": road_dist_km.round(4),
    })
    pll = gpd.GeoSeries(gpd.points_from_xy(pop["x"], pop["y"]), crs=CRS_M).to_crs(CRS_LL)
    population = pd.DataFrame({"lat": pll.y.round(6), "lon": pll.x.round(6),
                               "population": pop["population"].round(1)})

    candidates.to_csv(f"{args.out_dir}/candidates.csv", index=False)
    population.to_csv(f"{args.out_dir}/population.csv", index=False)

    ok = (candidates["land_area_m2"] >= 1000).mean() * 100
    print("\nSelesai.")
    print(f"  candidates.csv : {len(candidates)} baris ({ok:.0f}% lolos lahan >= 1.000 m2)")
    print(f"  population.csv : {len(population)} baris")
    print("  Cek: lat Surabaya ~ -7.2, lon ~ 112.7. Kalau jauh, ada yang salah.")


if __name__ == "__main__":
    main()
