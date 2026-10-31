## FP-KKA — Data riil Surabaya (bagian B: Data & Visualisasi)

Data dan peta untuk optimisasi lokasi Kopdes Merah Putih (Simulated Annealing). Kode algoritma (A) ada di repo terpisah: https://github.com/nailahikaru/final-project-kka

Isi repo
File	Isi
population.csv	Titik penduduk (grid)
candidates.csv	Kandidat lokasi Kopdes (grid)
surabaya.geojson	Batas Kota Surabaya (MultiPolygon, dari OpenStreetMap)
peta_data.py	Peta persebaran data (penduduk, kandidat, potensi ekonomi)
peta_hasil.py	Peta hasil optimisasi + map coloring + lokasi_rekomendasi.csv
results/surabaya/	Gambar dan CSV hasil dari run peta_hasil.py
Kamus data

population.csv — 1.556 titik, total 3.018.021 jiwa

Kolom	Arti	Satuan
lat	Lintang titik grid	derajat (WGS84)
lon	Bujur titik grid	derajat (WGS84)
population	Jumlah penduduk pada sel grid tsb	jiwa

Jarak antar titik sekitar 460 m. Ada 11 titik berpenduduk 0 dan 214 titik < 50 jiwa (tepi kota / area tak berpenduduk).

candidates.csv — 1.340 kandidat (grid sekitar 500 m)

Kolom	Arti	Satuan
id	Nomor kandidat (0..1339)	—
lat, lon	Koordinat kandidat	derajat (WGS84)
land_area_m2	Luas lahan tersedia	m²
econ_score	Potensi ekonomi, dinormalisasi	0..1
road_dist_km	Jarak ke jalan utama terdekat	km

Hasil cek data:

938 dari 1.340 kandidat lolos syarat lahan ≥ 1.000 m² (402 tidak lolos).
405 kandidat (30%) punya econ_score = 0 (279 di antaranya lolos syarat lahan). Artinya tidak ada aktivitas ekonomi terdeteksi, bukan data hilang.
Tidak ada nilai kosong dan tidak ada id ganda.
Semua titik penduduk dan kandidat berada di dalam poligon surabaya.geojson.
Sumber dan praproses (WAJIB DIISI sebelum Bab 4)
Penduduk: [ISI: sumber, tahun, cara membuat grid dari data kelurahan / raster]
Batas kota: OpenStreetMap relation 8225862 (Nominatim). Data © OpenStreetMap contributors, lisensi ODbL 1.0 (http://osm.org/copyright) — wajib dicantumkan di laporan.
land_area_m2: [ISI: sumber / proxy yang dipakai. Jika perkiraan, tulis jelas sebagai asumsi]
econ_score: [ISI: sumber / proxy (mis. jumlah POI/UMKM/pasar) dan cara normalisasi]
road_dist_km: [ISI: sumber jaringan jalan dan cara menghitung jarak]
Cara menjalankan

peta_hasil.py memakai modul A (data.py, model.py, optimizers.py)

pip install numpy matplotlib pandas

python peta_data.py population.csv candidates.csv
python peta_hasil.py --pop population.csv --cand candidates.csv --boundary surabaya.geojson
python peta_hasil.py --pop population.csv --cand candidates.csv --boundary surabaya.geojson --k 15 --radius 1.5 --seed 3

Keluaran peta_hasil.py (folder results/surabaya/): peta_hasil_sebelum_sesudah.png, peta_rekomendasi.png, lokasi_rekomendasi.csv.

Catatan
Koordinat diproyeksikan ke km (ekuirektangular) oleh load_csv() milik A; peta mengembalikannya ke lintang/bujur.
Hasil Simulated Annealing bersifat acak; angka bisa sedikit berbeda antar komputer walau seed sama.  rata-rata beberapa seed (python main.py experiments --seeds 5 --pop population.csv --cand candidates.csv).
