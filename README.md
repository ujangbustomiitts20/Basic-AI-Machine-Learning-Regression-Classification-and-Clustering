
##  README — Dasar AI & Machine Learning 

###  Deskripsi Proyek

Repositori ini berisi **contoh implementasi dasar AI & Machine Learning** menggunakan Python dan Scikit-Learn.
Tiga konsep utama yang ditampilkan:

1. **Regresi (Supervised Learning)** — Prediksi harga rumah berdasarkan fitur sintetis: luas, jumlah kamar, dan umur bangunan.
2. **Klasifikasi (Supervised Learning)** — Identifikasi jenis bunga *Iris* menggunakan *Logistic Regression*.
3. **Clustering (Unsupervised Learning)** — Pengelompokan 300 data tinggi & berat badan menggunakan algoritma *KMeans*.

Proyek ini ditujukan untuk pemula yang ingin memahami bagaimana model AI dibuat, dilatih, diuji, dan digunakan untuk prediksi data baru.

---

###  Struktur Proyek

```
ai_basics_demo/
├── tahap1_ai_ml_demo.py
├── model_regresi.pkl
├── model_iris.pkl
├── requirements.txt
└── README.md
```

---

###  Cara Menjalankan

#### Instalasi dependensi

```bash
pip install -r requirements.txt
```

####  Jalankan program utama

```bash
python tahap1_ai_ml_demo.py
```

####  Output utama:

* **Regresi:** Menampilkan nilai MAE dan R², serta grafik perbandingan harga asli vs prediksi.
* **Klasifikasi:** Akurasi model dan *confusion matrix* untuk dataset *Iris*.
* **Clustering:** Visualisasi pengelompokan 300 individu berdasarkan tinggi dan berat.

---

### Model yang Disimpan

* `model_regresi.pkl` → model Linear Regression yang sudah dilatih.
* `model_iris.pkl` → model Logistic Regression untuk klasifikasi bunga *Iris*.

Keduanya dapat dimuat kembali menggunakan `joblib.load()` untuk melakukan prediksi baru tanpa perlu melatih ulang.

---

###  Teknologi & Library

* **Python 3.10+**
* **scikit-learn**
* **pandas**
* **numpy**
* **matplotlib**
* **joblib**
* **Pillow** (untuk menampilkan gambar Iris)

---

###  Penulis

Dibuat oleh [Aolia Ikhwanudin](https://github.com/ujangbustomiitts20)
aikwanudin@gmail.com
Sebagai bagian dari seri pembelajaran *AI & Machine Learning dari Dasar hingga Implementasi*.

---
