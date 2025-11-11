# ===========================================
# TAHAP 1: DASAR AI & MACHINE LEARNING (DEMO)
# - Regresi (LinearRegression)
# - Klasifikasi (LogisticRegression pada Iris)
# - Clustering (KMeans pada data sintetis)
# ===========================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
   r2_score, mean_absolute_error,
   accuracy_score, classification_report, confusion_matrix
)
from sklearn.datasets import load_iris, make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
import joblib


np.random.seed(42)


# ---------------------------------------------------
# A) REGRESI — Prediksi harga rumah (data sintetis)
# ---------------------------------------------------
n = 160
luas = np.random.uniform(30, 150, size=n)        # m2
kamar = np.random.randint(1, 6, size=n)          # 1..5
umur = np.random.randint(0, 31, size=n)          # tahun (fitur baru)
# Rumus harga sintetis (ada noise):
#   1.6jt * luas + 18jt * kamar - 3jt * umur + noise
harga = 1.6e6*luas + 18e6*kamar - 3e6*umur + np.random.normal(0, 12e6, size=n)


df_reg = pd.DataFrame({"luas_m2": luas, "kamar": kamar, "umur_th": umur, "harga": harga})
X_train, X_test, y_train, y_test = train_test_split(
   df_reg[["luas_m2", "kamar", "umur_th"]], df_reg["harga"], test_size=0.2, random_state=42
)


reg_model = Pipeline([
   ("scale", StandardScaler()),
   ("linreg", LinearRegression())
])
reg_model.fit(X_train, y_train)
y_pred = reg_model.predict(X_test)


print("=== REGRESI: Prediksi Harga Rumah ===")
print("MAE  :", round(mean_absolute_error(y_test, y_pred), 2))
print("R^2  :", round(r2_score(y_test, y_pred), 4))
print("Contoh 5 baris prediksi:")
print(pd.DataFrame({
   "luas_m2": np.round(X_test["luas_m2"].values[:5], 1),
   "kamar": X_test["kamar"].values[:5],
   "umur_th": X_test["umur_th"].values[:5],
   "harga_asli": np.round(y_test.values[:5]),
   "harga_pred": np.round(y_pred[:5])
}))
print()


# Plot 1: y_asli vs y_pred (semakin dekat garis y=x semakin baik)
plt.figure()
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()])
plt.title("Regresi: Harga Asli vs Prediksi")
plt.xlabel("Harga Asli")
plt.ylabel("Harga Prediksi")
plt.show()


# Simpan model regresi
joblib.dump(reg_model, "model_regresi.pkl")


# ================================================
# UJI MODEL REGRESI YANG SUDAH DILATIH
# ================================================


# Pastikan reg_model dari tahap sebelumnya SUDAH ADA
# Misal setelah menjalankan kode tahap 1:
# reg_model.fit(X_train, y_train)


import pandas as pd


# 1️⃣ Data uji baru (input manual)
data_uji = pd.DataFrame({
   "luas_m2": [40, 75, 100, 135],   # contoh luas rumah
   "kamar": [2, 3, 4, 5]            # contoh jumlah kamar
})


# 2️⃣ Gunakan model untuk prediksi harga
prediksi_harga = reg_model.predict(data_uji)


# 3️⃣ Tampilkan hasil dalam format tabel rapi
hasil = data_uji.copy()
hasil["harga_prediksi (Rp)"] = prediksi_harga.round(0)
print("=== HASIL PREDIKSI DARI MODEL REGRESI ===")
print(hasil)


# 4️⃣ (Opsional) Format tampilan uang agar lebih mudah dibaca
print("\n=== PREDIKSI DALAM FORMAT RUPIAH ===")
for i, row in hasil.iterrows():
   print(f"Luas: {row['luas_m2']} m², Kamar: {row['kamar']} → "
         f"Harga diprediksi ≈ Rp {row['harga_prediksi (Rp)']:,.0f}")




# ---------------------------------------------------
# B) KLASIFIKASI — Iris (Logistic Regression)
# ---------------------------------------------------


import matplotlib.pyplot as plt
from PIL import Image
import urllib.request
import numpy as np


# URL gambar diagram bunga Iris
url = "https://editor.analyticsvidhya.com/uploads/51518iris%20img1.png"


# Buka URL dan baca gambar menggunakan PIL
with urllib.request.urlopen(url) as response:
   img = Image.open(response)
   img_array = np.array(img)


# Tampilkan gambar
plt.figure(figsize=(8,5))
plt.imshow(img_array)
plt.axis('off')
plt.title("Diagram Bunga Iris (Petal & Sepal)", fontsize=14, pad=10)


# Tambahkan keterangan di bawah gambar
plt.text(
   0, -25,
   """Gambar ini menunjukkan bagian-bagian bunga Iris:
- Sepal (kelopak)
- Petal (mahkota)
Fitur pada dataset Iris diambil dari pengukuran panjang & lebar dua bagian ini.""",
   fontsize=10,
   color="black",
   wrap=True
)


plt.show()




iris = load_iris(as_frame=True)
X = iris.data
y = iris.target


Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)


clf = Pipeline([
   ("scale", StandardScaler()),
   ("logreg", LogisticRegression(max_iter=1000))
])
clf.fit(Xtr, ytr)
yp = clf.predict(Xte)


print("=== KLASIFIKASI: IRIS Logistic Regression ===")
print("Akurasi:", round(accuracy_score(yte, yp), 4))
print("Laporan klasifikasi:")
print(classification_report(yte, yp, target_names=iris.target_names))


# Plot 2: Confusion matrix (tanpa seaborn)
cm = confusion_matrix(yte, yp)
plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix - Iris")
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.xticks(ticks=range(3), labels=iris.target_names, rotation=45)
plt.yticks(ticks=range(3), labels=iris.target_names)
for i in range(cm.shape[0]):
   for j in range(cm.shape[1]):
       plt.text(j, i, cm[i, j], ha="center", va="center")
plt.tight_layout()
plt.show()


# Simpan model klasifikasi
joblib.dump(clf, "model_iris.pkl")


# ==========================================================
# B. KLASIFIKASI (Supervised Learning) — Dataset IRIS
# ==========================================================
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
import pandas as pd


# 1️⃣ Muat dataset IRIS
iris = load_iris(as_frame=True)
X = iris.data          # fitur (panjang/lebar kelopak & mahkota)
y = iris.target        # label (0=setosa, 1=versicolor, 2=virginica)


# 2️⃣ Tampilkan isi dataset
print("=== DATASET IRIS ===")
print(f"Jumlah data : {X.shape[0]} baris, {X.shape[1]} fitur\n")
print("Nama fitur :", list(X.columns))
print("Nama kelas :", list(iris.target_names))
print("\nContoh 10 baris pertama:")
print(pd.concat([X, y.rename("target")], axis=1).head(10))
print("="*60, "\n")


# 3️⃣ Bagi data menjadi Train dan Test
Xtr, Xte, ytr, yte = train_test_split(
   X, y, test_size=0.25, random_state=42, stratify=y
)


# 4️⃣ Buat pipeline model
clf = Pipeline([
   ("scale", StandardScaler()),               # standarisasi data
   ("logreg", LogisticRegression(max_iter=1000))  # model klasifikasi
])


# 5️⃣ Latih model
clf.fit(Xtr, ytr)


# 6️⃣ Prediksi data test
yp = clf.predict(Xte)


# 7️⃣ Evaluasi hasil
print("=== KLASIFIKASI: IRIS Logistic Regression ===")
print("Akurasi:", accuracy_score(yte, yp))
print("\nLaporan klasifikasi:")
print(classification_report(yte, yp, target_names=iris.target_names))


# 8️⃣ (Opsional) Lihat contoh hasil prediksi
print("\nContoh hasil prediksi 10 data test:")
hasil = Xte.copy()
hasil["target_asli"] = [iris.target_names[i] for i in yte]
hasil["target_pred"] = [iris.target_names[i] for i in yp]
print(hasil.head(10))


# ================================================
# PREDIKSI DATA BARU — IRIS (gunakan model clf yang sudah fit)
# ================================================
import pandas as pd
import numpy as np


# 1) Contoh data baru (4 fitur sesuai urutan kolom X)
data_baru = pd.DataFrame([
   {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},  # mirip setosa
   {"sepal length (cm)": 6.0, "sepal width (cm)": 2.8, "petal length (cm)": 4.9, "petal width (cm)": 1.8},  # mirip virginica
   {"sepal length (cm)": 5.9, "sepal width (cm)": 3.0, "petal length (cm)": 4.2, "petal width (cm)": 1.3},  # mirip versicolor
])


# (opsional) pastikan kolomnya tepat & berurutan seperti X
data_baru = data_baru[X.columns]


# 2) Prediksi kelas & probabilitas
pred_idx = clf.predict(data_baru)                     # index kelas (0/1/2)
pred_proba = clf.predict_proba(data_baru)             # probabilitas per kelas
pred_nama = [iris.target_names[i] for i in pred_idx]  # nama kelas


# 3) Gabungkan ke tabel hasil
proba_df = pd.DataFrame(
   pred_proba,
   columns=[f"proba_{name}" for name in iris.target_names]
)
hasil_pred = pd.concat([data_baru.reset_index(drop=True), proba_df], axis=1)
hasil_pred["pred_class_idx"] = pred_idx
hasil_pred["pred_class_name"] = pred_nama


print("=== HASIL PREDIKSI DATA BARU (IRIS) ===")
print(hasil_pred)




# ---------------------------------------------------
# C) CLUSTERING — KMeans pada data 2D sintetis
# ---------------------------------------------------
# =======================================================
# CLUSTERING 300 ORANG: Tinggi vs Berat (3 cluster)
# =======================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


np.random.seed(42)


# 1️⃣ Buat dataset acak (realistis)
# cluster1: tubuh kecil, cluster2: sedang, cluster3: besar
tinggi = np.concatenate([
   np.random.normal(160, 5, 100),  # kecil
   np.random.normal(170, 5, 100),  # sedang
   np.random.normal(180, 5, 100)   # besar
])
berat = np.concatenate([
   np.random.normal(55, 5, 100),   # kecil
   np.random.normal(70, 5, 100),   # sedang
   np.random.normal(85, 5, 100)    # besar
])


# Buat DataFrame
df = pd.DataFrame({
   "Orang": [f"Org_{i+1}" for i in range(300)],
   "Tinggi (cm)": tinggi,
   "Berat (kg)": berat
})


print("=== CONTOH DATASET ORANG (10 baris) ===")
print(df.head(10))
print(f"\nJumlah data: {len(df)} orang\n")


# 2️⃣ Jalankan KMeans (3 cluster)
X = df[["Tinggi (cm)", "Berat (kg)"]]
km = KMeans(n_clusters=3, random_state=42, n_init=10)
km.fit(X)


# 3️⃣ Tambahkan label cluster
df["Cluster"] = km.labels_


# 4️⃣ Lihat ringkasan hasil
print("=== RINGKASAN JUMLAH ORANG PER CLUSTER ===")
print(df["Cluster"].value_counts().sort_index())
print()


print("=== PUSAT CLUSTER (CENTROIDS) ===")
print(pd.DataFrame(km.cluster_centers_, columns=["Tinggi (cm)", "Berat (kg)"]).round(2))
print()


print("=== CONTOH 10 HASIL CLUSTERING ===")
print(df.head(10))


# 5️⃣ Visualisasi hasil
plt.figure(figsize=(7,5))
plt.scatter(df["Tinggi (cm)"], df["Berat (kg)"],
           c=df["Cluster"], cmap="viridis", s=40)
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1],
           marker="X", color="red", s=200, label="Centroids")
plt.title("Clustering 300 Orang Berdasarkan Tinggi & Berat")
plt.xlabel("Tinggi (cm)")
plt.ylabel("Berat (kg)")
plt.legend()
plt.grid(True)
plt.show()


# ================================================
# PREDIKSI CLUSTER UNTUK DATA BARU (MODEL SUDAH JADI)
# ================================================
import pandas as pd


# Misal data baru:
data_baru = pd.DataFrame({
   "Orang": ["Orang_X", "Orang_Y", "Orang_Z"],
   "Tinggi (cm)": [162, 175, 183],
   "Berat (kg)": [58, 72, 88]
})


# Ambil hanya fitur numeriknya
X_new = data_baru[["Tinggi (cm)", "Berat (kg)"]]


# Gunakan model KMeans yang sudah dilatih sebelumnya (variabel km)
pred_cluster = km.predict(X_new)


# Tambahkan hasil ke data baru
data_baru["Cluster_Prediksi"] = pred_cluster


# Tampilkan hasilnya
print("=== HASIL PREDIKSI CLUSTER UNTUK DATA BARU ===")
print(data_baru)


