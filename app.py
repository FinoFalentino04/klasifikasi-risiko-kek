import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import datetime

# ==========================================
# 1. KONFIGURASI DAN DISCLAIMER MEDIS
# ==========================================
st.set_page_config(page_title="DSS Risiko KEK - Cisaat", layout="centered")

st.title("Sistem Pendukung Keputusan Skrining KEK")
st.markdown("Prototipe klasifikasi risiko Kurang Energi Kronis (KEK) menggunakan algoritma *K-Nearest Neighbor*.")

# Menjawab Revisi Pak Didik: Penurunan Klaim Medis
st.warning("⚠️ **Perhatian:** Sistem ini adalah prototipe skrining awal (Sistem Pendukung Keputusan), bukan alat diagnosis medis. Keputusan akhir tetap berada di tangan tenaga kesehatan yang berwenang.")

# ==========================================
# 2. INISIALISASI MODEL
# ==========================================
@st.cache_resource
def load_model():
    # Pastikan file .pkl berada di satu folder yang sama dengan app.py
    knn = joblib.load('knn_model_3class_tuned.pkl')
    scaler = joblib.load('scaler_3class_tuned.pkl')
    return knn, scaler

try:
    knn_model, scaler_model = load_model()
except Exception as e:
    st.error("Gagal memuat model. Pastikan file 'knn_model_3class_tuned.pkl' dan 'scaler_3class_tuned.pkl' ada di direktori.")

# ==========================================
# 3. ANTARMUKA INPUT DATA (REVISI USIA)
# ==========================================
st.subheader("Form Parameter Pasien")

# Menjawab Revisi Pak Didik: Usia berdasarkan tanggal pemeriksaan, bukan hari ini
col1, col2 = st.columns(2)
with col1:
    tgl_pemeriksaan = st.date_input("Tanggal Pemeriksaan")
with col2:
    # Perbaikan batas kalender untuk usia ibu hamil
    tgl_lahir = st.date_input(
        "Tanggal Lahir Pasien",
        value=datetime.date(1996, 1, 1),          # Nilai awal (default) saat web dibuka (usia ~30 tahun)
        min_value=datetime.date(1950, 1, 1),      # Batas kalender paling tua (tahun 1950)
        max_value=datetime.date.today()           # Batas kalender paling muda (hari ini)
    )

col3, col4 = st.columns(2)
with col3:
    bb = st.number_input("Berat Badan Sebelum Hamil (kg)", min_value=30.0, max_value=150.0, value=50.0)
with col4:
    tb = st.number_input("Tinggi Badan (cm)", min_value=100.0, max_value=200.0, value=150.0)

lila = st.number_input("Lingkar Lengan Atas / LiLA (cm)", min_value=15.0, max_value=40.0, value=23.5)

# ==========================================
# 4. PROSES KLASIFIKASI & VISUALISASI GRAFIK
# ==========================================
if st.button("Lakukan Skrining", type="primary"):
    
    # Kalkulasi Dinamis
    usia_tahun = (tgl_pemeriksaan - tgl_lahir).days // 365
    imt = bb / ((tb / 100) ** 2)
    
    # Tampilkan nilai yang dihitung
    st.info(f"Kalkulasi Sistem: **Usia:** {usia_tahun} tahun | **IMT:** {imt:.2f}")

    # Praproses input
    input_data = np.array([[usia_tahun, bb, imt, lila]])
    input_scaled = scaler_model.transform(input_data)
    
    # Prediksi K-NN
    prediksi = knn_model.predict(input_scaled)[0]
    probabilitas = knn_model.predict_proba(input_scaled)[0]
    
    # Menjawab Revisi Pak Didik: Perbaikan Label Kelas
    label_kelas = [
        "Risiko Rendah Berdasarkan Variabel yang Dianalisis", 
        "Risiko Sedang", 
        "Risiko Tinggi"
    ]
    
    hasil_akhir = label_kelas[prediksi]
    
    # Pewarnaan notifikasi berdasarkan hasil
    if prediksi == 0:
        st.success(f"**Hasil Klasifikasi:** {hasil_akhir}")
    elif prediksi == 1:
        st.warning(f"**Hasil Klasifikasi:** {hasil_akhir}")
    else:
        st.error(f"**Hasil Klasifikasi:** {hasil_akhir}")

    # Menjawab Revisi Pak Asep: Penambahan Grafik Informatif
    st.subheader("Grafik Probabilitas Kedekatan (K-NN)")
    
    fig, ax = plt.subplots(figsize=(7, 3))
    label_grafik = ['Rendah', 'Sedang', 'Tinggi']
    warna = ['#2ca02c', '#ff7f0e', '#d62728'] # Hijau, Oranye, Merah
    
    bars = ax.barh(label_grafik, probabilitas * 100, color=warna)
    ax.set_xlabel('Probabilitas Kedekatan (%)')
    ax.set_xlim(0, 100)
    
    for bar in bars:
        lebar = bar.get_width()
        ax.text(lebar + 2, bar.get_y() + bar.get_height()/2, f'{lebar:.1f}%', va='center', fontweight='bold')
    
    st.pyplot(fig)
