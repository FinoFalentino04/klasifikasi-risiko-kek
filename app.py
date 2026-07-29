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

# Menjawab Revisi Pak Didik: Menurunkan klaim medis
st.warning("⚠️ **Perhatian:** Sistem ini adalah prototipe skrining awal (Sistem Pendukung Keputusan), bukan alat diagnosis medis. Keputusan akhir tetap berada di tangan tenaga kesehatan yang berwenang.")

# ==========================================
# 2. INISIALISASI MODEL
# ==========================================
@st.cache_resource
def load_model():
    # Pastikan nama file .pkl sesuai dengan yang di-export dari Colab terbaru
    knn = joblib.load('knn_model_3class_tuned.pkl')
    scaler = joblib.load('scaler_3class_tuned.pkl')
    return knn, scaler

try:
    knn_model, scaler_model = load_model()
except Exception as e:
    st.error("Gagal memuat model. Pastikan file 'knn_model_3class_tuned.pkl' dan 'scaler_3class_tuned.pkl' berada di dalam folder yang sama dengan app.py.")

# ==========================================
# 3. ANTARMUKA INPUT DATA
# ==========================================
st.subheader("Form Parameter Pasien")

col1, col2 = st.columns(2)
with col1:
    tgl_pemeriksaan = st.date_input("Tanggal Pemeriksaan")
with col2:
    # Menjawab Revisi Pak Didik: Logika kalender diperbaiki bisa mundur ke tahun 1950
    tgl_lahir = st.date_input(
        "Tanggal Lahir Pasien",
        value=datetime.date(1996, 1, 1),
        min_value=datetime.date(1950, 1, 1),
        max_value=datetime.date.today()
    )

col3, col4 = st.columns(2)
with col3:
    bb = st.number_input("Berat Badan Sebelum Hamil (kg)", min_value=30.0, max_value=150.0, value=50.0)
with col4:
    imt = st.number_input("Indeks Massa Tubuh (IMT)", min_value=10.0, max_value=50.0, value=22.0)

lila = st.number_input("Lingkar Lengan Atas / LiLA (cm)", min_value=15.0, max_value=40.0, value=23.5)

# ==========================================
# 4. PROSES KLASIFIKASI & VISUALISASI
# ==========================================
if st.button("Lakukan Skrining", type="primary"):
    
    # Menghitung Usia
    usia_tahun = (tgl_pemeriksaan - tgl_lahir).days // 365
    
    st.markdown("---")
    
    # Menjawab Revisi Pak Asep: TABEL RINGKASAN DATA
    st.subheader("Tabel Ringkasan Data Klinis")
    df_hasil = pd.DataFrame({
        "Parameter Indikator": ["Usia Pasien", "Berat Badan", "Indeks Massa Tubuh (IMT)", "Lingkar Lengan Atas (LiLA)"],
        "Nilai Pasien": [f"{usia_tahun} Tahun", f"{bb} kg", f"{imt}", f"{lila} cm"],
        "Ambang Batas Normal": ["20 - 35 Tahun", ">= 45 kg", "18.5 - 25.0", ">= 23.5 cm"]
    })
    # Menampilkan tabel tanpa index default (0, 1, 2, 3) agar lebih rapi
    st.table(df_hasil.assign(hack='').set_index('hack'))

    # Praproses input untuk dimasukkan ke Model
    input_data = np.array([[usia_tahun, bb, imt, lila]])
    input_scaled = scaler_model.transform(input_data)
    
    # Melakukan Prediksi
    prediksi = knn_model.predict(input_scaled)[0]
    probabilitas = knn_model.predict_proba(input_scaled)[0]
    
    # Menjawab Revisi Pak Didik: Label Aman diubah
    label_kelas = [
        "Risiko Rendah Berdasarkan Variabel yang Dianalisis", 
        "Risiko Sedang", 
        "Risiko Tinggi"
    ]
    hasil_akhir = label_kelas[prediksi]
    
    st.subheader("Hasil Keputusan Sistem")
    if prediksi == 0:
        st.success(f"**Klasifikasi:** {hasil_akhir}")
    elif prediksi == 1:
        st.warning(f"**Klasifikasi:** {hasil_akhir}")
    else:
        st.error(f"**Klasifikasi:** {hasil_akhir}")

    # Menjawab Revisi Pak Asep: GRAFIK VISUALISASI
    st.markdown("---")
    st.subheader("Panel Visualisasi Data")
    
    col_grafik1, col_grafik2 = st.columns(2)
    
    # Grafik 1: Bar Chart Probabilitas Kedekatan K-NN
    with col_grafik1:
        st.markdown("**1. Probabilitas Kedekatan (K-NN)**")
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        label_grafik = ['Rendah', 'Sedang', 'Tinggi']
        warna = ['#2ca02c', '#ff7f0e', '#d62728']
        
        bars = ax1.bar(label_grafik, probabilitas * 100, color=warna)
        ax1.set_ylabel('Probabilitas (%)')
        ax1.set_ylim(0, 110)
        
        for bar in bars:
            tinggi = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2, tinggi + 2, f'{tinggi:.1f}%', ha='center', fontweight='bold')
        
        st.pyplot(fig1)

    # Grafik 2: Scatter Plot Posisi Gizi
    with col_grafik2:
        st.markdown("**2. Peta Posisi Gizi Pasien**")
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        
        # Garis Batas Kritis Medis
        ax2.axhline(y=23.5, color='red', linestyle='--', linewidth=1.5, label='Batas Kritis LiLA (23.5)')
        ax2.axvline(x=18.5, color='orange', linestyle='--', linewidth=1.5, label='Batas Bawah IMT (18.5)')
        ax2.axvline(x=25.0, color='orange', linestyle='--', linewidth=1.5, label='Batas Atas IMT (25.0)')
        
        # Titik Pasien yang di-input saat ini
        ax2.scatter(imt, lila, color='blue', s=150, zorder=5, label='Pasien Saat Ini')
        
        ax2.set_xlabel('Indeks Massa Tubuh (IMT)')
        ax2.set_ylabel('Lingkar Lengan Atas (cm)')
        # Mengatur rentang sumbu agar garis putus-putus selalu terlihat jelas
        ax2.set_xlim(10, 40)
        ax2.set_ylim(15, 35)
        
        ax2.legend(loc='lower right', fontsize=8)
        ax2.grid(True, linestyle=':', alpha=0.6)
        
        st.pyplot(fig2)
