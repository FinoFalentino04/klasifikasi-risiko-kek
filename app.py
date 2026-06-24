import streamlit as st
import numpy as np
import joblib

# Mengatur konfigurasi halaman web
st.set_page_config(page_title="Klasifikasi KEK K-NN", layout="centered")

# Judul Aplikasi (Sesuai judul skripsi final)
st.title("Klasifikasi Status Gizi dan Risiko KEK pada Ibu Hamil")
st.subheader("Menggunakan Algoritma K-Nearest Neighbor (K-NN) di Kecamatan Cisaat")

st.write("---")

# Memuat Model K-NN dan Z-Score Scaler 3 Kategori
@st.cache_resource
def load_model_and_scaler():
    try:
        model_knn = joblib.load('knn_model_3class.pkl')
        model_scaler = joblib.load('scaler_3class.pkl')
        return model_knn, model_scaler
    except FileNotFoundError:
        return None, None

knn, scaler = load_model_and_scaler()

if knn is None or scaler is None:
    st.error("⚠️ File 'knn_model_3class.pkl' atau 'scaler_3class.pkl' tidak ditemukan. Pastikan kedua file tersebut berada di direktori yang sama dengan app.py.")
else:
    # Membuat Form Input Data Pasien
    st.write("### Masukkan Data Klinis Warga / Pasien Baru")
    
    col1, col2 = st.columns(2)
    
    with col1:
        usia = st.number_input("Usia (Tahun)", min_value=10, max_value=60, value=25, step=1)
        bb = st.number_input("Berat Badan Sebelum Hamil (kg)", min_value=30.0, max_value=150.0, value=50.0, step=0.1)
        
    with col2:
        imt = st.number_input("Indeks Massa Tubuh (IMT)", min_value=10.0, max_value=50.0, value=21.5, step=0.1)
        lila = st.number_input("Lingkar Lengan Atas / LiLA (cm)", min_value=15.0, max_value=40.0, value=24.0, step=0.1)
        
    st.write("---")
    
    # Tombol Eksekusi Logika K-NN
    if st.button("Lakukan Klasifikasi Risiko", type="primary", use_container_width=True):
        
        # Membentuk array dari input pengguna
        data_input = np.array([[usia, bb, imt, lila]])
        
        # Normalisasi data input menggunakan Z-Score Scaler
        data_input_scaled = scaler.transform(data_input)
        
        # Mendapatkan kelas hasil klasifikasi akhir (0: Aman, 1: Sedang, 2: Tinggi)
        klasifikasi_hasil = knn.predict(data_input_scaled)[0]
        
        # Mendapatkan nilai probabilitas untuk ketiga kelas
        probabilitas = knn.predict_proba(data_input_scaled)[0]
        
        persentase_aman = probabilitas[0] * 100
        persentase_sedang = probabilitas[1] * 100
        persentase_tinggi = probabilitas[2] * 100
        
        # Menampilkan Hasil Klasifikasi Utama
        st.write("### Kesimpulan Klasifikasi K-NN:")
        
        if klasifikasi_hasil == 2:
            st.error(f"🚨 **Kategori: RISIKO TINGGI (Indikasi KEK)**")
            st.write("Sistem mengklasifikasikan probabilitas dominan pada kategori Risiko Tinggi. Pasien memiliki indikator kritis (seperti LiLA < 23.5 cm atau kombinasi usia ekstrem dan IMT berisiko) yang sangat identik dengan kelompok pasien Kurang Energi Kronis (KEK).")
        elif klasifikasi_hasil == 1:
            st.warning(f"⚠️ **Kategori: RISIKO SEDANG**")
            st.write("Sistem mengklasifikasikan adanya indikasi Risiko Sedang. Terdapat minimal satu parameter (usia, berat badan, atau IMT) yang kurang ideal, namun belum masuk tahap sangat kritis.")
        else:
            st.success(f"✅ **Kategori: AMAN / TIDAK BERISIKO**")
            st.write("Sistem mengklasifikasikan status kehamilan pasien aman. Seluruh parameter klinis berada dalam rentang normal dan memiliki jarak kedekatan dengan kelompok ibu hamil dengan gizi sehat.")
            
        st.write("---")
        
        # Menampilkan Rincian Persentase untuk Analisis Evaluasi
        st.write("#### Detail Probabilitas (Tingkat Keyakinan Algoritma):")
        st.markdown("Berikut adalah persentase tingkat kemiripan kondisi klinis pasien saat ini dengan riwayat data rekam medis ibu hamil lainnya di fasilitas kesehatan:")
        
        col_a, col_b, col_c = st.columns(3)
        col_a.metric(label="🟢 Aman", value=f"{persentase_aman:.1f}%")
        col_b.metric(label="🟡 Risiko Sedang", value=f"{persentase_sedang:.1f}%")
        col_c.metric(label="🔴 Risiko Tinggi", value=f"{persentase_tinggi:.1f}%")
