import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
import matplotlib.pyplot as plt
import seaborn as sns

# Set page configuratio
st.set_page_config(page_title="📊 Analisis Dataset Clustering", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigasi")
st.sidebar.markdown("---")  # Separator line

# Add icons to the sidebar options
page = st.sidebar.radio("Pilih Halaman:", 
                         options=["🏠 About", "📊 Analisis Data"])

# Halaman About
if page == "🏠 About":
    st.title("📊 Tentang Aplikasi")
    
    st.markdown("""
    Aplikasi ini dirancang untuk **menganalisis dataset clustering** yang berisi informasi demografis individu.
    
    ### 📋 Deskripsi Dataset
    Dataset ini mencakup berbagai atribut demografis individu yang dapat digunakan untuk analisis lebih lanjut dalam:
    - Pemasaran
    - Penelitian sosial
    - Pengembangan produk

    ### 📊 Kolom dalam Dataset
    - **ID**: Identifikasi unik untuk setiap entri.
    - **Sex**: Jenis kelamin (0 = Perempuan, 1 = Laki-laki).
    - **Marital Status**: Status pernikahan (0 = Belum Menikah, 1 = Menikah).
    - **Age**: Usia individu.
    - **Education**: Tingkat pendidikan (0 = Tidak berpendidikan, 1 = SD, 2 = SMP).
    - **Income**: Pendapatan tahunan individu.
    - **Occupation**: Kode untuk jenis pekerjaan.
    - **Settlement Size**: Ukuran pemukiman (0 = Kecil, 1 = Sedang, 2 = Besar).

    ### 🔍 Fitur Utama
    - **Filter Data**: Berdasarkan jenis kelamin, status pernikahan, usia, dan pendidikan.
    - **Tampilkan Data**: Data yang difilter dalam bentuk tabel.
    - **Statistik Deskriptif**: Ringkasan data untuk analisis cepat.
    - **Visualisasi**: Distribusi pendapatan dan usia menggunakan grafik histogram.
    - **Ekspor Data**: Simpan data yang telah difilter ke dalam file CSV.

    ### 🎯 Tujuan Aplikasi
    Memberikan wawasan yang lebih dalam tentang data demografis dan membantu dalam pengambilan keputusan berbasis data.

    ### 🛠️ Penggunaan
    Silakan pilih filter di sidebar untuk menganalisis data sesuai kebutuhan Anda.

    ### 📊 Jenis-Jenis Hierarchical Clustering
    Terdapat dua jenis utama dari hierarchical clustering:
    
    1. **Agglomerative Clustering**: 
       - Dikenal sebagai pendekatan bottom-up. 
       - Setiap data dianggap sebagai cluster tunggal pada awalnya, dan kemudian secara bertahap menggabungkan pasangan cluster hingga semua cluster digabungkan menjadi satu cluster yang berisi semua data.
       - Algoritma ini tidak memerlukan penentuan jumlah cluster sebelumnya.

    2. **Divisive Clustering**: 
       - Dikenal sebagai pendekatan top-down. 
       - Memulai dengan satu cluster yang berisi seluruh data dan kemudian membagi cluster tersebut secara rekursif hingga setiap data terpisah menjadi cluster tunggal.
       - Juga tidak memerlukan penentuan jumlah cluster sebelumnya.

    ### ⚖️ Perbandingan Agglomerative dan Divisive Clustering
    - **Kompleksitas**: 
      - Divisive clustering lebih kompleks dibandingkan dengan agglomerative clustering.
      - Agglomerative clustering memiliki kompleksitas waktu O(n³) dalam kasus naive, tetapi dapat dioptimalkan menjadi O(n²) dengan menggunakan struktur data priority queue.
      - Divisive clustering lebih efisien jika tidak menghasilkan hierarki lengkap hingga data individu.

    - **Akurasi**: 
      - Algoritma divisive lebih akurat karena mempertimbangkan distribusi global data saat membuat keputusan pemisahan.
      - Agglomerative clustering membuat keputusan berdasarkan pola lokal tanpa mempertimbangkan distribusi global data, sehingga keputusan awal tidak dapat dibatalkan.
    """)


# Halaman Analisis Data
elif page == "📊 Analisis Data":
    # Load dataset
    data = pd.read_csv('Clustering.csv')

    # Header
    st.title("📊 Analisis Dataset Clustering")
    st.markdown("""
    Aplikasi ini memungkinkan Anda untuk menganalisis dataset clustering berdasarkan berbagai parameter.
    Silakan pilih filter di bawah ini untuk melihat data yang relevan.
    """)

    # Sidebar for filters
    st.sidebar.header("Filter Data")
    st.sidebar.markdown("---")  # Separator line

    # Filter berdasarkan Jenis Kelamin
    sex_filter = st.sidebar.selectbox("Pilih Jenis Kelamin:", options=["Semua", "Laki-laki", "Perempuan"])
    if sex_filter == "Laki-laki":
        data = data[data['Sex'] == 1]
    elif sex_filter == "Perempuan":
        data = data[data['Sex'] == 0]

    # Filter berdasarkan Status Pernikahan
    marital_status_filter = st.sidebar.selectbox("Pilih Status Pernikahan:", options=["Semua", "Menikah", "Belum Menikah"])
    if marital_status_filter == "Menikah":
        data = data[data['Marital status'] == 1]
    elif marital_status_filter == "Belum Menikah":
        data = data[data['Marital status'] == 0]

    # Filter berdasarkan Usia
    age_filter = st.sidebar.slider("Pilih Rentang Usia:", min_value=int(data['Age'].min()), max_value=int(data['Age'].max()), value=(30, 50))
    data = data[(data['Age'] >= age_filter[0]) & (data['Age'] <= age_filter[1])]

    # Filter berdasarkan Pendidikan
    education_filter = st.sidebar.selectbox("Pilih Tingkat Pendidikan:", options=["Semua", "0", "1", "2 "])
    if education_filter == "0":
        data = data[data['Education'] == 0]
    elif education_filter == "1":
        data = data[data['Education'] == 1]
    elif education_filter == "2":
        data = data[data['Education'] == 2]

    # Tampilkan data yang difilter
    st.subheader("📊 Data yang Difilter")
    st.write(data)

    # Statistik deskriptif
    st.subheader("📈 Statistik Deskriptif")
    st.write(data.describe())

    #Visualisasi distribusi pendapatan
    st.subheader("💰 Distribusi Pendapatan")
    fig_income = px.histogram(data, x='Income', title='Distribusi Pendapatan', nbins=30)
    st.plotly_chart(fig_income)

    # Visualisasi distribusi usia
    st.subheader("👶 Distribusi Usia")
    fig_age = px.histogram(data, x='Age', title='Distribusi Usia', nbins=25)
    st.plotly_chart(fig_age)

    # Hierarchical Agglomerative Clustering
    st.subheader("🌳 Hierarchical Agglomerative Clustering")
    st.markdown("""
    Di bawah ini adalah visualisasi dendrogram untuk menunjukkan hasil clustering.
    """)

    # Menggunakan fitur yang relevan untuk clustering
    features = data[['Age', 'Income','Occupation','Education']]  # Misalkan kita menggunakan 'Age' dan 'Income' untuk clustering
    model = AgglomerativeClustering(n_clusters=3)  # Misalkan kita ingin 3 cluster
    model.fit(features)

    # Menambahkan hasil clustering ke DataFrame
    data['Cluster'] = model.labels_

    # Visualisasi Dendrogram
    plt.figure(figsize=(10, 7))
    dendrogram = sch.dendrogram(sch.linkage(features, method='ward'))
    st.pyplot(plt)

    # Penjelasan tentang dendrogram
    st.markdown("""
    Dendrogram di atas menunjukkan hasil dari **Hierarchical Agglomerative Clustering**. 
    Setiap cabang pada dendrogram mewakili pengelompokan individu berdasarkan kesamaan atribut yang dipilih, yaitu **Usia**, **Pendapatan**, **Education** dan **Occipation**. 
    Jarak antara cabang menunjukkan seberapa mirip atau berbeda kelompok tersebut. 
    Semakin dekat jarak antara dua cabang, semakin mirip individu dalam kelompok tersebut. 
    Anda dapat melihat bagaimana individu dikelompokkan ke dalam cluster yang berbeda berdasarkan karakteristik mereka.
    """)

    # Tampilkan hasil clustering
    st.subheader("Hasil Clustering")
    st.write(data[['Age', 'Income','Education','Occupation','Cluster']])


    from sklearn.metrics import silhouette_score
    silhouette_avg = silhouette_score(features, model.labels_)
    st.subheader("📈 Silhouette Score")
    st.write(f"Silhouette Score: {silhouette_avg:.2f}")

    # Drop 'Sex' and 'Marital status' columns for correlation analysis
    data_filtered = data.drop(columns=['Sex', 'Marital status'], errors='ignore')

    # Create a heatmap for the correlation matrix
    st.subheader("📊 Heatmap Korelasi")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data_filtered.corr(), annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
    st.pyplot(plt)

    # Opsi untuk menyimpan data yang difilter
    if st.button("Simpan Data yang Difilter ke CSV"):
        data.to_csv('Filtered_Data.csv', index=False)
        st.success("Data berhasil disimpan sebagai 'Filtered_Data.csv'")