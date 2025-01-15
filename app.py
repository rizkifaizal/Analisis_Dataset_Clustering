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
    ### 🔍 Tentang Aplikasi
    Aplikasi ini dirancang untuk **menganalisis dataset clustering** yang mencakup berbagai atribut demografis individu. 
    Aplikasi ini memanfaatkan algoritma clustering, khususnya **Hierarchical Agglomerative Clustering**, untuk membantu memahami pola dan pengelompokan dalam data.

    ### 📋 Deskripsi Dataset
    Dataset ini berisi informasi demografis seperti:
    - **ID**: Identifikasi unik untuk setiap individu.
    - **Sex**: Jenis kelamin (0 = Perempuan, 1 = Laki-laki).
    - **Marital Status**: Status pernikahan (0 = Belum Menikah, 1 = Menikah).
    - **Age**: Usia individu.
    - **Education**: Tingkat pendidikan (0 = Tidak berpendidikan, 1 = SD, 2 = SMP).
    - **Income**: Pendapatan tahunan.
    - **Occupation**: Jenis pekerjaan.
    - **Settlement Size**: Ukuran pemukiman (0 = Kecil, 1 = Sedang, 2 = Besar).

    ### 📊 Apa itu Hierarchical Agglomerative Clustering?
    **Hierarchical Agglomerative Clustering (HAC)** adalah algoritma clustering yang bekerja dengan pendekatan **bottom-up**:
    - Setiap individu dimulai sebagai cluster tunggal.
    - Cluster ini digabungkan secara berulang berdasarkan kesamaan hingga semua data berada dalam satu cluster besar.
    - Hasilnya divisualisasikan dalam bentuk **dendrogram**, yang menunjukkan hubungan antar individu atau kelompok.

    #### 🚀 Keunggulan HAC
    - **Tanpa Penentuan Jumlah Cluster Awal**: Tidak memerlukan jumlah cluster yang ditentukan sebelumnya.
    - **Analisis Visual dengan Dendrogram**: Memberikan wawasan mendalam tentang struktur data.
    - **Fleksibilitas dalam Metode Penggabungan**: Seperti metode ward, average, complete, atau single linkage.

    ### 🎯 Tujuan Analisis
    Dengan memanfaatkan HAC, aplikasi ini bertujuan:
    - Mengidentifikasi pola dan hubungan dalam data demografis.
    - Membantu dalam pengelompokan individu berdasarkan atribut seperti usia, pendapatan, pendidikan, dan pekerjaan.
    - Memberikan wawasan yang relevan untuk pengambilan keputusan di berbagai bidang, seperti pemasaran, penelitian sosial, dan pengembangan produk.

    ### 🛠️ Fitur Utama
    - **Filter Data**: Berdasarkan jenis kelamin, usia, status pernikahan, dan tingkat pendidikan.
    - **Visualisasi Data**: 
      - Distribusi pendapatan dan usia.
      - Visualisasi dendrogram untuk melihat hasil clustering.
    - **Evaluasi Model**: Menghitung nilai **Silhouette Score** untuk mengukur kualitas clustering.
    - **Ekspor Data**: Simpan data yang telah difilter dan diberi label cluster dalam format CSV.

    ### 🌳 Mengapa Hierarchical Clustering?
    Hierarchical clustering cocok digunakan saat:
    - Anda ingin memahami struktur data yang mendalam.
    - Analisis visual dan eksplorasi data adalah prioritas.
    - Tidak ada informasi awal mengenai jumlah cluster yang diinginkan.

    Kami berharap aplikasi ini dapat membantu Anda memahami data secara lebih baik dan memanfaatkan hasil analisis untuk tujuan strategis.
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