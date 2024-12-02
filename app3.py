import pandas as pd
import numpy as np
import sqlite3
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt

# --- Load Dataset ---
file_path = "FIX_PREPROSESING.xlsx"  # Replace with your dataset path
data = pd.read_excel(file_path)

# --- Validasi dan Pembersihan Data ---
required_columns = ['Nama Restoran', 'Cleaned_Text', 'Rating', 'Lokasi']
missing_columns = [col for col in required_columns if col not in data.columns]

if missing_columns:
    st.error(f"Kolom berikut tidak ditemukan dalam dataset: {', '.join(missing_columns)}")
    st.stop()

# Isi nilai kosong
data['Cleaned_Text'] = data['Cleaned_Text'].fillna('')
data['Rating'] = data['Rating'].fillna(data['Rating'].mean())
data['Lokasi'] = data['Lokasi'].fillna('')

# Normalisasi rating ke skala 0-1
data['Normalized_Rating'] = (data['Rating'] - data['Rating'].min()) / (data['Rating'].max() - data['Rating'].min())

# Gabungkan fitur untuk rekomendasi
data['Combined_Features'] = data['Cleaned_Text'] + " " + data['Lokasi']

# --- Vectorization ---
# TF-IDF untuk teks gabungan
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['Combined_Features'])

# TF-IDF untuk lokasi
tfidf_location = TfidfVectorizer(stop_words='english')
location_matrix = tfidf_location.fit_transform(data['Lokasi'])

# Similarity calculations
text_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)
location_similarity = cosine_similarity(location_matrix, location_matrix)
rating_similarity = cosine_similarity(data[['Normalized_Rating']])  # Similarity rating

# Kombinasikan similarity dengan bobot
def calculate_combined_similarity(alpha=0.5, beta=0.3, gamma=0.2):
    return (
        alpha * text_similarity +
        beta * location_similarity +
        gamma * rating_similarity
    )

# --- Recommendation Function ---
def recommend_items(item_name, alpha=0.5, beta=0.3, gamma=0.2, top_n=5):
    combined_similarity = calculate_combined_similarity(alpha, beta, gamma)

    # Temukan indeks restoran yang dipilih
    try:
        idx = data[data['Nama Restoran'] == item_name].index[0]
    except IndexError:
        return f"Restoran '{item_name}' tidak ditemukan dalam data."

    # Hitung skor similarity
    sim_scores = list(enumerate(combined_similarity[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Ambil top-N rekomendasi, kecuali restoran itu sendiri
    sim_scores = sim_scores[1:top_n + 1]
    recommended_indices = [score[0] for score in sim_scores]

    # Return recommended items
    recommendations = data.iloc[recommended_indices][['Nama Restoran', 'Lokasi', 'Rating', 'Cleaned_Text']]
    return recommendations

# --- Streamlit Interface ---
def run_streamlit():
    st.set_page_config(page_title="Sistem Rekomendasi Restoran", layout="wide")
    st.title("🍴 Sistem Rekomendasi Restoran - Content-Based Filtering")

    # Sidebar untuk pengaturan bobot
    st.sidebar.title("Pengaturan dan Filter")
    alpha = st.sidebar.slider("Bobot Teks", 0.0, 1.0, 0.5, 0.1)
    beta = st.sidebar.slider("Bobot Lokasi", 0.0, 1.0, 0.3, 0.1)
    gamma = st.sidebar.slider("Bobot Rating", 0.0, 1.0, 0.2, 0.1)
    top_n = st.sidebar.slider("Jumlah Rekomendasi", 1, 10, 5)

    # Filter lokasi
    location_filter = st.sidebar.multiselect("Pilih Lokasi", data['Lokasi'].unique())

    # Filter dataset berdasarkan lokasi
    filtered_data = data.copy()
    if location_filter:
        filtered_data = filtered_data[filtered_data['Lokasi'].isin(location_filter)]

    # Dropdown untuk memilih restoran
    if filtered_data.empty:
        st.error("Tidak ada restoran yang memenuhi kriteria filter.")
        st.stop()

    restaurant_name = st.selectbox("Pilih Nama Restoran", filtered_data['Nama Restoran'].tolist())

    # Tampilkan rekomendasi
    if st.button("Tampilkan Rekomendasi"):
        recommendations = recommend_items(restaurant_name, alpha, beta, gamma, top_n)

        if isinstance(recommendations, str):  # Jika terjadi error
            st.error(recommendations)
        else:
            st.markdown("### Rekomendasi untuk Anda:")
            for _, row in recommendations.iterrows():
                st.markdown(
                    f"""
                    **{row['Nama Restoran']}**
                    - 📍 Lokasi: {row['Lokasi']}
                    - ⭐ Rating: {row['Rating']}
                    ---
                    """
                )

    # Visualisasi heatmap
    if st.sidebar.checkbox("Tampilkan Heatmap Similarity Teks"):
        st.subheader("Heatmap Similarity Teks")
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(text_similarity, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# --- Jalankan Streamlit ---
if __name__ == '__main__':
    run_streamlit()
