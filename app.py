# =========================================
# STREAMLIT APP
# ANALISIS TOPIK ULASAN TOKOPEDIA
# TF-IDF + LDA
# =========================================

import streamlit as st
import pandas as pd
import re
import string
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# =========================================
# KONFIGURASI HALAMAN
# =========================================
st.set_page_config(
    page_title="Analisis Topik Ulasan Tokopedia",
    layout="wide"
)

st.title("📊 Analisis Topik Ulasan Produk Tokopedia")
st.write("Metode: TF-IDF dan Latent Dirichlet Allocation (LDA)")

# =========================================
# DOWNLOAD RESOURCE NLTK
# =========================================
nltk.download('punkt')
nltk.download("punkt_tab")
nltk.download('stopwords')

stop_words = set(stopwords.words("indonesian"))

# =========================================
# FUNGSI PREPROCESSING
# =========================================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

# =========================================
# UPLOAD DATASET
# =========================================
st.subheader("📁 Upload Dataset CSV")
uploaded_file = st.file_uploader(
    "Upload file CSV (harus memiliki kolom 'review')",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # Normalisasi nama kolom
    df.columns = df.columns.str.lower()

    if "review" not in df.columns:
        st.error("❌ Kolom 'review' tidak ditemukan di dataset!")
        st.stop()

    df["review"] = df["review"].astype(str)

    st.success("✅ Dataset berhasil dimuat")
    st.write("Contoh data:")
    st.dataframe(df.head())

    # =========================================
    # PREPROCESSING
    # =========================================
    st.subheader("🧹 Preprocessing Teks")
    df["clean_review"] = df["review"].apply(preprocess_text)

    st.write("Contoh hasil preprocessing:")
    st.dataframe(df[["review", "clean_review"]].head())

    # =========================================
    # TF-IDF
    # =========================================
    st.subheader("🔢 Representasi TF-IDF")
    tfidf = TfidfVectorizer(max_features=1000)
    X_tfidf = tfidf.fit_transform(df["clean_review"])

    st.write("Jumlah fitur TF-IDF:", X_tfidf.shape[1])

    # =========================================
    # LDA
    # =========================================
    st.subheader("🧠 Analisis Topik (LDA)")
    num_topics = st.slider("Jumlah Topik", 2, 10, 5)

    lda = LatentDirichletAllocation(
        n_components=num_topics,
        random_state=42
    )
    lda.fit(X_tfidf)

    feature_names = tfidf.get_feature_names_out()

    for idx, topic in enumerate(lda.components_):
        st.markdown(f"### Topik {idx + 1}")
        top_words = [feature_names[i] for i in topic.argsort()[-10:]]
        st.write(", ".join(top_words))

    st.success("🎉 Analisis topik berhasil dilakukan!")

