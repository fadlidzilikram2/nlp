# ===============================
# ANALISIS ULASAN TOKOPEDIA
# TF-IDF + LDA + NAIVE BAYES
# ===============================

import pandas as pd
import numpy as np
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# ===============================
# DOWNLOAD RESOURCE NLTK
# ===============================
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

# ===============================
# LOAD DATASET
# ===============================
# Dataset harus memiliki kolom:
# review    -> teks ulasan
# sentiment -> label sentimen (positif, negatif, netral)

df = pd.read_csv("ulasan_tokopedia.csv")
df["review"] = df["review"].astype(str)

# ===============================
# PREPROCESSING TEKS
# ===============================
stop_words = set(stopwords.words("indonesian"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # hapus angka
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)

df["clean_review"] = df["review"].apply(preprocess_text)

# ===============================
# TF-IDF
# ===============================
tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(df["clean_review"])

# ===============================
# LDA (ANALISIS TOPIK)
# ===============================
lda = LatentDirichletAllocation(
    n_components=5,
    random_state=42
)
lda.fit(X_tfidf)

print("\n=== HASIL TOPIK LDA ===")
feature_names = tfidf.get_feature_names_out()
for idx, topic in enumerate(lda.components_):
    print(f"Topik {idx+1}:")
    print([feature_names[i] for i in topic.argsort()[-10:]])
    print()

# ===============================
# ANALISIS SENTIMEN (NAIVE BAYES)
# ===============================
