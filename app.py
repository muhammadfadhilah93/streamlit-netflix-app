import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
@st.cache
def load_data():
    file_path = "NetflixOriginals.csv"  # Sesuaikan dengan lokasi file Anda
    data = pd.read_csv(file_path, encoding="latin1")
    return data

data = load_data()

# Streamlit Sidebar for User Inputs
st.sidebar.header("Filter Options")

# Konversi 'Premiere' ke format datetime
data['Premiere'] = pd.to_datetime(data['Premiere'], errors='coerce')

# Display basic info
st.title("Netflix Originals IMDB Score Prediction")
st.write("Dataset Info:")
st.write(data.info())
st.write(data.head())

# One-hot encoding untuk 'Genre' dan 'Language'
data_encoded = pd.get_dummies(data, columns=['Genre', 'Language'], drop_first=True)

# Fitur dan target
X = data_encoded.drop(columns=['Title', 'Premiere', 'IMDB Score'])
y = data_encoded['IMDB Score']

# Split data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

st.write(f"Shape of X_train: {X_train.shape}, X_test: {X_test.shape}")

# Model pelatihan
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Evaluasi model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
st.write(f"Mean Squared Error: {mse}")

# Simpan model
joblib.dump(model, "netflix_model.pkl")
st.write("Model saved as netflix_model.pkl")

# ===================================
# 1. Grafik Distribusi IMDB Score
# ===================================
st.subheader("Distribusi IMDB Score")
plt.figure(figsize=(10, 6))
sns.histplot(data['IMDB Score'], bins=20, kde=True, color="skyblue")
plt.title("Distribusi IMDB Score", fontsize=16)
plt.xlabel("IMDB Score", fontsize=12)
plt.ylabel("Frekuensi", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
st.pyplot()

# ===================================
# 2. Grafik Durasi vs IMDB Score
# ===================================
st.subheader("Hubungan Durasi dengan IMDB Score")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Runtime'], y=data['IMDB Score'], color="green", alpha=0.6)
plt.title("Hubungan Durasi dengan IMDB Score", fontsize=16)
plt.xlabel("Durasi (menit)", fontsize=12)
plt.ylabel("IMDB Score", fontsize=12)
plt.grid(alpha=0.7)
st.pyplot()

# ===================================
# 3. Jumlah Film per Genre
# ===================================
st.subheader("Jumlah Film per Genre")
plt.figure(figsize=(22, 6))
genre_counts = data['Genre'].value_counts()
genre_counts.plot(kind='bar', color="orange", alpha=0.8)
plt.title("Jumlah Film per Genre", fontsize=16)
plt.xlabel("Genre", fontsize=12)
plt.ylabel("Jumlah Film", fontsize=12)
plt.xticks(rotation=90, ha="center")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.subplots_adjust(bottom=0.35)
st.pyplot()

# ===================================
# 4. Perubahan IMDB Score Berdasarkan Tahun Rilis
# ===================================
st.subheader("Perubahan IMDB Score Berdasarkan Tahun Rilis")
data['Year'] = data['Premiere'].dt.year
plt.figure(figsize=(10, 6))
sns.lineplot(data=data, x='Year', y='IMDB Score', marker="o", color="purple")
plt.title("Perubahan IMDB Score Berdasarkan Tahun Rilis", fontsize=16)
plt.xlabel("Tahun Rilis", fontsize=12)
plt.ylabel("IMDB Score", fontsize=12)
plt.grid(alpha=0.7)
st.pyplot()