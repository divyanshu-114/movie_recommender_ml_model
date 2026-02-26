# 🎬 Movie Recommendation System

A content-based movie recommender built with **Python**, **scikit-learn**, and **Streamlit**. Select any movie and instantly get 5 similar recommendations powered by cosine similarity on movie metadata.

---

## 🚀 Demo

Run the app locally and interact via a clean Streamlit UI:

![App Screenshot](https://via.placeholder.com/800x400?text=Movie+Recommender+UI)

---

## 🧠 How It Works

The recommender uses a **content-based filtering** approach:

1. **Data**: Uses the [TMDB 5000 Movie Dataset](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata) (movies + credits CSVs).
2. **Feature Engineering**: Combines `overview`, `genres`, `keywords`, top-3 `cast` members, and `director` into a single `tags` string per movie.
3. **Vectorization**: Applies `CountVectorizer` (top 5000 features, English stop words removed) to convert tags into vectors.
4. **Similarity**: Computes **cosine similarity** across all movie vectors and stores the matrix.
5. **Inference**: Given a selected movie, returns the top 5 closest movies by cosine distance (excluding itself).

---

## 📁 Project Structure

```
movie-recommender/
├── app.py              # Streamlit web app
├── preprocess.py       # Data cleaning, feature engineering & model training
├── requirements.txt    # Python dependencies
├── data/
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
└── model/
    ├── movies.pkl      # Preprocessed movie DataFrame
    └── similarity.pkl  # Cosine similarity matrix
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender
```

### 2. Create a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add the dataset
Download the TMDB 5000 dataset from Kaggle and place both CSVs in the `data/` folder:
- `data/tmdb_5000_movies.csv`
- `data/tmdb_5000_credits.csv`

### 5. Train the model
```bash
python preprocess.py
```
This generates `model/movies.pkl` and `model/similarity.pkl`.

### 6. Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| Pandas | Data loading & manipulation |
| scikit-learn | CountVectorizer + Cosine Similarity |
| Streamlit | Interactive web UI |
| Pickle | Model serialization |

---

## 📦 Dependencies

```
pandas
numpy
scikit-learn
streamlit
nltk
```

---

## 📌 Notes

- The `similarity.pkl` file is large (~176 MB) — consider adding it to `.gitignore` and regenerating locally via `preprocess.py`.
- Only the **top 3 cast members** and the **director** are used for each movie to keep tag noise low.
- Spaces within multi-word names (e.g. `Sam Mendes`) are removed before vectorization to treat them as single tokens.

---

## 📄 License

This project is open-source and available under the [MIT License](LICENSE).
