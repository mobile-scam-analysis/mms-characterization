import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import unicodedata
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from umap import UMAP
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import spacy

# Setup
OUTPUT_DIR = "outputs_kmeans"
os.makedirs(OUTPUT_DIR, exist_ok=True)
tqdm.pandas()
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

# Define text processing dictionaries
CONTRACTIONS = {
    "ain't": "am not", "don't": "do not", "aren't": "are not", "can't": "cannot", "cant": "cannot",
    "can't've": "cannot have", "he'd": "he would", "he'd've": "he would have", "he'll": "he will",
    "he'll've": "he will have", "he's": "he is", "I'd": "I would", "mightn't": "might not",
    "must've": "must have", "mustn't": "must not", "needn't": "need not", "o'clock": "of the clock",
    "shan't": "shall not", "she'd": "she would", "she'll": "she will", "they're": "they are",
    "they've": "they have", "wasn't": "was not", "we'd": "we would", "we're": "we are",
    "we've": "we have", "weren't": "were not", "wouldn't": "would not", "you'd": "you would",
    "you'll": "you will", "you're": "you are", "you've": "you have", "lol": "laughing out loud",
    "lmao": "laughing my ass off", "rofl": "rolling on the floor laughing", "omg": "oh my god",
    "wtf": "what the heck", "idk": "I do not know", "ikr": "I know right", "imo": "in my opinion",
    "imho": "in my humble opinion", "fyi": "for your information", "brb": "be right back",
    "gtg": "got to go", "g2g": "got to go", "ttyl": "talk to you later", "bbl": "be back later",
    "np": "no problem", "jk": "just kidding", "atm": "at the moment", "asap": "as soon as possible",
    "bc": "because", "cuz": "because", "gr8": "great", "l8r": "later", "kk": "okay", "ok": "okay",
    "k": "okay", "msg": "message", "txt": "text",
}

SLANG = {
    "u": "you", "ur": "your", "urs": "yours", "r": "are", "y": "why", "tho": "though",
    "pls": "please", "plz": "please", "thx": "thanks", "ty": "thank you", "yw": "you are welcome",
    "omw": "on my way", "dm": "direct message", "irl": "in real life", "afaik": "as far as I know",
    "ftw": "for the win", "lmk": "let me know"
}

contractions = {
    "dont": "do not", "can't": "cannot", "cant": "cannot", "wont": "will not", "won't": "will not",
    "im": "i am", "i'm": "i am", "ive": "i have", "i've": "i have", "id": "i would", "i'd": "i would",
    "youre": "you are", "you're": "you are", "hes": "he is", "he's": "he is", "shes": "she is", "she's": "she is",
    "isnt": "is not", "isn't": "is not", "arent": "are not", "aren't": "are not", "wasnt": "was not", "wasn't": "was not",
    "werent": "were not", "weren't": "were not", "didnt": "did not", "didn't": "did not", "hasnt": "has not",
    "hasn't": "has not", "havent": "have not", "haven't": "have not", "couldnt": "could not", "couldn't": "could not",
    "shouldnt": "should not", "shouldn't": "should not", "wouldnt": "would not", "wouldn't": "would not",
    "mustnt": "must not", "mustn't": "must not", "neednt": "need not", "needn't": "need not", "lets": "let us",
    "let's": "let us", "thats": "that is", "that's": "that is"
}

REMOVE = {
    "num", "like", "com", "pm", "http", "just", "el", "know", "need", "want", "oh", "email", "did",
    "eyes", "okay", "min", "eye", "yes", "let", "scam", "good", "red", "follow", "new", "make",
    "text", "mug", "verizon", "old", "ee", "ae", "oe", "eee", "mt", "es", "nd", "bn", "spam",
    "instagram", "message", "ce", "se", "oo", "fr", "ea", "en", "aa", "regret", "ll"
}

def remove_numbers(text):
    return " ".join(w for w in text.split() if not re.fullmatch(r"\d+", w))

def normalize(text):
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    for patt, repl in CONTRACTIONS.items():
        text = re.sub(rf"\b{patt}\b", repl, text)
    for patt, repl in SLANG.items():
        text = re.sub(rf"\b{patt}\b", repl, text)
    text = re.sub(r"http\S+|www\S+", " url ", text)
    text = re.sub(r"\S+@\S+", " email ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def remove_terms(text):
    return " ".join(tok for tok in text.split() if tok not in REMOVE)

def lemmatize(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop])

CLEANED_FILE = "cleaned_text_no_single_chars.csv"
df = pd.read_csv("semantically_cleaned_messages.csv").rename(columns={"Semantically Cleaned Text": "message"})

if os.path.exists(CLEANED_FILE):
    print("Loading cached lemmatized text...")
    df = pd.read_csv(CLEANED_FILE)
else:
    print("Cleaning and lemmatizing text...")
    df["message"] = df["message"].astype(str)
    df["cleaned"] = (
        df["message"]
        .progress_apply(normalize)
        .progress_apply(remove_terms)
        .progress_apply(remove_numbers)
        .progress_apply(lemmatize)
    )
    df.to_csv(CLEANED_FILE, index=False)

df["cleaned"] = df["cleaned"].fillna("").astype(str)
df = df[df["cleaned"].apply(lambda x: isinstance(x, str))]

EMBEDDINGS_FILE = "sentence_embeddings.npy"
if os.path.exists(EMBEDDINGS_FILE):
    print("Loading cached embeddings file...")
    emb_np = np.load(EMBEDDINGS_FILE)
else:
    print("Generating embeddings file...")
    #model = SentenceTransformer("intfloat/e5-large-v2")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    emb_np = np.array(model.encode(df["cleaned"].tolist(), convert_to_tensor=False, show_progress_bar=True))
    np.save(EMBEDDINGS_FILE, emb_np)

subset_df = df.sample(n=1000, random_state=42)
subset_indices = subset_df.index
subset_np = emb_np[subset_indices]

param_grid = {
    'n_neighbors': [10, 15],
    'min_dist': [0.1],
    'spread': [1.0, 2.0],
    'n_components_umap': [10, 20],
    'n_components_pca': [50, 100],
    'apply_umap': [True, False]
}
grid = list(ParameterGrid(param_grid))
results = []

print("Running grid search (excluding n_clusters)...")
for params in tqdm(grid):
    try:
        # PCA
        pca = PCA(n_components=params['n_components_pca'], random_state=42)
        pca_result = pca.fit_transform(subset_np)

        # UMAP (optional)
        if params['apply_umap']:
            umap_model = UMAP(
                n_neighbors=params['n_neighbors'],
                min_dist=params['min_dist'],
                spread=params['spread'],
                n_components=params['n_components_umap'],
                metric='cosine',
                random_state=42
            )
            reduced = umap_model.fit_transform(pca_result)
        else:
            reduced = pca_result

        Ks = range(7, 21)
        best_k, best_score = None, -1
        for k in Ks:
            km = KMeans(n_clusters=k, random_state=42)
            labels = km.fit_predict(reduced)
            sil = silhouette_score(reduced, labels)
            if sil > best_score:
                best_score = sil
                best_k = k

        km = KMeans(n_clusters=best_k, random_state=42)
        labels = km.fit_predict(reduced)
        sil = silhouette_score(reduced, labels)
        cal = calinski_harabasz_score(reduced, labels)
        db = davies_bouldin_score(reduced, labels)

        results.append({
            "params": params,
            "best_k": best_k,
            "silhouette": sil,
            "calinski_harabasz": cal,
            "davies_bouldin": db
        })

    except Exception as e:
        print(f"Skipping config {params} due to error: {e}")

results_df = pd.DataFrame(results)
scaler = MinMaxScaler()
results_df["silhouette_norm"] = scaler.fit_transform(results_df[["silhouette"]])
results_df["calinski_harabasz_norm"] = scaler.fit_transform(results_df[["calinski_harabasz"]])
results_df["davies_bouldin_norm"] = scaler.fit_transform(results_df[["davies_bouldin"]])
results_df["composite_score"] = (
    results_df["silhouette_norm"] +
    results_df["calinski_harabasz_norm"] +
    (1 - results_df["davies_bouldin_norm"])
)

best_result = results_df.sort_values("composite_score", ascending=False).iloc[0]
best_params = best_result["params"]
best_k = best_result["best_k"]

print(f"\nBest Params: {best_params}")
print(f"Best K (from silhouette): {best_k}")
print(f"Composite Score: {best_result['composite_score']:.4f}")

pca = PCA(n_components=best_params['n_components_pca'], random_state=42)
pca_result = pca.fit_transform(emb_np)

if best_params['apply_umap']:
    umap_final = UMAP(
        n_neighbors=best_params['n_neighbors'],
        min_dist=best_params['min_dist'],
        spread=best_params['spread'],
        n_components=best_params['n_components_umap'],
        metric='cosine',
        random_state=42
    )
    low_dim = umap_final.fit_transform(pca_result)
else:
    low_dim = pca_result

kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(low_dim)
df["cluster"] = labels
df.to_csv("new_text_kmeans_pca_umap.csv", index=False)

print("Final Clustering Metrics:")
print("Silhouette Score:", silhouette_score(low_dim, labels))
print("Calinski-Harabasz Score:", calinski_harabasz_score(low_dim, labels))
print("Davies-Bouldin Score:", davies_bouldin_score(low_dim, labels))

viz_data = UMAP(n_neighbors=15, min_dist=0.5, spread=1.5, n_components=2, metric="cosine", random_state=42).fit_transform(pca_result)
plt.figure(figsize=(8, 6))
plt.scatter(viz_data[:, 0], viz_data[:, 1], c=labels, cmap="tab20", s=5, alpha=0.8)
plt.title(f"2D Clustering Visualization (k={best_k}, UMAP={best_params['apply_umap']})")
plt.tight_layout()
plt.savefig("cluster_plot.png", dpi=300)

with open(os.path.join(OUTPUT_DIR, "cluster_sizes.txt"), "w") as f:
    for cid in sorted(df["cluster"].unique()):
        size = (df["cluster"] == cid).sum()
        print(f"Cluster {cid} size: {size}")
        f.write(f"Cluster {cid} size: {size}\n")

with open(os.path.join(OUTPUT_DIR, "cluster_keywords.txt"), "w") as f:
    for cid in sorted(df["cluster"].unique()):
        texts = df[df["cluster"] == cid]["cleaned"].tolist()
        vec = TfidfVectorizer(max_features=30, ngram_range=(1, 2), stop_words="english", min_df=3, max_df=0.8)
        X = vec.fit_transform(texts)
        scores = X.sum(axis=0).A1
        feats = vec.get_feature_names_out()
        top10 = [f for _, f in sorted(zip(scores, feats), reverse=True)][:10]
        print(f"Cluster {cid} keywords: {top10}")
        f.write(f"Cluster {cid} keywords: {', '.join(top10)}\n")