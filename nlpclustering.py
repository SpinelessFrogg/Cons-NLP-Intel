import gc
import torch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

def cluster_similar_bills(df, distance_threshold=0.45):
    # creates nlp embeddings and clusters similar bills with cosine distance
    model = SentenceTransformer("all-MiniLM-L6-v2")
    texts = (df["title"] + ". " + df["description"]).tolist() # uses text and description as input for embedding
    embeddings = model.encode(texts, normalize_embeddings=True)

    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=distance_threshold
    )
    df["policy_cluster"] = clustering.fit_predict(embeddings)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return df