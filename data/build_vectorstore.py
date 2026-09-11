"""
build_vectorstore.py
--------------------
Embeds all historical Spotify brand replies into a FAISS vector store.
This store is later used by the retriever for RAG-based reply drafting.

Usage:
    python data/build_vectorstore.py

Outputs:
    data/vectorstore/index.faiss   – FAISS flat L2 index
    data/vectorstore/metadata.json – List of {"customer_text", "brand_reply"}
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

DATA_DIR = Path(__file__).parent
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
SUBSET_CSV = DATA_DIR / "spotify_subset.csv"


def load_model():
    """Load sentence-transformer embedding model."""
    from sentence_transformers import SentenceTransformer
    print("Loading embedding model (all-MiniLM-L6-v2)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Model loaded.")
    return model


def embed_texts(model, texts: list[str]) -> np.ndarray:
    """Embed a list of texts, showing progress."""
    print(f"Embedding {len(texts):,} texts...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,  # cosine similarity via dot product
    )
    return embeddings


def build_faiss_index(embeddings: np.ndarray):
    """Build a FAISS flat inner-product index (cosine similarity)."""
    import faiss
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product = cosine for normalized vecs
    index.add(embeddings.astype("float32"))
    print(f"  FAISS index built: {index.ntotal:,} vectors, dim={dim}")
    return index


def main():
    print("=" * 60)
    print("Building FAISS Vector Store")
    print("=" * 60)

    # Ensure data exists
    if not SUBSET_CSV.exists():
        print(f"ERROR: {SUBSET_CSV} not found.")
        print("Run:  python data/download_data.py  first.")
        raise SystemExit(1)

    df = pd.read_csv(SUBSET_CSV)
    print(f"Loaded {len(df):,} conversation pairs from {SUBSET_CSV.name}")

    # Drop rows with missing text
    df = df.dropna(subset=["customer_text", "brand_reply"])
    df["customer_text"] = df["customer_text"].astype(str).str.strip()
    df["brand_reply"] = df["brand_reply"].astype(str).str.strip()
    df = df[df["customer_text"].str.len() > 5]
    import argparse
    parser = argparse.ArgumentParser(description="Build FAISS vector store")
    parser.add_argument("--max-pairs", type=int, default=10000, help="Max pairs to embed (default 10,000)")
    args = parser.parse_args()

    if args.max_pairs and len(df) > args.max_pairs:
        print(f"  Subsampling top {args.max_pairs:,} pairs for efficient indexing...")
        df = df.sample(n=args.max_pairs, random_state=42).reset_index(drop=True)

    model = load_model()

    # We embed customer messages — retrieval is: given new customer msg,
    # find the most similar historical customer msg, return its brand reply.
    embeddings = embed_texts(model, df["customer_text"].tolist())

    index = build_faiss_index(embeddings)

    # Save
    import faiss
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(VECTORSTORE_DIR / "index.faiss"))
    print(f"[INFO] Saved FAISS index to {VECTORSTORE_DIR / 'index.faiss'}")

    metadata = df[["customer_text", "brand_reply"]].to_dict(orient="records")
    with open(VECTORSTORE_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved metadata to {VECTORSTORE_DIR / 'metadata.json'}")
    print(f"\nVector store ready with {len(metadata):,} entries.")


if __name__ == "__main__":
    main()
