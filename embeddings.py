"""Embedding model + ChromaDB access, shared by step 4 and the app."""
from functools import lru_cache

import chromadb

import config


@lru_cache(maxsize=1)
def get_embedder():
    """Load the Sentence Transformer once and reuse it."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed(texts: list[str]) -> list[list[float]]:
    """Turn a list of texts into normalized embedding vectors."""
    vectors = get_embedder().encode(
        texts, batch_size=64, normalize_embeddings=True, show_progress_bar=len(texts) > 64
    )
    return vectors.tolist()


def get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection(create: bool = False):
    """Return the ChromaDB collection (cosine distance)."""
    client = get_client()
    if create:
        try:
            client.delete_collection(config.COLLECTION_NAME)
        except Exception:  # collection did not exist yet
            pass
        return client.create_collection(
            config.COLLECTION_NAME,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": config.EMBEDDING_MODEL,
            },
        )
    collection = client.get_collection(config.COLLECTION_NAME)
    indexed_model = (collection.metadata or {}).get("embedding_model")
    if indexed_model != config.EMBEDDING_MODEL:
        raise RuntimeError(
            "The vector database was built with a different embedding model. "
            "Rebuild it with step4_build_vector_db.py."
        )
    return collection
