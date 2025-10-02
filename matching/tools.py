"""
Author: Simon Strycek
Date: 2025-09-12
Description: Tools for car matching.
"""

import numpy as np
import faiss
import math


def find_matches(
        query: np.ndarray,
        db: np.ndarray | faiss.IndexIVFFlat,
        number_of_clusters: int = None,
        move_to_gpu: bool = True
    ):
    """
    Finds the best matches between a query and a database. If the database is a
    faiss.IndexIVFFlat, it will be used directly. Otherwise, a new index will be
    created with number of clusters equal to the square root of the number of
    database features.

    Args:
        query: The query features.
        db: The database features or a faiss.IndexIVFFlat.
        batch_size: The batch size for the index.
        move_to_gpu: Whether to move the index to the GPU.

    Returns:
        The similarities and indices of the best matches.
    """
    if isinstance(db, faiss.IndexIVFFlat):
        emb_size = db.d
        index = db
    elif isinstance(db, np.ndarray):
        nlist = int(math.sqrt(len(db))) if number_of_clusters is None else number_of_clusters
        emb_size = db.shape[1]

        quantizer = faiss.IndexFlatIP(emb_size)
        index = faiss.IndexIVFFlat(quantizer, emb_size, nlist, faiss.METRIC_INNER_PRODUCT)
        
        db = np.ascontiguousarray(db.astype(np.float32, copy=False))
        index.train(db)
        index.add(db)

    if query.shape[1] != emb_size:
        raise ValueError(f"The number of features in query ({query.shape[1]}) and db ({emb_size}) must match.")

    index.nprobe = 10 # empirically chosen

    if move_to_gpu:
        index = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(), device=0, index=index
        )

    query = np.ascontiguousarray(query.astype(np.float32, copy=False))
    similarities, indices = index.search(query, 1)  # TOP1 per query

    return similarities.flatten().tolist(), indices.flatten().tolist()