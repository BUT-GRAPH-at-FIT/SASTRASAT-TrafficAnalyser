import numpy as np
import faiss
import math

def find_matches(query: np.ndarray, db: np.ndarray | faiss.IndexIVFFlat, emb_size: int = 128, batch_size: int = 10, move_to_gpu: bool = True):
    if isinstance(db, faiss.IndexIVFFlat):
        index = db
    elif isinstance(db, np.ndarray):
        nlist = int(math.sqrt(len(db)))

        quantizer = faiss.IndexFlatIP(emb_size)
        index = faiss.IndexIVFFlat(quantizer, emb_size, nlist, faiss.METRIC_INNER_PRODUCT)

        index.train(db)
        index.add(db)
    else:
        raise ValueError("`db` must be either a numpy array or a `faiss.IndexIVFFlat`.")

    index.nprobe = 10 # empirically chosen

    if not move_to_gpu:
        index = faiss.index_cpu_to_gpu(
            faiss.StandardGpuResources(), device=0, index=index
        )

    similarities, indices = [], []

    n_queries = len(query)
    for idx, batch_start in enumerate(range(0, n_queries, batch_size)):
        batch_end = min(batch_start + batch_size, n_queries)
        sim, ind = index.search(query[batch_start:batch_end], 1) # find TOP1
        similarities.extend(sim.flatten().tolist())
        indices.extend(ind.flatten().tolist())

    return similarities, indices