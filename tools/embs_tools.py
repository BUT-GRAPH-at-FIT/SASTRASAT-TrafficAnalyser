import csv
import logging
import os

import cv2
import h5py
import torch
from tqdm import tqdm


def get_file_embs(file_path, emb_name="features", identifier_name="track_ids"):
    with h5py.File(file_path, 'r') as f:
        if emb_name not in f:
            raise ValueError("{} not found in {}".format(emb_name, file_path))

        if identifier_name not in f:
            raise ValueError("{} not found in {}".format(identifier_name, file_path))

        embeddings = f[emb_name][:]
        identifiers = f[identifier_name][:]

        return list(zip(identifiers, embeddings))


def get_detection_to_track_map(file_path):
    detection_to_track_map = {}

    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            detection_id = int(row['record_id'])
            track_id = int(row['track_id'])
            detection_to_track_map[detection_id] = track_id

    return detection_to_track_map


def get_detection_to_bb_size_map(file_path):
    detection_to_bb_size_map = {}

    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)

        if not 'bb_size' in reader.fieldnames:
            return {}

        for row in reader:
            detection_id = int(row['record_id'])
            bb_size = tuple(map(int, row['bb_size'].strip('()').split(',')))
            detection_to_bb_size_map[detection_id] = bb_size

    return detection_to_bb_size_map


def get_embs(dir_path, meta_file_name="track_meta.csv", emb_name="features", identifier_name="track_ids"):

    embs = {}
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.h5'):
                try:
                    meta_file = os.path.join(root, meta_file_name)
                    track_map = get_detection_to_track_map(meta_file)
                    bb_size_map = get_detection_to_bb_size_map(meta_file)

                    if len(track_map) == 0:
                        logging.error(f"No meta file found in {meta_file}, skipping...")
                        continue

                    file_path = os.path.join(root, file)

                    embs[root] = [
                        (
                            track_map[id],
                            bb_size_map[id] if len(bb_size_map) > 0 else None,
                            emb
                        )
                        for id, emb in get_file_embs(file_path, emb_name, identifier_name)
                    ]
                except ValueError as e:
                    logging.error(f"Error while reading file {file_path}: {e}")
    return embs


def accelerated_cosine_similarity(matrix_A, matrix_B, batch_size=512, device='cuda'):
    M, E = matrix_A.shape
    N, E_B = matrix_B.shape

    if E != E_B:
        raise ValueError(f"The number of features in A ({E}) and B ({E_B} must match.")

    result = torch.empty((M, N), dtype=matrix_A.dtype)

    for i in tqdm(range(0, M, batch_size), desc="Computing cosine similarity"):
        a_batch = matrix_A[i:i+batch_size].to(device)
        a_batch = torch.nn.functional.normalize(a_batch, p=2, dim=1)

        for j in range(0, N, batch_size):
            b_batch = matrix_B[j:j+batch_size].to(device)
            b_batch = torch.nn.functional.normalize(b_batch, p=2, dim=1)

            sim = a_batch @ b_batch.T

            result[i:i+a_batch.size(0), j:j+b_batch.size(0)] = sim.cpu()

    return result


def aggregate_embeddings(embeddings, aggregation_fn):
    aggregated_embeddings = {}

    for track_id, bb_size, emb in tqdm(embeddings, desc="Aggregating embeddings"):
        if track_id not in aggregated_embeddings:
            aggregated_embeddings[track_id] = []

        aggregated_embeddings[track_id].append((bb_size, emb))

    averaged_embeddings = [
        (track_id, (aggregation_fn(embs) if len(embs) > 1 else embs[0][1]))
        for track_id, embs in aggregated_embeddings.items()
    ]

    return averaged_embeddings


def _bb_weights(bb_size):
    bb_widhts = torch.tensor([w for w, _ in bb_size])
    bb_heights = torch.tensor([h for _, h in bb_size])
    bb_widhts = bb_widhts / bb_widhts.max()
    bb_heights = bb_heights / bb_heights.max()
    weights = bb_widhts + bb_heights
    weights /= weights.sum()

    return weights


def bb_weighted_average(embeddings):
    bb_size = [(w, h) for (w, h), _ in embeddings]
    embeddings = [torch.tensor(emb) for _, emb in embeddings]

    if any(size is None for size in bb_size):
        logging.warning(
            f"Records have missing bounding box sizes, using simple average for aggregation."
        )
        return torch.mean(embeddings, dim=0)

    weights = _bb_weights(bb_size)

    return (torch.stack(embeddings) * weights.unsqueeze(0).T).sum(dim=0)


def bb_greedy(embeddings):
    bb_size = [(w, h) for (w, h), _ in embeddings]
    embeddings = [torch.tensor(emb) for _, emb in embeddings]

    if any(size is None for size in bb_size):
        logging.warning(
            f"Records have missing bounding box sizes, using simple average for aggregation."
        )
        return torch.mean(embeddings, dim=0)

    weights = _bb_weights(bb_size)

    return embeddings[torch.argmax(weights)]


def get_crops_for_id(root_path, track_id, meta_file_name="track_meta.csv"):
    crops = []

    with open(os.path.join(root_path, meta_file_name), 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Searching for the right crops"):
            if int(row['track_id']) == int(track_id):
                crop_file_name = row['crop_path']
                crop_path = os.path.join(root_path, "vehicle_crops", crop_file_name)

                if os.path.exists(crop_path):
                    crops.append(cv2.imread(crop_path))
                else:
                    logging.warning(f"Crop file {crop_path} does not exist.")

    return crops
