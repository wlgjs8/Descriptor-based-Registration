import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.neighbors import NearestNeighbors

import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.metrics.pairwise import cosine_similarity

def find_most_similar_features(ct_features, scan_features):
    N, _ = ct_features.shape
    most_similar_indices = np.zeros(N, dtype=int)

    for i in range(N):
        # Compute the cosine similarity between the i-th scan feature and all ct features
        similarities = cosine_similarity(scan_features[i].reshape(1, -1), ct_features.reshape(N, -1))
        
        # Find the index of the most similar ct feature
        most_similar_indices[i] = np.argmax(similarities)

    return most_similar_indices

def compute_rigid_transform(ct_coords, scan_coords):
    # Compute the centroids of the ct and scan coordinates
    centroid_ct = np.mean(ct_coords, axis=0)
    centroid_scan = np.mean(scan_coords, axis=0)

    # Center the ct and scan coordinates
    ct_centered = ct_coords - centroid_ct
    scan_centered = scan_coords - centroid_scan

    # Compute the rotation matrix using SVD
    H = np.dot(scan_centered.T, ct_centered)
    U, S, Vt = np.linalg.svd(H)
    rotation = np.dot(Vt.T, U.T)

    # Ensure a right-handed coordinate system
    if np.linalg.det(rotation) < 0:
        Vt[-1,:] *= -1
        rotation = np.dot(Vt.T, U.T)

    # Compute the translation vector
    translation = centroid_ct - np.dot(centroid_scan, rotation)

    # Combine the rotation and translation into a 4x4 transformation matrix
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation

    return transform

def align_features(ct_features, scan_features, ct_coords, scan_coords):
    # Find the most similar ct feature for each scan feature
    most_similar_indices = find_most_similar_features(ct_features, scan_features)

    # Gather the corresponding ct and scan coordinates
    ct_coords_matched = ct_coords[most_similar_indices]
    scan_coords_matched = scan_coords

    # Compute the rigid transform from the scan coordinates to the ct coordinates
    transform = compute_rigid_transform(ct_coords_matched, scan_coords_matched)

    return transform

ct_features = np.random.rand(1200, 32)
scan_features = np.random.rand(1200, 32)

ct_coords = np.random.rand(1200, 3)
scan_coords = np.random.rand(1200, 3)

T = align_features(ct_features, scan_features, ct_coords, scan_coords)
print(T)