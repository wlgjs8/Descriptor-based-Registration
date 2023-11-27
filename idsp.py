import numpy as np
from scipy.optimize import minimize
from sklearn.metrics.pairwise import cosine_similarity

def find_most_similar_features(ct_features, scan_features_transformed):
    N = ct_features.shape[0]
    most_similar_indices = np.zeros(N, dtype=int)

    for i in range(N):
        # Compute the cosine similarity between the i-th transformed scan feature and all ct features
        similarities = cosine_similarity(scan_features_transformed[i].reshape(1, -1), ct_features.reshape(N, -1))
        
        # Find the index of the most similar ct feature
        most_similar_indices[i] = np.argmax(similarities)

    return most_similar_indices

def objective_function(transform, ct_features, scan_features, ct_coords, scan_coords):
    # Reshape the transform into a 4x4 matrix
    transform = transform.reshape((4, 4))
    
    # Apply the transform to the scan features
    scan_coords_transformed = apply_transform(scan_coords, transform)
    
    # Find the most similar ct feature for each transformed scan feature
    most_similar_indices = find_most_similar_features(ct_features, scan_features)

    # Gather the corresponding ct and scan coordinates
    ct_coords_matched = ct_coords[most_similar_indices]
    scan_coords_matched = scan_coords_transformed
    
    # Compute the distance between the matched coordinates
    dist = np.linalg.norm(ct_coords_matched - scan_coords_matched, axis=1).sum()
    print('CUR DISTANCE : ', dist)

    return dist

# def apply_transform(features, transform):

#     print('transform : ', transform)
#     # Apply the 4x4 transform to the features
#     # This implementation depends on the specifics of your features and transform
    
#     # return transformed_features

#     transformed_features = np.dot(vertices, transform_matrix[:3, :3].T) + transform_matrix[:3, 3]
#     return features

def apply_transform(coords, transform):
    print('transform : ', transform)

    # Convert the 3D coordinates to 4D homogeneous coordinates
    homogeneous_coords = np.hstack([coords, np.ones((coords.shape[0], 1))])
    
    # Apply the 4x4 transform
    transformed_homogeneous_coords = np.dot(homogeneous_coords, transform.T)
    
    # Convert the 4D homogeneous coordinates back to 3D coordinates
    transformed_coords = transformed_homogeneous_coords[:, :3] / transformed_homogeneous_coords[:, 3, np.newaxis]
    
    return transformed_coords

# Initial guess for the transform
initial_guess = np.eye(4).flatten()

ct_features = np.random.rand(12000, 32)
scan_features = np.random.rand(12000, 32)

ct_coords = np.random.rand(12000, 3)
scan_coords = np.random.rand(12000, 3)


'''

torch tensor 로 구성된 12000 개의 Point 에 대해서 CT 와 Scan 각각에서 x, y, z 좌표와 Feature 가 각각 3차원과 32차원으로 있어.
Iterative Closest Point 알고리즘과 유사하게, Scan 의 점을 CT의 점으로 Transform 하는 최적의 Transformation 을 구하려고 해.
최적화해야하는 Energy Function 은 ICP와는 다르게 Scan의 Feature와 가장 유사한 CT Feature 를 찾고, 그 점으로 Transform할 수 있는 Transformation 을 
반복적으로 최적화하는 코드를 작성해줘. torch 라이브러리의 tensor를 사용해줘.!
ct_coords :  (12000, 3)
scan_coords :  (12000, 3)
ct_features :  (12000, 32)
scan_features :  (12000, 32)
'''


print('ct_coords : ', ct_coords.shape)
print('scan_coords : ', scan_coords.shape)
print('ct_features : ', ct_features.shape)
print('scan_features : ', scan_features.shape)

import torch
ct_features = torch.tensor(ct_features).cuda()
scan_features = torch.tensor(scan_features).cuda()
ct_coords = torch.tensor(ct_coords).cuda()
scan_coords = torch.tensor(scan_coords).cuda()

# Run the optimization
res = minimize(objective_function, initial_guess, args=(ct_features, scan_features, ct_coords, scan_coords), method='Nelder-Mead')

if res.success:
    print("Optimization converged successfully.")
else:
    print("Optimization did not converge.")
    print("Reason:", res.message)

# Reshape the optimized transform into a 4x4 matrix
optimized_transform = res.x.reshape((4, 4))

print('optimized_transform : ', optimized_transform)