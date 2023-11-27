import numpy as np
from sklearn.neighbors import NearestNeighbors

def nearest_neighbor(src, dst):
    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()

def best_fit_transform(A, B):
    assert A.shape == B.shape

    # 차원 얻기
    m = A.shape[1]

    # 각 점군 중심점 및 중심 편차 계산
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A    # A행렬 편차
    BB = B - centroid_B     # B행렬 편차

    # SVD이용한 회전 행렬 계산
    H = np.dot(AA.T, BB)    # 분산
    U, S, Vt = np.linalg.svd(H)   # SVD 계산
    R = np.dot(Vt.T, U.T)     # 회전 행렬

    # 반사된 경우 행렬 계산
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # 이동 행렬 계산
    t = centroid_B.T - np.dot(R, centroid_A.T)   # 

    # 동차 변환행렬 계산
    T = np.identity(m+1)
    T[:m, :m] = R # 회전 행렬
    T[:m, m] = t  # 이동 행렬

    return T, R, t

def iterative_closet_point(A, B, max_iterations=100, tolerance=1e-4):
    # src: numpy 형태 Nx32 행렬. 소스(Src) mD points
    # dst: numpy 형태 Nx32 행렬. 대상(Dst) mD points

    m = A.shape[1]

    src = np.ones((m+1, A.shape[0]))
    dst = np.ones((m+1, B.shape[0]))

    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # src 점군에 초기 자세 적용
    prev_error = 0

    for i in range(max_iterations):  # 정합될때까지 반복 계산
        # 소스와 목적 점군 간에 가장 근처 이웃점 탐색. 계산량이 많음. 
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # 소스 점군에서 대상 점군으로 정합 시 필요한 변환행렬 계산
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # 소스 점군에 변환행렬 적용해 좌표 갱신
        src = np.dot(T, src)

        # 에러값 계산
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:  # 허용치보다 에러 작으면 탈출
            print('Converge Iteration {}'.format(i+1))
            break
        prev_error = mean_error


    # 변환행렬 계산
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T

ct_features = np.random.rand(2400, 3)
scan_features = np.random.rand(2400, 3)

T = iterative_closet_point(ct_features, scan_features)
print(T.shape)