import torch
import torch.nn.functional as F
from torch.autograd import Variable

# 입력 데이터 설정
num_points = 12000
scan_points = torch.randn(num_points, 3)  # Scan의 x, y, z 좌표
scan_features = torch.randn(num_points, 32)  # Scan의 Feature

ct_points = torch.randn(num_points, 3)  # CT의 x, y, z 좌표
ct_features = torch.randn(num_points, 32)  # CT의 Feature

# Transformation 변수 초기화
translation = Variable(torch.zeros(1, 3), requires_grad=True)  # 이동 변환 (x, y, z)
rotation = Variable(torch.eye(3), requires_grad=True)  # 회전 변환 (3x3)

# 최적화 설정
optimizer = torch.optim.Adam([translation, rotation], lr=0.1)

# 최적화 반복
num_iterations = 100
for iteration in range(num_iterations):
    optimizer.zero_grad()

    # Scan과 CT의 좌표 변환
    transformed_scan_points = scan_points + translation
    transformed_scan_points = torch.matmul(transformed_scan_points, rotation)

    # Scan과 CT의 Feature 간 유사도 계산
    feature_distances = torch.cdist(scan_features, ct_features)  # Feature 간 거리 계산
    closest_indices = torch.argmin(feature_distances, dim=1)  # 가장 가까운 CT Feature의 인덱스 찾기
    closest_ct_points = ct_points[closest_indices]

    # Energy Function 계산
    point_distances = torch.norm(transformed_scan_points - closest_ct_points, dim=1)  # 좌표 간 거리 계산
    energy = torch.mean(point_distances)

    # 역전파 및 가중치 업데이트
    energy.backward()
    optimizer.step()

    # 로그 출력
    if (iteration + 1) % 10 == 0:
        print(f"Iteration [{iteration+1}/{num_iterations}]: Energy = {energy.item()}")

# 최적화된 결과 출력
print("최적화된 Transformation:")
print("Translation:", translation.data)
print("Rotation:", rotation.data)