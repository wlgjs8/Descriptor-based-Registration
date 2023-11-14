import os
import numpy as np
import nibabel as nib
import torch
from einops import rearrange

data_sample = np.random.choice(110829, 12000, replace=False)
neg_sample = np.random.choice(12000, 1000, replace=False)

np.save('data_sample.npy', data_sample)
np.save('neg_sample.npy', neg_sample)


# ct_features = torch.rand(128, 128, 128, 32).cuda()
# scan_features = torch.rand(12000, 32).cuda()
# # matched_pairs = torch.rand(24000, 3).cuda()
# matched_pairs = torch.ones(12000, 3).cuda().to(int)

# # 결과를 저장할 빈 리스트 생성
# result = []

# # 각 scan feature에 대한 유사도 계산 및 순위 추출
# flat_ct_features = rearrange(ct_features, 'd h w c -> (d h w) c')
# for i in range(scan_features.size(0)):
#     # 현재 scan_feature 선택
#     current_scan_feature = scan_features[i].unsqueeze(0)

#     cosine_similarities = torch.nn.functional.cosine_similarity(current_scan_feature, flat_ct_features)

#     '''
#     current_scan_feature :  torch.Size([1, 32])
#     ct_features :  torch.Size([128, 128, 128, 32])
#     cosine_similarities :  torch.Size([128, 128, 32])
#     '''

#     # 유사도를 기반으로 순위 계산
#     _, rank = cosine_similarities.sort(descending=True)

#     target_coordinate = (64, 64, 64)
#     index = (target_coordinate[0] * 128 * 128 + target_coordinate[1] * 128 + target_coordinate[2])

#     matched_rank = rank[index].cpu()
#     print('matched_rank : ', matched_rank)
#     # 결과 리스트에 추가
#     result.append(matched_rank)

# # 결과를 하나의 텐서로 변환
# final_result = torch.cat(result, dim=0).view(-1, 1)

# # 결과 출력
# print(final_result)

'''
torch.tensor 로 정의된 (128, 128, 128, 32) 형태의 ct_features 와 (24000, 32) 형태의 scan_features 가 있어.
scan_features 의 24000의 feature에 대해서 각각 ct_features 의 어떤 좌표의 Voxel 이랑 매칭되어야 하는지 torch.tensor의 (24000, 3) 이란 matched_pairs 정의가 되어있어.

위의 상황에서, scan_features 들이 ct 매칭되어 있는 점 24000 과 얼마나 유사한 feature (32,) 를 갖는지 각 점들에 대한 유사도 rank 를 구하고 싶어.
유사도는 cosine similarity 로 구하고, output tensor 는 (24000, 1)의 형태로, 각 점에 대한 유사도 순위를 (i.e. 1, 2, 3) 으로 나타내는 python 코드를 작성해줘
'''