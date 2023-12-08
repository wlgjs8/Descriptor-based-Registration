import os
import torch
import trimesh
import numpy as np
import nibabel as nib
from sklearn.linear_model import RANSACRegressor
from sklearn.preprocessing import StandardScaler
from skimage import filters
import sys

sys.path.append('../')
sys.path.append('.')

import gen_utils as gu

return_matrix = np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])

def __get_pt_and_gradient_rotation__(affine_coords, otsu_image, pixel_spacing=0.2, slice_spacing=0.2, ct_resize_shape=(128, 128, 128)):
    ct_original_resize_pair = []

    original_shape = (224, 224, 224)
    d_affine_coord, h_affine_coord, w_affine_coord = affine_coords

    voxel_d = 128
    voxel_h = 128
    voxel_w = 128

    check_array = np.zeros((128, 128, 128))

    # for vd in range(10, 60):
    #     for vh in range(10, 100):
    #         for vw in range(10, 100):
    for vd in range(0, 90):
        for vh in range(0, 110):
            for vw in range(0, 110):
                
                # scan_d, scan_x, scan_y = scan_image[i]
                ct_d, ct_h, ct_w = vd, vh, vw

                spacing_d = ct_d * (1 / slice_spacing)
                spacing_w = ct_w * (1 / pixel_spacing)
                spacing_h = ct_h * (1 / pixel_spacing)

                spacing_d = int(np.round(spacing_d))
                spacing_h = int(np.round(spacing_h))
                spacing_w = int(np.round(spacing_w))

                if otsu_image[spacing_w][spacing_h][spacing_d] == 0:
                    continue

                affine_h = h_affine_coord[spacing_w][spacing_h][spacing_d]
                affine_w = w_affine_coord[spacing_w][spacing_h][spacing_d]
                affine_d = d_affine_coord[spacing_w][spacing_h][spacing_d]

                affine_h = int(np.round(affine_h))
                affine_w = int(np.round(affine_w))
                affine_d = int(np.round(affine_d))

                # print('original {} {} {} -> resize {} {} {}'.format(spacing_d, spacing_h, spacing_w, affine_d, affine_h, affine_w))

                resized_h = affine_h * (ct_resize_shape[1] / original_shape[0])
                resized_w = affine_w * (ct_resize_shape[2] / original_shape[1])
                resized_d = affine_d * (ct_resize_shape[0] / original_shape[2])

                resized_d = int(np.round(resized_d))
                resized_h = int(np.round(resized_h))
                resized_w = int(np.round(resized_w))

                # matched_pair_gradient.append([resized_w, resized_h, 128-resized_d, gradient_image[spacing_w, spacing_h, spacing_d]])

                resized_d = 128 - resized_d

                if (resized_d < 5) or (resized_d > 123):
                    continue
                if (resized_h < 5) or (resized_h > 123):
                    continue
                if (resized_w < 5) or (resized_w > 123):
                    continue

                
                if check_array[resized_d, resized_h, resized_w] == 1:
                    continue
                check_array[resized_d, resized_h, resized_w] = 1

                # ct_original_resize_pair.append([
                #     resized_d, resized_h, resized_w, ct_d, ct_h, ct_w
                # ])
                ct_original_resize_pair.append([
                    ct_d, ct_h, ct_w
                ])

                # print('mm : ', spacing_d, spacing_h, spacing_w)
                # print('reshape : ', affine_d, affine_h, affine_w)
                # print()
                # if flag == True:
                #     if gradient_image[spacing_w, spacing_h, spacing_d] > 0.5:
                #         check_volume[resized_d][resized_h][resized_w] += 1

    ct_original_resize_pair = np.array(ct_original_resize_pair)
    # utils._check(check_volume)

    return ct_original_resize_pair

# 입력 데이터 설정
# num_points = 12000
# scan_points = torch.randn(num_points, 3)  # Scan의 x, y, z 좌표
# scan_features = torch.randn(num_points, 32)  # Scan의 Feature

# ct_points = torch.randn(num_points, 3)  # CT의 x, y, z 좌표
# ct_features = torch.randn(num_points, 32)  # CT의 Feature

CUR_DIR = os.getcwd()

stl_file_path = os.path.join(CUR_DIR, 'datasets/lower_registration/LOWER_Result_sota_augment.stl')
original_scan_image = trimesh.load_mesh(stl_file_path)

scan_normals = original_scan_image.vertex_normals
curvatures = np.linalg.norm(np.gradient(scan_normals, axis=0), axis=1)
mean_curvature = np.mean(curvatures)
high_curvature_indices = np.array(curvatures > mean_curvature)
high_curvature_vertices = original_scan_image.vertices[high_curvature_indices]

SAMPLE_CNT = 6000
# scan_image, idx = gu.resample_pcd_with_idx([high_curvature_vertices], SAMPLE_CNT, "fps")
scan_image = gu.resample_pcd([high_curvature_vertices], SAMPLE_CNT, "fps")

scan_points = np.array(scan_image)[0]
print('scan_points : ', scan_points.shape)


DATA_DIR = os.path.abspath('/home/jeeheon/Documents/point-transformer/datasets')

case_idx, flag = 'Case_18', 'lower'
d_affine_coord_path = os.path.join(DATA_DIR, case_idx, 'd_{}_label.nii.gz'.format(flag))
h_affine_coord_path = os.path.join(DATA_DIR, case_idx, 'h_{}_label.nii.gz'.format(flag))
w_affine_coord_path = os.path.join(DATA_DIR, case_idx, 'w_{}_label.nii.gz'.format(flag))

d_affine_coord = nib.load(d_affine_coord_path)
h_affine_coord = nib.load(h_affine_coord_path)
w_affine_coord = nib.load(w_affine_coord_path)
d_affine_coord = d_affine_coord.get_fdata()
h_affine_coord = h_affine_coord.get_fdata()
w_affine_coord = w_affine_coord.get_fdata()

d_affine_coord = np.transpose(d_affine_coord, (2, 1, 0))
h_affine_coord = np.transpose(h_affine_coord, (2, 1, 0))
w_affine_coord = np.transpose(w_affine_coord, (2, 1, 0))
d_affine_coord = np.flip(d_affine_coord, 2) 
h_affine_coord = np.flip(h_affine_coord, 2) 
w_affine_coord = np.flip(w_affine_coord, 2)

otsu_path = os.path.join(DATA_DIR, case_idx, '{}_original_dcm_arr.nii.gz'.format(case_idx))
otsu_image = nib.load(otsu_path)
otsu_image = otsu_image.get_fdata()
otsu_image = np.transpose(otsu_image, (2, 1, 0))
otsu_image = np.flip(otsu_image, 2)

otsu_ct_image = otsu_image

for i in range(1):
    otsu_threshold = filters.threshold_otsu(otsu_ct_image[otsu_ct_image>0])
    otsu_ct_index = otsu_ct_image > otsu_threshold

    temp_arr = np.zeros(otsu_image.shape)
    temp_arr[otsu_ct_index == True] = otsu_image[otsu_ct_index == True]
    otsu_ct_image = temp_arr

otsu_image = otsu_ct_image

# nii_otsu_image = nib.Nifti1Image(otsu_image, affine=np.eye(4))
nii_otsu_image = nib.Nifti1Image(otsu_image, affine=return_matrix)
nib.save(nii_otsu_image, os.path.join(DATA_DIR, case_idx, '{}_otsu.nii.gz'.format(case_idx)))


matched_pair_gradient = __get_pt_and_gradient_rotation__([d_affine_coord, h_affine_coord, w_affine_coord], otsu_image)


ct_points = matched_pair_gradient
print('ct_points : ', ct_points.shape)

print('scan_points : ', scan_points)
print('ct_points : ', ct_points)

restored_mesh = trimesh.Trimesh(vertices=matched_pair_gradient,
    )
restored_mesh.export('datasets/Case_18/Case_18_ctpoint.ply')



ransac = RANSACRegressor().fit(scan_points, ct_points)

# 최적의 Transformation 출력
print("최적의 Transformation:")
print("Estimator coefficients (rotation):", ransac.estimator_.coef_)
print("Estimator intercept (translation):", ransac.estimator_.intercept_)




stl_file_path = os.path.join(CUR_DIR, 'datasets/lower_registration/LOWER_Result_sota_augment.stl')
scan_image = trimesh.load_mesh(stl_file_path)

transform_matrix = ransac.estimator_.coef_
U, S, Vt = np.linalg.svd(transform_matrix)
rotation_mat = np.matmul(U, Vt)

translation_mat = ransac.estimator_.intercept_


predict_mesh = np.matmul(scan_image.vertices, rotation_mat)
# predict_mesh = np.matmul(scan_image.vertices, np.linalg.inv(rotation_mat).T)
# predict_mesh = np.add(scan_image.vertices, translation_mat)
# predict_mesh = np.add(predict_mesh, translation_mat)
predict_mesh = predict_mesh + translation_mat


# scan_image.vertices = predict_mesh + trans_bef
scan_image.vertices = predict_mesh

# scan_image.vertices = predict_mesh
scan_image.export('datasets/lower_registration/only_ransac.stl')