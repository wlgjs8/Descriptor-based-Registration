import os
import numpy as np
import torch
import nibabel as nib
import torch.nn as nn

from stl import mesh
from einops import rearrange
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def load_mesh(filename):
    mesh_data = mesh.Mesh.from_file(filename)
    vertices = mesh_data.vectors.reshape(-1, 3)
    unique_vertices, indices = np.unique(vertices, axis=0, return_index=True)
    return unique_vertices


def __get_pt_and_gradient224__(gradient_image, scan_image, affine_coords, pixel_spacing=0.2, slice_spacing=0.2, ct_resize_shape=(128, 128, 128)):
    matched_pair_gradient = []

    original_shape = (224, 224, 224)
    d_affine_coord, h_affine_coord, w_affine_coord = affine_coords
    out_of_range_list = []
    # check_volume = np.zeros((128, 128, 128))

    print('len(scan_image) : ', len(scan_image))

    for i in range(len(scan_image)):
        # scan_d, scan_x, scan_y = scan_image[i]
        scan_x, scan_y, scan_d = scan_image[i]
        spacing_d = scan_d * (1 / slice_spacing)
        spacing_w = scan_x * (1 / pixel_spacing)
        spacing_h = scan_y * (1 / pixel_spacing)

        spacing_d = int(np.round(spacing_d))
        spacing_h = int(np.round(spacing_h))
        spacing_w = int(np.round(spacing_w))

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
        matched_pair_gradient.append([
            resized_d, resized_h, resized_w, gradient_image[spacing_w, spacing_h, spacing_d],
            affine_d, affine_h, affine_w
        ])

        flag=True
        if (resized_d < 1) or (resized_d > 126):
            flag = False
        if (resized_h < 1) or (resized_h > 126):
            flag = False
        if (resized_w < 1) or (resized_w > 126):
            flag = False

        out_of_range_list.append(flag)
        # if flag == True:
        #     if gradient_image[spacing_w, spacing_h, spacing_d] > 0.5:
        #         check_volume[resized_d][resized_h][resized_w] += 1

    out_of_range_list = np.array(out_of_range_list)
    matched_pair_gradient = np.array(matched_pair_gradient).astype(int)
    # utils._check(check_volume)

    return matched_pair_gradient, out_of_range_list

def find_nearest_neighbor(single_scan_image, scan_image):
    # single_scan_image의 좌표

    print('single_scan_image : ', single_scan_image.shape)
    print('scan_image : ', scan_image.shape)
    single_scan_coords = single_scan_image.flatten()

    # scan_image의 좌표
    scan_coords = scan_image.reshape(-1, 3)

    # 각각의 좌표에서의 거리 계산
    distances = cdist([single_scan_coords], scan_coords)

    # 가장 가까운 좌표의 인덱스 찾기
    nearest_neighbor_index = np.argmin(distances)

    # 가장 가까운 좌표 반환
    nearest_neighbor_coords = scan_coords[nearest_neighbor_index]

    return nearest_neighbor_coords

def pred_correspondence_with_gt_top1_diff():
    case_idx, flag = 'Case_18', 'lower'
    print('Case : {}, Flag : {}'.format(case_idx, flag))
    DATA_DIR = os.path.abspath('/home/jeeheon/Documents/point-transformer/datasets')
    CASE_DIR = os.path.join(DATA_DIR, case_idx)
    CUR_IDX = 0
    np_pred_gt_diff = nib.load('./pred_correspondence_with_gt_top1_diff{}.nii.gz'.format(CUR_IDX))
    np_pred_gt_diff = np_pred_gt_diff.get_fdata()

    np_pred_gt_diff[np_pred_gt_diff > 360] = 0 

    bool_pred_gt_diff = np.zeros((224, 224, 224))
    bool_pred_gt_diff[np_pred_gt_diff>0] = 1

    ct_file_path = os.path.join(CASE_DIR, '{}_crop_image_concat.nii.gz'.format(case_idx))
    gradient_file_path = os.path.join(CASE_DIR, 'edge_gradient_normalize.nii.gz'.format(case_idx))

    gradient_image = nib.load(gradient_file_path)
    gradient_image = gradient_image.get_fdata()


    file_list = os.listdir(CASE_DIR)
    for single_file in file_list:
        if single_file.lower()[:5] == flag:
            stl_file_path = os.path.join(CASE_DIR, single_file)

    scan_image = load_mesh(stl_file_path)

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

    matched_pair_gradient, out_of_range_list = __get_pt_and_gradient224__(gradient_image, scan_image, [d_affine_coord, h_affine_coord, w_affine_coord])
    matched_pair_gradient = matched_pair_gradient[out_of_range_list]

    '''
    '''

    for idx, single_matched_pair_gradient in enumerate(matched_pair_gradient):
        if idx % 10000 == 0:
            print(idx)
        _, _, _, single_gradient, dd, hh, ww = single_matched_pair_gradient
        single_scan_image = scan_image[idx]

        if bool_pred_gt_diff[dd][hh][ww] != 0:
            # print('FUCK!')
            continue
        
        print('No FUCK!')
        print('No FUCK!')

        bool_pred_gt_diff[dd][hh][ww] = 1

        nearest_neighbor_coords = find_nearest_neighbor(single_scan_image, np.delete(scan_image, idx, axis=0))
        # break

        scan_x, scan_y, scan_d = nearest_neighbor_coords

        spacing_d = scan_d * (1 / 0.2)
        spacing_w = scan_x * (1 / 0.2)
        spacing_h = scan_y * (1 / 0.2)

        spacing_d = int(np.round(spacing_d))
        spacing_h = int(np.round(spacing_h))
        spacing_w = int(np.round(spacing_w))

        affine_h = h_affine_coord[spacing_w][spacing_h][spacing_d]
        affine_w = w_affine_coord[spacing_w][spacing_h][spacing_d]
        affine_d = d_affine_coord[spacing_w][spacing_h][spacing_d]

        affine_h = int(np.round(affine_h))
        affine_w = int(np.round(affine_w))
        affine_d = int(np.round(affine_d))
        np_pred_gt_diff[affine_d][affine_h][affine_w] = np_pred_gt_diff[dd][hh][ww]

        print('From (', dd, hh, ww, ') value : {} -> -> To (', affine_d, affine_h, affine_w, ') value : {}'.format(np_pred_gt_diff[dd][hh][ww], np_pred_gt_diff[affine_d][affine_h][affine_w]))



    nii_pred_gt_diff = nib.Nifti1Image(np_pred_gt_diff, affine=np.eye(4))
    nib.save(nii_pred_gt_diff, './pred_correspondence_with_gt_top1_diff{}.nii.gz'.format(CUR_IDX + 101))


pred_correspondence_with_gt_top1_diff()