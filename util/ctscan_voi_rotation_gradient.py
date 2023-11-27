import os
import numpy as np
import torch
from torch.utils.data import Dataset

import gen_utils as gu

import nibabel as nib
from stl import mesh
import utils
from utils import resize_img

from skimage import filters
from skimage.measure import marching_cubes
from vedo import show, Points, Line
from monai.transforms import (
    Compose,
    NormalizeIntensity,
    NormalizeIntensityd,
    ThresholdIntensity,
)

from augmentation import augmentator as aug
# from trimesh import Trimesh
import trimesh

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ct_transform = NormalizeIntensity(nonzero=True)
ct_transform = Compose(
    [
        ThresholdIntensity(threshold=1000, above=True, cval=0.0),
        NormalizeIntensity(nonzero=True, channel_wise=True)
    ]
)
return_matrix = np.array([
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])

# def load_mesh(filename):
#     mesh_data = mesh.Mesh.from_file(filename)
#     vertices = mesh_data.vectors.reshape(-1, 3)
#     unique_vertices, indices = np.unique(vertices, axis=0, return_index=True)
#     return unique_vertices

class CTScanDataset_Center(Dataset):
    def __init__(self, split='train', transform=ct_transform):
        super().__init__()
        
        self.transform = transform
        self.VIS_GRADIENT_THRESHOLD = 0.3
        self.VIS_FLAG = False
        self.VIS_RADIUS = 10
        self.VIS_SAMPLE_CNT = 350
        
        if split =='train':
            self.case_list = ['Case_18']
        else:
            self.case_list = ['Case_18']

        self.data_list = []
        for case in self.case_list:
            # self.data_list.append((case, 'upper'))
            self.data_list.append((case, 'lower'))

        self.DATA_DIR = os.path.abspath('/home/jeeheon/Documents/point-transformer/datasets')
        self.SAMPLE_CNT = 12000

    def __getitem__(self, idx):
        case_idx, flag = self.data_list[idx]
        print('Case : {}, Flag : {}'.format(case_idx, flag))
        CASE_DIR = os.path.join(self.DATA_DIR, case_idx)
        # ct_file_path = os.path.join(CASE_DIR, '{}_original_dcm_arr.nii.gz'.format(case_idx))
        ct_file_path = os.path.join(CASE_DIR, '{}_crop_image_concat.nii.gz'.format(case_idx))
        gradient_file_path = os.path.join(CASE_DIR, 'edge_gradient_normalize.nii.gz'.format(case_idx))

        # file_list = os.listdir(CASE_DIR)
        # for single_file in file_list:
        #     if single_file.lower()[:5] == flag:
        #         # stl_file_path = os.path.join(CASE_DIR, single_file)
        #         stl
        stl_file_path = os.path.join('/home/jeeheon/Documents/point-transformer/datasets/lower_registration', 'LOWER_Result_sota.stl') #(11973, 3)
        # stl_file_path = os.path.join('/home/jeeheon/Documents/point-transformer/datasets/lower_registration', 'LOWER_Result_sota_augment.stl') #(235, 3)
        
        # stl_file_path = os.path.join('/home/jeeheon/Documents/point-transformer/datasets/lower_registration', 'LOWER_Result_sota_restored.stl') #(11973, 3)


        ct_image = nib.load(ct_file_path)
        ct_image = ct_image.get_fdata()

        ct_image = resize_img(ct_image, (128, 128, 128))
        
        ct_image = np.transpose(ct_image, (2, 1, 0))
        ct_image = np.flip(ct_image, 2)

        gradient_image = nib.load(gradient_file_path)
        gradient_image = gradient_image.get_fdata()

        # scan_image = load_mesh(stl_file_path)
        original_scan_image = trimesh.load_mesh(stl_file_path)

        scan_normals = original_scan_image.vertex_normals
        curvatures = np.linalg.norm(np.gradient(scan_normals, axis=0), axis=1)
        mean_curvature = np.mean(curvatures)
        high_curvature_indices = np.array(curvatures > mean_curvature)
        high_curvature_vertices = original_scan_image.vertices[high_curvature_indices]
        high_curvature_vertices = np.array(high_curvature_vertices)

        if high_curvature_vertices.shape[0]>self.SAMPLE_CNT:
            scan_image = gu.resample_pcd([high_curvature_vertices], self.SAMPLE_CNT, "fps")[0]
        



        d_affine_coord_path = os.path.join(self.DATA_DIR, case_idx, 'd_{}_label.nii.gz'.format(flag))
        h_affine_coord_path = os.path.join(self.DATA_DIR, case_idx, 'h_{}_label.nii.gz'.format(flag))
        w_affine_coord_path = os.path.join(self.DATA_DIR, case_idx, 'w_{}_label.nii.gz'.format(flag))

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

        matched_pair_gradient = self.__get_pt_and_gradient_rotation__(gradient_image, scan_image, [d_affine_coord, h_affine_coord, w_affine_coord])
        
        # print('out_of_range_list[:,:3]  : ', out_of_range_list.shape)
        # print('out_of_range_list Faslse : ', out_of_range_list[out_of_range_list==False].shape)
        # scan_image = scan_image[out_of_range_list]
        # matched_pair_gradient = matched_pair_gradient[out_of_range_list]
 
        # print('scan_image[:,:3]  : ', scan_image[:,:3].shape)
        translation2origin = np.mean(scan_image[:,:3], axis=0)
        scan_image[:,:3] -= translation2origin
        
        aug_obj = aug.Augmentator([aug.Rotation([-30,30], 'rand')])
        aug_obj.reload_vals()
        transformed_mesh, aug_mats = aug_obj.run(scan_image[:,:3])
        '''save'''
        # save_image = trimesh.load_mesh(stl_file_path)
        # print('save_image : ', save_image.vertices.shape)
        # print('transformed_mesh : ', transformed_mesh.shape)

        # save_image.vertices = transformed_mesh
        # save_image.export('datasets/lower_registration/LOWER_Result_sota_augment.stl')


        # self.__ct2mesh__(ct_image, CASE_DIR)
        # self.__gt_correspondence__(gradient_image, transformed_mesh, [d_affine_coord, h_affine_coord, w_affine_coord], CASE_DIR, os.path.join('/home/jeeheon/Documents/point-transformer/datasets/lower_registration', 'LOWER_Result_sota_augment.stl'))

        # aug_mats = np.array([[ 0.99518056, -0.06589451,  0.07261936],
        #                     [ 0.06871358,  0.99694903, -0.03702806],
        #                     [-0.06995785,  0.04183954,  0.99667214]]
        #                     )
        
        # coord = scan_image

        feat = torch.ones([transformed_mesh.shape[0], 3])

        point_cloud_size = (scan_image.shape[0], 3)
        label = torch.rand(point_cloud_size)
        label = label*13
        label = torch.round(label)
        
        offset = [scan_image.shape[0]]
        offset = torch.IntTensor(offset)

        ct_image = self.transform(ct_image)

        # return ct_image, matched_pair_gradient, transformed_mesh, feat, label, offset
        return ct_image, matched_pair_gradient, transformed_mesh, feat, label, offset, translation2origin

    def __len__(self):
        # return len(self.data_idx) * self.loop
        return len(self.data_list)


    def __get_pt_and_gradient_rotation__(self, gradient_image, scan_image, affine_coords, pixel_spacing=0.2, slice_spacing=0.2, ct_resize_shape=(128, 128, 128)):
        ct_original_resize_pair = []

        original_shape = (224, 224, 224)
        d_affine_coord, h_affine_coord, w_affine_coord = affine_coords
        # check_volume = np.zeros((128, 128, 128))

        # for i in range(len(scan_image)):

        voxel_d = 128
        voxel_h = 128
        voxel_w = 128

        check_array = np.zeros((128, 128, 128))

        # for vd in range(90):
        #     for vh in range(110):
        #         for vw in range(110):
        # for vd in range(15, 85):
        #     for vh in range(15, 85):
        #         for vw in range(15, 85):
        # for vd in range(15, 55):
        #     for vh in range(15, 85):
        #         for vw in range(15, 85):
        for vd in range(10, 60):
        # for vd in range(60, 90):
            for vh in range(10, 100):
                for vw in range(10, 100):
                    
                    # scan_d, scan_x, scan_y = scan_image[i]
                    ct_d, ct_h, ct_w = vd, vh, vw

                    spacing_d = ct_d * (1 / slice_spacing)
                    spacing_w = ct_w * (1 / pixel_spacing)
                    spacing_h = ct_h * (1 / pixel_spacing)

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

                    if (resized_d < 5) or (resized_d > 123):
                        continue
                    if (resized_h < 5) or (resized_h > 123):
                        continue
                    if (resized_w < 5) or (resized_w > 123):
                        continue

                    
                    if check_array[resized_d, resized_h, resized_w] == 1:
                        continue
                    check_array[resized_d, resized_h, resized_w] = 1

                    ct_original_resize_pair.append([
                        resized_d, resized_h, resized_w, ct_d, ct_h, ct_w
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