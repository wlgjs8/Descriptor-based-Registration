import os
import numpy as np
import torch
from torch.utils.data import Dataset

import gen_utils as gu

import nibabel as nib
from stl import mesh
import utils
from utils import resize_img
from monai.transforms import (
    Compose,
    NormalizeIntensity,
    NormalizeIntensityd,
)

ct_transform = NormalizeIntensity(nonzero=True)
# ct_transform = Compose(
#     [
#         NormalizeIntensityd(keys=["ct_image"], nonzero=True, channel_wise=True)
#     ]
# )
return_matrix = np.array([
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
])

def load_mesh(filename):
    mesh_data = mesh.Mesh.from_file(filename)
    return mesh_data.vectors.reshape(-1, 3)

def write_ply(filename, vertices):
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(vertices)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("end_header\n")
        
        for vertex in vertices:
            f.write(f"{vertex[0]} {vertex[1]} {vertex[2]}\n")

class CTScanDataset(Dataset):
    def __init__(self, split='train', transform=ct_transform):
        super().__init__()
        
        self.transform = transform

        if split =='train':
            # self.data_list = [('Case_01', 'Upper'), ('Case_01', 'Lower')]
            self.case_list = ['Case_02', 'Case_09', 'Case_10', 'Case_12', 'Case_14']
        else:
            # self.data_list = [('Case_03', 'Upper'), ('Case_03', 'Lower')]
            self.case_list = ['Case_01', 'Case_03', 'Case_11', 'Case_15', 'Case_18']

        self.data_list = []
        for case in self.case_list:
            self.data_list.append((case, 'upper'))
            self.data_list.append((case, 'lower'))

        self.DATA_DIR = os.path.abspath('/home/jeeheon/Documents/point-transformer/datasets')
        self.SAMPLE_CNT = 24000

    def __getitem__(self, idx):
        '''
        point_cloud_size = (12000, 3)
        coord = torch.rand(point_cloud_size)
        point_cloud_size = (12000, 3)
        feat = torch.rand(point_cloud_size)
        point_cloud_size = (12000, 3)
        label = torch.rand(point_cloud_size)
        label = label*13
        label = torch.round(label)
        return coord, feat, label
        '''

        case_idx, flag = self.data_list[idx]
        print('Case : {}, Flag : {}'.format(case_idx, flag))
        CASE_DIR = os.path.join(self.DATA_DIR, case_idx)
        ct_file_path = os.path.join(CASE_DIR, '{}_original_dcm_arr.nii.gz'.format(case_idx))
        gradient_file_path = os.path.join(CASE_DIR, 'edge_gradient_normalize.nii.gz'.format(case_idx))

        file_list = os.listdir(CASE_DIR)
        for single_file in file_list:
            if single_file.lower()[:5] == flag:
                stl_file_path = os.path.join(CASE_DIR, single_file)

        ct_image = nib.load(ct_file_path)
        ct_image = ct_image.get_fdata()

        gradient_image = nib.load(gradient_file_path)
        gradient_image = gradient_image.get_fdata()

        scan_image = load_mesh(stl_file_path)
        # random_list = np.random.choice(scan_image.shape[0], 12000)
        # fps_sampling

        if scan_image.shape[0]>self.SAMPLE_CNT:
            scan_image = gu.resample_pcd([scan_image], self.SAMPLE_CNT, "fps")[0]

        # print('random_list : ', len(random_list))
        # scan_image = scan_image[random_list, :]

        matched_pair_gradient = self.__get_pt_and_gradient__(gradient_image, scan_image)        

        
        # ct_image <= resize


        coord = scan_image
        feat = torch.ones([coord.shape[0], 3])

        '''
        print('ct_image : ', ct_image.shape)
        print('coord : ', coord.shape)
        print('feat : ', feat.shape)
        ct_image :  (496, 800, 800)
        coord :  (1279902, 3)
        feat :  torch.Size([1279902, 3])
        '''

        # label = torch.rand(len(scan_image))
        point_cloud_size = (scan_image.shape[0], 3)
        label = torch.rand(point_cloud_size)
        label = label*13
        label = torch.round(label)
        
        offset = [scan_image.shape[0]]
        offset = torch.IntTensor(offset)

        '''
        ct_image, coord, feat, target, offset
        '''
        # ct_image = torch.tensor(ct_image).cuda()
        # coord = torch.tensor(coord).cuda()
        # feat = feat.cuda()
        # label = label.cuda()
        # offset = offset.cuda()
        ct_image = resize_img(ct_image, (128, 128, 128))
        
        # nii_ct_image = np.transpose(ct_image, (2, 1, 0))
        # nii_ct_image = np.flip(nii_ct_image, 2)
        # nii_ct_image = nib.Nifti1Image(nii_ct_image, affine=return_matrix)
        # nib.save(nii_ct_image, './ct_image.nii.gz')

        # proccessed_out = {
        #     'ct_image' : ct_image,
        # }
        # proccessed_out = self.transform(proccessed_out)

        # ct_image = proccessed_out['ct_image']
        ct_image = self.transform(ct_image)
        # print('ct_image : ', ct_image.shape)
        # print('ct_image : ', ct_image.shape)
        # print('ct_image : ', ct_image.shape)
        # print('ct_image : ', ct_image.shape)
        matched_pair_gradient = np.array(matched_pair_gradient)
        # print('matched_pair_gradient : ', np.mean(matched_pair_gradient))

        return ct_image, matched_pair_gradient, coord, feat, label, offset



    def __len__(self):
        # return len(self.data_idx) * self.loop
        return len(self.data_list)

    def __get_pt_and_gradient__(self, gradient_image, scan_image, pixel_spacing=0.2, slice_spacing=0.2, ct_resize_shape=(128, 128, 128)):
        matched_pair_gradient = []
        check_volume = np.zeros((800, 800, 496))
        resize_check_volume = np.zeros(ct_resize_shape)

        '''
        Inputs:
            gradient_image : (800, 800, 496)
            scan_image : (Sampled_cnt, 3)

        Outputs : 
            matched_pair_gradient : (Sampled_cnt, 4)
            [
                [d, x, y, gradient_value],
                [d, x, y, gradient_value],
            ]
        '''

        original_shape = gradient_image.shape


        for i in range(len(scan_image)):
            scan_d, scan_x, scan_y = scan_image[i]
            
            spacing_d = scan_d * (1 / slice_spacing)
            spacing_x = scan_x * (1 / pixel_spacing)
            spacing_y = scan_y * (1 / pixel_spacing)

            spacing_d = int(np.round(spacing_d))
            spacing_x = int(np.round(spacing_x))
            spacing_y = int(np.round(spacing_y))

            resized_d = spacing_d * (ct_resize_shape[0] / original_shape[0])
            resized_x = spacing_x * (ct_resize_shape[1] / original_shape[1])
            resized_y = spacing_y * (ct_resize_shape[2] / original_shape[2])

            resized_d = int(np.round(resized_d))
            resized_x = int(np.round(resized_x))
            resized_y = int(np.round(resized_y))

            matched_pair_gradient.append([resized_d, resized_x, resized_y, gradient_image[spacing_d, spacing_x, spacing_y]])
            # check_volume[spacing_d, spacing_x, spacing_y] = 1
            # resize_check_volume[resized_d, resized_x, resized_y] = gradient_image[spacing_d, spacing_x, spacing_y]


        # nii_check_volume = nib.Nifti1Image(check_volume, affine=return_matrix)
        # nib.save(nii_check_volume, './check_volume.nii.gz')

        # nii_resize_check_volume = nib.Nifti1Image(resize_check_volume, affine=return_matrix)
        # nib.save(nii_resize_check_volume, './resize_check_volume_gradient.nii.gz')


        return matched_pair_gradient


class CTScanDataset_Center(Dataset):
    def __init__(self, split='train', transform=ct_transform):
        super().__init__()
        
        self.transform = transform
        
        if split =='train':
            # self.data_list = [('Case_01', 'Upper'), ('Case_01', 'Lower')]
            # self.case_list = ['Case_02', 'Case_09', 'Case_10', 'Case_12', 'Case_14']
            self.case_list = ['Case_18']
        else:
            # self.data_list = [('Case_03', 'Upper'), ('Case_03', 'Lower')]
            # self.case_list = ['Case_15', 'Case_01', 'Case_03', 'Case_11', 'Case_18']
            # self.case_list = ['Case_18', 'Case_01', 'Case_03', 'Case_11', 'Case_15']
            self.case_list = ['Case_02']

        self.data_list = []
        for case in self.case_list:
            self.data_list.append((case, 'upper'))
            # self.data_list.append((case, 'lower'))

        self.DATA_DIR = os.path.abspath('/home/jeeheon/Documents/point-transformer/datasets')
        self.SAMPLE_CNT = 12000

        # self.Y_AXIS_MAX = 33.15232091532151
        # self.Y_AXIS_MIN = -36.9843781139949

    def __getitem__(self, idx):
        '''
        point_cloud_size = (12000, 3)
        coord = torch.rand(point_cloud_size)
        point_cloud_size = (12000, 3)
        feat = torch.rand(point_cloud_size)
        point_cloud_size = (12000, 3)
        label = torch.rand(point_cloud_size)
        label = label*13
        label = torch.round(label)
        return coord, feat, label
        '''

        case_idx, flag = self.data_list[idx]
        print('Case : {}, Flag : {}'.format(case_idx, flag))
        CASE_DIR = os.path.join(self.DATA_DIR, case_idx)
        ct_file_path = os.path.join(CASE_DIR, '{}_original_dcm_arr.nii.gz'.format(case_idx))
        gradient_file_path = os.path.join(CASE_DIR, 'edge_gradient_normalize.nii.gz'.format(case_idx))

        file_list = os.listdir(CASE_DIR)
        for single_file in file_list:
            if single_file.lower()[:5] == flag:
                stl_file_path = os.path.join(CASE_DIR, single_file)

        ct_image = nib.load(ct_file_path)
        ct_image = ct_image.get_fdata()

        gradient_image = nib.load(gradient_file_path)
        gradient_image = gradient_image.get_fdata()

        scan_image = load_mesh(stl_file_path)
        # random_list = np.random.choice(scan_image.shape[0], 12000)
        # fps_sampling

        if scan_image.shape[0]>self.SAMPLE_CNT:
            scan_image = gu.resample_pcd([scan_image], self.SAMPLE_CNT, "fps")[0]

        # print('random_list : ', len(random_list))
        # scan_image = scan_image[random_list, :]

        matched_pair_gradient = self.__get_pt_and_gradient__(gradient_image, scan_image)
        # print('scan_image : ', scan_image.shape)
        # print('bef scan : ', scan_image[:3])
        # write_ply("bef trans.ply", scan_image)
        # write_ply("bef upper trans.ply", scan_image)
        scan_image[:,:3] -= np.mean(scan_image[:,:3], axis=0)
        # write_ply("aft trans.ply", scan_image)

        # print('aft scan : ', scan_image[:3])
        # scan_image[:, :3] = ((scan_image[:, :3]-self.Y_AXIS_MIN)/(self.Y_AXIS_MAX - self.Y_AXIS_MIN))*2-1
        # scan_image[:, :3] = ((scan_image[:, :3]-scan_image[:, 1].min())/(scan_image[:, 1].max() - scan_image[:, 1].min()))*2-1
        
        # ct_image <= resize


        coord = scan_image
        feat = torch.ones([coord.shape[0], 3])

        '''
        print('ct_image : ', ct_image.shape)
        print('coord : ', coord.shape)
        print('feat : ', feat.shape)
        ct_image :  (496, 800, 800)
        coord :  (1279902, 3)
        feat :  torch.Size([1279902, 3])
        '''

        # label = torch.rand(len(scan_image))
        point_cloud_size = (scan_image.shape[0], 3)
        label = torch.rand(point_cloud_size)
        label = label*13
        label = torch.round(label)
        
        offset = [scan_image.shape[0]]
        offset = torch.IntTensor(offset)

        '''
        ct_image, coord, feat, target, offset
        '''
        # ct_image = torch.tensor(ct_image).cuda()
        # coord = torch.tensor(coord).cuda()
        # feat = feat.cuda()
        # label = label.cuda()
        # offset = offset.cuda()
        ct_image = resize_img(ct_image, (128, 128, 128))

        # proccessed_out = {
        #     'ct_image' : ct_image,
        # }
        # proccessed_out = self.transform(proccessed_out)

        # ct_image = proccessed_out['ct_image']
        ct_image = self.transform(ct_image)
        # print('ct_image : ', ct_image.shape)
        # print('ct_image : ', ct_image.shape)
        # print('ct_image : ', ct_image.shape)
        # print('ct_image : ', ct_image.shape)
        matched_pair_gradient = np.array(matched_pair_gradient)
        # print('matched_pair_gradient : ', np.mean(matched_pair_gradient))

        return ct_image, matched_pair_gradient, coord, feat, label, offset



    def __len__(self):
        # return len(self.data_idx) * self.loop
        return len(self.data_list)

    def __get_pt_and_gradient__(self, gradient_image, scan_image, pixel_spacing=0.2, slice_spacing=0.2, ct_resize_shape=(128, 128, 128)):
        matched_pair_gradient = []
        # check_volume = np.zeros((800, 800, 496))
        # resize_check_volume = np.zeros(ct_resize_shape)

        '''
        Inputs:
            gradient_image : (800, 800, 496)
            scan_image : (Sampled_cnt, 3)

        Outputs : 
            matched_pair_gradient : (Sampled_cnt, 4)
            [
                [d, x, y, gradient_value],
                [d, x, y, gradient_value],
            ]
        '''

        original_shape = gradient_image.shape
        check_volume = np.zeros((128, 128, 128))


        for i in range(len(scan_image)):
            scan_d, scan_x, scan_y = scan_image[i]
            
            spacing_d = scan_d * (1 / slice_spacing)
            spacing_x = scan_x * (1 / pixel_spacing)
            spacing_y = scan_y * (1 / pixel_spacing)

            spacing_d = int(np.round(spacing_d))
            spacing_x = int(np.round(spacing_x))
            spacing_y = int(np.round(spacing_y))

            resized_d = spacing_d * (ct_resize_shape[0] / original_shape[0])
            resized_x = spacing_x * (ct_resize_shape[1] / original_shape[1])
            resized_y = spacing_y * (ct_resize_shape[2] / original_shape[2])

            resized_d = int(np.round(resized_d))
            resized_x = int(np.round(resized_x))
            resized_y = int(np.round(resized_y))
            
            check_volume[resized_d][resized_x][resized_y] += 1

            matched_pair_gradient.append([resized_d, resized_x, resized_y, gradient_image[spacing_d, spacing_x, spacing_y]])

        utils._check(check_volume)

        return matched_pair_gradient