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

def load_mesh(filename):
    mesh_data = mesh.Mesh.from_file(filename)
    vertices = mesh_data.vectors.reshape(-1, 3)
    unique_vertices, indices = np.unique(vertices, axis=0, return_index=True)
    return unique_vertices

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

        file_list = os.listdir(CASE_DIR)
        for single_file in file_list:
            if single_file.lower()[:5] == flag:
                stl_file_path = os.path.join(CASE_DIR, single_file)

        ct_image = nib.load(ct_file_path)
        ct_image = ct_image.get_fdata()

        ct_image = resize_img(ct_image, (128, 128, 128))
        ct_image = np.transpose(ct_image, (2, 1, 0))
        ct_image = np.flip(ct_image, 2)

        gradient_image = nib.load(gradient_file_path)
        gradient_image = gradient_image.get_fdata()

        scan_image = load_mesh(stl_file_path)

        if scan_image.shape[0]>self.SAMPLE_CNT:
            scan_image = gu.resample_pcd([scan_image], self.SAMPLE_CNT, "fps")[0]

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

        # self.__ct2mesh__(ct_image, CASE_DIR)
        # self.__gt_correspondence__(gradient_image, scan_image, [d_affine_coord, h_affine_coord, w_affine_coord], CASE_DIR, stl_file_path)

        matched_pair_gradient, out_of_range_list = self.__get_pt_and_gradient__(gradient_image, scan_image, [d_affine_coord, h_affine_coord, w_affine_coord])
        scan_image = scan_image[out_of_range_list]
        matched_pair_gradient = matched_pair_gradient[out_of_range_list]
 
        scan_image[:,:3] -= np.mean(scan_image[:,:3], axis=0)
        coord = scan_image
        feat = torch.ones([coord.shape[0], 3])

        point_cloud_size = (scan_image.shape[0], 3)
        label = torch.rand(point_cloud_size)
        label = label*13
        label = torch.round(label)
        
        offset = [scan_image.shape[0]]
        offset = torch.IntTensor(offset)

        ct_image = self.transform(ct_image)

        return ct_image, matched_pair_gradient, coord, feat, label, offset

    def __ct2mesh__(self, resize_ct_image, SAVE_DIR):
        otsu_ct_image = resize_ct_image
        for i in range(2):
            otsu_threshold = filters.threshold_otsu(otsu_ct_image[otsu_ct_image>0])
            otsu_ct_index = otsu_ct_image > otsu_threshold

            temp_arr = np.zeros(otsu_ct_image.shape)
            temp_arr[otsu_ct_index == True] = otsu_ct_image[otsu_ct_index == True]
            otsu_ct_image = temp_arr

        otsu_ct_image[:, :, 75:] = 0
        vertices, faces, _, _ = marching_cubes(otsu_ct_image, level=0.5)

        stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

        for i, f in enumerate(faces):
            for j in range(3):
                stl_mesh.vectors[i][j] = vertices[f[j], :]

        # STL 파일로 저장
        stl_mesh.save(os.path.join(SAVE_DIR, 'CT2Mesh_dataloader.stl'))

    def __gt_correspondence__(self, gradient_image, scan_image, affine_coords, SAVE_DIR, stl_file_path, pixel_spacing=0.2, slice_spacing=0.2, ct_resize_shape=(128, 128, 128)):
        matched_pair_gradient = []
        matched_scan_image = []

        original_shape = (224, 224, 224)
        d_affine_coord, h_affine_coord, w_affine_coord = affine_coords

        for i in range(len(scan_image)):
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

            resized_h = affine_h * (ct_resize_shape[1] / original_shape[0])
            resized_w = affine_w * (ct_resize_shape[2] / original_shape[1])
            resized_d = affine_d * (ct_resize_shape[0] / original_shape[2])

            resized_d = int(np.round(resized_d))
            resized_h = int(np.round(resized_h))
            resized_w = int(np.round(resized_w))

            if gradient_image[spacing_w, spacing_h, spacing_d] < self.VIS_GRADIENT_THRESHOLD:
                continue
            flag=False
            if resized_d < 1 or resized_d > 126:
                flag = True
            if resized_h < 1 or resized_h > 126:
                flag = True
            if resized_w < 1 or resized_w > 126:
                flag = True

            if flag==False:
                matched_pair_gradient.append([resized_w, resized_h, 128-resized_d, gradient_image[spacing_w, spacing_h, spacing_d]])
                matched_scan_image.append(scan_image[i])

        matched_scan_image = np.array(matched_scan_image)
        matched_pair_gradient = np.array(matched_pair_gradient)
        # matched_scan_image, matched_pair_gradient[:, :3]
        # gradient_mesh_index, gradient_ct_index

        gradient_mesh_index = matched_scan_image
        gradient_ct_index = matched_pair_gradient[:, :3]

        mesh1 = trimesh.load(stl_file_path)
        mesh2 = trimesh.load(os.path.join(SAVE_DIR, 'CT2Mesh_dataloader.stl'))

        translation_vector = [100, 0, 0]
        mesh2 = mesh2.apply_translation(translation_vector)
        mesh1.visual.face_colors = [200, 200, 250, 100]
        mesh2.visual.face_colors = [200, 200, 250, 100]

        rand_index = np.random.choice(gradient_mesh_index.shape[0], size=self.VIS_SAMPLE_CNT, replace=False)
        sampled_vertices1 = gradient_mesh_index[rand_index]
        pc1 = Points(sampled_vertices1, r=self.VIS_RADIUS)
        pc1.cmap("jet", list(range(len(sampled_vertices1))))

        sampled_vertices2 = gradient_ct_index[rand_index]
        sampled_vertices2 += translation_vector
        pc2 = Points(sampled_vertices2, r=self.VIS_RADIUS)
        pc2.cmap("jet", list(range(len(sampled_vertices2))))

        lines = []
        for p1, p2 in zip(sampled_vertices1, sampled_vertices2):
            line = Line(p1, p2, c="green")
            lines.append(line)
        
        # from vedo import Sphere
        # figure = Sphere()
        show([(mesh1, pc1, mesh2, pc2, lines)], N=1, bg="black", axes=0)
        # show([(mesh1, pc1, mesh2, pc2, lines)], N=1, bg="black", axes=0, interactive=False, new=True, offscreen=True).screenshot(os.path.join(SAVE_DIR, 'gt_correspondence(30).png'))

    def __len__(self):
        # return len(self.data_idx) * self.loop
        return len(self.data_list)


    def __get_pt_and_gradient__(self, gradient_image, scan_image, affine_coords, pixel_spacing=0.2, slice_spacing=0.2, ct_resize_shape=(128, 128, 128)):
        matched_pair_gradient = []

        original_shape = (224, 224, 224)
        d_affine_coord, h_affine_coord, w_affine_coord = affine_coords
        out_of_range_list = []
        # check_volume = np.zeros((128, 128, 128))

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
        matched_pair_gradient = np.array(matched_pair_gradient)
        # utils._check(check_volume)

        return matched_pair_gradient, out_of_range_list