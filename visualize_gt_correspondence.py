import os
import copy
import numpy as np
import nibabel as nib

import trimesh
from stl import mesh
from skimage import filters
from skimage.measure import marching_cubes
# from pymesh import Subdivider, form_mesh

import gen_utils as gu

DATA_DIR = os.path.join(os.getcwd(), 'datasets')
# ct_file_path = os.path.join(DATA_DIR, 'Case_18', "Case_18_original_dcm_arr.nii.gz")
ct_file_path = os.path.join(DATA_DIR, 'Case_18', "Case_18_crop_image_concat.nii.gz")
gradient_file_path = os.path.join(DATA_DIR, 'Case_18', "edge_gradient_normalize.nii.gz")

mesh1_path = os.path.join(DATA_DIR, 'Case_18', "LOWER_Result_sota.stl")
# save_mesh_path = os.path.join(DATA_DIR, 'Case_18', "CT2Mesh_800.stl")
save_mesh_path = os.path.join(DATA_DIR, 'Case_18', "CT2Mesh_dataloader.stl")

SAMPLE_CNT = 12000
SAMPLE_SHOW = 100
RADIUS = 10
GRADIENT = 0.01
ps, ss = 0.2, 0.2
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

def get_pt_and_gradient(gradient_image, scan_image, affine_coords, pixel_spacing=0.2, slice_spacing=0.2, ct_resize_shape=(128, 128, 128)):
    matched_pair_gradient = []
    matched_scan_image = []

    # original_shape = gradient_image.shape
    original_shape = (224, 224, 224)
    # print('gradient_image : ', gradient_image.shape)

    d_affine_coord, h_affine_coord, w_affine_coord = affine_coords


    for i in range(len(scan_image)):
        # scan_d, scan_x, scan_y = scan_image[i]
        # scan_x, scan_y, scan_d = scan_image[i]
        scan_x, scan_y, scan_d = scan_image[i]

        spacing_d = scan_d * (1 / slice_spacing)
        # spacing_h = scan_x * (1 / pixel_spacing)
        # spacing_w = scan_y * (1 / pixel_spacing)

        spacing_w = scan_x * (1 / pixel_spacing)
        spacing_h = scan_y * (1 / pixel_spacing)

        spacing_d = int(np.round(spacing_d))
        spacing_h = int(np.round(spacing_h))
        spacing_w = int(np.round(spacing_w))

        # affine_h = h_affine_coord[spacing_d][spacing_h][spacing_w]
        # affine_w = w_affine_coord[spacing_d][spacing_h][spacing_w]
        # affine_d = d_affine_coord[spacing_d][spacing_h][spacing_w]

        # affine_h = h_affine_coord[spacing_h][spacing_w][spacing_d]
        # affine_d = w_affine_coord[spacing_h][spacing_w][spacing_d]
        # affine_w = d_affine_coord[spacing_h][spacing_w][spacing_d]

        # affine_h = h_affine_coord[spacing_w][spacing_h][spacing_d]
        # affine_w = w_affine_coord[spacing_w][spacing_h][spacing_d]
        # affine_d = d_affine_coord[spacing_w][spacing_h][spacing_d]
        
        # affine_d = h_affine_coord[spacing_w][spacing_h][spacing_d]
        # affine_w = w_affine_coord[spacing_w][spacing_h][spacing_d]
        # affine_h = d_affine_coord[spacing_w][spacing_h][spacing_d]

        affine_h = h_affine_coord[spacing_w][spacing_h][spacing_d]
        affine_w = w_affine_coord[spacing_w][spacing_h][spacing_d]
        affine_d = d_affine_coord[spacing_w][spacing_h][spacing_d]

        # affine_d = d_affine_coord[spacing_w][spacing_h][spacing_d]
        # affine_h = h_affine_coord[spacing_w][spacing_h][spacing_d]
        # affine_w = w_affine_coord[spacing_w][spacing_h][spacing_d]

        affine_h = int(np.round(affine_h))
        affine_w = int(np.round(affine_w))
        affine_d = int(np.round(affine_d))

        # affine_h, affine_w, affine_d = spacing_h, spacing_w, spacing_d

        # print('original {} {} {} -> resize {} {} {}'.format(spacing_d, spacing_h, spacing_w, affine_d, affine_h, affine_w))

        resized_h = affine_h * (ct_resize_shape[1] / original_shape[0])
        resized_w = affine_w * (ct_resize_shape[2] / original_shape[1])
        resized_d = affine_d * (ct_resize_shape[0] / original_shape[2])

        resized_d = int(np.round(resized_d))
        resized_h = int(np.round(resized_h))
        resized_w = int(np.round(resized_w))
        

        # matched_pair_gradient.append([resized_d, resized_h, resized_w, gradient_image[spacing_h, spacing_w, spacing_d]])
        # matched_pair_gradient.append([resized_h, resized_w, 128 - resized_d, gradient_image[spacing_h, spacing_w, spacing_d]])
        # matched_pair_gradient.append([resized_d, resized_h, resized_w, gradient_image[spacing_h, spacing_w, spacing_d]])
        # print('gradient_image[spacing_w, spacing_h, spacing_d] : ', gradient_image[spacing_w, spacing_h, spacing_d])
        # if gradient_image[spacing_w, spacing_h, spacing_d] < GRADIENT:
        if gradient_image[spacing_w, spacing_h, spacing_d] > GRADIENT:
            continue
        matched_pair_gradient.append([resized_w, resized_h, 128-resized_d, gradient_image[spacing_w, spacing_h, spacing_d]])
        matched_scan_image.append(scan_image[i])
        # matched_pair_gradient.append([resized_h, resized_w, resized_d, gradient_image[spacing_h, spacing_w, spacing_d]])

        # matched_pair_gradient.append([resized_w, resized_h, resized_d, gradient_image[spacing_h, spacing_w, spacing_d]])

        # matched_pair_gradient.append([spacing_h, spacing_w, spacing_d, gradient_image[spacing_h, spacing_w, spacing_d]])

    return matched_scan_image, matched_pair_gradient


def gt_correspondence(stl_file_path, gradient_file_path):
    '''
    stl_file_path : (224, 224, 224)
    '''
    gradient_image = nib.load(gradient_file_path)
    gradient_image = gradient_image.get_fdata()
    
    scan_image = load_mesh(stl_file_path)
    if scan_image.shape[0]> SAMPLE_CNT:
        scan_image = gu.resample_pcd([scan_image], SAMPLE_CNT, "fps")[0]

    case_idx = 'Case_18'
    flag = 'lower'

    d_affine_coord_path = os.path.join(DATA_DIR, case_idx, 'd_{}_label.nii.gz'.format(flag))
    h_affine_coord_path = os.path.join(DATA_DIR, case_idx, 'h_{}_label.nii.gz'.format(flag))
    w_affine_coord_path = os.path.join(DATA_DIR, case_idx, 'w_{}_label.nii.gz'.format(flag))

    d_affine_coord = nib.load(d_affine_coord_path)
    h_affine_coord = nib.load(h_affine_coord_path)
    w_affine_coord = nib.load(w_affine_coord_path)
    d_affine_coord = d_affine_coord.get_fdata()
    h_affine_coord = h_affine_coord.get_fdata()
    w_affine_coord = w_affine_coord.get_fdata()

    # print('bef tranpose : ', d_affine_coord.shape)

    d_affine_coord = np.transpose(d_affine_coord, (2, 1, 0))
    h_affine_coord = np.transpose(h_affine_coord, (2, 1, 0))
    w_affine_coord = np.transpose(w_affine_coord, (2, 1, 0))
    d_affine_coord = np.flip(d_affine_coord, 2) 
    h_affine_coord = np.flip(h_affine_coord, 2) 
    w_affine_coord = np.flip(w_affine_coord, 2)

    # print('aft tranpose : ', d_affine_coord.shape)
    # print('aft flip : ', d_affine_coord.shape)

    matched_scan_image, matched_pair_gradient = get_pt_and_gradient(gradient_image, scan_image, [d_affine_coord, h_affine_coord, w_affine_coord])
    matched_scan_image = np.array(matched_scan_image)
    matched_pair_gradient = np.array(matched_pair_gradient)

    print('matched_scan_image : ', matched_scan_image.shape)
    print('matched_pair_gradient : ', matched_pair_gradient.shape)

    return matched_scan_image, matched_pair_gradient[:, :3]

def ct2mesh(ct_path, save_path):
    ct_image = nib.load(ct_path)
    ct_image = ct_image.get_fdata()

    ct_image = np.transpose(ct_image, (2, 1, 0))
    ct_image = np.flip(ct_image, 2)

    from utils import resize_img
    ct_image = resize_img(ct_image, (128, 128, 128))

    nii_otsu_ct_image = nib.Nifti1Image(ct_image, affine=return_matrix)
    nib.save(nii_otsu_ct_image, DATA_DIR + '/otsu{}.nii.gz'.format('0'))

    otsu_ct_image = ct_image
    for i in range(2):
        otsu_threshold = filters.threshold_otsu(otsu_ct_image[otsu_ct_image>0])
        otsu_ct_index = otsu_ct_image > otsu_threshold

        temp_arr = np.zeros(ct_image.shape)
        temp_arr[otsu_ct_index == True] = otsu_ct_image[otsu_ct_index == True]
        otsu_ct_image = temp_arr
                
        nii_otsu_ct_image = nib.Nifti1Image(otsu_ct_image, affine=return_matrix)
        nib.save(nii_otsu_ct_image, DATA_DIR + '/otsu{}.nii.gz'.format(i+1))


    # otsu_ct_image[:, :, 62:] = 0
    # vertices, faces, _, _ = marching_cubes(otsu_ct_image, level=0.5)

    # # vertices[:, 0] = vertices[:, 0] * ps
    # # vertices[:, 1] = vertices[:, 1] * ps
    # # vertices[:, 2] = vertices[:, 2] * ss

    # # stl_mesh = mesh.Mesh(np.zeros(faces.shape[0]))
    # stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    # for i, f in enumerate(faces):
    #     for j in range(3):
    #         stl_mesh.vectors[i][j] = vertices[f[j], :]

    # # STL 파일로 저장
    # stl_mesh.save(save_path)



# ct2mesh(ct_file_path, save_mesh_path)
gradient_mesh_index, gradient_ct_index = gt_correspondence(mesh1_path, gradient_file_path)

# stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

# from vedo import *
from vedo import show, Points, Line


mesh1 = trimesh.load(mesh1_path)
mesh2 = trimesh.load(save_mesh_path)

# translation_vector = [-50, 0, 0]
# mesh1 = mesh1.apply_translation(translation_vector)
translation_vector = [100, 0, 0]
mesh2 = mesh2.apply_translation(translation_vector)

# translation_vector = [100, 0, 0]
# print('bef : ', mesh2.vertices[:3])
# mesh2.vertices += translation_vector
# print('aft : ', mesh2.vertices[:3])

mesh1.visual.face_colors = [200, 200, 250, 100]
mesh2.visual.face_colors = [200, 200, 250, 100]

# n1 = mesh1.vertices.shape[0]
# n1 = len(gradient_mesh_index)
# sampled_vertices1 = mesh1.vertices[np.random.choice(n1, 100)]
rand_index = np.random.choice(gradient_mesh_index.shape[0], size=SAMPLE_SHOW, replace=False)
sampled_vertices1 = gradient_mesh_index[rand_index]
pc1 = Points(sampled_vertices1, r=RADIUS)
pc1.cmap("jet", list(range(len(sampled_vertices1))))

# n2 = mesh2.vertices.shape[0]
# sampled_vertices2 = mesh2.vertices[np.random.choice(n2, 100)]
# sampled_vertices2 = gradient_mesh_index[rand_index]
sampled_vertices2 = gradient_ct_index[rand_index]
sampled_vertices2 += translation_vector
pc2 = Points(sampled_vertices2, r=RADIUS)
pc2.cmap("jet", list(range(len(sampled_vertices2))))

# lines = Line(pc1.points(), pc2.points(), c="red")
# translation_vector = [100, 0, 0]
# pc2.x(100)
# lines = Line(pc1.points(), pc2.points(), c="red")

lines = []
for p1, p2 in zip(sampled_vertices1, sampled_vertices2):
    line = Line(p1, p2, c="green")
    lines.append(line)
# print('pc2.points() : ', pc2.points()[:3])
# mesh1.vertices.translate([-50, 0, 0])
# mesh2.vertices.translate([50, 0, 0])
show([(mesh1, pc1, mesh2, pc2, lines)], N=1, bg="black", axes=0)