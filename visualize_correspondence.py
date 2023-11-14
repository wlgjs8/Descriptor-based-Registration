import os
import copy
import numpy as np
import nibabel as nib

import trimesh
from stl import mesh
from skimage import filters
from skimage.measure import marching_cubes
# from pymesh import Subdivider, form_mesh


DATA_DIR = os.path.join(os.getcwd(), 'datasets')
mesh1_path = os.path.join(DATA_DIR, 'Case_18', "LOWER_Result_sota.stl")
# ct_path = os.path.join(DATA_DIR, 'Case_18', "Case_18_original_dcm_arr.nii.gz")
ct_path = os.path.join(DATA_DIR, 'Case_18', "Case_18_crop_image_concat.nii.gz")
# mesh2_path = os.path.join(DATA_DIR, 'Case_18', "CT2Mesh_lower.stl")
mesh2_path = os.path.join(DATA_DIR, 'Case_18', "CT2Mesh_voi_all.stl")
mesh22_path = os.path.join(DATA_DIR, 'Case_18', "CT2Mesh2.stl")
gradient_ct_path = os.path.join(DATA_DIR, 'Case_18', "edge_gradient_normalize.nii.gz")
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

def gt_correspondence(gradient_ct_path):
    ct_image = nib.load(gradient_ct_path)
    ct_image = ct_image.get_fdata()

    ct_image[:, :, 235:] = 0
    ct_image[:, :, :10] = 0

    gradient_index_x, gradient_index_y, gradient_index_z = np.where(ct_image > 0.5)
    gradient_index = np.array([gradient_index_x, gradient_index_y, gradient_index_z], dtype=np.float64)
    gradient_index = np.swapaxes(gradient_index, 1, 0)
    print('gradient_index : ', gradient_index.shape)

    gradient_ct_index = copy.deepcopy(gradient_index)
    print('gradient_index[:,0] : ', gradient_index[:,0])
    gradient_index[:,0] *= ps
    gradient_index[:,1] *= ps
    gradient_index[:,2] *= ss
    print('gradient_index[:,0] : ', gradient_index[:,0])

    

    # for single_gradient_index in gradient_index:
    #     x, y, z = single_gradient_index



    return gradient_index, gradient_ct_index


def ct2mesh(ct_path, save_path):
    ct_image = nib.load(ct_path)
    ct_image = ct_image.get_fdata()
    
    # ct_image = np.transpose(ct_image, (2, 1, 0))
    # ct_image = np.flip(ct_image, 2)

    from utils import resize_img
    print('bef : ', ct_image.shape)
    ct_image = resize_img(ct_image, (128, 128, 128))
    print('aft : ', ct_image.shape)

    # otsu_threshold = filters.threshold_otsu(ct_image)
    # otsu_ct_image = ct_image > otsu_threshold

    otsu_ct_image = ct_image
    # for i in range(3):
    #     otsu_threshold = filters.threshold_otsu(otsu_ct_image)
    #     otsu_ct_index = otsu_ct_image > otsu_threshold

    #     temp_arr = np.zeros(ct_image.shape)
    #     temp_arr[otsu_ct_index == True] = otsu_ct_image[otsu_ct_index == True]
    #     otsu_ct_image = temp_arr
    #     # otsu_ct_image = otsu_ct_image.astype(int)
        
    #     # nii_otsu_ct_image = nib.Nifti1Image(otsu_ct_image, affine=return_matrix)
    #     # nib.save(nii_otsu_ct_image, DATA_DIR + '/otsu{}.nii.gz'.format(i+1))

    # verts, faces, normals, values = marching_cubes(ellip_double, 0)
    print('otsu_ct_image : ', otsu_ct_image.shape)
    # otsu_ct_image[:, :, 235:] = 0
    # otsu_ct_image[:, :, :10] = 0
    # otsu_ct_image[:, :, 107:] = 0
    # otsu_ct_image[:, :, :5] = 0
    vertices, faces, _, _ = marching_cubes(otsu_ct_image, level=0.5)

    # vertices[:, 0] = vertices[:, 0] * ps
    # vertices[:, 1] = vertices[:, 1] * ps
    # vertices[:, 2] = vertices[:, 2] * ss

    # stl_mesh = mesh.Mesh(np.zeros(faces.shape[0]))
    stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

    for i, f in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = vertices[f[j], :]

    # STL 파일로 저장
    stl_mesh.save(save_path)

    # # PyMesh를 사용하여 STL 데이터를 smoothing
    # pymesh_mesh = form_mesh(stl_mesh.vectors.copy(), stl_mesh.faces.copy())

    # # Subdivider를 사용하여 STL 데이터를 세분화 및 smoothing
    # subdivider = Subdivider(pymesh_mesh)
    # subdivider.uniform_remeshing(target_edge_length=0.1)  # target_edge_length를 조절하여 smoothing 강도를 조절할 수 있음

    # # Smoothing된 데이터를 얻음
    # smoothed_vertices = subdivider.vertices()
    # smoothed_faces = subdivider.faces()

    # # Smoothing된 데이터로 새로운 STL 메시 생성
    # smoothed_stl_mesh = mesh.Mesh(np.zeros(smoothed_faces.shape[0], dtype=mesh.Mesh.dtype))
    # for i, f in enumerate(smoothed_faces):
    #     for j in range(3):
    #         smoothed_stl_mesh.vectors[i][j] = smoothed_vertices[f[j], :]

    # stl_mesh.save(mesh22_path)

ct2mesh(ct_path, mesh2_path)
gradient_mesh_index, gradient_ct_index = gt_correspondence(gradient_ct_path)

# stl_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))

# from vedo import *
from vedo import show, Points, Line


mesh1 = trimesh.load(mesh1_path)
mesh2 = trimesh.load(mesh2_path)

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
rand_index = np.random.choice(gradient_mesh_index.shape[0], size=50, replace=False)
sampled_vertices1 = gradient_mesh_index[rand_index]
pc1 = Points(sampled_vertices1, r=15)
pc1.cmap("jet", list(range(len(sampled_vertices1))))

# n2 = mesh2.vertices.shape[0]
# sampled_vertices2 = mesh2.vertices[np.random.choice(n2, 100)]
# sampled_vertices2 = gradient_mesh_index[rand_index]
sampled_vertices2 = gradient_mesh_index[rand_index]
sampled_vertices2 += translation_vector
pc2 = Points(sampled_vertices2, r=15)
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