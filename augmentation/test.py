import os
import nibabel as nib
from stl import mesh
import numpy as np

import sys
sys.path.append('..')
import augmentator as aug
import trimesh

def load_mesh(filename):
    mesh_data = mesh.Mesh.from_file(filename)
    vertices = mesh_data.vectors.reshape(-1, 3)
    unique_vertices, indices = np.unique(vertices, axis=0, return_index=True)
    return unique_vertices

def save_mesh(filename, vertices, faces):
    original_shape = (-1, 3)
    vertices = vertices.reshape(original_shape)

    # Create a mesh object
    mesh_data = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
    # for i, (face, vertex) in enumerate(zip(faces, vertices)):
    #     for j, vertex_index in enumerate(face):
    #         mesh_data.vectors[i][j] = vertex[vertex_index]

    for i, f in enumerate(faces):
        for j in range(3):
            mesh_data.vectors[i][j] = vertices[f[j], :]

    mesh_data.save(filename)

def save_trimesh(filename, vertices, faces):
    # Ensure vertices are of shape (-1, 3)
    vertices = vertices.reshape(-1, 3)
    faces = faces.reshape(-1, 3)
    
    # Create a trimesh object
    mesh = trimesh.Trimesh(vertices)
    mesh.vertices = vertices
    mesh.vertex_normals = faces
    mesh.export(filename)

# aug_obj = aug.Augmentator([aug.Scaling([0.85, 1.15]), aug.Rotation([-10,10], 'rand'), aug.Translation([-10, 10])])
aug_obj = aug.Augmentator([aug.Rotation([-10,10], 'rand')])

CUR_DIR = os.getcwd()
stl_file_path = os.path.join(CUR_DIR, 'datasets/lower_registration/LOWER_Result_sota.stl')
# scan_image = load_mesh(stl_file_path)
# scan_image = trimesh.load_mesh(stl_file_path)

# vertices = scan_image.vertices
# normals = scan_image.vertex_normals

# scan_image = np.concatenate([vertices, normals], axis=-1)

# for i in range(10):
#     scan_image = trimesh.load_mesh(stl_file_path)

#     vertices = scan_image.vertices
#     normals = scan_image.vertex_normals

#     np_scan_image = np.concatenate([vertices, normals], axis=-1)
#     aug_obj.reload_vals()
#     src_pcd, aug_mats = aug_obj.run(np_scan_image)

#     scan_image.vertices = src_pcd[:, :3]
#     scan_image.vertex_normals = src_pcd[:, 3:]

#     scan_image.export('datasets/Case_18/Augmentation{}.stl'.format(i))

scan_image = trimesh.load_mesh(stl_file_path)

move_mean = np.mean(scan_image.vertices)
scan_image.vertices -= np.mean(scan_image.vertices)

vertices = scan_image.vertices
normals = scan_image.vertex_normals

np_scan_image = np.concatenate([vertices, normals], axis=-1)

print('bef vertices ', np_scan_image[:, :3])
print('bef normals ', np_scan_image[:, 3:])
print()

aug_obj.reload_vals()
transformed_mesh, aug_mats = aug_obj.run(np_scan_image)
# print('transformed_mesh : ', transformed_mesh)

transformed_mesh_vertices, transformed_mesh_normals = transformed_mesh[:, :3], transformed_mesh[:, 3:]
scan_image.vertices = transformed_mesh_vertices
scan_image.export('datasets/lower_registration/LOWER_Result_sota_augment.stl')

scan_image.vertices = transformed_mesh_vertices + move_mean
scan_image.export('datasets/lower_registration/LOWER_Result_sota_augment_nottrans.stl')
restored_mesh_vertices = np.matmul(transformed_mesh_vertices, np.linalg.inv(aug_mats[0]).T)
restored_mesh_normals = np.matmul(transformed_mesh_normals, np.linalg.inv(aug_mats[0]).T)

print('aug_mats[0] : ', aug_mats[0])
# print('aft scan_image.vertices ', scan_image.vertices)

# label_mat = np.matmul(aug_mats, np.linalg.inv(gt_mat[0]))

print('aft vertices ', restored_mesh_vertices)
print('aft normals ', restored_mesh_normals)
print()

# restored_mesh = trimesh.Trimesh(vertices=restored_mesh_vertices, faces=restored_mesh_normals)

restored_mesh = trimesh.Trimesh(vertices=np.array(restored_mesh_vertices),
    vertex_normals=np.array(restored_mesh_normals),
    faces=scan_image.faces,
    )
restored_mesh.export('datasets/Case_18/LOWER_Result_sota_restored.stl')


mesh = trimesh.load_mesh(stl_file_path)
# curvatures = mesh.vertex_curvature
vertex_normals = mesh.vertex_normals
curvatures = np.linalg.norm(np.gradient(vertex_normals, axis=0), axis=1)

mean_curvature = np.mean(curvatures)
print('mean_curvature : ', mean_curvature)
filtering_indices = curvatures > mean_curvature
print('bef : ', mesh.vertices.shape)
high_curvature_vertices = mesh.vertices[filtering_indices]
high_curvature_normals = mesh.vertex_normals[filtering_indices]
# high_curvature_faces = mesh.faces[filtering_indices]

mesh.vertices = high_curvature_vertices
mesh.vertex_normals = high_curvature_normals
print('aft : ', mesh.vertices.shape)
print('mesh : ', mesh)
# mesh.faces = high_curvature_faces

curvature_mesh = trimesh.Trimesh(vertices=np.array(high_curvature_vertices),
    vertex_normals=np.array(high_curvature_normals),
    )
# restored_mesh.export('datasets/Case_18/LOWER_Result_sota_restored.stl')
curvature_mesh.export('datasets/lower_registration/LOWER_Result_sota_curvature.ply')


# mesh.export('datasets/lower_registration/LOWER_Result_sota_curvature.stl')
# restored_mesh.show()


# scan_image.export('datasets/Case_18/LOWER_Result_sota_augment.stl')
# print('aug_mats : ', aug_mats)