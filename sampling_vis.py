import os
import pymeshlab
from skimage import measure

CUR_DIR = os.getcwd()
bef_stl_path = os.path.join(CUR_DIR, 'datasets/Case_01/Lower_Result.stl')
save_stl_path = os.path.join(CUR_DIR, 'datasets/Case_01/output.stl')

# osstem_mesh = pymeshlab.MeshSet()
# osstem_mesh.load_new_mesh(bef_stl_path)
# osstem_mesh.set_selection_all(allfaces=False)
# osstem_mesh.show_polyscope()
import gen_utils as gu
from stl import mesh
import numpy as np

def load_mesh(filename):
    mesh_data = mesh.Mesh.from_file(filename)
    print('mesh_data : ', mesh_data.vectors.shape)
    return mesh_data.vectors.reshape(-1, 3)

def save_stl(vertices, stl_file_path):
    new_mesh = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
    new_mesh.vectors = vertices
    # new_vertices = np.repeat(vertices, 3, axis=1)
    # print('new_vertices : ', new_vertices.shape)
    # new_mesh.vectors = new_vertices.reshape(-1, 3, 3)
    new_mesh.save(stl_file_path)

def write_ply(filename, vertices, faces):
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
        

scan_image = load_mesh(bef_stl_path)
write_ply("input.ply", scan_image, None)
scan_image = gu.resample_pcd([scan_image], 24000, "fps")[0]

write_ply("output24000.ply", scan_image, None)

# save_stl(scan_image, save_stl_path)