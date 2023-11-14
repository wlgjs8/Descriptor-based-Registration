import os
import numpy as np
import nibabel as nib

from scipy.ndimage import sobel

return_matrix = np.array([
                [-1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ])

DATA_DIR = os.path.join(os.getcwd(), 'datasets')
# case_list = ['Case_01', 'Case_03']
case_list = [
    'Case_02', 'Case_09', 'Case_10', 'Case_11', 'Case_12',
    'Case_14', 'Case_15', 'Case_18'
]

for case_idx in case_list:
    print('case_idx : >> ', case_idx)
    CASE_DIR = os.path.join(DATA_DIR, case_idx)
    image_path = os.path.join(CASE_DIR, '{}_original_dcm_arr.nii.gz'.format(case_idx))
    image = nib.load(image_path)
    image = image.get_fdata()

    gradient_x = sobel(image, axis=0)
    gradient_y = sobel(image, axis=1)
    gradient_z = sobel(image, axis=2)

    # Gradient 벡터의 크기 계산 (3D 엣지 강도)
    edge_magnitude = np.sqrt(gradient_x**2 + gradient_y**2 + gradient_z**2)

    edge_magnitude = (edge_magnitude-np.min(edge_magnitude))/(np.max(edge_magnitude)-np.min(edge_magnitude))
    mean_gradient = np.mean(edge_magnitude)

    edge_magnitude = np.transpose(edge_magnitude, (2, 1, 0))
    edge_magnitude = np.flip(edge_magnitude, 2)

    nii_edge_magnitude = nib.Nifti1Image(edge_magnitude, affine=return_matrix)
    nib.save(nii_edge_magnitude, CASE_DIR + '/edge_gradient_normalize.nii.gz')