from sklearn.neighbors import KDTree
import torch
import numpy as np

# import gen_utils as gu
from sklearn.decomposition import PCA

# np.random.seed(42)

class Augmentator:
    def __init__(self, augmentation_list):
        self.augmentation_list = augmentation_list
    
    def run(self, mesh_arr_input):
        mats = []
        for augmentation in self.augmentation_list:
            mesh_arr, mat = augmentation.augment(mesh_arr_input)
            mats.append(mat)
        return mesh_arr, mats

    def reload_vals(self):
        for augmentation in self.augmentation_list:
            augmentation.reload_val()

class Scaling:
    def __init__(self, trans_range):
        self.trans_range = trans_range
        assert self.trans_range[1] > self.trans_range[0]

    def augment(self, vert_arr):
        vert_arr[:,:3] = vert_arr[:,:3] * self.scale_val
        return vert_arr, self.scale_val

    def reload_val(self):
        scale_val = np.random.rand(1)
        scale_val = (scale_val) * (self.trans_range[1]-self.trans_range[0]) + self.trans_range[0]
        self.scale_val = scale_val

def axis_rotation(axis, angle):
    angle = 6.06598083
    # angle = 29.5
    # print('angle : ', angle)
    ang = np.radians(angle) 
    # ang = np.radians(180) 

    R=np.zeros((3,3))
    ux, uy, uz = axis
    cos = np.cos
    sin = np.sin
    R[0][0] = cos(ang)+ux*ux*(1-cos(ang))
    R[0][1] = ux*uy*(1-cos(ang)) - uz*sin(ang)
    R[0][2] = ux*uz*(1-cos(ang)) + uy*sin(ang)
    R[1][0] = uy*ux*(1-cos(ang)) + uz*sin(ang)
    R[1][1] = cos(ang) + uy*uy*(1-cos(ang))
    R[1][2] = uy*uz*(1-cos(ang))-ux*sin(ang)
    R[2][0] = uz*ux*(1-cos(ang))-uy*sin(ang)
    R[2][1] = uz*uy*(1-cos(ang))+ux*sin(ang)
    R[2][2] = cos(ang) + uz*uz*(1-cos(ang))
    return R

class Rotation:
    def __init__(self, angle_range, angle_axis):
        self.angle_range = angle_range
        self.angle_axis = angle_axis
        assert self.angle_range[1] > self.angle_range[0]

    def augment(self, vert_arr):
        if self.angle_axis == "pca":
            pca_axis = PCA(n_components=3).fit(vert_arr[:,:3]).components_
            rotation_mat = pca_axis
            flap_rand = ((np.random.rand(3)>0.5).astype(np.float)-0.5)*2
            pca_axis[0] *= flap_rand[0]
            pca_axis[1] *= flap_rand[1]
            pca_axis[2] *= flap_rand[2]
        else:
            # rotation_mat = gu.axis_rotation(self.angle_axis_val, self.rot_val)
            rotation_mat = axis_rotation(self.angle_axis_val, self.rot_val)
        if type(vert_arr) == torch.Tensor:
            rotation_mat = torch.from_numpy(rotation_mat).type(torch.float32).cuda()
        vert_arr[:,:3] = (rotation_mat @ vert_arr[:,:3].T).T
        if vert_arr.shape[1]==6:
            vert_arr[:,3:] = (rotation_mat @ vert_arr[:,3:].T).T
        return vert_arr, rotation_mat

    def reload_val(self):
        if self.angle_axis == "rand":
            self.angle_axis_val = np.random.rand(3)
            self.angle_axis_val = np.array([0.28108159, 0.50814053, 0.47973886])
            print('self.angle_axis_val : ' , self.angle_axis_val)
            self.angle_axis_val /= np.linalg.norm(self.angle_axis_val)
        elif self.angle_axis == "fixed":
            self.angle_axis_val = np.array([0,0,1])
        elif self.angle_axis == "pca":
            pass
        else:
            raise "rotation augmentation parameter error"
        rot_val = np.random.rand(1)
        rot_val = (rot_val) * (self.angle_range[1]-self.angle_range[0]) + self.angle_range[0]
        self.rot_val = rot_val

class Translation:
    def __init__(self, trans_range):
        self.trans_range = trans_range
        assert self.trans_range[1] > self.trans_range[0]

    def augment(self, vert_arr):
        vert_arr[:,:3] = vert_arr[:,:3] + self.trans_val
        return vert_arr, self.trans_val

    def reload_val(self):
        trans_val = np.random.rand(1,3)
        trans_val = (trans_val) * (self.trans_range[1]-self.trans_range[0]) + self.trans_range[0]
        self.trans_val = trans_val