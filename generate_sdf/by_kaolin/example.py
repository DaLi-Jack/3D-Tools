from copy import deepcopy
import json
import os
import time
import gzip

import kaolin
import numpy as np
import torch
import trimesh
from kaolin.metrics.trianglemesh import point_to_mesh_distance
from kaolin.ops.mesh import check_sign, index_vertices_by_faces

def to_tensor(data, device='cuda'):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    else:
        raise NotImplementedError()

def compute_unit_cube_spacing_padding(mesh, padding, voxel_resolution):
    """
    returns spacing for marching cube
    add padding, attention!!! padding must be same with mesh_to_voxels_padding
    """
    spacing = (np.max(mesh.bounding_box.extents) + padding) / (voxel_resolution - 1)
    return spacing

def scale_to_unit_cube_padding(mesh, padding):
    """
    add padding
    """
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump().sum()

    vertices = mesh.vertices - mesh.bounding_box.centroid
    vertices *= 2 / (np.max(mesh.bounding_box.extents) + padding)

    return trimesh.Trimesh(vertices=vertices, faces=mesh.faces)


class KaolinMeshModel():
    def __init__(self, store_meshes=None, device="cuda"):
        """
        Args:
            `store_meshes` Optional, `list` of `Mesh`.
        """
        self.device = device
        if store_meshes is not None:
            self.update_meshes(store_meshes)
        
    def update_meshes(self, meshes):
        if meshes is not None:
            self.object_mesh_list = []
            self.object_verts_list = []
            self.object_faces_list = []
            self.object_face_verts_list = []
            for mesh in meshes:
                self.object_mesh_list.append(mesh)
                self.object_verts_list.append(torch.Tensor(self.object_mesh_list[-1].vertices).to(self.device))
                self.object_faces_list.append(torch.Tensor(self.object_mesh_list[-1].faces).long().to(self.device))
                self.object_face_verts_list.append(index_vertices_by_faces(self.object_verts_list[-1].unsqueeze(0), self.object_faces_list[-1]))
        self.num_meshes = len(meshes)
    
    def mesh_points_sd(self, mesh_idx, points):
        """
        Compute the signed distance of a specified point cloud (`points`) to a mesh (specified by `mesh_idx`).

        Args:
            `mesh_idx`: Target mesh index in stored.
            `points`: Either `list`(B) of `ndarrays`(N x 3) or `Tensor` (B x N x 3).

        Returns:
            `signed_distance`: `Tensor`(B x N)
        """
        points = to_tensor(points)
        verts = self.object_verts_list[mesh_idx].clone().unsqueeze(0).tile((points.shape[0], 1, 1))
        faces = self.object_faces_list[mesh_idx].clone()
        face_verts = self.object_face_verts_list[mesh_idx]
        
        signs = check_sign(verts, faces, points)
        dis, _, _ = point_to_mesh_distance(points, face_verts)      # Note: The calculated distance is the squared euclidean distance.
        dis = torch.sqrt(dis)                  
        return torch.where(signs, -dis, dis)
        



mesh_path = 'data/watertight_model.obj'
save_path = 'generate_sdf/by_kaolin/output'
os.makedirs(save_path, exist_ok=True)

voxels_path = os.path.join(save_path, 'voxels.npy.gz')
json_spacing_path = os.path.join(save_path, 'spacing_centroid.json')
save_spacing_centroid_dic = {}

t1 = time.time()

mesh = trimesh.load(mesh_path)

padding = 0.1
voxel_resolution = 64
device = 'cuda'                 # NOTE: must use gpu!

###### calculate spacing before mesh scale to unit cube
spacing = compute_unit_cube_spacing_padding(mesh, padding, voxel_resolution)
save_spacing_centroid_dic['spacing'] = str(spacing)
save_spacing_centroid_dic['padding'] = str(padding)
save_spacing_centroid_dic['centroid'] = np.array(mesh.bounding_box.centroid).tolist()


mesh = scale_to_unit_cube_padding(mesh, padding)


# voxelize unit cube
xs = np.linspace(-1, 1, voxel_resolution)
ys = np.linspace(-1, 1, voxel_resolution)
zs = np.linspace(-1, 1, voxel_resolution)
xx, yy, zz = np.meshgrid(xs, ys, zs, indexing='ij')
points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float32).cuda()

obj_meshes = []
obj_meshes.append(mesh)
kl = KaolinMeshModel(store_meshes=obj_meshes, device=device)
sdf = kl.mesh_points_sd(0, points.unsqueeze(0).contiguous())
voxels = sdf.reshape((voxel_resolution, voxel_resolution, voxel_resolution)).detach().cpu().numpy()



# save spacing and centroid
json.dump(save_spacing_centroid_dic, open(json_spacing_path, 'w'))

f = gzip.GzipFile(voxels_path, "w")
np.save(file=f, arr=voxels)
f.close()

t2  =time.time()
print('running time {}s'.format(t2 - t1))

