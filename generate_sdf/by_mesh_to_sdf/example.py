'''
Description: if use 'mesh_to_voxels_padding', need to change mesh_to_sdf source code, in envs 'blender'
Author: Junfeng Ni
Date: 2023-03-17 10:20:59
LastEditTime: 2023-04-24 19:15:22
'''

import trimesh
import numpy as np
import gzip
import os
from mesh_to_sdf import sample_sdf_near_surface
from mesh_to_sdf import mesh_to_voxels, mesh_to_voxels_padding
import json
import time

def compute_unit_cube_spacing(mesh, voxel_resolution):
    """
    returns spacing for marching cube
    """
    spacing = np.max(mesh.bounding_box.extents) / (voxel_resolution - 1)
    return spacing

def compute_unit_cube_spacing_padding(mesh, padding, voxel_resolution):
    """
    returns spacing for marching cube
    add padding, attention!!! padding must be same with mesh_to_voxels_padding
    """
    spacing = (np.max(mesh.bounding_box.extents) + padding) / (voxel_resolution - 1)
    return spacing



mesh_path = 'data/watertight_model.obj'
save_path = 'generate_sdf/by_mesh_to_sdf/output'
os.makedirs(save_path, exist_ok=True)

voxels_path = os.path.join(save_path, 'voxels.npy.gz')
json_spacing_path = os.path.join(save_path, 'spacing_centroid.json')
save_spacing_centroid_dic = {}

t1 = time.time()

mesh = trimesh.load(mesh_path)


padding = 0.1
voxel_resolution = 64
spacing = compute_unit_cube_spacing_padding(mesh, padding, voxel_resolution)
save_spacing_centroid_dic['spacing'] = str(spacing)
save_spacing_centroid_dic['padding'] = str(padding)
save_spacing_centroid_dic['centroid'] = np.array(mesh.bounding_box.centroid).tolist()

voxels = mesh_to_voxels_padding(mesh, padding, voxel_resolution=voxel_resolution, pad=False, sign_method='normal', normal_sample_count=30)

# save spacing and centroid
json.dump(save_spacing_centroid_dic, open(json_spacing_path, 'w'))

f = gzip.GzipFile(voxels_path, "w")
np.save(file=f, arr=voxels)
f.close()

t2  =time.time()
print('running time {}s'.format(t2 - t1))

