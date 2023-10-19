'''
Description: 
Author: Junfeng Ni
Date: 2023-02-03 12:48:09
LastEditTime: 2023-04-24 19:11:48
'''
import numpy as np
import os, gzip, json
import skimage
import trimesh


# obj_path = 'generate_sdf/by_mesh_to_sdf/output'
obj_path = 'generate_sdf/by_kaolin/output'

voxels_path = os.path.join(obj_path, 'voxels.npy.gz')
f = gzip.GzipFile(voxels_path, 'r')
voxels = np.load(f)

json_path = os.path.join(obj_path, 'spacing_centroid.json')
with open(json_path, 'r') as f:
    dic = json.load(f)


spacing = float(dic['spacing'])
centroid = np.array(dic['centroid'])

vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0.0, spacing=(spacing,spacing,spacing))


recover_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

vertices = recover_mesh.vertices - recover_mesh.bounding_box.centroid + centroid
recover_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

# recover_mesh.show()
recover_mesh.export(f'{obj_path}/recover_mesh.obj')
print('done')

