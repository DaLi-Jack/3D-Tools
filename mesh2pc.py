import trimesh

mesh_path = './data/raw_model.obj'
mesh = trimesh.load(mesh_path)
# mesh.show()

pc = mesh.sample(10000)
pc = trimesh.PointCloud(pc)
# pc.show()
pc.export('./data/pc.obj')

print('done')
