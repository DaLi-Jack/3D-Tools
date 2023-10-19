import os

Manifold_path = '/home/nijunfeng/mycode/Manifold/build/manifold'
input_path = 'data/raw_model.obj'
output_path = 'data/watertight_model.obj'

os.system("timeout 10 %s %s %s 10000"%(Manifold_path,input_path,output_path))
print('done')
