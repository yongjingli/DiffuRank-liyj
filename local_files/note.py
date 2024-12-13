# provide the code for extracting colorful pointclouds (support Objaverse objects)
# python extract_pointcloud.py

# LVIS数据子集
# https://huggingface.co/datasets/tiange/Cap3D/discussions/16
# https://huggingface.co/datasets/tiange/Cap3D/tree/main/misc/LVIS_diffurank

# 这些下载的数据是objaverse-XL的高质量subset
# all the objects captioned by our two papers are from Objaverse-XL 1.3 million highquality subset
# Also, all the Objaverse-1.0 objects are included in this highquality subset of Objaverse-XL.
#  Its format corresponding to Blender output.
# blender的版本
# https://huggingface.co/datasets/tiange/Cap3D/blob/main/misc/blender.zip

# 下载blender
# wget https://huggingface.co/datasets/tiange/Cap3D/resolve/main/misc/blender.zip
# unzip blender.zip

# 需要将use_workbench设置True保存的图像才是RGBA的,要不然会保存为4张图像

# 运行生成渲染图片的命令
# ./blender/blender-3.4.1-linux-x64/blender -b -P render_script_type2.py -- --object_path_pkl './example_material/example_object_path.pkl' --parent_dir './example_material'

# 缺少mathutils库的安装
# 采用+号的方式发现缺少Python.h的文件
# mathutils的安装方式
# git clone https://gitlab.com/ideasman42/blender-mathutils.git
# cd blender-mathutils
# # To build you can choose between pythons distutils or CMake.
# # distutils:
# python setup.py build
# sudo python setup.py install

# 缺少Python.h的文件的解决方式，将其他python3.10的头文件都复制过来
# cp /home/pxn-lyj/anaconda3/envs/sam2/include/python3.10/*  /home/pxn-lyj/Egolee/programs/DiffuRank-liyj/blender/blender-3.4.1-linux-x64/3.4/python/include/python3.10
# python指定python解释器的位置
# /home/pxn-lyj/Egolee/programs/DiffuRank-liyj/blender/blender-3.4.1-linux-x64/3.4/python/bin/python3.10 setup.py build
# /home/pxn-lyj/Egolee/programs/DiffuRank-liyj/blender/blender-3.4.1-linux-x64/3.4/python/bin/python3.10 setup.py install

# https://gltf-viewer.donmccurdy.com/ GLB在线查看网站
# 采用GLB文件，对比了zero-123和type3的渲染效果，感觉差不多，采用.obj文件 设置觉得type2的效果更好一些，应该是对.obj文件有些处理
# 采用cap3d渲染的图像
#python render_script_type2.py -- --object_path_pkl ./example_material/example_object_path.pkl --parent_dir ./example_material

# 采用objverse-zero12的自带的渲染脚本
# https://github.com/cvlab-columbia/zero123/tree/main/objaverse-rendering/scripts
# python blender_scripts.py -- --object_path /home/pxn-lyj/Egolee/programs/DiffuRank-liyj/example_material/glbs/3b6197d570974a07ac9cc4e8b5feb1af.glb --output_dir ./example_material

# blender 的api 文档  https://docs.blender.org/api/current/

# 发现很多mesh模型都是场景级别的，需要看下是否可以找出一些比较好的object级别的
#  7k LVIS objects
# https://huggingface.co/datasets/tiange/Cap3D/discussions/16
# https://huggingface.co/datasets/tiange/Cap3D/tree/main/misc/LVIS_diffurank

# 在splatter-image中也是采用LVIS subset，这些质量更高一些
# Additionally, please download lvis-annotations-filtered.json from the model repository.
# This json which holds the list of IDs of objects from the LVIS subset. These assets are of higher quality.

# 如何从objaverse-xl中获取lvis的类别
# Objaverse-LVIS categories
