import cv2
# import matplotlib.pyplot as plt
from PIL import Image
import bpy
import bpy
from mathutils import Vector
from mathutils.noise import random_unit_vector
import pickle
import math


def read_img():
    # 生成的数据
    img_path0 = "/home/pxn-lyj/Egolee/programs/DiffuRank-liyj/example_material/Cap3D_imgs/9bc465a181f349908b95d030f541663a/00000.png"
    # cap3d的数据
    img_path1 = "/home/pxn-lyj/Egolee/data/cap3d_data/Cap3D_Objaverse_renderimgs/0a0bec73aeb9466e875ae760fd75e64a/00000.png"
    # objverse的数据
    img_path2 = "/home/pxn-lyj/Egolee/data/objaverse_data/views_release/0a0c6d3b5f58499db8d6d649ba8de189/000.png"
    # sharpnet的数据
    img_path3 = "/home/pxn-lyj/Egolee/data/shapenet_srn_data/srn_cars/cars_train/1a1dcd236a1e6133860800e6696b8284/rgb/000001.png"


    # img0 = cv2.imread(img_path0)   # 这种方式读取背景是黑的 BGR
    # img1 = cv2.imread(img_path1)
    # img2 = cv2.imread(img_path2)
    # img3 = cv2.imread(img_path3)

    img0 = Image.open(img_path0)     # 采用这种方式读取 RGB
    img1 = Image.open(img_path1)
    img2 = Image.open(img_path2)
    img3 = Image.open(img_path3)     # 背景填充是白色没问题

    # 修改use_workbench=True 保存的才是RGBA的图像
    # def render_scene(output_path, fast_mode: bool, extract_material: bool, basic_lighting: bool):
    #     use_workbench = True
    #     # use_workbench = False

    # plt.subplot(4, 1, 1)  # 按照RGB的方式可视化
    # plt.imshow(img0)
    #
    # plt.subplot(4, 1, 2)
    # plt.imshow(img1)
    #
    # plt.subplot(4, 1, 3)
    # plt.imshow(img2)
    #
    # plt.subplot(4, 1, 4)
    # plt.imshow(img3)
    #
    # plt.show()

def debug_load_obj():
    path = "/home/pxn-lyj/Egolee/data/meshs/obj_000490/taizi1kgxiyiye.obj"
    # bpy.ops.import_scene.obj(filepath=path)
    bpy.ops.wm.obj_import(filepath=path)

def clear_lights():
    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, bpy.types.Light):
            obj.select_set(True)
    bpy.ops.object.delete()

def create_light(location, energy=1.0, angle=0.5 * math.pi / 180):
    # https://blender.stackexchange.com/questions/215624/how-to-create-a-light-with-the-python-api-in-blender-2-92
    light_data = bpy.data.lights.new(name="Light", type="SUN")
    light_data.energy = energy
    light_data.angle = angle
    light_object = bpy.data.objects.new(name="Light", object_data=light_data)

    direction = -location
    rot_quat = direction.to_track_quat("-Z", "Y")
    light_object.rotation_euler = rot_quat.to_euler()
    bpy.context.view_layer.update()

    bpy.context.collection.objects.link(light_object)
    light_object.location = location

    
def create_uniform_light(backend):
    clear_lights()
    # Random direction to decorrelate axis-aligned sides.
    pos = Vector(UNIFORM_LIGHT_DIRECTION)
    angle = 0.0092 if backend == "CYCLES" else math.pi
    create_light(pos, energy=5.0, angle=angle)    # 目前只是添加了两个光源
    create_light(-pos, energy=5.0, angle=angle)


def debug_create_uniform_light():
    create_uniform_light(backend="BLENDER_EEVEE")


if __name__ == "__main__":
    print("Start")
    # read_img()
    # debug_load_obj()
    UNIFORM_LIGHT_DIRECTION = [0.09387503, -0.63953443, -0.7630093]
    debug_create_uniform_light()
    print("End")

print("ffff")