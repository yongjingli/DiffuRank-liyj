import copy
from mathutils import Vector, Matrix
import cv2
# import matplotlib.pyplot as plt
from PIL import Image
import bpy
import bpy
from mathutils import Vector
from mathutils.noise import random_unit_vector
import pickle
import math
import numpy as np
import mathutils
import os
import random
UNIFORM_LIGHT_DIRECTION = [0.09387503, -0.63953443, -0.7630093]

bpy.ops.object.light_add(type="AREA")
light2 = bpy.data.lights["Area"]
light2.energy = 3000
bpy.data.objects["Area"].location[2] = 0.5
bpy.data.objects["Area"].scale[0] = 100
bpy.data.objects["Area"].scale[1] = 100
bpy.data.objects["Area"].scale[2] = 100


def read_render_img():
    # 读取渲染出来的数据，RGBA读取，以及读取的背景颜色
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
    # 读取obj文件，不同的版本的blender采用读取的接口不一样
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
    # 创建光源，可以在blender中查看具体光源的位置
    clear_lights()
    # Random direction to decorrelate axis-aligned sides.
    pos = Vector(UNIFORM_LIGHT_DIRECTION)
    angle = 0.0092 if backend == "CYCLES" else math.pi
    create_light(pos, energy=5.0, angle=angle)    # 目前只是添加了两个光源
    create_light(-pos, energy=5.0, angle=angle)


def debug_create_uniform_light():
    create_uniform_light(backend="BLENDER_EEVEE")


def load_depth():
    # 读取渲染出来的深度,
    # depth_path = "/home/pxn-lyj/Egolee/programs/DiffuRank-liyj/example_material/Cap3D_imgs/9bc465a181f349908b95d030f541663a/00001_depth.png"
    depth_path = "/home/pxn-lyj/Egolee/programs/DiffuRank-liyj/local_files/example_material/taizi1kgxiyiye/000_depth.png"
    depth = Image.open(depth_path)
    depth = np.array(depth)
    print(depth.shape)

    # 获取深度的方式如下
    max_store_num = np.iinfo(depth.dtype).max
    max_depth = 5.0
    cap3d_depth = depth / max_store_num * max_depth


def debug_camera_matrix_world_rot_matrix():
    # 对比bpy.context.scene.camera.matrix_world.decompose() 与bpy.context.scene.camera.matrix_world的区别
    # 最大的区别在于bpy.context.scene.camera.matrix_world的旋转矩阵包含了scale在里面，实际上相机位姿不需要包含scale，如何scale不为1时，需要对矩阵进行进一步的/scale
    import bpy
    # rotation_matrix_1_scale 与 rotation_matrix_2等价
    # 方法 1
    # camera = bpy.data.objects['Camera']  # 替换为你的摄像机名称
    # location, rotation, scale = camera.matrix_world.decompose()[0:3]
    location, rotation, scale = bpy.context.scene.camera.matrix_world.decompose()[0:3]
    rotation_matrix_1 = rotation.to_matrix()

    # 设置旋转+scale
    rotation_matrix_1_scale = copy.deepcopy(rotation_matrix_1)
    rotation_matrix_1_scale.col[0] *= scale.x
    rotation_matrix_1_scale.col[1] *= scale.y
    rotation_matrix_1_scale.col[2] *= scale.z

    # 方法 2
    matrix = bpy.context.scene.camera.matrix_world
    rotation_matrix_2 = matrix.to_3x3()

    # 比较两个旋转矩阵
    are_equal = (rotation_matrix_1 == rotation_matrix_2)

    m_scale_0 = (rotation_matrix_1.col[0][0] + 1e-6)/(rotation_matrix_2.col[0][0] + 1e-6)
    m_scale_1 = (rotation_matrix_1.col[0][1] + 1e-6)/(rotation_matrix_2.col[0][1] + 1e-6)
    m_scale_2 = (rotation_matrix_1.col[0][2] + 1e-6)/(rotation_matrix_2.col[0][2] + 1e-6)
    m_scale_3 = (rotation_matrix_1.col[1][0] + 1e-6)/(rotation_matrix_2.col[1][0] + 1e-6)

    # 打印结果
    print("scale:", scale)
    print("m_scale:", m_scale_0, m_scale_1, m_scale_2, m_scale_3)
    print("Rotation Matrix 1:", rotation_matrix_1)
    print("Rotation Matrix 1 scale:", rotation_matrix_1_scale)
    print("Rotation Matrix 2:", rotation_matrix_2)
    print("Are the rotation matrices equal?", are_equal)
    exit(1)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def debug_camera_matrix_world_scale():
    # camera.matrix_world.decompose()是不包含scale的
    # 对于cap3d，为scale后创建了相机，此时的相机scale为1,不受前面的影响，所有保存旋转矩阵时
    # 采用bpy.context.scene.camera.matrix_world.col的方式保存矩阵也是正确的，此时的scale为1

    location, rotation, scale = bpy.context.scene.camera.matrix_world.decompose()[0:3]
    print(location, rotation, scale)

    scale = 0.6
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    location, rotation, scale = bpy.context.scene.camera.matrix_world.decompose()[0:3]
    print(location, rotation, scale)
    # 相机的位置是不需要考虑scale的, 如果是通过matrix = bpy.context.scene.camera.matrix_world得到位姿，需要注意当前的scale是否为1
    # 如果不为1,需要对旋转矩阵进行scale的除

    camera_data = bpy.data.cameras.new(name="Camera")
    camera_object = bpy.data.objects.new("Camera", camera_data)
    bpy.context.scene.collection.objects.link(camera_object)
    bpy.context.scene.camera = camera_object
    location, rotation, scale = bpy.context.scene.camera.matrix_world.decompose()[0:3]
    print(location, rotation, scale)
    # 在zero123中是在scale(normalize_scene)前就创建了相机，scale为整体的scale尺寸
    # 在render_script_type2(cap3d)中,是在normalize_scene后，创建相机create_camera,采用bpy.context.scene.camera.matrix_world获取相机位姿时scale为1

def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    elif object_path.endswith(".obj"):
        try:
            bpy.ops.import_scene.obj(filepath=object_path)
        except:
            bpy.ops.wm.obj_import(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")

def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)

def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def randomize_lighting() -> None:
    light2.energy = random.uniform(300, 600)
    bpy.data.objects["Area"].location[0] = random.uniform(-1., 1.)
    bpy.data.objects["Area"].location[1] = random.uniform(-1., 1.)
    bpy.data.objects["Area"].location[2] = random.uniform(0.5, 1.5)

def set_camera_location(elevation, azimuth, distance):
    # from https://blender.stackexchange.com/questions/18530/
    # 用于在 Blender 中随机生成相机的位置，并根据该位置计算相机的旋转，使相机朝向原点（0, 0, 0）
    x, y, z = sample_spherical(radius_min=1.5, radius_max=2.2, maxz=2.2, minz=-2.2)
    # print(x, y, z)
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z
    # print(camera.location)
    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    # print(camera.location)
    return camera

def sample_spherical(radius=3.0, maxz=3.0, minz=0.):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        vec[2] = np.abs(vec[2])
        vec = vec / np.linalg.norm(vec, axis=0) * radius
        if maxz > vec[2] > minz:
            correct = True
    return vec

def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
#         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec

def randomize_camera():
    elevation = random.uniform(0., 90.)
    azimuth = random.uniform(0., 360)
    distance = random.uniform(0.8, 1.6)
    return set_camera_location(elevation, azimuth, distance)


def show_object_in_blender():
    # 在blender中调用，查看物体与光源等物体
    # import bpy
    # import os
    # filename = "/home/pxn-lyj/Egolee/programs/DiffuRank-liyj/test_blender_scripts.py"
    # exec(compile(open(filename).read(), filename, 'exec'))

    # from blender_script_zero123 import reset_scene, load_object, normalize_scene, randomize_lighting, randomize_camera
    object_file = "/home/pxn-lyj/Egolee/data/meshs/obj_000490/taizi1kgxiyiye.obj"
    # object_file = "/home/pxn-lyj/Egolee/programs/DiffuRank-liyj/example_material/glbs/822202b6b8594342b632b0a39432642a.glb"
    context = bpy.context
    scene = context.scene
    render = scene.render

    cam = scene.objects["Camera"]
    cam.location = (0, 1.2, 0)
    cam.data.lens = 35
    cam.data.sensor_width = 32

    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"

    reset_scene()
    load_object(object_file)
    normalize_scene()

    # create an empty object to track
    # 将之前创建的相机约束的目标设置为刚刚创建的空对象。这样，相机将会跟踪这个空对象的位置, 可以在 3D 视图中看到并操作这个空对象
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    randomize_lighting()

    camera = randomize_camera()

    bpy.ops.render.render(write_still=True)    # 需要调用这个才会刷新在世界坐标的位置  matrix_world

    print(camera.location)   # camera.location 是相机对象在其本地坐标系中的位置。它表示相机在其父对象（如果有的话）坐标系中的位置。
    location, rotation, scale = camera.matrix_world.decompose()[0:3]  # 用途：用于获取相机在全局坐标系中的确切位置。它不受父对象的影响，直接反映了相机在场景中的位置。
    print(location, rotation, scale)
    # location 与randomize_camera中设置的位置保持一致


def debug_save_rt_zero123_cap3d():
    # 对比zero123与cap3d的保存相机位姿时的区别

    # get_3x4_RT_matrix_from_blender
    context = bpy.context
    scene = context.scene
    cam = scene.objects["Camera"]

    scale = 0.6
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale

    # Apply scale to matrix_world.
    bpy.context.view_layer.update()

    location, rotation, scale = cam.matrix_world.decompose()[0:3]

    R_world2bcam = rotation.to_matrix().transposed()   # 将c2w转为w2c
    T_world2bcam = -1*R_world2bcam @ location          # T_w2c = -1 * R_w2c * T_c2w
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
    ))

    RT_hom = np.eye(4)
    RT_hom[:3, :] = np.array(RT)
    opengl_to_colmap = np.array([[1, 0, 0, 0],
                                 [0, -1, 0, 0],
                                 [0, 0, -1, 0],
                                 [0, 0, 0, 1]], dtype=np.float32)


    RT_colmap = opengl_to_colmap @ RT_hom

    print(location, rotation, scale)

    matrix = cam.matrix_world
    origin = list(matrix.col[3])[:3]
    x_vector = [t/scale.x for t in list(matrix.col[0])[:3]]
    y_vector = [t/scale.y for t in list(-matrix.col[1])[:3]]
    z_vector = [t/scale.z for t in list(-matrix.col[2])[:3]]
    rotation_matrix = np.array([x_vector, y_vector, z_vector]).T   # c2w  这里的转置是为了将行向量转为列向量
    translation_vector = np.array(origin)

    rt_matrix = np.eye(4)
    rt_matrix[:3, :3] = rotation_matrix
    rt_matrix[:3, 3] = translation_vector
    w2c = np.linalg.inv(rt_matrix)

    print(np.allclose(w2c, RT_colmap))
    print(w2c-RT_colmap)
    print("ff")


if __name__ == "__main__":
    print("Start")
    # read_render_img()
    # debug_load_obj()
    # debug_create_uniform_light()
    # load_depth()
    # debug_camera_matrix_world_rot_matrix()
    # debug_camera_matrix_world_scale()
    # show_object_in_blender()
    debug_save_rt_zero123_cap3d()
    print("End")

print("ffff")