"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import json
import math
import os
import random
import sys
import time
import urllib.request
import uuid
from typing import Tuple
from mathutils import Vector, Matrix
# from mathutils import Matrix
import numpy as np

import bpy
from mathutils import Vector
from debug_others import debug_rot_matrix

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default="~/.objaverse/hf-objaverse-v1/views_whole_sphere")
parser.add_argument(
    "--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"]
)
parser.add_argument("--scale", type=float, default=0.8)
parser.add_argument("--num_images", type=int, default=8)
parser.add_argument("--camera_dist", type=int, default=1.2)
    
argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print('===================', args.engine, '===================')

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO") # 创建了一个新的约束，类型为 TRACK_TO，这意味着相机将会朝向指定的目标对象
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"        # TRACK_NEGATIVE_Z 表示相机的负 Z 轴将朝向目标对象
cam_constraint.up_axis = "UP_Y"                       # 意味着相机的 Y 轴将始终保持朝上方向

# setup lighting
bpy.ops.object.light_add(type="AREA")                 # 添加一个类型为区域光源的光源对象
light2 = bpy.data.lights["Area"]                      # 获取名为 "Area" 的光源数据。请注意，光源的名称是自动生成的，通常为 "Area"
light2.energy = 3000
bpy.data.objects["Area"].location[2] = 0.5
bpy.data.objects["Area"].scale[0] = 100
bpy.data.objects["Area"].scale[1] = 100
bpy.data.objects["Area"].scale[2] = 100

# light_names = [obj.name for obj in scene.objects if obj.type =="LIGHT"]  # 获取所有光源名称


render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128                # 设置每帧渲染的样本数为 128, 样本数越高，图像质量通常越好，但渲染时间也会增加
scene.cycles.diffuse_bounces = 1          # 设置漫反射光线反弹的最大次数为 1, 较低的反弹次数可以减少渲染时间，但可能会影响场景的真实感
scene.cycles.glossy_bounces = 1           # 设置镜面反射光线反弹的最大次数为 1
scene.cycles.transparent_max_bounces = 3  # 设置透明光线的最大反弹次数为 3, 这对于处理透明物体（如玻璃）时的光线传播很重要
scene.cycles.transmission_bounces = 3     # 设置透射光线的反弹次数为 3，适用于透明材料
scene.cycles.filter_width = 0.01          # 设置过滤宽度为 0.01，影响光线采样的模糊程度
scene.cycles.use_denoising = True         # 启用去噪功能，可以在渲染后减少图像中的噪点
scene.render.film_transparent = True      # 将渲染背景设置为透明，这在合成时非常有用

UNIFORM_LIGHT_DIRECTION = [0.09387503, -0.63953443, -0.7630093]
BASIC_AMBIENT_COLOR = 0.3                # 设置基础环境光颜色（BASIC_AMBIENT_COLOR）
BASIC_DIFFUSE_COLOR = 0.7

bpy.context.preferences.addons["cycles"].preferences.get_devices()
# Set the device_type
bpy.context.preferences.addons[
    "cycles"
].preferences.compute_device_type = "CUDA"  # or "OPENCL"


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )

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

def set_camera_location(elevation, azimuth, distance):
    # from https://blender.stackexchange.com/questions/18530/
    # 用于在 Blender 中随机生成相机的位置，并根据该位置计算相机的旋转，使相机朝向原点（0, 0, 0）
    x, y, z = sample_spherical(radius_min=1.5, radius_max=2.2, maxz=2.2, minz=-2.2)
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    direction = - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()
    return camera

def randomize_lighting() -> None:
    light2.energy = random.uniform(300, 600)
    bpy.data.objects["Area"].location[0] = random.uniform(-1., 1.)
    bpy.data.objects["Area"].location[1] = random.uniform(-1., 1.)
    bpy.data.objects["Area"].location[2] = random.uniform(0.5, 1.5)


def reset_lighting() -> None:
    light2.energy = 1000
    bpy.data.objects["Area"].location[0] = 0
    bpy.data.objects["Area"].location[1] = 0
    bpy.data.objects["Area"].location[2] = 0.5


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


# load the glb model
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


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj

# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    # R_bcam2cv = Matrix(
    #     ((1, 0,  0),
    #     (0, 1, 0),
    #     (0, 0, 1)))

    # Transpose since the rotation is object rotation, 
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]

    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:     
    T_world2bcam = -1*R_world2bcam @ location

    # # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix((
        R_world2bcam[0][:] + (T_world2bcam[0],),
        R_world2bcam[1][:] + (T_world2bcam[1],),
        R_world2bcam[2][:] + (T_world2bcam[2],)
        ))
    return RT

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


def setup_nodes(output_path, MAX_DEPTH=5.0, capturing_material_alpha: bool = False, basic_lighting: bool = False):
    # 一个用于设置 Blender 的节点系统的函数 setup_nodes，该函数在合成器中创建节点以处理图像和深度信息
    # output_path: 输出文件的基本路径，用于保存生成的图像文件。
    # MAX_DEPTH: 最大深度值，用于深度图的归一化。
    # capturing_material_alpha: 布尔值，指示是否捕获材质的 alpha 通道。
    # basic_lighting: 布尔值，指示是否启用基础光照计算。

    tree = bpy.context.scene.node_tree
    links = tree.links

    # 这段代码删除当前节点树中的所有节点，确保每次调用该函数时都是从一个干净的状态开始
    for node in tree.nodes:
        tree.nodes.remove(node)

    # Helpers to perform math on links and constants.
    def node_op(op: str, *args, clamp=False):
        node = tree.nodes.new(type="CompositorNodeMath")
        node.operation = op
        if clamp:
            node.use_clamp = True
        for i, arg in enumerate(args):
            if isinstance(arg, (int, float)):
                node.inputs[i].default_value = arg
            else:
                links.new(arg, node.inputs[i])
        return node.outputs[0]

    def node_clamp(x, maximum=1.0):
        return node_op("MINIMUM", x, maximum)

    def node_mul(x, y, **kwargs):
        return node_op("MULTIPLY", x, y, **kwargs)

    def node_add(x, y, **kwargs):
        return node_op("ADD", x, y, **kwargs)

    def node_abs(x, **kwargs):
        return node_op("ABSOLUTE", x, **kwargs)

    # 创建一个输入节点，用于接收场景的渲染层。
    input_node = tree.nodes.new(type="CompositorNodeRLayers")
    input_node.scene = bpy.context.scene
    # CompositorNodeRLayers 是 Blender 中合成节点系统中的一个节点，用于访问场景的渲染层信息。它提供了对渲染结果的各种输出，包括颜色、深度、法线等数据。这个节点是合成工作流程中的重要组成部分，特别是在后期制作和特效处理中。
    input_sockets = {}
    for output in input_node.outputs:
        input_sockets[output.name] = output

    if capturing_material_alpha:
        color_socket = input_sockets["Image"]
    else:
        raw_color_socket = input_sockets["Image"]
        # 在 Blender 的合成节点系统中计算基础光照的部分
        if basic_lighting:
            # Compute diffuse lighting
            normal_xyz = tree.nodes.new(type="CompositorNodeSeparateXYZ")
            tree.links.new(input_sockets["Normal"], normal_xyz.inputs[0])
            normal_x, normal_y, normal_z = [normal_xyz.outputs[i] for i in range(3)]
            dot = node_add(
                node_mul(UNIFORM_LIGHT_DIRECTION[0], normal_x),
                node_add(
                    node_mul(UNIFORM_LIGHT_DIRECTION[1], normal_y),
                    node_mul(UNIFORM_LIGHT_DIRECTION[2], normal_z),
                ),
            )
            diffuse = node_abs(dot)
            # Compute ambient + diffuse lighting
            brightness = node_add(BASIC_AMBIENT_COLOR, node_mul(BASIC_DIFFUSE_COLOR, diffuse))
            # Modulate the RGB channels using the total brightness.
            rgba_node = tree.nodes.new(type="CompositorNodeSepRGBA")
            tree.links.new(raw_color_socket, rgba_node.inputs[0])
            combine_node = tree.nodes.new(type="CompositorNodeCombRGBA")
            for i in range(3):
                tree.links.new(node_mul(rgba_node.outputs[i], brightness), combine_node.inputs[i])
            tree.links.new(rgba_node.outputs[3], combine_node.inputs[3])
            raw_color_socket = combine_node.outputs[0]

        # We apply sRGB here so that our fixed-point depth map and material
        # alpha values are not sRGB, and so that we perform ambient+diffuse
        # lighting in linear RGB space.
        color_node = tree.nodes.new(type="CompositorNodeConvertColorSpace")
        # color_node.from_color_space = "Linear"
        color_node.to_color_space = "sRGB"
        tree.links.new(raw_color_socket, color_node.inputs[0])
        color_socket = color_node.outputs[0]

    split_node = tree.nodes.new(type="CompositorNodeSepRGBA")
    tree.links.new(color_socket, split_node.inputs[0])
    # Create separate file output nodes for every channel we care about.
    # The process calling this script must decide how to recombine these
    # channels, possibly into a single image.
    for i, channel in enumerate("rgba") if not capturing_material_alpha else [(0, "MatAlpha")]:
        output_node = tree.nodes.new(type="CompositorNodeOutputFile")
        output_node.base_path = f"{output_path}_{channel}"
        links.new(split_node.outputs[i], output_node.inputs[0])

    if capturing_material_alpha:
        # No need to re-write depth here.
        return

    depth_out = node_clamp(node_mul(input_sockets["Depth"], 1 / MAX_DEPTH))    # 创建深度估计输出的节点
    output_node = tree.nodes.new(type="CompositorNodeOutputFile")
    output_node.base_path = f"{output_path}_depth"
    links.new(depth_out, output_node.inputs[0])


def render_scene(render_path):
    # render the image
    scene.render.filepath = render_path

    # output depth
    use_workbench = True

    # depth
    bpy.context.scene.render.engine == "CYCLES"
    bpy.context.scene.cycles.samples = 16
    bpy.context.scene.cycles.use_denoising = True
    bpy.context.scene.cycles.denoiser = 'OPTIX'

    bpy.context.view_layer.update()
    bpy.context.scene.use_nodes = True
    bpy.context.scene.view_layers["ViewLayer"].use_pass_z = True

    bpy.context.scene.view_settings.view_transform = "Raw"  # sRGB done in graph nodes
    bpy.context.scene.render.film_transparent = True
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.context.scene.render.image_settings.color_mode = "BW"   # 这行代码将图像的颜色模式设置为黑白（BW）。这意味着渲染将输出灰度图像，所有颜色信息将被转换为亮度值。
    bpy.context.scene.render.image_settings.color_depth = "16"  # 将图像的颜色深度设置为 16 位。这意味着每个颜色通道将有 65536 个可能的值
    bpy.context.scene.render.filepath = render_path

    setup_nodes(render_path)
    bpy.ops.render.render(write_still=True)

    # bpy.ops.render.render(write_still=True)

    for channel_name in ["r", "g", "b", "a", "depth"]:
        sub_dir = f"{render_path}_{channel_name}"
        image_path = os.path.join(sub_dir, os.listdir(sub_dir)[0])
        name, ext = os.path.splitext(render_path)
        if channel_name == "depth" or not use_workbench:
            os.rename(image_path, f"{name}_{channel_name}{ext}")
        else:
            os.remove(image_path)
        os.removedirs(sub_dir)

    # RGB
    if use_workbench:
        # bpy.context.scene.use_nodes 是一个布尔属性，用于控制当前场景是否使用合成节点。如果将其设置为 False，则表示禁用节点系统。
        # 在这种情况下，Blender 将不再使用合成节点来处理渲染结果，所有的图像处理将依赖于传统的渲染设置。

        bpy.context.scene.use_nodes = False
        bpy.context.scene.render.engine == "CYCLES"
        bpy.context.scene.cycles.samples = 16
        bpy.context.scene.cycles.use_denoising = True
        bpy.context.scene.cycles.denoiser = 'OPTIX'
        bpy.context.scene.render.image_settings.color_mode = "RGBA"
        bpy.context.scene.render.image_settings.color_depth = "16"
        os.remove(render_path)
        bpy.ops.render.render(write_still=True)


def scene_fov():
    x_fov = bpy.context.scene.camera.data.angle_x
    y_fov = bpy.context.scene.camera.data.angle_y
    width = bpy.context.scene.render.resolution_x
    height = bpy.context.scene.render.resolution_y
    if bpy.context.scene.camera.data.angle == x_fov:
        y_fov = 2 * math.atan(math.tan(x_fov / 2) * height / width)
    else:
        x_fov = 2 * math.atan(math.tan(y_fov / 2) * width / height)
    return x_fov, y_fov


def write_camera_metadata(camera, path):
    x_fov, y_fov = scene_fov()
    bbox_min, bbox_max = scene_bbox()
    # matrix = bpy.context.scene.camera.matrix_world
    matrix = camera.matrix_world
    _, _, scale = camera.matrix_world.decompose()[0:3]
    with open(path, "w") as f:
        json.dump(
            dict(
                format_version=6,
                max_depth=5.0,
                bbox=[list(bbox_min), list(bbox_max)],
                origin=list(matrix.col[3])[:3],
                x_fov=x_fov,
                y_fov=y_fov,
                x=[t/scale.x for t in list(matrix.col[0])[:3]],
                y=[t/scale.y for t in list(-matrix.col[1])[:3]],
                z=[t/scale.z for t in list(-matrix.col[2])[:3]],
            ),
            f,
        )

def rotation_matrix_to_quaternion(R):
    # 确保输入是一个 3x3 矩阵
    assert R.shape == (3, 3)

    # 计算四元数的分量
    w = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    x = (R[2, 1] - R[1, 2]) / (4.0 * w)
    y = (R[0, 2] - R[2, 0]) / (4.0 * w)
    z = (R[1, 0] - R[0, 1]) / (4.0 * w)

    return np.array([w, x, y, z])

def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)

    reset_scene()

    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene()

    # create an empty object to track
    # 将之前创建的相机约束的目标设置为刚刚创建的空对象。这样，相机将会跟踪这个空对象的位置, 可以在 3D 视图中看到并操作这个空对象
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    randomize_lighting()
    for i in range(args.num_images):
        # # set the camera position
        # theta = (i / args.num_images) * math.pi * 2
        # phi = math.radians(60)
        # point = (
        #     args.camera_dist * math.sin(phi) * math.cos(theta),
        #     args.camera_dist * math.sin(phi) * math.sin(theta),
        #     args.camera_dist * math.cos(phi),
        # )
        # # reset_lighting()
        # cam.location = point

        # set camera
        camera = randomize_camera()

        # render the image
        render_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.png")
        render_scene(render_path)

        # scene.render.filepath = render_path
        # bpy.ops.render.render(write_still=True)

        # save camera RT matrix
        RT = get_3x4_RT_matrix_from_blender(camera)
        RT_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.npy")
        np.save(RT_path, RT)

        write_camera_metadata(camera, os.path.join(args.output_dir, object_uid, f"{i:03d}.json"))


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path

        # local_path = "/home/pxn-lyj/Egolee/data/meshs/obj_000490/taizi1kgxiyiye.obj"
        save_images(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
