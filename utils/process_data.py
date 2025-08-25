import open3d as o3d
import glob
import pickle
import third_patry.taichi_three as t3
import numpy as np
from third_patry.taichi_three.transform import *
from pathlib import Path
import trimesh
from tqdm import tqdm
import os
import cv2
import pickle

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

np.random.seed(12138)


def save(pid, data_id, vid, save_path, extr, intr, depth, img, mask, img_hr=None):
    img_save_path = os.path.join(save_path, 'img', data_id + '_' + '%03d' % pid)
    depth_save_path = os.path.join(save_path, 'depth', data_id + '_' + '%03d' % pid)
    mask_save_path = os.path.join(save_path, 'mask', data_id + '_' + '%03d' % pid)
    parm_save_path = os.path.join(save_path, 'parm', data_id + '_' + '%03d' % pid)
    Path(img_save_path).mkdir(exist_ok=True, parents=True)
    Path(parm_save_path).mkdir(exist_ok=True, parents=True)
    Path(mask_save_path).mkdir(exist_ok=True, parents=True)
    Path(depth_save_path).mkdir(exist_ok=True, parents=True)

    depth = depth * 2.0 ** 15
    cv2.imwrite(os.path.join(depth_save_path, '{}.png'.format(vid)), depth.astype(np.uint16))
    img = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)[:, :, ::-1]
    mask = (np.clip(mask, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    cv2.imwrite(os.path.join(img_save_path, '{}.jpg'.format(vid)), img)
    if img_hr is not None:
        img_hr = (np.clip(img_hr, 0, 1) * 255.0 + 0.5).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(img_save_path, '{}_hr.jpg'.format(vid)), img_hr)
    cv2.imwrite(os.path.join(mask_save_path, '{}.png'.format(vid)), mask)
    np.save(os.path.join(parm_save_path, '{}_intrinsic.npy'.format(vid)), intr)
    np.save(os.path.join(parm_save_path, '{}_extrinsic.npy'.format(vid)), extr)


def save_func(data_id, vid, save_path, extr, intr, depth, img, normal_map, mask, img_tar1=None):
    img_save_path = os.path.join(save_path, 'img', data_id)
    depth_save_path = os.path.join(save_path, 'depth', data_id)
    mask_save_path = os.path.join(save_path, 'mask', data_id)
    parm_save_path = os.path.join(save_path, 'parm', data_id)
    Path(img_save_path).mkdir(exist_ok=True, parents=True)
    Path(parm_save_path).mkdir(exist_ok=True, parents=True)
    Path(mask_save_path).mkdir(exist_ok=True, parents=True)
    Path(depth_save_path).mkdir(exist_ok=True, parents=True)

    if depth is not None:
        depth = depth * 2.0 ** 15
        cv2.imwrite(os.path.join(depth_save_path, f'{vid}.png'), depth.astype(np.uint16),
                    [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if normal_map is not None:
        normal_map = (normal_map + 1.0) * 0.5
        normal_map = (np.clip(normal_map, 0, 1) * 255.0).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(img_save_path, f'{vid}_normal.png'), normal_map, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if img is not None:
        img = (np.clip(img, 0, 1) * 255.0 + 0.5).astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(img_save_path, f'{vid}.png'), img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if mask is not None:
        mask = (np.clip(mask, 0, 1) * 255.0 + 0.5).astype(np.uint8)
        cv2.imwrite(os.path.join(mask_save_path, f'{vid}.png'), mask, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    # if img_tar1 is not None:
    #     img_tar1 = (np.clip(img_tar1, 0, 1) * 255.0 + 0.5).astype(np.uint8)[:, :, ::-1]
    #     cv2.imwrite(os.path.join(img_save_path, f'{vid}.png'.format(vid)), img_tar1,
    #                     [cv2.IMWRITE_PNG_COMPRESSION, 9])
        # if vid in ["source", "view1", "view2", "view3", "view1_mesh", "view2_mesh", "view3_mesh"]:
        #     # cv2.imwrite(os.path.join(img_save_path, 'tar_0.png'.format(vid)), img_tar1, [cv2.IMWRITE_PNG_COMPRESSION, 9])
        #     cv2.imwrite(os.path.join(img_save_path, 'tar_{}.png'.format(vid)), img_tar1,
        #                 [cv2.IMWRITE_PNG_COMPRESSION, 9])
        #     if intr is not None:
        #         np.save(os.path.join(parm_save_path, 'tar_{}_intrinsic.npy'.format(vid)), intr)
        #     if extr is not None:
        #         np.save(os.path.join(parm_save_path, 'tar_{}_extrinsic.npy'.format(vid)), extr)
        # else:
        #     cv2.imwrite(os.path.join(img_save_path, '{}.png'.format(vid)), img_tar1, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if intr is not None:
        np.save(os.path.join(parm_save_path, '{}_intrinsic.npy'.format(vid)), intr)
    if extr is not None:
        np.save(os.path.join(parm_save_path, '{}_extrinsic.npy'.format(vid)), extr)


class StaticRenderer:
    def __init__(self, src_res, double_res=False):
        ti.init(arch=ti.cuda, device_memory_fraction=0.8)
        self.scene = t3.Scene()
        self.N = 10
        self.src_res = src_res
        self.tar_res = (src_res[0] * 2, src_res[1] * 2) if double_res else (src_res[0], src_res[1])

    def change_all(self):
        save_obj = []
        save_tex = []
        for model in self.scene.models:
            save_obj.append(model.init_obj)
            save_tex.append(model.init_tex)
        ti.init(arch=ti.cuda, device_memory_fraction=0.8)
        print('init')
        self.scene = t3.Scene()
        for i in range(len(save_obj)):
            model = t3.StaticModel(self.N, obj=save_obj[i], tex=save_tex[i])
            self.scene.add_model(model)

    def check_update(self, obj):
        temp_n = self.N
        self.N = max(obj['vi'].shape[0], self.N)
        self.N = max(obj['f'].shape[0], self.N)
        if not (obj['vt'] is None):
            self.N = max(obj['vt'].shape[0], self.N)

        if self.N > temp_n:
            self.N *= 2
            self.change_all()
            self.camera_light()

    def add_model(self, obj, tex=None):
        self.check_update(obj)
        model = t3.StaticModel(self.N, obj=obj, tex=tex)
        self.scene.add_model(model)

    def modify_model(self, index, obj, tex=None):
        self.check_update(obj)
        self.scene.models[index].init_obj = obj
        self.scene.models[index].init_tex = tex
        self.scene.models[index]._init()

    def camera_light(self):
        camera = t3.Camera(res=self.src_res)
        self.scene.add_camera(camera)

        camera_tar1 = t3.Camera(res=self.tar_res)
        self.scene.add_camera(camera_tar1)

        light_dir = np.array([0, 0, 1])
        light_list = []
        for l in range(6):
            rotate = np.matmul(rotationX(math.radians(np.random.uniform(-30, 30))),
                               rotationY(math.radians(360 // 6 * l)))
            dir = [*np.matmul(rotate, light_dir)]
            light = t3.Light(dir, color=[1.0, 1.0, 1.0])
            light_list.append(light)
        lights = t3.Lights(light_list)
        self.scene.add_lights(lights)


def render_data_myself(renderer, data_path, data_id, save_path, cam_nums, res, dis=1.0, is_thuman=False):
    obj_path = os.path.join(data_path, '%s.obj' % data_id)
    img_path = os.path.join(data_path, 'material0.jpeg')
    texture = cv2.imread(img_path)[:, :, ::-1]
    texture = np.ascontiguousarray(texture)
    texture = texture.swapaxes(0, 1)[:, ::-1, :]
    obj = t3.readobj(obj_path, scale=1)

    # height normalization
    vy_max = np.max(obj['vi'][:, 1])
    vy_min = np.min(obj['vi'][:, 1])
    human_height = 1.80 + np.random.uniform(-0.05, 0.05, 1)
    obj['vi'][:, :3] = obj['vi'][:, :3] / (vy_max - vy_min) * human_height
    obj['vi'][:, 1] -= np.min(obj['vi'][:, 1])
    look_at_center = np.array([0, 0.85, 0])
    base_cam_pitch = -8

    # randomly move the scan
    move_range = 0.1 if human_height < 1.80 else 0.05
    delta_x = np.max(obj['vi'][:, 0]) - np.min(obj['vi'][:, 0])
    delta_z = np.max(obj['vi'][:, 2]) - np.min(obj['vi'][:, 2])
    if delta_x > 1.0 or delta_z > 1.0:
        move_range = 0.01
    obj['vi'][:, 0] += np.random.uniform(-move_range, move_range, 1)
    obj['vi'][:, 2] += np.random.uniform(-move_range, move_range, 1)

    if len(renderer.scene.models) >= 1:
        renderer.modify_model(0, obj, texture)
    else:
        renderer.add_model(obj, texture)

    degree_interval = 360 / cam_nums
    angle_list1 = list(range(360 - int(degree_interval // 2), 360))
    angle_list2 = list(range(0, 0 + int(degree_interval // 2)))
    angle_list = angle_list1 + angle_list2

    # angle_base = np.random.choice(angle_list, 1)

    def render(dis, angle, look_at_center, p, renderer, render_hr=False):
        ori_vec = np.array([0, 0, dis])
        rotate = np.matmul(rotationY(math.radians(angle)), rotationX(math.radians(p)))
        fwd = np.matmul(rotate, ori_vec)
        cam_pos = look_at_center + fwd

        x_min = 0
        y_min = -25
        cx = res[0] * 0.5
        cy = res[1] * 0.5
        fx = res[0] * 0.8
        fy = res[1] * 0.8
        _cx = cx - x_min
        _cy = cy - y_min
        renderer.scene.cameras[0].set_intrinsic(fx, fy, _cx, _cy)
        renderer.scene.cameras[0].set(pos=cam_pos, target=look_at_center)
        renderer.scene.cameras[0]._init()

        if render_hr:
            fx = res[0] * 0.8 * 2
            fy = res[1] * 0.8 * 2
            _cx = (res[0] * 0.5 - x_min) * 2
            _cy = (res[1] * 0.5 - y_min) * 2

        renderer.scene.cameras[1].set_intrinsic(fx, fy, _cx, _cy)
        renderer.scene.cameras[1].set(pos=cam_pos, target=look_at_center)
        renderer.scene.cameras[1]._init()

        renderer.scene.render()
        camera = renderer.scene.cameras[0]
        camera_hr = renderer.scene.cameras[1]
        extrinsic = camera.export_extrinsic()
        intrinsic = camera.export_intrinsic()
        depth_map = camera.zbuf.to_numpy().swapaxes(0, 1)
        img = camera.img.to_numpy().swapaxes(0, 1)
        img_hr = camera_hr.img.to_numpy().swapaxes(0, 1)
        mask = camera.mask.to_numpy().swapaxes(0, 1)
        return extrinsic, intrinsic, depth_map, img, mask, img_hr

    def check(angle):
        if angle == 360:
            return 0
        elif angle > 360:
            return angle - 360
        else:
            return angle

    for index, angle_base in enumerate(angle_list):
        for pid in range(1, 37):
            angle = check(angle_base + pid * degree_interval)
            extrinsic, intrinsic, depth_map, img, mask, img_tar = render(dis, angle, look_at_center, base_cam_pitch,
                                                                         renderer, renderer.src_res != renderer.tar_res)
            save_func(f"{data_id}_{index}", angle, save_path, extrinsic, intrinsic, depth_map, img, mask, img_tar)


def render_data_version2(renderer, data_path, data_id, save_path, cam_nums, res, dis=1.0, phase="train",
                         is_thuman=False):
    obj_path = os.path.join(data_path, '%s.obj' % data_id)
    img_path = os.path.join(data_path, 'material0.jpeg')
    texture = cv2.imread(img_path)[:, :, ::-1]
    texture = np.ascontiguousarray(texture)
    texture = texture.swapaxes(0, 1)[:, ::-1, :]
    obj = t3.readobj(obj_path, scale=1)

    # height normalization
    vy_max = np.max(obj['vi'][:, 1])
    vy_min = np.min(obj['vi'][:, 1])
    human_height = 1.80 + np.random.uniform(-0.05, 0.05, 1)
    obj['vi'][:, :3] = obj['vi'][:, :3] / (vy_max - vy_min) * human_height
    obj['vi'][:, 1] -= np.min(obj['vi'][:, 1])
    look_at_center = np.array([0, 0.85, 0])
    base_cam_pitch = -8

    # randomly move the scan
    move_range = 0.1 if human_height < 1.80 else 0.05
    delta_x = np.max(obj['vi'][:, 0]) - np.min(obj['vi'][:, 0])
    delta_z = np.max(obj['vi'][:, 2]) - np.min(obj['vi'][:, 2])
    if delta_x > 1.0 or delta_z > 1.0:
        move_range = 0.01
    move1 = np.random.uniform(-move_range, move_range, 1)
    move2 = np.random.uniform(-move_range, move_range, 1)
    obj['vi'][:, 0] += move1
    obj['vi'][:, 2] += move2

    if len(renderer.scene.models) >= 1:
        renderer.modify_model(0, obj, texture)
    else:
        renderer.add_model(obj, texture)

    degree_interval = 360 / cam_nums
    angle_list1 = list(range(360 - int(degree_interval // 2), 360))
    angle_list2 = list(range(0, 0 + int(degree_interval // 2)))
    angle_list = angle_list1 + angle_list2

    angle_base = np.random.choice(angle_list, 1)[0]

    def render(dis, angle, look_at_center, p, renderer, render_hr=False):
        ori_vec = np.array([0, 0, dis])
        rotate = np.matmul(rotationY(math.radians(angle)), rotationX(math.radians(p)))
        fwd = np.matmul(rotate, ori_vec)
        cam_pos = look_at_center + fwd

        x_min = 0
        y_min = -25
        cx = res[0] * 0.5
        cy = res[1] * 0.5
        fx = res[0] * 0.8
        fy = res[1] * 0.8
        _cx = cx - x_min
        _cy = cy - y_min
        renderer.scene.cameras[0].set_intrinsic(fx, fy, _cx, _cy)
        renderer.scene.cameras[0].set(pos=cam_pos, target=look_at_center)
        renderer.scene.cameras[0]._init()

        if render_hr:
            fx = res[0] * 0.8 * 2
            fy = res[1] * 0.8 * 2
            _cx = (res[0] * 0.5 - x_min) * 2
            _cy = (res[1] * 0.5 - y_min) * 2

        renderer.scene.cameras[1].set_intrinsic(fx, fy, _cx, _cy)
        renderer.scene.cameras[1].set(pos=cam_pos, target=look_at_center)
        renderer.scene.cameras[1]._init()

        renderer.scene.render()
        camera = renderer.scene.cameras[0]
        camera_hr = renderer.scene.cameras[1]
        extrinsic = camera.export_extrinsic()
        intrinsic = camera.export_intrinsic()
        depth_map = camera.zbuf.to_numpy().swapaxes(0, 1)
        img = camera.img.to_numpy().swapaxes(0, 1)
        normal_map = camera.normal_map.to_numpy().swapaxes(0, 1)
        img_hr = camera_hr.img.to_numpy().swapaxes(0, 1)
        mask = camera.mask.to_numpy().swapaxes(0, 1)
        return extrinsic, intrinsic, depth_map, img, normal_map, mask, img_hr

    save_path_gps = os.path.join(os.path.split(save_path)[0],"GPS_render",phase)
    for pid in range(2):
        paid = {"angle_base": angle_base, "move1": move1, "move2": move1,
                "human_height": human_height}
        # angle = check(angle_base + pid * degree_interval)
        view0 = angle_base + pid * degree_interval
        view1 = (view0 + 90) % 360
        view2 = (view0 + 180) % 360
        view3 = (view0 + 270) % 360

        viewgps = (view0+degree_interval) % 360
        paid['view0'] = view0
        paid['view1'] = view1
        paid['view2'] = view2
        paid['view3'] = view3

        extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, view0, look_at_center,
                                                                                 base_cam_pitch,
                                                                                 renderer,
                                                                                 renderer.src_res != renderer.tar_res)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "source", save_path, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "source_0", save_path_gps, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)
        extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, viewgps, look_at_center,
                                                                                 base_cam_pitch,
                                                                                 renderer,
                                                                                 renderer.src_res != renderer.tar_res)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "source_1", save_path_gps, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)
        extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, view1, look_at_center,
                                                                                 base_cam_pitch,
                                                                                 renderer,
                                                                                 renderer.src_res != renderer.tar_res)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "view1", save_path, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)

        extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, view2, look_at_center,
                                                                                 base_cam_pitch,
                                                                                 renderer,
                                                                                 renderer.src_res != renderer.tar_res)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "view2", save_path, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)

        extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, view3, look_at_center,
                                                                                 base_cam_pitch,
                                                                                 renderer,
                                                                                 renderer.src_res != renderer.tar_res)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "view3", save_path, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)

        nv = []
        for n in range(1, cam_nums + 1):
            viewt = (view0 + np.random.uniform() * n * degree_interval) % 360
            nv.append(viewt)
            extrinsic, intrinsic, depth_map, img, _, mask, img_tar = render(dis, viewt, look_at_center, base_cam_pitch,
                                                                            renderer,
                                                                            renderer.src_res != renderer.tar_res)
            save_func(f"{data_id}_{str(pid).zfill(4)}", f"tar_{n}", save_path, extrinsic, intrinsic, depth_map,
                      None, None, mask, img_tar)
        paid['nv'] = nv
        os.makedirs(f"{save_path}/angles", exist_ok=True)
        with open(f"{save_path}/angles/{data_id}_{str(pid).zfill(4)}.pkl", "wb") as f:
            pickle.dump(paid, f)


def render_data_360(renderer, data_path, data_id, save_path, cam_nums, res, dis=1.0, phase="train",
                         is_thuman=False):
    obj_path = os.path.join(data_path, '%s.obj' % data_id)
    img_path = os.path.join(data_path, 'material0.jpeg')
    texture = cv2.imread(img_path)[:, :, ::-1]
    texture = np.ascontiguousarray(texture)
    texture = texture.swapaxes(0, 1)[:, ::-1, :]
    obj = t3.readobj(obj_path, scale=1)

    # height normalization
    vy_max = np.max(obj['vi'][:, 1])
    vy_min = np.min(obj['vi'][:, 1])
    human_height = 1.80 + np.random.uniform(-0.05, 0.05, 1)
    obj['vi'][:, :3] = obj['vi'][:, :3] / (vy_max - vy_min) * human_height
    obj['vi'][:, 1] -= np.min(obj['vi'][:, 1])
    look_at_center = np.array([0, 0.85, 0])
    base_cam_pitch = -8

    # randomly move the scan
    move_range = 0.1 if human_height < 1.80 else 0.05
    delta_x = np.max(obj['vi'][:, 0]) - np.min(obj['vi'][:, 0])
    delta_z = np.max(obj['vi'][:, 2]) - np.min(obj['vi'][:, 2])
    if delta_x > 1.0 or delta_z > 1.0:
        move_range = 0.01
    move1 = np.random.uniform(-move_range, move_range, 1)
    move2 = np.random.uniform(-move_range, move_range, 1)
    obj['vi'][:, 0] += move1
    obj['vi'][:, 2] += move2

    if len(renderer.scene.models) >= 1:
        renderer.modify_model(0, obj, texture)
    else:
        renderer.add_model(obj, texture)

    def render(dis, angle, look_at_center, p, renderer, render_hr=False):
        ori_vec = np.array([0, 0, dis])
        rotate = np.matmul(rotationY(math.radians(angle)), rotationX(math.radians(p)))
        fwd = np.matmul(rotate, ori_vec)
        cam_pos = look_at_center + fwd

        x_min = 0
        y_min = -25
        cx = res[0] * 0.5
        cy = res[1] * 0.5
        fx = res[0] * 0.8
        fy = res[1] * 0.8
        _cx = cx - x_min
        _cy = cy - y_min
        renderer.scene.cameras[0].set_intrinsic(fx, fy, _cx, _cy)
        renderer.scene.cameras[0].set(pos=cam_pos, target=look_at_center)
        renderer.scene.cameras[0]._init()

        if render_hr:
            fx = res[0] * 0.8 * 2
            fy = res[1] * 0.8 * 2
            _cx = (res[0] * 0.5 - x_min) * 2
            _cy = (res[1] * 0.5 - y_min) * 2

        renderer.scene.cameras[1].set_intrinsic(fx, fy, _cx, _cy)
        renderer.scene.cameras[1].set(pos=cam_pos, target=look_at_center)
        renderer.scene.cameras[1]._init()

        renderer.scene.render()
        camera = renderer.scene.cameras[0]
        camera_hr = renderer.scene.cameras[1]
        extrinsic = camera.export_extrinsic()
        intrinsic = camera.export_intrinsic()
        depth_map = camera.zbuf.to_numpy().swapaxes(0, 1)
        img = camera.img.to_numpy().swapaxes(0, 1)
        normal_map = camera.normal_map.to_numpy().swapaxes(0, 1)
        img_hr = camera_hr.img.to_numpy().swapaxes(0, 1)
        mask = camera.mask.to_numpy().swapaxes(0, 1)
        return extrinsic, intrinsic, depth_map, img, normal_map, mask, img_hr

    save_path_gps = os.path.join(os.path.split(save_path)[0],"GPS_render")
    degree_interval = 360 / 36
    for i in range(4):
        angle_base = np.random.uniform(0, 360)
        paid = {"angle_base": angle_base, "move1": move1, "move2": move1,
                "human_height": human_height}
        for pid in range(36):
            view = (angle_base + pid * degree_interval) % 360
            paid[f'view{pid}'] = view

            extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, view, look_at_center,
                                                                                     base_cam_pitch,
                                                                                     renderer,
                                                                                     renderer.src_res != renderer.tar_res)
            save_func(f"{data_id}_{str(i).zfill(4)}", f"view{pid}", save_path, extrinsic, intrinsic, depth_map, img,
                      normal_map, mask,
                      None)
            if pid == 0:
                save_func(f"{data_id}_{str(i).zfill(4)}", "source_0", save_path_gps, extrinsic, intrinsic, depth_map, img,
                          normal_map, mask, None)
                viewgps = (view + degree_interval) % 360
                extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, viewgps, look_at_center,
                                                                                         base_cam_pitch,
                                                                                         renderer,
                                                                                         renderer.src_res != renderer.tar_res)
                save_func(f"{data_id}_{str(i).zfill(4)}", "source_1", save_path_gps, extrinsic, intrinsic, depth_map,
                          img,
                          normal_map, mask,
                          None)

            os.makedirs(f"{save_path}/angles", exist_ok=True)
            with open(f"{save_path}/angles/{data_id}_{str(i).zfill(4)}.pkl", "wb") as f:
                pickle.dump(paid, f)

def render_smplxdata_360(renderer, data_path, data_id, save_path, cam_nums, res, dis=1.0, phase="train",
                         is_thuman=False):
    obj_path = os.path.join(data_path, '%s.obj' % data_id)
    img_path = os.path.join(data_path, 'material0.jpeg')
    texture = cv2.imread(img_path)[:, :, ::-1]
    texture = np.ascontiguousarray(texture)
    texture = texture.swapaxes(0, 1)[:, ::-1, :]
    obj = t3.readobj(obj_path, scale=1)

    # height normalization
    vy_max = np.max(obj['vi'][:, 1])
    vy_min = np.min(obj['vi'][:, 1])
    human_height = 1.80 + np.random.uniform(-0.05, 0.05, 1)
    obj['vi'][:, :3] = obj['vi'][:, :3] / (vy_max - vy_min) * human_height
    obj['vi'][:, 1] -= np.min(obj['vi'][:, 1])
    look_at_center = np.array([0, 0.85, 0])
    base_cam_pitch = -8

    # randomly move the scan
    move_range = 0.1 if human_height < 1.80 else 0.05
    delta_x = np.max(obj['vi'][:, 0]) - np.min(obj['vi'][:, 0])
    delta_z = np.max(obj['vi'][:, 2]) - np.min(obj['vi'][:, 2])
    if delta_x > 1.0 or delta_z > 1.0:
        move_range = 0.01
    move1 = np.random.uniform(-move_range, move_range, 1)
    move2 = np.random.uniform(-move_range, move_range, 1)
    obj['vi'][:, 0] += move1
    obj['vi'][:, 2] += move2

    if len(renderer.scene.models) >= 1:
        renderer.modify_model(0, obj, texture)
    else:
        renderer.add_model(obj, texture)

    def render(dis, angle, look_at_center, p, renderer, render_hr=False):
        ori_vec = np.array([0, 0, dis])
        rotate = np.matmul(rotationY(math.radians(angle)), rotationX(math.radians(p)))
        fwd = np.matmul(rotate, ori_vec)
        cam_pos = look_at_center + fwd

        x_min = 0
        y_min = -25
        cx = res[0] * 0.5
        cy = res[1] * 0.5
        fx = res[0] * 0.8
        fy = res[1] * 0.8
        _cx = cx - x_min
        _cy = cy - y_min
        renderer.scene.cameras[0].set_intrinsic(fx, fy, _cx, _cy)
        renderer.scene.cameras[0].set(pos=cam_pos, target=look_at_center)
        renderer.scene.cameras[0]._init()

        if render_hr:
            fx = res[0] * 0.8 * 2
            fy = res[1] * 0.8 * 2
            _cx = (res[0] * 0.5 - x_min) * 2
            _cy = (res[1] * 0.5 - y_min) * 2

        renderer.scene.cameras[1].set_intrinsic(fx, fy, _cx, _cy)
        renderer.scene.cameras[1].set(pos=cam_pos, target=look_at_center)
        renderer.scene.cameras[1]._init()

        renderer.scene.render()
        camera = renderer.scene.cameras[0]
        camera_hr = renderer.scene.cameras[1]
        extrinsic = camera.export_extrinsic()
        intrinsic = camera.export_intrinsic()
        depth_map = camera.zbuf.to_numpy().swapaxes(0, 1)
        img = camera.img.to_numpy().swapaxes(0, 1)
        normal_map = camera.normal_map.to_numpy().swapaxes(0, 1)
        img_hr = camera_hr.img.to_numpy().swapaxes(0, 1)
        mask = camera.mask.to_numpy().swapaxes(0, 1)
        return extrinsic, intrinsic, depth_map, img, normal_map, mask, img_hr

    save_path_gps = os.path.join(os.path.split(save_path)[0],"GPS_render")
    degree_interval = 360 / 36
    for i in range(4):
        angle_base = np.random.uniform(0, 360)
        paid = {"angle_base": angle_base, "move1": move1, "move2": move1,
                "human_height": human_height}
        for pid in range(36):
            view = (angle_base + pid * degree_interval) % 360
            paid[f'view{pid}'] = view

            extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, view, look_at_center,
                                                                                     base_cam_pitch,
                                                                                     renderer,
                                                                                     renderer.src_res != renderer.tar_res)
            save_func(f"{data_id}_{str(i).zfill(4)}", f"view{pid}", save_path, extrinsic, intrinsic, depth_map, img,
                      normal_map, mask,
                      None)
            if pid == 0:
                save_func(f"{data_id}_{str(i).zfill(4)}", "source_0", save_path_gps, extrinsic, intrinsic, depth_map, img,
                          normal_map, mask, None)
                viewgps = (view + degree_interval) % 360
                extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, viewgps, look_at_center,
                                                                                         base_cam_pitch,
                                                                                         renderer,
                                                                                         renderer.src_res != renderer.tar_res)
                save_func(f"{data_id}_{str(i).zfill(4)}", "source_1", save_path_gps, extrinsic, intrinsic, depth_map,
                          img,
                          normal_map, mask,
                          None)

            os.makedirs(f"{save_path}/angles", exist_ok=True)
            with open(f"{save_path}/angles/{data_id}_{str(i).zfill(4)}.pkl", "wb") as f:
                pickle.dump(paid, f)

def render_data_human43d(renderer, data_path, data_id, save_path, cam_nums, res, dis=1.0, phase="train",
                         is_thuman=False):
    obj_path = os.path.join(data_path, 'norm_mtl.obj')
    img_path = os.path.join(data_path, 'mNORMAL_u1_v1.png')
    texture = cv2.imread(img_path)[:, :, ::-1]
    texture = np.ascontiguousarray(texture)
    texture = texture.swapaxes(0, 1)[:, ::-1, :]
    obj = t3.readobj(obj_path, scale=1)

    # height normalization
    vy_max = np.max(obj['vi'][:, 1])
    vy_min = np.min(obj['vi'][:, 1])
    human_height = 1.80 + np.random.uniform(-0.05, 0.05, 1)
    obj['vi'][:, :3] = obj['vi'][:, :3] / (vy_max - vy_min) * human_height
    obj['vi'][:, 1] -= np.min(obj['vi'][:, 1])
    look_at_center = np.array([0, 0.85, 0])
    base_cam_pitch = -8

    # randomly move the scan
    move_range = 0.1 if human_height < 1.80 else 0.05
    delta_x = np.max(obj['vi'][:, 0]) - np.min(obj['vi'][:, 0])
    delta_z = np.max(obj['vi'][:, 2]) - np.min(obj['vi'][:, 2])
    if delta_x > 1.0 or delta_z > 1.0:
        move_range = 0.01
    move1 = np.random.uniform(-move_range, move_range, 1)
    move2 = np.random.uniform(-move_range, move_range, 1)
    obj['vi'][:, 0] += move1
    obj['vi'][:, 2] += move2

    if len(renderer.scene.models) >= 1:
        renderer.modify_model(0, obj, texture)
    else:
        renderer.add_model(obj, texture)

    degree_interval = 360 / cam_nums
    angle_list1 = list(range(360 - int(degree_interval // 2), 360))
    angle_list2 = list(range(0, 0 + int(degree_interval // 2)))
    angle_list = angle_list1 + angle_list2

    angle_base = np.random.choice(angle_list, 1)[0]

    def render(dis, angle, look_at_center, p, renderer, render_hr=False):
        ori_vec = np.array([0, 0, dis])
        rotate = np.matmul(rotationY(math.radians(angle)), rotationX(math.radians(p)))
        fwd = np.matmul(rotate, ori_vec)
        cam_pos = look_at_center + fwd

        x_min = 0
        y_min = -25
        cx = res[0] * 0.5
        cy = res[1] * 0.5
        fx = res[0] * 0.8
        fy = res[1] * 0.8
        _cx = cx - x_min
        _cy = cy - y_min
        renderer.scene.cameras[0].set_intrinsic(fx, fy, _cx, _cy)
        renderer.scene.cameras[0].set(pos=cam_pos, target=look_at_center)
        renderer.scene.cameras[0]._init()

        if render_hr:
            fx = res[0] * 0.8 * 2
            fy = res[1] * 0.8 * 2
            _cx = (res[0] * 0.5 - x_min) * 2
            _cy = (res[1] * 0.5 - y_min) * 2

        renderer.scene.cameras[1].set_intrinsic(fx, fy, _cx, _cy)
        renderer.scene.cameras[1].set(pos=cam_pos, target=look_at_center)
        renderer.scene.cameras[1]._init()

        renderer.scene.render()
        camera = renderer.scene.cameras[0]
        camera_hr = renderer.scene.cameras[1]
        extrinsic = camera.export_extrinsic()
        intrinsic = camera.export_intrinsic()
        depth_map = camera.zbuf.to_numpy().swapaxes(0, 1)
        img = camera.img.to_numpy().swapaxes(0, 1)
        normal_map = camera.normal_map.to_numpy().swapaxes(0, 1)
        img_hr = camera_hr.img.to_numpy().swapaxes(0, 1)
        mask = camera.mask.to_numpy().swapaxes(0, 1)
        return extrinsic, intrinsic, depth_map, img, normal_map, mask, img_hr

    save_path_gps = os.path.join(os.path.split(save_path)[0],"GPS_render",phase)
    for pid in range(2):
        paid = {"angle_base": angle_base, "move1": move1, "move2": move1,
                "human_height": human_height}
        # angle = check(angle_base + pid * degree_interval)
        view0 = angle_base + pid * degree_interval
        viewgps = (view0+degree_interval) % 360
        paid['view0'] = view0
        view1 = (view0 + 90) % 360
        paid['view1'] = view1
        view2 = (view0 + 180) % 360
        paid['view2'] = view2
        view3 = (view0 + 270) % 360
        paid['view3'] = view3

        extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, view0, look_at_center,
                                                                                 base_cam_pitch,
                                                                                 renderer,
                                                                                 renderer.src_res != renderer.tar_res)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "source", save_path, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "source_0", save_path_gps, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)
        extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, viewgps, look_at_center,
                                                                                 base_cam_pitch,
                                                                                 renderer,
                                                                                 renderer.src_res != renderer.tar_res)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "source_1", save_path_gps, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)
        extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, view1, look_at_center,
                                                                                 base_cam_pitch,
                                                                                 renderer,
                                                                                 renderer.src_res != renderer.tar_res)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "view1", save_path, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)

        extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, view2, look_at_center,
                                                                                 base_cam_pitch,
                                                                                 renderer,
                                                                                 renderer.src_res != renderer.tar_res)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "view2", save_path, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)

        extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, view3, look_at_center,
                                                                                 base_cam_pitch,
                                                                                 renderer,
                                                                                 renderer.src_res != renderer.tar_res)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "view3", save_path, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)

        nv = []
        for n in range(1, cam_nums + 1):
            viewt = (view0 + np.random.uniform() * n * degree_interval) % 360
            nv.append(viewt)
            extrinsic, intrinsic, depth_map, img, _, mask, img_tar = render(dis, viewt, look_at_center, base_cam_pitch,
                                                                            renderer,
                                                                            renderer.src_res != renderer.tar_res)
            save_func(f"{data_id}_{str(pid).zfill(4)}", f"tar_{n}", save_path, extrinsic, intrinsic, depth_map,
                      None, None, mask, img_tar)
        paid['nv'] = nv
        os.makedirs(f"{save_path}/angles", exist_ok=True)
        with open(f"{save_path}/angles/{data_id}_{str(pid).zfill(4)}.pkl", "wb") as f:
            pickle.dump(paid, f)


def render_data_2k2k(renderer, data_path, data_id, save_path, cam_nums, res, dis=1.0, phase="train",
                         is_thuman=False):
    mesh = o3d.io.read_triangle_mesh("/home/xtf/data/2k2k/train/00007/00007.ply")
    mesh.compute_vertex_normals()

    # 确保是三角网格
    if not mesh.has_triangles():
        raise ValueError("PLY 文件不是三角网格")

    # 顶点和三角面
    vertices = np.asarray(mesh.vertices)  # (N, 3)
    triangles = np.asarray(mesh.triangles)  # (F, 3)

    obj = {
        'vi': vertices.astype(np.float32),
        'face': triangles.astype(np.int32),  # 如果你的渲染器用的是 face 索引
    }


    # obj_path = os.path.join(data_path, 'norm_mtl.obj')
    # img_path = os.path.join(data_path, 'mNORMAL_u1_v1.png')
    # texture = cv2.imread(img_path)[:, :, ::-1]
    # texture = np.ascontiguousarray(texture)
    # texture = texture.swapaxes(0, 1)[:, ::-1, :]
    # obj = t3.readobj(obj_path, scale=1)

    # height normalization
    vy_max = np.max(obj['vi'][:, 1])
    vy_min = np.min(obj['vi'][:, 1])
    human_height = 1.80 + np.random.uniform(-0.05, 0.05, 1)
    obj['vi'][:, :3] = obj['vi'][:, :3] / (vy_max - vy_min) * human_height
    obj['vi'][:, 1] -= np.min(obj['vi'][:, 1])
    look_at_center = np.array([0, 0.85, 0])
    base_cam_pitch = -8

    # randomly move the scan
    move_range = 0.1 if human_height < 1.80 else 0.05
    delta_x = np.max(obj['vi'][:, 0]) - np.min(obj['vi'][:, 0])
    delta_z = np.max(obj['vi'][:, 2]) - np.min(obj['vi'][:, 2])
    if delta_x > 1.0 or delta_z > 1.0:
        move_range = 0.01
    move1 = np.random.uniform(-move_range, move_range, 1)
    move2 = np.random.uniform(-move_range, move_range, 1)
    obj['vi'][:, 0] += move1
    obj['vi'][:, 2] += move2
    texture = None
    if len(renderer.scene.models) >= 1:
        renderer.modify_model(0, obj, texture)
    else:
        renderer.add_model(obj, texture)

    degree_interval = 360 / cam_nums
    angle_list1 = list(range(360 - int(degree_interval // 2), 360))
    angle_list2 = list(range(0, 0 + int(degree_interval // 2)))
    angle_list = angle_list1 + angle_list2

    angle_base = np.random.choice(angle_list, 1)[0]

    def render(dis, angle, look_at_center, p, renderer, render_hr=False):
        ori_vec = np.array([0, 0, dis])
        rotate = np.matmul(rotationY(math.radians(angle)), rotationX(math.radians(p)))
        fwd = np.matmul(rotate, ori_vec)
        cam_pos = look_at_center + fwd

        x_min = 0
        y_min = -25
        cx = res[0] * 0.5
        cy = res[1] * 0.5
        fx = res[0] * 0.8
        fy = res[1] * 0.8
        _cx = cx - x_min
        _cy = cy - y_min
        renderer.scene.cameras[0].set_intrinsic(fx, fy, _cx, _cy)
        renderer.scene.cameras[0].set(pos=cam_pos, target=look_at_center)
        renderer.scene.cameras[0]._init()

        if render_hr:
            fx = res[0] * 0.8 * 2
            fy = res[1] * 0.8 * 2
            _cx = (res[0] * 0.5 - x_min) * 2
            _cy = (res[1] * 0.5 - y_min) * 2

        renderer.scene.cameras[1].set_intrinsic(fx, fy, _cx, _cy)
        renderer.scene.cameras[1].set(pos=cam_pos, target=look_at_center)
        renderer.scene.cameras[1]._init()

        renderer.scene.render()
        camera = renderer.scene.cameras[0]
        camera_hr = renderer.scene.cameras[1]
        extrinsic = camera.export_extrinsic()
        intrinsic = camera.export_intrinsic()
        depth_map = camera.zbuf.to_numpy().swapaxes(0, 1)
        img = camera.img.to_numpy().swapaxes(0, 1)
        normal_map = camera.normal_map.to_numpy().swapaxes(0, 1)
        img_hr = camera_hr.img.to_numpy().swapaxes(0, 1)
        mask = camera.mask.to_numpy().swapaxes(0, 1)
        return extrinsic, intrinsic, depth_map, img, normal_map, mask, img_hr

    save_path_gps = os.path.join(os.path.split(save_path)[0],"GPS_render",phase)
    for pid in range(2):
        paid = {"angle_base": angle_base, "move1": move1, "move2": move1,
                "human_height": human_height}
        # angle = check(angle_base + pid * degree_interval)
        view0 = angle_base + pid * degree_interval
        viewgps = (view0+degree_interval) % 360
        paid['view0'] = view0
        view1 = (view0 + 90) % 360
        paid['view1'] = view1
        view2 = (view0 + 180) % 360
        paid['view2'] = view2
        view3 = (view0 + 270) % 360
        paid['view3'] = view3

        extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, view0, look_at_center,
                                                                                 base_cam_pitch,
                                                                                 renderer,
                                                                                 renderer.src_res != renderer.tar_res)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "source", save_path, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "source_0", save_path_gps, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)
        extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, viewgps, look_at_center,
                                                                                 base_cam_pitch,
                                                                                 renderer,
                                                                                 renderer.src_res != renderer.tar_res)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "source_1", save_path_gps, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)
        extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, view1, look_at_center,
                                                                                 base_cam_pitch,
                                                                                 renderer,
                                                                                 renderer.src_res != renderer.tar_res)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "view1", save_path, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)

        extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, view2, look_at_center,
                                                                                 base_cam_pitch,
                                                                                 renderer,
                                                                                 renderer.src_res != renderer.tar_res)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "view2", save_path, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)

        extrinsic, intrinsic, depth_map, img, normal_map, mask, img_tar = render(dis, view3, look_at_center,
                                                                                 base_cam_pitch,
                                                                                 renderer,
                                                                                 renderer.src_res != renderer.tar_res)
        save_func(f"{data_id}_{str(pid).zfill(4)}", "view3", save_path, extrinsic, intrinsic, depth_map, img,
                  normal_map, mask,
                  img_tar)

        nv = []
        for n in range(1, cam_nums + 1):
            viewt = (view0 + np.random.uniform() * n * degree_interval) % 360
            nv.append(viewt)
            extrinsic, intrinsic, depth_map, img, _, mask, img_tar = render(dis, viewt, look_at_center, base_cam_pitch,
                                                                            renderer,
                                                                            renderer.src_res != renderer.tar_res)
            save_func(f"{data_id}_{str(pid).zfill(4)}", f"tar_{n}", save_path, extrinsic, intrinsic, depth_map,
                      None, None, mask, img_tar)
        paid['nv'] = nv
        os.makedirs(f"{save_path}/angles", exist_ok=True)
        with open(f"{save_path}/angles/{data_id}_{str(pid).zfill(4)}.pkl", "wb") as f:
            pickle.dump(paid, f)

# def render_smplx(renderer, data_path, angles_path, data_id, save_path, cam_nums, res, dis=1.0, phase="train",
#                  is_thuman=False):
def render_smplx():
    # obj_path = os.path.join(data_path, '%s.obj' % data_id)
    obj_path = "/home/xtf/data/Thuman2.1/fit_smpl/0000_0000/smpl/smpl_mesh.obj"
    # img_path = os.path.join(data_path, 'material0.jpeg')
    # texture = cv2.imread(img_path)[:, :, ::-1]
    # texture = np.ascontiguousarray(texture)
    texture = None  # texture.swapaxes(0, 1)[:, ::-1, :]
    obj = t3.readobj(obj_path, scale=1)

    with open(angles_path, 'rb') as f:
        data = pickle.load(f)

    with open(obj_path.replace("smpl_mesh.obj","smpl_param.pkl"), "rb") as f:
        param = pickle.load(f)

    vy_max = np.max(obj['vi'][:, 1])
    vy_min = np.min(obj['vi'][:, 1])
    human_height = param["body_scale"]

    obj['vi'][:, :3] = obj['vi'][:, :3] / (vy_max - vy_min) * human_height
    obj['vi'][:, 1] -= np.min(obj['vi'][:, 1])
    look_at_center = np.array([0, 0.85, 0])
    base_cam_pitch = -8
    #
    # obj['vi'][:, 0] += data["move1"]
    # obj['vi'][:, 2] += data["move2"]

    if len(renderer.scene.models) >= 1:
        renderer.modify_model(0, obj)
    else:
        renderer.add_model(obj)

    view1 = data['view1']
    view2 = data['view2']
    view3 = data['view3']

    # look_at_center = data['look_at_center']
    # base_cam_pitch = data['base_cam_pitch']

    def render(dis, angle, look_at_center, p, renderer, render_hr=False):
        ori_vec = np.array([0, 0, dis])
        rotate = np.matmul(rotationY(math.radians(angle)), rotationX(math.radians(p)))
        fwd = np.matmul(rotate, ori_vec)
        cam_pos = look_at_center + fwd

        x_min = 0
        y_min = -25
        cx = res[0] * 0.5
        cy = res[1] * 0.5
        fx = res[0] * 0.8
        fy = res[1] * 0.8
        _cx = cx - x_min
        _cy = cy - y_min
        renderer.scene.cameras[0].set_intrinsic(fx, fy, _cx, _cy)
        renderer.scene.cameras[0].set(pos=cam_pos, target=look_at_center)
        renderer.scene.cameras[0]._init()

        if render_hr:
            fx = res[0] * 0.8 * 2
            fy = res[1] * 0.8 * 2
            _cx = (res[0] * 0.5 - x_min) * 2
            _cy = (res[1] * 0.5 - y_min) * 2

        renderer.scene.cameras[1].set_intrinsic(fx, fy, _cx, _cy)
        renderer.scene.cameras[1].set(pos=cam_pos, target=look_at_center)
        renderer.scene.cameras[1]._init()

        renderer.scene.render()
        camera = renderer.scene.cameras[0]
        camera_hr = renderer.scene.cameras[1]
        extrinsic = camera.export_extrinsic()
        intrinsic = camera.export_intrinsic()
        depth_map = camera.zbuf.to_numpy().swapaxes(0, 1)
        img = camera.img.to_numpy().swapaxes(0, 1)
        # normal_map = camera.normal_map.to_numpy().swapaxes(0, 1)
        img_hr = camera_hr.img.to_numpy().swapaxes(0, 1)
        mask = camera.mask.to_numpy().swapaxes(0, 1)
        return extrinsic, intrinsic, depth_map, img, mask, img_hr

    extrinsic, intrinsic, depth_map, img, mask, img_tar = render(dis, view1, look_at_center, base_cam_pitch,
                                                                 renderer, renderer.src_res != renderer.tar_res)
    save_func(f"{os.path.basename(angles_path)[:-4]}", "view1_mesh", save_path, None, None, depth_map, img, None, mask,
              img_tar)

    extrinsic, intrinsic, depth_map, img, mask, img_tar = render(dis, view2, look_at_center, base_cam_pitch,
                                                                 renderer, renderer.src_res != renderer.tar_res)
    save_func(f"{os.path.basename(angles_path)[:-4]}", "view2_mesh", save_path, None, None, depth_map, img, None, mask,
              img_tar)

    extrinsic, intrinsic, depth_map, img, mask, img_tar = render(dis, view3, look_at_center, base_cam_pitch,
                                                                 renderer, renderer.src_res != renderer.tar_res)
    save_func(f"{os.path.basename(angles_path)[:-4]}", "view3_mesh", save_path, None, None, depth_map, img, None, mask,
              img_tar)

def check(path,name):
    img = glob.glob(os.path.join(path,"img",f"{name}_*"))
    mask = glob.glob(os.path.join(path,"mask",f"{name}_*"))
    depth = glob.glob(os.path.join(path,"depth",f"{name}_*"))
    if len(img) != 16:
        return False
    for i in img:
        if len(os.listdir(i)) != 28:
            return False
    for i in mask:
        if len(os.listdir(i)) != 4:
            return False
    for i in depth:
        if len(os.listdir(i)) != 4:
            return False
    return True


def render_data_copy():
    cam_nums = 16
    scene_radius = 2.0
    res = (512, 512)
    thuman_root = '/home/xtf/data/Thuman2.1'

    np.random.seed(1314)
    renderer = StaticRenderer(src_res=res, double_res=False)
    trainvallist = os.listdir("/home/xtf/data/Thuman2.1/thuman2.1_val")
    trainvallist = list(set([i.split("_")[0] for i in trainvallist]))
    # render_smplx(renderer,)


    for index, name in enumerate(trainvallist):
        print(index, "|", len(trainvallist))
        path = os.path.join(thuman_root, "thuman2.1_val_render_512")
        oldatapath = os.path.join(thuman_root, "thuman2.1_val", name)
        # render_data_myself(renderer, oldatapath, name, path, cam_nums, res, dis=scene_radius, is_thuman=True)
        render_data_version2(renderer, oldatapath, name, path, cam_nums, res, dis=scene_radius, phase="val",
                             is_thuman=True)
        # for angles_path in glob.glob(f"/home/xtf/data/Thuman2.1/thuman2.1_val/angles/{name}_*.pkl"):
        #      render_smplx(renderer, oldatapath, angles_path, name, path, cam_nums, res, dis=scene_radius, is_thuman=True)
    #
    #
    # trainvallist = os.listdir("/home/xtf/data/Thuman2.1/thuman2.1_train")
    # trainvallist = list(set([i.split("_")[0] for i in trainvallist]))
    # for index, name in enumerate(trainvallist):
    #     print(index, "|", len(trainvallist))
    #
    #     path = os.path.join(thuman_root, "thuman2.1_train_render_512")
    #     if check(path, name):
    #         continue
    #     oldatapath = os.path.join(thuman_root, "thuman2.1_train", name)
    #     render_data_version2(renderer, oldatapath, name, path, cam_nums, res, dis=scene_radius, phase="train",
    #                      is_thuman=True)
    #     # for angles_path in glob.glob(f"/home/xtf/data/Thuman2.1/thuman2.1_train/angles/{name}_*.pkl"):
    #     #      render_smplx(renderer, oldatapath, angles_path, name, path, cam_nums, res, dis=scene_radius, is_thuman=True)


def render_Thu_data_360():
    cam_nums = 16
    scene_radius = 2.0
    res = (512, 512)
    thuman_root = '/xtf/data/Thuman2.1/all'

    np.random.seed(1314)
    renderer = StaticRenderer(src_res=res, double_res=False)
    trainvallist = os.listdir(thuman_root)

    for index, name in enumerate(trainvallist):
        print(index, "|", len(trainvallist))
        path = "/xtf/data/Thuman2.1/thuman2.1_render_512"
        oldatapath = os.path.join(thuman_root, name)
        render_data_360(renderer, oldatapath, name, path, cam_nums, res, dis=scene_radius, phase="val",
                             is_thuman=True)


def render_Thu_smplxdata_360():
    cam_nums = 16
    scene_radius = 2.0
    res = (512, 512)
    thuman_root = '/xtf/data/Thuman2.1/all'

    np.random.seed(1314)
    renderer = StaticRenderer(src_res=res, double_res=False)
    trainvallist = os.listdir(thuman_root)

    for index, name in enumerate(trainvallist):
        print(index, "|", len(trainvallist))
        path = "/xtf/data/Thuman2.1/thuman2.1_render_512"
        oldatapath = os.path.join(thuman_root, name)
        render_data_360(renderer, oldatapath, name, path, cam_nums, res, dis=scene_radius, phase="val",
                             is_thuman=True)


def render_data_human4dit3d():
    cam_nums = 16
    scene_radius = 2.0
    res = (512, 512)
    thuman_root = '/home/xtf/data/human4dit3d'

    np.random.seed(1314)
    renderer = StaticRenderer(src_res=res, double_res=False)
    trainvallist = os.listdir(thuman_root)

    for index, name in enumerate(trainvallist):
        print(index, "|", len(trainvallist))
        path = os.path.join(thuman_root, "human4dit3d_render_512")
        oldatapath = os.path.join(thuman_root, name)
        render_data_human43d(renderer, oldatapath, name, path, cam_nums, res, dis=scene_radius, phase="val",
                             is_thuman=True)

def render_data_2k():
    cam_nums = 16
    scene_radius = 2.0
    res = (512, 512)
    thuman_root = '/home/xtf/data/human4dit3d'

    np.random.seed(1314)
    renderer = StaticRenderer(src_res=res, double_res=False)
    render_data_2k2k(renderer, thuman_root, thuman_root, thuman_root, cam_nums, res, dis=scene_radius, phase="val",
                         is_thuman=True)
    trainvallist = os.listdir(thuman_root)

    for index, name in enumerate(trainvallist):
        print(index, "|", len(trainvallist))
        path = os.path.join(thuman_root, "human4dit3d_render_512")
        oldatapath = os.path.join(thuman_root, name)
        render_data_human43d(renderer, oldatapath, name, path, cam_nums, res, dis=scene_radius, phase="val",
                             is_thuman=True)

# render_data_2k()



def get_c(angle,upper,lower):
    if lower<=upper:
        return lower <= angle <= upper
    else:
        return angle >= lower or angle <= upper


def align_mesh_to_upright(mesh, target_axis='Y', rotation_angle=None):
    """
    将平躺的mesh转换为直立姿态
    参数：
    mesh - trimesh.Trimesh对象
    target_axis - 目标直立轴 ('Y'或'Z')
    rotation_angle - 手动指定旋转角度 (弧度制)
    """
    # 自动检测旋转角度（如果未指定）
    if rotation_angle is None:
        # 通过包围盒确定主方向
        extents = mesh.bounding_box.extents
        horizontal_axes = [i for i in sorted(range(3), key=lambda x: extents[x]) if i != 1][-1]

        # 计算需要旋转的轴和角度
        if target_axis == 'Y':
            rotation_angle = -np.pi / 2  # 默认绕X轴旋转-90度
        else:
            rotation_angle = np.pi / 2  # Z-up情况

    # 创建旋转矩阵（绕X轴旋转）
    rotation = trimesh.transformations.rotation_matrix(rotation_angle, [1, 0, 0])

    # 应用旋转
    mesh.apply_transform(rotation)

    # 重置位置到原点
    mesh = reset_mesh_position(mesh)

    return mesh


def reset_mesh_position(mesh):
    """将模型底部对齐到原点"""
    # 计算Y轴最小点
    min_y = mesh.vertices[:, 1].min()
    translation = [0, -min_y, 0]
    mesh.apply_translation(translation)
    return mesh


def process_dataset(input_dir, output_dir, target_axis='Y'):
    """
    批量处理数据集
    参数：
    input_dir - 输入目录路径
    output_dir - 输出目录路径
    target_axis - 目标直立轴 ('Y'或'Z')
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if not filename.lower().endswith(('.obj', '.ply', '.stl')):
            continue

        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        try:
            # 加载mesh
            mesh = trimesh.load(input_path)

            # 执行对齐
            aligned_mesh = align_mesh_to_upright(mesh, target_axis=target_axis)

            # 导出结果
            aligned_mesh.export(output_path)
            print(f"Processed: {filename}")

        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

