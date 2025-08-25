import numpy as np
import os
from PIL import Image
import cv2
import torch
import json
import trimesh
import open3d as o3d

def save_np_to_json(parm, save_name):
    for key in parm.keys():
        parm[key] = parm[key].tolist()
    with open(save_name, 'w') as file:
        json.dump(parm, file, indent=1)


def load_json_to_np(parm_name):
    with open(parm_name, 'r') as f:
        parm = json.load(f)
    for key in parm.keys():
        parm[key] = np.array(parm[key])
    return parm

def get_pointcloud(mesh_path,total=282000):
    mesh_path = mesh_path.replace('img', 'smpl_pc') + '.ply'
    assert os.path.exists(mesh_path), mesh_path

    pcd = o3d.io.read_point_cloud(mesh_path)
    pc = np.array(pcd.points).astype(np.float32)

    pc_len = pc.shape[0]
    pad_len = total - pc_len

    if pad_len > 0:
        padded = np.concatenate([pc,np.zeros((pad_len,pc.shape[1]),dtype=np.float32)], axis=0)
    elif pc_len < 0:
        padded = pc[:total]
        pad_len = 0
    else:
        padded = pc
    mask = np.array(pad_len)
    return padded, mask

def depth2pts(depth, extrinsic, intrinsic):
    # depth H W extrinsic 3x4 intrinsic 3x3 pts map H W 3
    rot = extrinsic[:3, :3]
    trans = extrinsic[:3, 3:]
    S, S = depth.shape

    y, x = torch.meshgrid(torch.linspace(0.5, S-0.5, S, device=depth.device),
                          torch.linspace(0.5, S-0.5, S, device=depth.device))
    pts_2d = torch.stack([x, y, torch.ones_like(x)], dim=-1)  # H W 3

    pts_2d[..., 2] = 1.0 / (depth + 1e-8)
    pts_2d[..., 0] -= intrinsic[0, 2]
    pts_2d[..., 1] -= intrinsic[1, 2]
    pts_2d_xy = pts_2d[..., :2] * pts_2d[..., 2:]
    pts_2d = torch.cat([pts_2d_xy, pts_2d[..., 2:]], dim=-1)

    pts_2d[..., 0] /= intrinsic[0, 0]
    pts_2d[..., 1] /= intrinsic[1, 1]
    pts_2d = pts_2d.reshape(-1, 3).T
    pts = rot.T @ pts_2d - rot.T @ trans
    return pts.T.view(S, S, 3)


def pts2depth(ptsmap, extrinsic, intrinsic):
    S, S, _ = ptsmap.shape
    pts = ptsmap.view(-1, 3).T
    calib = intrinsic @ extrinsic
    pts = calib[:3, :3] @ pts
    pts = pts + calib[:3, 3:4]
    pts[:2, :] /= (pts[2:, :] + 1e-8)
    depth = 1.0 / (pts[2, :].view(S, S) + 1e-8)
    return depth


def stereo_pts2flow(pts0, pts1, rectify0, rectify1, Tf_x):
    new_extr0, new_intr0, rectify_mat0_x, rectify_mat0_y = rectify0
    new_extr1, new_intr1, rectify_mat1_x, rectify_mat1_y = rectify1
    new_depth0 = pts2depth(torch.FloatTensor(pts0), torch.FloatTensor(new_extr0), torch.FloatTensor(new_intr0))
    new_depth1 = pts2depth(torch.FloatTensor(pts1), torch.FloatTensor(new_extr1), torch.FloatTensor(new_intr1))
    new_depth0 = new_depth0.detach().numpy()
    new_depth1 = new_depth1.detach().numpy()
    new_depth0 = cv2.remap(new_depth0, rectify_mat0_x, rectify_mat0_y, cv2.INTER_LINEAR)
    new_depth1 = cv2.remap(new_depth1, rectify_mat1_x, rectify_mat1_y, cv2.INTER_LINEAR)

    offset0 = new_intr1[0, 2] - new_intr0[0, 2]
    disparity0 = -new_depth0 * Tf_x
    flow0 = offset0 - disparity0

    offset1 = new_intr0[0, 2] - new_intr1[0, 2]
    disparity1 = -new_depth1 * (-Tf_x)
    flow1 = offset1 - disparity1

    flow0[new_depth0 < 0.05] = 0
    flow1[new_depth1 < 0.05] = 0

    return flow0, flow1
def get_c(angle,upper,lower):
    if lower<=upper:
        return lower <= angle <= upper
    else:
        return angle >= lower or angle <= upper


def read_img(name):
    img = np.array(Image.open(name))
    return img


def read_depth(name):
    return cv2.imread(name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 2.0 ** 15

# def get_pointcloud(mesh_path,num_points=80000, with_normal=True):
#     if num_points== 262144:
#         pc_path = mesh_path.replace('_smpl_mesh.obj','_smpl_pc.ply')
#     else:
#         pc_path = mesh_path.replace('_smpl_mesh.obj',f'_smpl_pc{num_points}.ply')
#
#     if os.path.exists(pc_path):
#         pcd = o3d.io.read_point_cloud(pc_path)
#         return np.asarray(pcd.points)
#     else:
#         return sample_pointcloud_from_mesh(mesh_path,num_points,with_normal)


def sample_pointcloud_from_mesh(mesh_path,num_points=10000, with_normal=True):
    # 1. 加载三角网格
    mesh = trimesh.load(mesh_path, force='mesh')
    output_path = mesh_path.replace('_smpl_mesh.obj',f'_smpl_pc{num_points}.ply')
    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError(f"{mesh_path} is not a valid mesh.")

    # 2. 均匀表面采样点
    points, _ = trimesh.sample.sample_surface(mesh, num_points)

    # 4. 转为 Open3D 点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    # 5. 保存点云
    o3d.io.write_point_cloud(output_path, pcd)
    return np.asarray(pcd.points)
