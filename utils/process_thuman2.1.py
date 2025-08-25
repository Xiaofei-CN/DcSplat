from pathlib import Path
import numpy as np
import trimesh
import glob
import shutil
import os
def align_mesh_to_up(mesh_path, save_path=None, verbose=True):
    mesh = trimesh.load(mesh_path)

    # 若为 scene 类型，提取 mesh
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    cov = np.cov(vertices.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    up_axis = eigvecs[:, np.argmax(eigvals)]  # 方向向量

    target_axis = np.array([0, 1, 0])  # Y轴向上

    # 判断方向：若朝下，取反向
    if np.dot(up_axis, target_axis) < 0:
        up_axis = -up_axis

    # 计算夹角
    cos_theta = np.dot(up_axis, target_axis) / (np.linalg.norm(up_axis) * np.linalg.norm(target_axis))
    angle_deg = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    if verbose:
        print(f"主轴与 Y 轴夹角（方向考虑后）：{angle_deg:.2f}°")

    # 角度在阈值内则不处理
    if angle_deg < 5:
        if verbose:
            print("已接近直立方向，跳过旋转。")
        shutil.copytree(os.path.split(mesh_path)[0], os.path.split(save_path)[0])
        return 0

    # 构建旋转矩阵
    v = np.cross(up_axis, target_axis)
    c = np.dot(up_axis, target_axis)
    if np.linalg.norm(v) < 1e-6:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx @ vx * ((1 - c) / (np.linalg.norm(v) ** 2))

    # 应用旋转
    mesh.vertices = mesh.vertices @ R.T

    if save_path:
        dir = os.path.split(save_path)[0]
        Path(dir).mkdir(exist_ok=True, parents=True)
        mesh.export(save_path)
    return mesh

for oldpath in glob.glob("/home/xtf/data/Thuman2.1/model/*/*.obj"):
    basename = int(os.path.basename(oldpath)[:-4])
    if basename <= 525:
        continue
    newpath = oldpath.replace("model","newmoel")
    align_mesh_to_up(oldpath,newpath)
