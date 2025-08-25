import shutil
import os
import glob
from sklearn.model_selection import train_test_split
import random
random.seed(42)

def random_select_n(n, low=0, high=15):
    """
    从 [low, high] 范围内随机选择 n 个不重复整数

    参数:
        n    : 需要选择的数量
        low  : 范围下限（包含）
        high : 范围上限（包含）

    返回:
        List[int]，长度为 n 的不重复整数列表
    """
    total = high - low + 1
    if n > total:
        raise ValueError(f"n={n} 超过范围 [{low}, {high}] 内的可选数量 {total}")

    return random.sample(range(low, high + 1), n)


def get_list():
    txt = []
    for i in range(0,2285):
        for j in range(36):
            for t in range(4):
                txt.append(f"{str(i).zfill(4)}_{str(t).zfill(4)}_2_view{j}.png\n")

    for i in range(2305,2320):
        for j in range(36):
            for t in range(4):
                txt.append(f"{str(i).zfill(4)}_{str(t).zfill(4)}_2_view{j}.png\n")

    for i in range(2415,2430):
        for j in range(36):
            for t in range(4):
                txt.append(f"{str(i).zfill(4)}_{str(t).zfill(4)}_2_view{j}.png\n")


    with open(f"/xtf/data/Thuman2.1/thuman2.1_train.txt", "w") as f:
        f.writelines(txt)
    # txt = []
    # for i in range(2285,2445):
    #     for j in range(36):
    #         if j in [0,9,18,27]:
    #             continue
    #         for t in range(4):
    #             txt.append(f"{str(i).zfill(4)}_{str(t).zfill(4)}_2_view{j}.png\n")

    # with open(f"/xtf/data/Thuman2.1/thuman2.1_val.txt", "w") as f:
    #     f.writelines(txt)
    # txt = []
    # for i in range(2285,2445):
    #     for j in range(36):
    #         if j in [0,9,18,27]:
    #             continue
    #         for t in range(1):
    #             txt.append(f"{str(i).zfill(4)}_{str(t).zfill(4)}_2_view{j}.png\n")
    # with open(f"/xtf/data/Thuman2.1/thuman2.1_val_test.txt", "w") as f:
    #     f.writelines(txt)
get_list()


def split_train_test_thuman():
    files = glob.glob(f"/home/xtf/data/Thuman2.1/newmodel/*")
    train_data, val_data = train_test_split(files, test_size=0.08, random_state=42)
    os.makedirs(f"/home/xtf/data/Thuman2.1/newmodel/train/", exist_ok=True)
    os.makedirs(f"/home/xtf/data/Thuman2.1/newmodel/val/", exist_ok=True)
    for i in train_data:
        destpath = i.replace("newmodel", "newmodel/train")
        shutil.copytree(i, destpath)
    for i in val_data:
        destpath = i.replace("newmodel", "newmodel/val")
        shutil.copytree(i, destpath)


def split_train_test_renderpeople():
    files = glob.glob(f"/home/xtf/data/renderpeople/*")
    # files.remove("/home/xtf/data/renderpeople/human_list.txt")
    train_data, val_data = train_test_split(files, test_size=0.08, random_state=42)
    os.makedirs(f"/home/xtf/data/renderpeople/train/", exist_ok=True)
    os.makedirs(f"/home/xtf/data/renderpeople/val/", exist_ok=True)
    for i in train_data:
        destpath = i.replace("renderpeople", "renderpeople/train")
        shutil.copytree(i, destpath)
    for i in val_data:
        destpath = i.replace("renderpeople", "renderpeople/val")
        shutil.copytree(i, destpath)


def get_val_list_for_Nerf():
    filelist = os.listdir(f"/home/xtf/data/Thuman2.1/thuman2.1_val_render/img")

    fol = []
    for file in filelist:
        fol.append(file.split("_")[0])

    fol = list(set(fol))
    txt = []
    for f in fol:
        a1 = random_select_n(1, low=0, high=15)[0]
        for n in random_select_n(8, low=1, high=16):
            txt.append(f"{f}_{str(a1).zfill(4)}_2_tar_{n}\n")

    with open(f"/home/xtf/data/Thuman2.1/thuman2.1_val_nerf.txt", "w") as f:
        f.writelines(txt)