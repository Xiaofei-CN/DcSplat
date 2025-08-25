from data.base_dataset import BaseDataset
import os
import numpy as np
from utils.data_utils import *
import torchvision.transforms.functional as TF
from utils.utils import *
from utils.graphics_utils import *
import glob
import pickle
class Thuman_easyDataset(BaseDataset):

    def initialize(self, opt, phase):
        self.opt = opt
        self.other_view = opt.model.other_view
        # self.use_processed_data = opt.use_processed_data
        self.phase = phase
        self.data_root = os.path.join(opt.dataset.data_root, f'thuman2.1_{phase}_render')
        self.sample_list = self.get_list()

        self.img_path = os.path.join(self.data_root, 'img/%s')
        self.mask_path = os.path.join(self.data_root, 'mask/%s')
        self.depth_path = os.path.join(self.data_root, 'depth/%s')
        self.intr_path = os.path.join(self.data_root, 'parm/%s')
        self.extr_path = os.path.join(self.data_root, 'parm/%s')
        self.angle_path = os.path.join(self.data_root, 'angles/%s')


    def name(self):
        return 'ThumanDataset'


    def __len__(self):

        return 10000 if self.phase == 'train' else len(self.sample_list)

    def get_list(self):
        data_list = []
        for angles_path in glob.glob(os.path.join(self.data_root,"angles","*.pkl")):
            with open(angles_path, 'rb') as f:
                data = pickle.load(f)
            basename = angles_path.split('/')[-1][:-4]
            base = data["angle_base"]
            down = (base - 22.5) % 360
            up = (base + 22.5) % 360
            for index,n in enumerate(data["nv"]):
                if get_c(n, up, down):
                    data_list.append([basename,index+1])

        return data_list

    def train_data(self, index):
        id,target = self.sample_list[index]
        img_path = self.img_path % id
        dcit_tensor = {}
        dcit_tensor['source_view'] = self.load_multi_view(img_path,
                                                          ["source.png", "view1.png", "view2.png", "view3.png"],
                                                          require_spmlx=False, phase=self.phase, is_pose=False)

        dcit_tensor["pose"] = self.load_multi_view(img_path, ["source.png", "view1.png", "view2.png", "view3.png"],
                                                   require_spmlx=False, phase=self.phase)

        dcit_tensor['target_view_novel'] = self.get_single_novel_view_tensor(img_path, f"tar_{target}.png")
        dcit_tensor['target_view_rec'] = self.get_single_novel_view_tensor(img_path, "tar_source.png")
        dcit_tensor["pair"] = f"{id}_2_{target}"
        return dcit_tensor

    def val_data(self, index):
        id, target = self.sample_list[index]
        img_path = self.img_path % id
        dcit_tensor = {}
        dcit_tensor['source_view'] = self.load_multi_view(img_path,
                                                          ["source.png", "view1.png", "view2.png", "view3.png"],
                                                          require_spmlx=False, phase=self.phase, is_pose=False)

        dcit_tensor["pose"] = self.load_multi_view(img_path, ["source.png", "view1.png", "view2.png", "view3.png"],
                                                   require_spmlx=False, phase=self.phase)
        dcit_tensor['target_view_novel'] = self.get_single_novel_view_tensor(img_path, f"tar_{target}.png")
        dcit_tensor['target_view_rec'] = self.get_single_novel_view_tensor(img_path, "tar_source.png")
        dcit_tensor["pair"] = f"{id}_2_{target}"
        return dcit_tensor

    def load_multi_view(self, img_path, source_list, resize=None, require_mask=True,is_pose=True,
                        require_depth=True, require_spmlx=True, phase="test"):
        source_dict = {}

        for source in source_list:
            out = self.load_single_view(img_path, source, resize=resize, require_mask=require_mask,is_pose=is_pose,
                                        require_depth=require_depth, require_spmlx=require_spmlx, phase=phase)
            for key, value in out.items():
                if key not in source_dict:
                    source_dict[key] = []
                source_dict[key].append(value)
        for k, v in source_dict.items():
            if k == "view":
                continue
            source_dict[k] = torch.cat(v, dim=0)
        return source_dict


    def load_single_view(self, img_path, sample_name, resize=None, require_mask=True,is_pose=False,
                         require_depth=True, require_spmlx=True, phase='train'):

        img_name = os.path.join(img_path, sample_name)
        mask_name = img_name.replace('img', 'mask')
        depth_name = mask_name.replace('mask', 'depth')
        intr_name = img_name.replace('img', 'parm').replace('.png', '_intrinsic.npy')
        extr_name = intr_name.replace('_intrinsic.npy', "_extrinsic.npy")
        spmlx_name = img_name.replace('img', 'smplx').replace('.png', '_smpl_mesh.obj')
        if is_pose:
            img_name = img_name[:-4]+"_normal.png"
            # mask_name = mask_name[:-4]+"_normal.png"
            # depth_name = depth_name[:-4]+"_normal.png"

        intr, extr = np.load(intr_name), np.load(extr_name)
        extr = np.concatenate([extr, np.array([[0, 0, 0, 1]])], axis=0)
        img = read_img(img_name)
        img = torch.from_numpy(img).permute(2, 0, 1) / 255.0 * 2.0 - 1.0

        if resize:
            img = TF.resize(img, resize)
        mask, depth, pc = None, None, None
        if require_depth:
            depth = cv2.imread(depth_name, cv2.IMREAD_UNCHANGED).astype(np.float32) / 2.0 ** 15
            # depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            # depth = depth * 2.0 - 1.0
            depth = torch.from_numpy(depth)
        if require_mask:
            mask = read_img(mask_name)
            mask = torch.from_numpy(mask).permute(2, 0, 1).float()
            if resize:
                mask = TF.resize(mask, resize)
            mask = mask / 255.0
            img = img * mask
            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0
            # mask = mask[:1]
        if require_spmlx:
            pc = get_pointcloud(spmlx_name,num_points=20000)

        dict = {"view": sample_name,
                "img": img.unsqueeze(0),
                "intr": torch.FloatTensor(intr).unsqueeze(0),
                "extr": torch.FloatTensor(extr).unsqueeze(0),
                }
        if mask is not None:
            dict["mask"] = mask.unsqueeze(0)
        if depth is not None:
            dict["depth"] = torch.FloatTensor(depth).unsqueeze(0)
        if pc is not None:
            dict["pc"] = torch.FloatTensor(pc).unsqueeze(0)
        if phase == 'val':
            or_img = read_img(img_name)
            or_img = torch.from_numpy(or_img).permute(2, 0, 1) / 255.0
            dict["or_img"] = or_img.unsqueeze(0)

        return dict

    def get_single_novel_view_tensor(self, img_path, sample_name):
        img_name = os.path.join(img_path, sample_name)
        intr_name = img_name.replace('img', 'parm').replace('.png', '_intrinsic.npy')
        extr_name = intr_name.replace('_intrinsic.npy', "_extrinsic.npy")
        intr, extr = np.load(intr_name), np.load(extr_name)
        # extr = np.concatenate([extr, np.array([[0, 0, 0, 1]])], axis=0)
        if self.opt.model.tar_res == 1024:
            intr[:2] *= 2
        img = read_img(img_name)
        height, width = img.shape[:2]
        img = torch.from_numpy(img).permute(2, 0, 1) / 255.0

        R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(extr[:3, 3], np.float32)

        FovX = focal2fov(intr[0, 0], width)
        FovY = focal2fov(intr[1, 1], height)
        projection_matrix = getProjectionMatrix(znear=self.opt.dataset.znear, zfar=self.opt.dataset.zfar, K=intr,
                                                h=height,
                                                w=width).transpose(0, 1)
        # world_view_transform = torch.tensor(getWorld2View2(R, T, np.array(self.opt.dataset.trans), self.opt.dataset.scale)).transpose(0,
        #                                                                                                               1)
        world_view_transform = torch.tensor(getWorld2View2(R, T)).transpose(0, 1)
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        novel_view_data = {
            'view': sample_name,
            'img': img,
            'extr': torch.FloatTensor(extr),
            'FovX': FovX,
            'FovY': FovY,
            'width': width,
            'height': height,
            'world_view_transform': world_view_transform,
            'full_proj_transform': full_proj_transform,
            'camera_center': camera_center
        }
        return novel_view_data