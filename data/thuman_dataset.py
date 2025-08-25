from data.base_dataset import BaseDataset
import os
import numpy as np
from utils.data_utils import *
import torchvision.transforms.functional as TF
from utils.utils import *
from utils.graphics_utils import *

class ThumanDataset(BaseDataset):

    def initialize(self, opt, phase):
        self.opt = opt
        self.other_view = opt.model.other_view
        # self.use_processed_data = opt.use_processed_data
        self.phase = phase
        self.data_root = os.path.join(opt.dataset.data_root, f'thuman2.1_{phase}_render')
        self.esasy_list,self.hard_list = self.get_list()

        self.img_path = os.path.join(self.data_root, 'img/%s')
        self.mask_path = os.path.join(self.data_root, 'mask/%s')
        self.depth_path = os.path.join(self.data_root, 'depth/%s')
        self.intr_path = os.path.join(self.data_root, 'parm/%s')
        self.extr_path = os.path.join(self.data_root, 'parm/%s')
        self.mode = "easy"
        self.sample_list = []
        self.set_phase(self.mode)

    def name(self):
        return 'ThumanDataset'

    def set_phase(self, phase):
        self.mode = phase
        if self.mode == 'easy':
            self.sample_list = self.esasy_list
        elif self.mode == 'hard':
            self.sample_list = self.hard_list
        elif self.mode == 'mix':
            self.sample_list = self.esasy_list + self.hard_list

    def __len__(self):
        return len(self.sample_list)

    def get_list(self,):
        with open(os.path.join(self.opt.dataset.data_root,f"thuman2.1_{self.phase}.txt"), 'r') as f:
            lines = f.readlines()
        easydata,hard = [],[]
        for i in lines:
            if i.startswith("easy_"):
                easydata.append(i.lstrip("easy_").rstrip("\n").split("_2_"))
            else:
                hard.append(i.lstrip("hard_").rstrip("\n").split("_2_"))
        return easydata,hard

    def train_data(self, index):
        id, target = self.sample_list[index]
        img_path = self.img_path % id
        dcit_tensor = {}
        dcit_tensor['source_view'] = None
        if self.other_view:
            source = ["source.png", "view1.png", "view2.png", "view3.png"]
            dcit_tensor['source_view'] = self.load_multi_view(img_path, source, require_spmlx=False, phase=self.phase)
        else:
            dcit_tensor['source_view'] = self.load_single_view(img_path, "source.png", require_spmlx=False,
                                                               phase=self.phase)

        dcit_tensor['target_view'] = self.get_single_novel_view_tensor(img_path, target+".png")
        dcit_tensor["pair"] = f"{id}_2_{target}"
        return dcit_tensor

    def val_data(self, index):
        id, target = self.sample_list[index]
        img_path = self.img_path % id
        dcit_tensor = {}
        dcit_tensor['source_view'] = None
        if self.other_view:
            source = ["source.png", "view1.png", "view2.png", "view3.png"]
            dcit_tensor['source_view'] = self.load_multi_view(img_path, source, require_spmlx=False, phase=self.phase)
        else:
            dcit_tensor['source_view'] = self.load_single_view(img_path, "source.png", require_spmlx=False,
                                                               phase=self.phase)
        dcit_tensor['target_view'] = self.get_single_novel_view_tensor(img_path, target+".png")
        dcit_tensor["pair"] = f"{id}_2_{target}"
        return dcit_tensor

    def load_single_view(self, img_path, sample_name, resize=None, require_mask=True,
                         require_depth=True, require_spmlx=True, phase='train'):

        img_name = os.path.join(img_path, sample_name)
        mask_name = img_name.replace('img', 'mask')
        depth_name = mask_name.replace('mask', 'depth')
        intr_name = img_name.replace('img', 'parm').replace('.png', '_intrinsic.npy')
        extr_name = intr_name.replace('_intrinsic.npy', "_extrinsic.npy")
        spmlx_name = img_name.replace('img', 'smplx').replace('.png', '_smpl_mesh.obj')
        if sample_name in ["view1.png", "view2.png", "view3.png"]:
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
        if require_spmlx and sample_name == "source.png":
            pc = get_pointcloud(spmlx_name)

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