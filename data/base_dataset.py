import os.path
import pickle
from utils.graphics_utils import *
import open3d as o3d
import glob
import numpy as np
from torch.utils.data import Dataset
from utils.data_utils import *
from utils.utils import *
import torchvision.transforms.functional as TF


class BaseDataset(Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def initialize(self, opt, phase):
        self.opt = opt
        self.other_view = opt.model.other_view
        # self.use_processed_data = opt.use_processed_data
        self.phase = phase
        self.data_root = os.path.join(opt.dataset.data_root, f'big_{self.phase}_s512_t{opt.model.tar_res}')
        self.sample_list = sorted(list(os.listdir(os.path.join(self.data_root, 'img')))) if self.phase == 'train' else self.get_list()

        self.img_path = os.path.join(self.data_root, 'img/%s')
        self.mask_path = os.path.join(self.data_root, 'mask/%s')
        self.depth_path = os.path.join(self.data_root, 'depth/%s')
        self.intr_path = os.path.join(self.data_root, 'parm/%s')
        self.extr_path = os.path.join(self.data_root, 'parm/%s')

    def __getitem__(self, index):
        if self.phase == 'train':
            return self.train_data(index)
        else:
            return self.val_data(index)

    def get_list(self,):
        data = []
        for i in glob.glob(os.path.join(self.data_root, "img", "*", "tar_*.png")):
            path, name = os.path.split(i)
            if not self.opt.model.src_rec and name == "tar_0.png":
                continue
            data.append([os.path.basename(path), name])
        return data

    def train_data(self, index):
        id = self.sample_list[index]
        img_path = self.img_path % id
        tar_list = [i for i in os.listdir(img_path) if i.startswith('tar')]
        if not self.opt.model.src_rec:
            tar_list.remove("tar_0.png")
        dcit_tensor = {}
        dcit_tensor['source_view'] = None
        if self.other_view:
            source = ["source.png", "view1.png", "view2.png", "view3.png"]
            dcit_tensor['source_view'] = self.load_multi_view(img_path, source, require_spmlx=False, phase=self.phase)
        else:
            dcit_tensor['source_view'] = self.load_single_view(img_path, "source.png", require_spmlx=False,
                                                               phase=self.phase)

        target = np.random.choice(tar_list, 1)[0]
        dcit_tensor['target_view'] = self.get_single_novel_view_tensor(img_path, target)
        dcit_tensor["pair"] = f"{id}_2_{target}"
        return dcit_tensor

    def val_data(self, index):
        id, target = self.sample_list[index]
        img_path = self.img_path % id
        dcit_tensor = {}
        dcit_tensor['source_view'] = None
        if self.other_view:
            source = ["source.png", "view1.png", "view2.png", "view3.png"]
            dcit_tensor['source_view'] = self.load_multi_view(img_path, source, require_spmlx=True, phase=self.phase)
        else:
            dcit_tensor['source_view'] = self.load_single_view(img_path, "source.png", require_spmlx=True,
                                                               phase=self.phase)

        dcit_tensor['target_view'] = self.get_single_novel_view_tensor(img_path, target)
        dcit_tensor["pair"] = f"{id}_2_{target}"
        return dcit_tensor

    def load_multi_view(self, img_path, source_list, resize=None, require_mask=True,
                        require_depth=True, require_spmlx=True, phase="test"):
        source_dict = {}

        for source in source_list:
            out = self.load_single_view(img_path, source, resize=resize, require_mask=require_mask,
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

    def __len__(self):
        return len(self.sample_list)

    # TODO we need to know how the function works
    def load_single_view(self, img_path, sample_name, resize=None, require_mask=True,
                         require_depth=True, require_spmlx=True, phase='train'):

        img_name = os.path.join(img_path, sample_name)
        mask_name = img_name.replace('img', 'mask')
        depth_name = mask_name.replace('mask', 'depth')
        intr_name = img_name.replace('img', 'parm').replace('.png', '_intrinsic.npy')
        extr_name = intr_name.replace('_intrinsic.npy', "_extrinsic.npy")
        spmlx_name = img_name.replace('img', 'smplx').replace('.png', '_smpl_mesh.obj')

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

    def stereo_to_dict_tensor(self, stereo_data, subject_name):
        img_tensor, mask_tensor = [], []
        for (img_view, mask_view) in [('img0', 'mask0'), ('img1', 'mask1')]:
            img = torch.from_numpy(stereo_data[img_view]).permute(2, 0, 1)
            img = 2 * (img / 255.0) - 1.0
            mask = torch.from_numpy(stereo_data[mask_view]).permute(2, 0, 1).float()
            mask = mask / 255.0

            img = img * mask
            mask[mask < 0.5] = 0.0
            mask[mask >= 0.5] = 1.0
            img_tensor.append(img)
            mask_tensor.append(mask)

        lmain_data = {
            'img': img_tensor[0],
            'mask': mask_tensor[0],
            'intr': torch.FloatTensor(stereo_data['camera']['intr0']),
            'ref_intr': torch.FloatTensor(stereo_data['camera']['intr1']),
            'extr': torch.FloatTensor(stereo_data['camera']['extr0']),
            'Tf_x': torch.FloatTensor(stereo_data['camera']['Tf_x'])
        }

        rmain_data = {
            'img': img_tensor[1],
            'mask': mask_tensor[1],
            'intr': torch.FloatTensor(stereo_data['camera']['intr1']),
            'ref_intr': torch.FloatTensor(stereo_data['camera']['intr0']),
            'extr': torch.FloatTensor(stereo_data['camera']['extr1']),
            'Tf_x': -torch.FloatTensor(stereo_data['camera']['Tf_x'])
        }

        if 'flow0' in stereo_data:
            flow_tensor, valid_tensor = [], []
            for (flow_view, valid_view) in [('flow0', 'valid0'), ('flow1', 'valid1')]:
                flow = torch.from_numpy(stereo_data[flow_view])
                flow = torch.unsqueeze(flow, dim=0)
                flow_tensor.append(flow)

                valid = torch.from_numpy(stereo_data[valid_view])
                valid = torch.unsqueeze(valid, dim=0)
                valid = valid / 255.0
                valid_tensor.append(valid)

            lmain_data['flow'], lmain_data['valid'] = flow_tensor[0], valid_tensor[0]
            rmain_data['flow'], rmain_data['valid'] = flow_tensor[1], valid_tensor[1]

        return {'name': subject_name, 'lmain': lmain_data, 'rmain': rmain_data}

    def get_single_novel_view_tensor(self, img_path, sample_name):
        img_name = os.path.join(img_path, sample_name)
        if self.opt.model.src_rec and sample_name == "tar_0.png":
            intr_name = img_name.replace('img', 'parm').replace('tar_0.png', 'source_intrinsic.npy')
        else:
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
    # def get_novel_view_tensor(self, img_path, sample_name):
    #     img_list, mask_list, intr_list, extr_list = [], [], [], []
    #     if not isinstance(sample_name, list):
    #         sample_name = [sample_name]
    #     for tn in sample_name:
    #         img, mask, intr, extr = self.load_single_view(img_path, tn, require_mask=True, require_dict=False)
    #         img_list.append(img)
    #         mask_list.append(mask)
    #         intr_list.append(intr)
    #         extr_list.append(extr)
    #     img = torch.stack(img_list)
    #     mask = torch.stack(mask_list)
    #     intr = torch.stack(intr_list)
    #     extr = torch.stack(extr_list)
    #     # height, width = img.shape[:2]
    #
    #     # R = np.array(extr[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
    #     # T = np.array(extr[:3, 3], np.float32)
    #     #
    #     # FovX = focal2fov(intr[0, 0], width)
    #     # FovY = focal2fov(intr[1, 1], height)
    #     # projection_matrix = getProjectionMatrix(znear=self.opt.dataset.znear, zfar=self.opt.dataset.zfar, K=intr,
    #     #                                         h=height,
    #     #                                         w=width).transpose(0, 1)
    #     # world_view_transform = torch.tensor(
    #     #     getWorld2View2(R, T, np.array(self.opt.dataset.trans), self.opt.dataset.scale)).transpose(0,
    #     #                                                                                               1)
    #     # full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
    #     # camera_center = world_view_transform.inverse()[3, :3]
    #
    #     novel_view_data = {
    #         'view': sample_name,
    #         'img': img,
    #         "mask": mask,
    #         "intr": intr,
    #         'extr': extr,
    #         # 'FovX': FovX,
    #         # 'FovY': FovY,
    #         # 'width': width,
    #         # 'height': height,
    #         # 'world_view_transform': world_view_transform,
    #         # 'full_proj_transform': full_proj_transform,
    #         # 'camera_center': camera_center
    #     }
    #     return novel_view_data
