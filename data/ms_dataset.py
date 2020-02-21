import os.path
import random
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch
import numpy as np
from data.base_dataset import BaseDataset


def get_2d_paths(dir):
    arrays = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith('.pkl'):
                path = os.path.join(root, fname)
                arrays.append(path)

    return arrays


class MsDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.opt = opt
        self.dir_data = os.path.join(opt.dataroot, opt.phase)
        self.all_paths = sorted(get_2d_paths(self.dir_data))

    def __getitem__(self, index):
        path_this_sample = self.all_paths[index]
        data_all_modalities = np.load(path_this_sample, allow_pickle=True)
        data_return = {'paths': path_this_sample}

        for modality in data_all_modalities.keys():
            data_return[modality] = transforms.ToTensor()(data_all_modalities[modality])
        mask = data_return['mask']
        _, h, w2 = mask.size()

        if w2 > self.opt.trainSize and h > self.opt.trainSize:
            cnt = 0
            while True:
                w_offset = random.randint(0, max(0, w2 - self.opt.trainSize - 1))
                h_offset = random.randint(0, max(0, h - self.opt.trainSize - 1))
                if torch.sum(torch.nonzero(mask[:, h_offset:h_offset + self.opt.trainSize,
                                 w_offset:w_offset + self.opt.trainSize])) >= 1 or cnt == 9:
                    for modality in data_all_modalities.keys():
                        data_return[modality] = data_return[modality][:, h_offset:h_offset + self.opt.trainSize,
                                                w_offset:w_offset + self.opt.trainSize]
                    break
                cnt += 1
        elif w2 < self.opt.trainSize and h < self.opt.trainSize:
            w_offset = random.randint(0, max(0, self.opt.trainSize - w2 - 1))
            h_offset = random.randint(0, max(0, self.opt.trainSize - h - 1))
            pad_param = [w_offset, self.opt.trainSize - w2 - w_offset, h_offset, self.opt.trainSize - h - h_offset]
            for modality in data_all_modalities.keys():
                data_return[modality] = F.pad(data_return[modality], pad_param, 'constant', 0)
        elif w2 == self.opt.trainSize and h == self.opt.trainSize:
            pass
        else:
            raise ValueError('w and h should be both larger than trainSize or smaller than trainSize, '
                             'but got w=%d, h=%d, trainSize=%d' % (w2, h, self.opt.trainSize))

        for modality in data_all_modalities.keys():
            if modality == 'mask':
                data_return[modality] = data_return[modality] * 2 - 1
            else:
                data_return[modality] = data_return[modality] / 2 - 1

        for flip_axis in [2, 1]:
            if (not self.opt.no_flip) and random.random() < 0.3:
                idx = [i for i in range(data_return['mask'].size(flip_axis) - 1, -1, -1)]
                idx = torch.LongTensor(idx)
                for modality in data_all_modalities.keys():
                    data_return[modality] = data_return[modality].index_select(flip_axis, idx)

        if (not self.opt.no_flip) and random.random() < 0.5:
            for modality in data_all_modalities.keys():
                data_return[modality] = data_return[modality].transpose(1, 2)

        return data_return

    def __len__(self):
        return len(self.all_paths)

    def name(self):
        return 'MsDataset'
