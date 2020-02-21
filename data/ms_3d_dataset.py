import os.path
import torchvision.transforms as transforms
import numpy as np
import json
import nibabel as nib
from data.base_dataset import BaseDataset
from configurations import *
from util.image_property import hash_file, normalize_image, slice_with_neighborhood


def get_3d_paths(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if fname.endswith(MODALITIES[0]+'.' + SUFFIX):
                images.append([])
                images[-1].append(os.path.join(root, fname))
                for i in range(1, len(MODALITIES)):
                    images[-1].append(os.path.join(root, fname.replace(MODALITIES[0]+'.'+SUFFIX, MODALITIES[i]+'.' + SUFFIX)))
                images[-1].append(os.path.join(root, fname.replace(MODALITIES[0] + '.' + SUFFIX, 'mask.' + SUFFIX)))

    images.sort(key=lambda x: x[0])
    return images


def flip_by_times(np_array, times):
    for i in range(times):
        np_array = np.flip(np_array, axis=1)
    return np_array


# currently, this dataset is for test use only
class Ms3dDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def initialize(self, opt):
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.all_paths = get_3d_paths(self.dir_AB)
        self.neighbors = opt.input_nc // 2
        with open(os.path.join(opt.dataroot, 'properties.json'), 'r') as f:
            self.dic_properties = json.load(f)

    def __getitem__(self, index):
        paths_this_scan = self.all_paths[index]
        data_all_modalities = {}
        if os.path.exists(paths_this_scan[-1]):
            data_all_modalities['mask'] = nib.load(paths_this_scan[-1]).get_fdata()
        for i, modality in enumerate(MODALITIES):
            path_modality = paths_this_scan[i]
            label_modality = hash_file(path_modality)
            data_modality = nib.load(path_modality).get_fdata()
            if label_modality in self.dic_properties:
                peak_modality = self.dic_properties[label_modality]['peak']
            else:
                peak_modality = normalize_image(data_modality, modality)
            data_all_modalities[modality] = np.array(data_modality / peak_modality, dtype=np.float32)

        data_return = {mod: {'axial': [], 'sagittal': [], 'coronal': []} for mod in MODALITIES+['mask']}
        data_return['mask_paths'] = paths_this_scan[-1]
        data_return['alt_paths'] = paths_this_scan[0]

        for k, orientation in enumerate(['axial', 'sagittal', 'coronal']):
            slices_per_image = data_all_modalities[MODALITIES[0]].shape[AXIS_TO_TAKE[k]]
            for i in range(slices_per_image):
                for modality in MODALITIES:
                    slice_modality = slice_with_neighborhood(data_all_modalities[modality], AXIS_TO_TAKE[k], i, self.neighbors)
                    slice_modality = transforms.ToTensor()(slice_modality)
                    slice_modality = slice_modality.float() / 2 - 1
                    data_return[modality][orientation].append(slice_modality)
                if os.path.exists(paths_this_scan[-1]):
                    slice_modality = slice_with_neighborhood(data_all_modalities['mask'], AXIS_TO_TAKE[k], i, 0)
                    slice_modality = transforms.ToTensor()(slice_modality)
                    slice_modality = slice_modality.float() * 2 - 1
                    data_return['mask'][orientation].append(slice_modality)

        return data_return

    def __len__(self):
        return len(self.all_paths)

    def name(self):
        return 'Ms3dDataset'
