import os
import json
import shutil
import nibabel as nib
import numpy as np
from util.util import mkdir
from util.image_property import normalize_image, hash_file
from configurations import *


def get_ids():
    dic_ids = {}
    mkdir(os.path.join(PATH_DATASET, 'raw'))
    fnames = sorted(os.listdir(PATH_DATASET))
    for fname in fnames:
        if not fname.endswith(SUFFIX):
            continue
        fname_tmp = fname.split('.')[0]
        prefix, patient_str, timepoint_str, modality = fname_tmp.split('_')
        patient_id, timepoint_id = int(patient_str), int(timepoint_str)
        # patient_id, timepoint_id = int(patient_id), int(timepoint_id)
        if patient_id in dic_ids and timepoint_id in dic_ids[patient_id]:
            continue

        # if there is new patient id and timepoint id, we get all the modalities and masks based on the constants
        # and we will move the files into the 'raw' subdirectory
        if patient_id not in dic_ids:
            dic_ids[patient_id] = {}
        dic_ids[patient_id][timepoint_id] = {'modalities':{}, 'mask':{}}
        for mod in MODALITIES+MASKS:
            fname_modality = '_'.join((prefix, patient_str, timepoint_str, mod)) + '.' + SUFFIX
            path_modality_src = os.path.join(PATH_DATASET, fname_modality)
            path_modality_dst = os.path.join(PATH_DATASET, 'raw', fname_modality)
            category = 'modalities' if mod in MODALITIES else 'mask'
            assert os.path.exists(path_modality_src)
            shutil.move(path_modality_src, path_modality_dst)
            dic_ids[patient_id][timepoint_id][category][mod] = path_modality_dst
    fname_json = os.path.join(PATH_DATASET, 'ids.json')
    with open(fname_json, 'w') as f:
        json.dump(dic_ids, f, indent=2)
    return dic_ids


def get_properties():
    fname_json = os.path.join(PATH_DATASET, 'ids.json')
    with open(fname_json, 'r') as f:
        dic_ids = json.load(f)
    dic_properties = {}
    for patient_id in dic_ids:
        for timepoint_id in dic_ids[patient_id]:
            for modality in dic_ids[patient_id][timepoint_id]['modalities']:
                path_modality = dic_ids[patient_id][timepoint_id]['modalities'][modality]
                label = hash_file(path_modality)
                data = nib.load(path_modality).get_fdata()
                peak = normalize_image(data, modality)
                peak = peak[0] if isinstance(peak, np.ndarray) else peak
                dic_properties[label] = {}
                dic_properties[label]['path'] = path_modality
                dic_properties[label]['peak'] = peak
    fname_json = os.path.join(PATH_DATASET, 'properties.json')
    with open(fname_json, 'w') as f:
        json.dump(dic_properties, f, indent=2)
    return dic_properties


if __name__ == '__main__':
    assert os.path.exists(PATH_DATASET)

    dic_ids = get_ids()
    dic_properties = get_properties()
