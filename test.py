import torch
import time
import numpy as np
import nibabel as nib
import torch.nn.functional as F
from options.test_options import TestOptions
from data import CreateDataLoader
from models import create_model
from collections import OrderedDict, defaultdict
from skimage import measure
from scipy.stats import pearsonr
from configurations import *


def pad_images(opt_test, *image_list):
    padNum = -1
    pad_pos_y = (opt_test.testSize - image_list[0][0].shape[-2]) // 2
    pad_pos_x = (opt_test.testSize - image_list[0][0].shape[-1]) // 2
    pad_param = [pad_pos_x, opt_test.testSize - image_list[0].shape[-1] - pad_pos_x,
                 pad_pos_y, opt_test.testSize - image_list[0].shape[-2] - pad_pos_y]

    var_return = []
    image_list = list(image_list)
    for one_image in image_list:
        pad_image = F.pad(one_image, pad_param, 'constant', padNum)
        var_return += [pad_image]

    sl = [slice(None)] * 2
    sl[0] = slice(pad_pos_y, pad_pos_y + image_list[0][0].shape[-2], 1)
    sl[1] = slice(pad_pos_x, pad_pos_x + image_list[0][0].shape[-1], 1)
    var_return += [tuple(sl)]
    return var_return


def seg_metrics(seg_vol, truth_vol, output_errors=False):
    time_start = time.time()
    seg_total = np.sum(seg_vol)
    truth_total = np.sum(truth_vol)
    tp = np.sum(seg_vol[truth_vol == 1])
    dice = 2 * tp / (seg_total + truth_total)
    ppv = tp / (seg_total + 0.001)
    tpr = tp / (truth_total + 0.001)
    vd = abs(seg_total - truth_total) / truth_total

    # calculate LFPR
    seg_labels, seg_num = measure.label(seg_vol, return_num=True, connectivity=2)
    lfp_cnt = 0
    tmp_cnt = 0
    for label in range(1, seg_num + 1):
        tmp_cnt += np.sum(seg_vol[seg_labels == label])
        if np.sum(truth_vol[seg_labels == label]) == 0:
            lfp_cnt += 1
    lfpr = lfp_cnt / (seg_num + 0.001)

    # calculate LTPR
    truth_labels, truth_num = measure.label(truth_vol, return_num=True, connectivity=2)
    ltp_cnt = 0
    for label in range(1, truth_num + 1):
        if np.sum(seg_vol[truth_labels == label]) > 0:
            ltp_cnt += 1
    ltpr = ltp_cnt / truth_num

    # calculate Pearson's correlation coefficient
    corr = pearsonr(seg_vol.flatten(), truth_vol.flatten())[0]
    # print("Timed used calculating metrics: ", time.time() - time_start)

    return OrderedDict([('dice', dice), ('ppv', ppv), ('tpr', tpr), ('lfpr', lfpr),
                        ('ltpr', ltpr), ('vd', vd), ('corr', corr)])


def print_metrics(prefix, metrics):
    message = prefix + ' '
    for k, v in metrics.items():
        message += '%s: %.3f ' % (k, v)
    print(message)


def model_test(models, dataset_test, opt_test, num_test, save_images=False, models_weight=None,
               mask_suffix='pred', save_membership=False):
    if not num_test:
        print("no %s subjects" % opt_test.phase)
    assert len(models), "no models loaded"

    start_time = time.time()
    orientations = ['axial', 'sagittal', 'coronal']
    transpose = {2: (1, 2, 0), 0: (0, 1, 2), 1: (1, 0, 2)}
    orientation_weight = [1, 1, 1]
    ret_metrics = defaultdict(float)
    metrics = []

    for i, data in enumerate(dataset_test):
        if i >= num_test:
            break

        mask, mask_path, alt_path = data['mask'], data['mask_paths'][0], data['alt_paths'][0]
        basename = os.path.basename(data['alt_paths'][0])
        basename = basename[:len(basename) - len(MODALITIES[0]) - len(SUFFIX) - 1]

        mask_pred = 0
        for k, orientation in enumerate(orientations):
            mask_cur_orientation = []
            num_slices = len(data[MODALITIES[0]][orientation])
            for j in range(num_slices):
                pad_data ={}
                if os.path.exists(mask_path):
                    pad_data['mask'], sl = pad_images(opt_test, data['mask'][orientation][j])
                else:
                    pad_data['mask'] = torch.zeros_like(data[MODALITIES[0]][orientation][j])
                for modality in MODALITIES:
                    pad_data[modality], sl = pad_images(opt_test, data[modality][orientation][j])

                slice_all_models = 0
                for m, current_model in enumerate(models):
                    current_model.set_input({mod: pad_data[mod] for mod in MODALITIES + ['mask']})
                    current_model.test()
                    current_visuals = current_model.get_current_visuals()
                    weight_this_model = 1 if models_weight is None else models_weight[m]
                    slice_this_model = np.squeeze(current_visuals['fake_mask'].cpu().numpy())[sl]
                    slice_all_models += slice_this_model * weight_this_model
                numerator = len(models) if models_weight is None else np.sum(models_weight)
                slice_all_models = np.array(slice_all_models) / numerator
                mask_cur_orientation.append(slice_all_models)
            mask_pred += np.transpose(np.squeeze(mask_cur_orientation), transpose[AXIS_TO_TAKE[k]]) * \
                         orientation_weight[k]
        mask_pred = np.array(mask_pred) / np.sum(orientation_weight)

        alt_image = nib.load(alt_path)
        if save_membership:
            mask_membership_name = alt_path.replace('%s.%s' % (MODALITIES[0], SUFFIX),
                                                    'membership_%s.%s' % (mask_suffix, SUFFIX))
            nib.Nifti1Image(mask_pred, alt_image.affine, alt_image.header).to_filename(mask_membership_name)
        mask_pred = (mask_pred > 0).astype(np.int8)

        if os.path.exists(mask_path):
            mask_data = nib.load(mask_path).get_fdata().astype(np.int8)
            res_this_mask = seg_metrics(mask_pred, mask_data, output_errors=False)
            metrics = list(res_this_mask.keys())
            for k in metrics:
                ret_metrics[k] += res_this_mask[k]
            print_metrics('processed ' + basename + '*,', res_this_mask)
        else:
            print('processed ' + basename + '*')

        if save_images:
            mask_pred_name = alt_path.replace('%s.%s' % (MODALITIES[0], SUFFIX), 'pred_%s.%s' % (mask_suffix, SUFFIX))
            nib.Nifti1Image(mask_pred, alt_image.affine, alt_image.header).to_filename(mask_pred_name)

    for k in metrics:
        ret_metrics[k] = ret_metrics[k] / num_test if num_test != 0 else ret_metrics[k]
    print("time used for validation: ", time.time() - start_time)
    return ret_metrics


if __name__ == '__main__':
    opt_test = TestOptions().parse()

    # hard-code some parameters for test
    opt_test.num_threads = 1   # test code only supports num_threads = 1
    opt_test.batch_size = 1    # test code only supports batch_size = 1
    opt_test.serial_batches = True  # no shuffle
    opt_test.no_flip = True    # no flip
    opt_test.display_id = -1   # no visdom display
    opt_test.dataset_mode = 'ms_3d'
    data_loader = CreateDataLoader(opt_test)
    dataset_test = data_loader.load_data()

    models = []
    models_indx = opt_test.load_str.split(',')
    models_weight = [1] * len(models_indx)
    for i in models_indx:
        current_model = create_model(opt_test, i)
        current_model.setup(opt_test)
        if opt_test.eval:
            current_model.eval()
        models.append(current_model)

    losses = model_test(models, dataset_test, opt_test, len(data_loader), save_images=True,
                       models_weight=models_weight, mask_suffix=opt_test.name, save_membership=True)
    print_metrics('test results', losses)

