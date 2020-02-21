import os
import time
import pickle
import hashlib
import numpy as np
import nibabel as nib
import statsmodels.api as sm
from scipy.signal import argrelextrema


def hash_file(filename):
    """"This function returns the SHA-1 hash
    of the file passed into it"""

    # make a hash object
    h = hashlib.sha1()

    # open file for reading in binary mode
    with open(filename, 'rb') as file:
        # loop till the end of the file
        chunk = 0
        while chunk != b'':
            # read only 1024 bytes at a time
            chunk = file.read(1024)
            h.update(chunk)

    # return the hex representation of digest
    return h.hexdigest()


def normalize_image(vol, contrast):
    # copied from FLEXCONN
    # slightly changed to fit our implementation
    temp = vol[np.nonzero(vol)].astype(float)
    q = np.percentile(temp, 99)
    temp = temp[temp <= q]
    temp = temp.reshape(-1, 1)
    bw = q / 80
    # print("99th quantile is %.4f, gridsize = %.4f" % (q, bw))

    kde = sm.nonparametric.KDEUnivariate(temp)

    kde.fit(kernel='gau', bw=bw, gridsize=80, fft=True)
    x_mat = 100.0 * kde.density
    y_mat = kde.support

    indx = argrelextrema(x_mat, np.greater)
    indx = np.asarray(indx, dtype=int)
    heights = x_mat[indx][0]
    peaks = y_mat[indx][0]
    peak = 0.00
    # print("%d peaks found." % (len(peaks)))

    # norm_vol = vol
    if contrast.lower() in ["t1", "mprage"]:
        peak = peaks[-1]
        # print("Peak found at %.4f for %s" % (peak, contrast))
        # norm_vol = vol/peak
        # norm_vol[norm_vol > 1.25] = 1.25
        # norm_vol = norm_vol/1.25
    elif contrast.lower() in ['t2', 'pd', 'flair', 'fl']:
        peak_height = np.amax(heights)
        idx = np.where(heights == peak_height)
        peak = peaks[idx]
        # print("Peak found at %.4f for %s" % (peak, contrast))
        # norm_vol = vol / peak
        # norm_vol[norm_vol > 3.5] = 3.5
        # norm_vol = norm_vol / 3.5
    else:
        print("Contrast must be either t1,t2,pd, or flair. You entered %s. Returning 0." % contrast)

    # return peak, norm_vol
    return peak


def slice_with_neighborhood(data_3d, axis_to_take, idx, neighborhood=0):
    axis_len = data_3d.shape[axis_to_take]
    assert axis_to_take in [0, 1, 2]
    assert axis_len > idx >= 0
    transpose = [[1, 2, 0], [0, 2, 1], [0, 1, 2]]
    sl = [slice(None)] * 3
    if idx - neighborhood < 0:
        sl[axis_to_take] = slice(0, idx + neighborhood + 1, 1)
        slice_tmp = np.transpose(np.copy(data_3d[tuple(sl)]), transpose[axis_to_take])
        shape = slice_tmp.shape
        array_pad = np.zeros((shape[0], shape[1], neighborhood-idx))
        # print(slice_tmp.shape, array_pad.shape)
        slice_to_return = np.concatenate((array_pad, slice_tmp), axis=2)
        return slice_to_return

    if idx + neighborhood >= axis_len:
        sl[axis_to_take] = slice(idx - neighborhood, axis_len, 1)
        slice_tmp = np.transpose(np.copy(data_3d[tuple(sl)]), transpose[axis_to_take])
        shape = slice_tmp.shape
        array_pad = np.zeros((shape[0], shape[1], idx + neighborhood - axis_len + 1))
        slice_to_return = np.concatenate((slice_tmp, array_pad), axis=2)
        return slice_to_return

    sl[axis_to_take] = slice(idx - neighborhood, idx + neighborhood + 1, 1)
    return np.transpose(np.copy(data_3d[tuple(sl)]), transpose[axis_to_take])