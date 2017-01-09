import scipy.io as scio
import skimage
import numpy as np
import math
from PIL import Image
import os
from skimage import transform, io as skio

mean_rgb = [122.675, 116.669, 104.008]
scales = [0.6, 0.8, 1.2, 1.5]
rorations = [-45, -22, 22, 45]
gammas = [.05, 0.8, 1.2, 1.5]

def gen_data(name):
    reftracker = scio.loadmat('data/images_tracker.00047.mat')
    desttracker = scio.loadmat('data/images_tracker/'+name+'.mat')
    refpos = np.floor(np.mean(reftracker, 0))
    xxc, yyc = np.meshgrid(np.arange(1, 1801, dtype=np.int), np.arange(1, 2001, dtype=np.int))
    #normalize x and y channels
    xxc = (xxc - 600 - refpos[0]) * 1.0 / 600
    yyc = (yyc - 600 - refpos[1]) * 1.0 / 600
    maskimg = Image.open('data/meanask.png')
    maskc = np.array(maskimg, dtype=np.float)
    maskc = np.pad(maskc, (600, 600), 'minimum')
    tform = transform.ProjectiveTransform()
    tform.estimate(reftracker + 600, desttracker + 600)

    img_data = skio.imread('data/images_data/'+name+'.jpg')
    # save org mat
    warpedxx = transform.warp(img_data, tform, output_shape=xxc.shape)
    warpedyy = transform.warp(img_data, tform, output_shape=xxc.shape)
    warpedmask = transform.warp(img_data, tform, output_shape=xxc.shape)
    warpedxx = warpedxx[600:1400, 600:1200, :]
    warpedyy = warpedyy[600:1400, 600:1200, :]
    warpedmask = warpedmask[600:1400, 600:1200, :]
    img_h, img_w, _ = img_data.shape
    mat = np.zeros((img_h, img_w, 6), dtype=np.float)
    mat[:, :, 0] = (img_data[2] * 1.0 - 104.008) / 255
    mat[:, :, 1] = (img_data[1] * 1.0 - 116.669) / 255
    mat[:, :, 2] = (img_data[0] * 1.0 - 122.675) / 255
    scio.savemat('portraitFCN_data/' + name + '.mat', {'img':mat})
    mat_plus = np.zeros((img_h, img_w, 6), dtype=np.float)
    mat_plus[:, :, 0:3] = mat
    mat_plus[:, :, 3] = warpedxx
    mat_plus[:, :, 4] = warpedyy
    mat_plus[:, :, 5] = warpedmask

def gamma_trans(mat, gamma):
    gamma_mean = np.pow(mean_rgb / 255, gamma)
    tmp_mat = np.pow(mat / 255, gamma)
    gamma_mat = np.zeros(mat.shape, dtype=np.float)
    gamma_mat[:, :, 0] = tmp_mat[:, :, 2] - gamma_mean[:, :, 2]
    gamma_mat[:, :, 1] = tmp_mat[:, :, 1] - gamma_mean[:, :, 1]
    gamma_mat[:, :, 2] = tmp_mat[:, :, 0] - gamma_mean[:, :, 0]
    return gamma_mat

def crop_all():
    files = os.listdir('data/images_data_crop')
    if not os.path.exists('data/portraitFCN_data'):
        os.mkdir('data/portraitFCN_data')
    if not os.path.exists('data/portraitFCN+_data'):
        os.mkdir('data/portraitFCN+_data')
