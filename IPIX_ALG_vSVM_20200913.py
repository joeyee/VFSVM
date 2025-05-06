# Local Binary Pattern function
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import numpy as np
import Load_IPIX_200701 as ipixmodel
import cfar_segmentation_200527 as cfar_model     # cfar
from   PIL import Image

from netCDF4 import Dataset
from scipy import io
from scipy import ndimage
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_curve, auc
import cv2
import os
import utilities_200611         as uti            # personal tools
import time

def get_tdoppl_mat(am, rangebin, Nf=512, Nt=2**17, PRF=1000):
    N = len(am)
    timeStep = Nf
    # wdw      = Nf
    # timeStep = 128
    # M        = int((N - wdw) / timeStep)

    xpix = int(N / timeStep)  # image's cols is xpix, 512->xpix=256, 1024->xpix=128
    #xpix = int(N / Nf)  # image's cols is xpix, 512->xpix=256, 1024->xpix=128
    ypix = 64  # /2        #image's row is ypix = 64

    hw       = np.hamming(Nf)
    #TD = np.zeros((Nf, M))


    yavg = max(1, int(Nf / ypix))  # the averaging len on the y axis.

    # adjust fft window and averaging to match image size
    # % short time fourier transforms
    # hw = np.hamming(Nf)
    TD = np.zeros((ypix, xpix))
    y = np.zeros((int(Nf / yavg), yavg))
    leny = y.size


    for m in range(xpix):
    #for m in range(M):

        # i  = m * timeStep
        # # be attention, the timestep Nf is not equal to the local observations.
        # x  = hw * am[i:wdw + i]
        # fftx = abs(np.fft.fftshift(np.fft.fft(x)))
        # TD[:, m] = fftx

        i = m * timeStep
        #i = m * Nf
        x = hw * am[i:Nf + i]
        fftx = abs(np.fft.fftshift(np.fft.fft(x)))
        y = np.reshape(fftx.ravel()[0:leny], (int(Nf / yavg), yavg))
        TD[:, m] = np.mean(y, 1)

    # % enhance log-plot with noise floor
    mn = np.mean(TD, 1)
    indx = np.argsort(mn)
    mn = mn[indx]
    noise = TD[indx[0:2], :]
    noiseFloor = np.median(noise[:])
    TD[TD < noiseFloor] = noiseFloor
    TD = np.flipud(TD)
    #logTD = TD
    logTD = np.log(TD)
    #logTD = (logTD - np.min(logTD)) * 63 / (np.max(logTD) - np.min(logTD))
    # % time and normalized frequency
    observ_time = Nt / PRF
    #plot_td(logTD, PRF, observ_time, rangebin)

    logTD = (logTD - np.min(logTD)) * 1 / (np.max(logTD) - np.min(logTD))
    #logTD = (TD - np.min(TD))*1. / (np.max(TD) - np.min(TD))
    #logTD = logTD.astype(np.uint8)
    return logTD

def compute_lbp_histogram(frame, P=8, R=1, method='uniform'):
    '''
    Compute lbp histogram for a tds image @frame
    :param frame:
    :param P:
    :param R:
    :param method:
    :return:
    '''
    lbp_frame = local_binary_pattern(frame, P=P, R=R, method=method)
    # # #side_lbp_hist, lpb_bins = np.histogram(side_lbp.ravel(), bins=10, density=True)
    bin_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # weights   = np.array([2, 1, 1, 3, 3, 1, 1, 1, 2])
    weights = np.array([2, 1, 1, 3, 3, 1, 1, 1, 2])
    # weights = np.array([1.5, 1, 1, 3, 3, 1, 1, 1, 1.5])
    weights = weights / np.sum(weights)
    lbp_hist, lpb_bins = np.histogram(lbp_frame.ravel(), bins=int(np.max(lbp_frame)), density=False)
    # lbp_hist = lbp_hist / lbp_frame.size
    lbp_hist_wt = lbp_hist * weights
    lbp_hist = lbp_hist_wt / np.sum(lbp_hist_wt)
    return lbp_hist

def training_svm(xfeatures):
    # Training vSVM
    clf = OneClassSVM(nu=0.4, kernel='rbf', gamma='auto')
    clf.fit(xfeatures)
    y_score = clf.score_samples(xfeatures)
    # y_test = clf.predict(xfeatures)
    least_votes = np.argsort(y_score)
    minvote = least_votes[0]
    return minvote

def alg_vfsvm():
    fileprefix = '/Users/yizhou/Radar_Datasets/IPIX/'
    #Nf = 512#512#1024#512  # 64#128#256#512*2
    #Nt = 2 ** 17  # 60000 for 1998file,  #131072 for 1993file

    #loading csv file for listing the wanted filenames, primary and secondary cells
    samples_info = pd.read_csv('/Users/yizhou/Radar_Datasets/IPIX/filename_labels.csv',
                               index_col='Label', sep=",")
    file_names = samples_info['file_name'].tolist()
    primaries = samples_info['primary'].tolist()
    secondary_st = samples_info['sec_start'].tolist()
    secondary_nd = samples_info['sec_end'].tolist()

    for txmode in ['hh', 'hv', 'vh', 'vv']:
        correct_cell_93 = 0
        correct_cell_98 = 0
        costTime93 = 0
        costTime98 = 0
        for findx, fileName in enumerate(file_names):
            filekey = fileName.split('_')[1]
            fileyear = fileName.split('_')[0][:4]

            primary_rangebin = primaries[findx] - 1
            if fileyear == '1993':
                Nt = 2 ** 17
                Nf = 512
            if fileyear == '1998':
                Nt = 60000
                Nf = 256
            ##'1635'

            nc_file = fileprefix + fileName
            nc = Dataset(nc_file, mode='r')
            # turn off the maskarray set.
            [nc.variables[k].set_auto_mask(False) for k in nc.variables]
            ranges = nc['range'][:]
            range_num = len(ranges)

            PRF = nc['PRF'][0]

            xfeatures = []
            ylabels   = []
            ts = time.clock()
            for rangebin in range(range_num):
                # # omit the secondary range cells
                if rangebin in range(secondary_st[findx] - 1, secondary_nd[findx]):
                    if rangebin != primary_rangebin:
                        continue

                I, Q, meanIQ, stdIQ, inbal = ipixmodel.ipixload(nc, txmode, rangebin, 'auto')
                am    = np.abs(I + 1j * Q)
                tdmat = get_tdoppl_mat(am, rangebin, Nf=Nf, Nt=Nt, PRF=1000)

                tdmat = (tdmat*255).astype('uint8')

                lbp_hist = compute_lbp_histogram(tdmat)
                xfeatures.append(lbp_hist)
                if rangebin == primary_rangebin:
                    ylabels.append(-1)  # Abnormal Mark for the primary cell with target
                else:
                    ylabels.append(1)
            est_target_bin = training_svm(xfeatures)
            te = time.clock()
            if ylabels[est_target_bin]== -1:
                str_res = 'C'
            else:
                str_res = 'W'
            print(fileyear, filekey, txmode, str_res, 'cost_time_perfile_permode: ', (te - ts))
            if fileyear =='1993':
                costTime93 = te - ts + costTime93
                if ylabels[est_target_bin]== -1:
                    correct_cell_93 += 1
            if fileyear =='1998':
                costTime98 = te - ts + costTime98
                if ylabels[est_target_bin]== -1:
                    correct_cell_98 += 1
        print('Total time    %s  for 1993 %f, for 1998 %f' %(txmode, costTime93, costTime98))
        print('correct_cells %s  for 1993 %d, for 1998 %d' %(txmode, correct_cell_93, correct_cell_98))

if __name__=='__main__':
    alg_vfsvm()