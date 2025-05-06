'''
Here， we use the source cdf IPIX radar data to compute the LBP features.
Select the best pre-processing parameters for the detector.
Decide:
1. optimal length of the segments (Nf).
2. any pre-processing of the Time-Doppler Spectra (TPS). Try AR on TPS.
3. Try the LBP_Var on the source image.
4. Illustrate the clutter only's histogram and the clutter with target.
   Checking the energy changing of the TPS.
5. Try a single one-class or two-class SVM detector on the whole data_sets.
'''

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

def get_tds_nonop(am):
    '''
    Get Time-Doppler spectra without overlap
    :param am:
    :return: Time-Doppler spectra
    '''
    N = len(am)
    #print('Length of the staring: %d'%N)
    Nf = 512 # length of the FT.

    timeStep = Nf
    height   = Nf
    HW       = np.hamming(Nf)

    width = int(N/Nf) # width of the TPS image.

    TD    = np.zeros((height, width))
    for m in range(width):
        i = m * timeStep
        # i = m * Nf
        x = HW * am[i:Nf + i]
        fftx = abs(np.fft.fftshift(np.fft.fft(x)))
        #y = np.reshape(fftx.ravel()[0:leny], (int(Nf / yavg), yavg))
        #TD[:, m] = np.mean(y, 1)
        TD[:, m] = fftx
    return TD

def get_tdoppl_mat(am, rangebin, Nf=512, Nt=2**17, PRF=1000):
    #N = len(am)
    timeStep = Nf
    # wdw      = Nf
    # timeStep = 128
    # M        = int((N - wdw) / timeStep)

    xpix = int(Nt / timeStep)  # image's cols is xpix, 512->xpix=256, 1024->xpix=128
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

def save_time_doppler_images(tdmat, dirPath, fname):
    sframe = Image.fromarray(tdmat)
    if not os.path.exists(dirPath):
        os.makedirs(dirPath)
    sframe.save('%s/%s' % (dirPath, fname), compress_level=0)

def distMat(xfeatures):

    histarrays = np.array(xfeatures)
    histarrays = histarrays.astype(np.float32)
    nsams   = len(xfeatures)#.shape[0]
    distMat = np.zeros((nsams, nsams))
    for i in range(nsams):
       for j in range(i+1, nsams):
           #CV_COMP_CORREL, CV_COMP_CHISQR, CV_COMP_INTERSECT, CV_COMP_BHATTACHARYYAHISTCMP_BHATTACHARYYA = 3
           #HISTCMP_CHISQR = 1 #HISTCMP_CHISQR_ALT = 4 #HISTCMP_CORREL = 0  #HISTCMP_HELLINGER = 3
           #HISTCMP_INTERSECT = 2 #HISTCMP_KL_DIV = 5
           distMat[i, j] = cv2.compareHist(histarrays[i, :], histarrays[j, :], cv2.HISTCMP_BHATTACHARYYA)
           distMat[j, i] = distMat[i, j]

    scoreVec = np.mean(distMat, 0)
    minId    = np.argmin(scoreVec)
    return minId, distMat






def make_tdmat_image():
    fileprefix = '/Users/yizhou/Radar_Datasets/IPIX/'
    Nf = 512#512#1024#512  # 64#128#256#512*2
    Nt = 2 ** 17  # 60000 for 1998file,  #131072 for 1993file

    #loading csv file for listing the wanted filenames, primary and secondary cells
    samples_info = pd.read_csv('/Users/yizhou/Radar_Datasets/IPIX/filename_labels.csv',
                               index_col='Label', sep=",")
    file_names = samples_info['file_name'].tolist()
    primaries = samples_info['primary'].tolist()
    secondary_st = samples_info['sec_start'].tolist()
    secondary_nd = samples_info['sec_end'].tolist()

    for findx, fileName in enumerate(file_names):
        filekey  = fileName.split('_')[1]
        fileyear = fileName.split('_')[0][:4]
        if fileyear=='1998':
             continue
        # if filekey!='191449':
        #      continue
        # if filekey != '202525':
        #     continue
        # if filekey not in ['141630','145028','213827','035737']:
        #     continue
        # if filekey not in ['191449', '162155','174259', '163113', '164055', '173950', '184537']:
        #     continue
        if fileyear == '1993':
            Nt = 2 ** 17
        if fileyear == '1998':
            Nt = 60000
        ##'1635'
        primary_rangebin = primaries[findx] - 1
        print(fileName + 'target range index is %d' % (primaries[findx] - 1))
        nc_file = fileprefix + fileName
        nc = Dataset(nc_file, mode='r')
        # turn off the maskarray set.
        [nc.variables[k].set_auto_mask(False) for k in nc.variables]
        ranges = nc['range'][:]
        range_num = len(ranges)
        PRF = nc['PRF'][0]

        for txmode in ['hv', 'vh', 'hh', 'vv']:
            # if txmode != 'hh':
            #     continue
            # figlbp, lbpax = plt.subplots()
            # figvar, varax = plt.subplots()
            # figjd, jdax = plt.subplots()
            xfeatures = []
            ylabels   = []

            for rangebin in range(range_num):

                if (secondary_st[findx] - 1) <= rangebin <= (secondary_nd[findx] - 1):
                    if rangebin != primary_rangebin:
                        continue

                I, Q, meanIQ, stdIQ, inbal = ipixmodel.ipixload(nc, txmode, rangebin, 'auto')
                am = np.abs(I + 1j * Q)
                #tdmat = get_tds_nonop(am)
                tdmat = get_tdoppl_mat(am, rangebin, Nf=Nf, Nt=Nt, PRF=1000)
                #plt.imshow(tdmat)
                #plt.show()
                #lbp_hist = get_lbp_feature(tdmat[:,64:64+64])
                lbp_hist = get_lbp_feature(tdmat)
                xfeatures.append((lbp_hist))

                if rangebin == primary_rangebin:
                    ylabels.append(-1)
                    # lbpax.plot(lbp_hist, 'r-.')
                    # varax.plot(lbp_var_hist, 'r-.')
                    # jdax.plot(lbp_joint_hist, 'r-.')
                else:
                    ylabels.append(1)
                    # lbpax.plot(lbp_hist)
                    # varax.plot(lbp_var_hist)
                    # jdax.plot(lbp_joint_hist)
                ##saving image.
                # tdmat = (tdmat*255).astype('uint8')
                # plt.imshow(tdmat, cmap='jet')
                # plt.draw()
                # dirPath = '%s%s_%s/%s' % (fileprefix,fileyear,filekey, txmode)
                # imgName = '%02d.png' % rangebin
                # save_time_doppler_images(tdmat, dirPath, imgName)

            clf = OneClassSVM(nu=0.4, kernel='rbf', gamma='auto')
            xfeatures = np.array(xfeatures)
            ylabels   = np.array(ylabels)
            clf.fit(xfeatures)
            y_score = clf.score_samples(xfeatures)
            least_votes = np.argsort(y_score)
            #yhat = clf.predict(xfeatures[ylabels==-1])
            yhat = clf.predict(xfeatures)

            ## Matrix comparison:
            #minId, distM  = distMat(xfeatures)
            print('')
            #if ylabels[least_votes[0]] == -1:
            #if yhat[0] == -1:
            if yhat[ylabels==-1] == -1:
            #if ylabels[minId] == -1:
                print(fileyear, filekey, txmode, '(gt:%2d), %s' % (primary_rangebin, 'Correct'))
            else:
                print(fileyear, filekey, txmode, '(gt:%2d), %s' % (primary_rangebin, 'Wrong'))


            # figPath = '/Users/yizhou/code/inesa_it_radar_singal_process/result_images/lbp/'
            # figlbp.savefig(os.path.join(figPath, '%s_%s_lbp.png'%(filekey, txmode)),bbox_inches='tight', pad_inches=2, dpi=200)
            # figvar.savefig(os.path.join(figPath, '%s_%s_var.png'%(filekey, txmode)), bbox_inches='tight', pad_inches=2, dpi=200)
            # figjd.savefig(os.path.join(figPath, '%s_%s_jd.png'%(filekey, txmode)), bbox_inches='tight', pad_inches=2, dpi=200)
            # plt.close(figlbp)
            # plt.close(figvar)
            # plt.close(figjd)

def get_lbp_feature(frame, Plbp=8, Rlbp=1, Pvar=8, Rvar=1):

    # lbp_var = local_binary_pattern(frame, P=Pvar, R=Rvar, method='var')
    # lbp_var[np.isnan(lbp_var)] = 0
    # lbp_var_hist, lpb_var_bins = np.histogram(lbp_var.ravel(), bins=10, density=False)
    # lbp_var_hist = lbp_var_hist / lbp_var.size

    lbp_frame = local_binary_pattern(frame, P=Plbp, R=Rlbp, method='uniform')
    # bin_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # weights   = np.array([2, 1, 1, 3, 3, 1, 1, 1, 2])
    # weights = weights/np.sum(weights)
    lbp_hist, lpb_bins = np.histogram(lbp_frame.ravel(), bins=int(np.max(lbp_frame)), density=False)
    lbp_hist = lbp_hist / lbp_frame.size
    # lbp_hist_wt = lbp_hist * weights
    # lbp_hist = lbp_hist_wt/np.sum(lbp_hist_wt)
    #
    # lbp_joint = np.outer(lbp_var_hist, lbp_hist)
    # lbp_joint_hist = lbp_joint.ravel() / np.sum(lbp_joint)

    #return lbp_hist, lbp_var_hist, lbp_joint_hist
    return lbp_hist

if __name__=='__main__':
    make_tdmat_image()
    exit(0)
