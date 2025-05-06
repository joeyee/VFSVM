'''
Time Doppler plot
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
from sklearn.svm     import LinearSVC
from sklearn.svm     import OneClassSVM
from sklearn.metrics import roc_curve, auc
import cv2
import os
import utilities_200611         as uti            # personal tools
import time


def make_tdmat_image():
    fileprefix = '/Users/yizhou/Radar_Datasets/IPIX/'
    Nf = 512#512#1024#512  # 64#128#256#512*2
    Nt = 2 ** 17  # 60000 for 1998file,  #131072 for 1993file

    #loading csv file for listing the wanted filenames, primary and secondary cells
    samples_info = pd.read_csv('/Users/yizhou/Radar_Datasets/IPIX/filename_labels.csv',
                               index_col='Label', sep=",")
    file_names   = samples_info['file_name'].tolist()
    primaries    = samples_info['primary'].tolist()
    secondary_st = samples_info['sec_start'].tolist()
    secondary_nd = samples_info['sec_end'].tolist()

    for findx, fileName in enumerate(file_names):
        # if findx >= 13:  # current not handle anstep files
        #     continue
        filekey = fileName.split('_')[1]
        fileyear = fileName.split('_')[0][:4]
        # if fileyear=='1998':
        #     continue
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
        print(fileName + 'target range index is %d' % (primaries[findx] - 1))
        nc_file = fileprefix + fileName
        nc = Dataset(nc_file, mode='r')
        # turn off the maskarray set.
        [nc.variables[k].set_auto_mask(False) for k in nc.variables]
        ranges = nc['range'][:]
        range_num = len(ranges)

        PRF = nc['PRF'][0]
        costTime = 0
        ntds = 0
        for txmode in ['hv', 'vh', 'hh', 'vv']:
            for rangebin in range(range_num):
                ts = time.clock()
                I, Q, meanIQ, stdIQ, inbal = ipixmodel.ipixload(nc, txmode, rangebin, 'auto')
                am    = np.abs(I + 1j * Q)
                tdmat = get_tdoppl_mat(am, rangebin, Nf=Nf, Nt=Nt, PRF=1000)
                tdmat = (tdmat*255).astype('uint8')
                costTime = time.clock() - ts + costTime
                ntds += 1
                plt.imshow(tdmat, cmap='jet')
                plt.draw()
                dirPath = '%s%s_%s/%s' % (fileprefix,fileyear,filekey, txmode)
                sframe = Image.fromarray(tdmat)
                if not os.path.exists(dirPath):
                    os.makedirs(dirPath)
                sframe.save('%s/%02d.png' % (dirPath, rangebin), compress_level=0)
        print('Total time for making tds %f, average time %f'%(costTime, costTime/ntds))

def preprocessing(I, Q, kval=1., nref=8, mguide=4):
    '''
    preprocessing data, return the amplitude echos.
    :param nc:
    :param kval:
    :param nref:
    :param mguide:
    :return:
    '''

    ## ndimages.conv_mean_filter 128 length sliding window
    # weights = np.ones(128)/128
    # aI = ndimage.convolve1d(I, weights)
    # aQ = ndimage.convolve1d(I, weights)
    am = np.abs(I + 1j * Q)
    #reshape am to the 2d array for cfar model
    am = np.reshape(am, (len(am), 1))
    cfar = cfar_model.CFAR(kval=1.0, nref=8, mguide=4)
    ave_cfar = cfar.cfar_ave(am, cfar.nref, cfar.mguide)
    mask_cfar, am_cfar = cfar.cfar_thresh(am, ave_cfar, cfar.kval)

    # plt.plot(am[0:500],'r')
    # plt.plot(am_cfar[0:500],'g')
    # plt.show()
    return am, am_cfar

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

def tdoppl(nc, txpol, rangebin, Nf=512, Nt=2**17):
    '''

    :param nc:
    :param txpol:
    :param rangebin:
    :param Nf: the window length for doing fft.  512  or 1024
    :param Nt: the used pulses of the returns.   6000 or 2^17
    :return:
    '''

    I, Q, meanIQ, stdIQ, inbal = ipixmodel.ipixload(nc, txpol, rangebin, 'auto')
    N = len(I)
    ## ndimages.conv_mean_filter 128 length sliding window
    # weights = np.ones(128)/128
    # aI = ndimage.convolve1d(I, weights)
    # aQ = ndimage.convolve1d(I, weights)

    xpix = int(N/Nf) #image's cols is xpix, 512->xpix=256, 1024->xpix=128
    ypix = 64#/2        #image's row is ypix = 64
    # adjust fft window and averaging to match image size
    #wdw = int(max(128, 2 ** np.ceil(np.log(4 * N / xpix) / np.log(2))))
    yavg = max(1, int(Nf / ypix)) # the averaging len on the y axis.
    # timeStep = int(wdw / 4)
    # M = int((N - wdw) / timeStep)

    # % short time fourier transforms
    hw = np.hamming(Nf)
    TD = np.zeros((ypix, xpix))
    y  = np.zeros((int(Nf / yavg), yavg))
    leny = y.size
    for m in range(xpix):
        i = m * Nf
        x = hw * (I[i:Nf + i] + 1j * Q[i:Nf + i])
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
    # #logTD = TD
    logTD = np.log(TD)
    logTD = np.flipud(logTD)
    logTD = (logTD - np.min(logTD)) * 63 / (np.max(logTD) - np.min(logTD))
    #logTD = (TD - np.min(TD))/(np.max(TD)-np.min(TD))

    #% time and normalized frequency
    PRF=nc['PRF'][0]             # Pulse Repetition Frequency [Hz]
    observ_time = Nt / PRF
    #plot_td(logTD, PRF, observ_time, rangebin)
    return logTD

def tdoppl_200702(nc,txpol,rangebin,xpix=250,ypix=80):
    #set the length of the axis-x
    #set the length of the axis-y

    # load I and Q data
    #[I,Q]=ipixcdf(nc,txpol,rangebin);
    I, Q, meanIQ, stdIQ, inbal = ipixmodel.ipixload(nc, txpol, rangebin, 'auto')
    N = len(I)

    #adjust fft window and averaging to match image size
    wdw =int(max(128,2**np.ceil(np.log(4*N/xpix)/np.log(2))))
    yavg=int(max(1,np.ceil(wdw/ypix)))
    timeStep=int(wdw/4)
    M=int((N-wdw)/timeStep)

    # % wdw=max(128,2^ceil(log(N/xpix)/log(2)));
    # % yavg=max(1,ceil(wdw/ypix));
    # % timeStep=wdw;
    # % M=floor((N-wdw)/timeStep);

    #% short time fourier transforms
    hw=np.hamming(wdw)
    TD=np.zeros((int(wdw/yavg),M))
    y=np.zeros((int(wdw/yavg), yavg))
    leny=y.size
    for m in range(M):
      i=m*timeStep
      x=I[i:wdw+i]+1j*Q[i:wdw+i]
      x=hw*(I[i:wdw+i]+1j*Q[i:wdw+i])
      fftx=abs(np.fft.fftshift(np.fft.fft(x)))
      y   =np.reshape(fftx.ravel()[0:leny], (int(wdw/yavg), yavg))
      TD[:,m]=np.mean(y,1)


    #% time and normalized frequency
    PRF=nc['PRF'][0]             # Pulse Repetition Frequency [Hz]
    time=np.array([0, M])/PRF
    freq=np.array([-0.5, 0.5])*PRF
    #% convert to doppler velocity
    #%doppl=freq*3e8/(2*nc{'RF_frequency'}(1)*1e9);
    doppl=freq*3e8/(2*nc['RF_frequency'][0]*1e9)  #%fd*c/2f_c

    #% enhance log-plot with noise floor
    mn=np.mean(TD,1)
    indx=np.argsort(mn)
    mn = mn[indx]
    noise=TD[indx[0:2],:]
    noiseFloor=np.median(noise[:])
    TD[TD<noiseFloor]=noiseFloor
    logTD=np.log(TD)
    logTD=np.flipud(logTD)
    logTD=(logTD-np.min(logTD))*63/(np.max(logTD)-np.min(logTD))




    # matlogTDdict = io.loadmat('/Users/yizhou/code/Matlab/IPIX/tdoppl.mat')
    # matlogTD=matlogTDdict['logTD']
    #mdif = np.sum(logTD-matlogTD)
    #% display image
    # time=(wdw/2+(0:M-1)*timeStep)/PRF
    # freq=((0:wdw/yavg-1)/(wdw/yavg)-0.5)*PRF;
    # doppl=freq*3e8/(2*get_var(nc,'RF_frequency')*1e9);
    # image(time,doppl,logTD); set(gca,'ydir','normal');
    # colormap jet(64)

    fig, ax = plt.subplots()
    im = ax.imshow(logTD, cmap='jet', interpolation='none')
    xticks = np.linspace(0, M / PRF, 5)
    xticklabels = ["{:1.2f}".format(i) for i in xticks]
    x_positions = np.linspace(0, logTD.shape[1] - 1, 5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xticklabels)

    yticks = np.linspace(-0.5 * PRF, 0.5 * PRF, 5)
    yticklabels = ["{:1.0f}".format(i) for i in yticks]
    y_positions = np.linspace(0, logTD.shape[0] - 1, 5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('Dopller freq.(Hz)')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('logTD intensity', rotation=-90, va="bottom")
    plt.show()
    plt.waitforbuttonpress()

def plot_td(logTD, PRF, observ_time,rangebin):
    fig, ax = plt.subplots()
    im = ax.imshow(logTD, cmap='jet', interpolation='none')
    xticks = np.linspace(0, observ_time, 5)
    xticklabels = ["{:1.2f}".format(i) for i in xticks]
    x_positions = np.linspace(0, logTD.shape[1] - 1, 5)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(xticklabels)
    yticks = np.linspace(-0.5 * PRF, 0.5 * PRF, 5)
    yticklabels = ["{:1.0f}".format(i) for i in yticks]
    y_positions = np.linspace(0, logTD.shape[0] - 1, 5)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel('Time(s)')
    ax.set_ylabel('Dopller freq.(Hz)')
    ax.set_title('%d'%rangebin)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('logTD intensity', rotation=-90, va="bottom")
    plt.draw()
    # key_press = False
    # while not key_press:
    #     key_press = plt.waitforbuttonpress()

def make_feature_label_pair(fileid, Nf, mat, bpostive=False):
    '''
    Compute lbp feature from the input matrix
    :param findx:
    :param Nf:
    :param mat:
    :param bpostive:
    :return:
    '''
    type = 0
    if bpostive:
        type = 1
    lbp = local_binary_pattern(mat, P=24, R=3, method='uniform')
    lbp[np.isnan(lbp)] = 0
    bhist, bins = np.histogram(lbp.ravel(), bins=8, range=(0, 8), density=True)
    # feature infor keeps the information of the samples
    feature_info = np.array([fileid, Nf, type])
    feature = np.concatenate([feature_info, bhist])
    #features.append(feature)
    return feature


def make_feature_label_pairs(fileid, Nf, matlist, positive_ids):
    '''
    Making the feature and label pairs for the TF matlist of all the rangebins, len 14 in Dartmouth.
    primary or secondary range id are marked for positive_ids
    :param matlist:
    :param positive_ids:
    :return:
    '''
    features = []
    for i,mat in enumerate(matlist):
        lbp = local_binary_pattern(mat, P=24, R=3, method='uniform')
        lbp[np.isnan(lbp)] = 0
        bhist, bins = np.histogram(lbp.ravel(), bins=8, range=(0, 8), density=True)
        if i not in positive_ids:
            type = 0
        else:
            type = 1
        # feature infor keeps the information of the samples
        feature_info = np.array([fileid, Nf, type])

        feature =  np.concatenate([feature_info, bhist])
        features.append(feature)
    return features
def svm_1class(features):
    '''
    define svm in one class to detect abnormal
    :param features:
    :return:
    '''
    if type(features) is list:
        #convert featuers to list.
        features = np.array(features)
    xtrain = features[:, 3::]
    y_gt   = (features[:,2]==1)*1+(features[:,2]==0)*(-1)
    clf = OneClassSVM(kernel='rbf',gamma='auto').fit(xtrain)
    y_test   = clf.predict(xtrain)
    y_score  = clf.score_samples(xtrain)
    fpr, tpr, thresh = roc_curve(y_gt[:], y_score[:])

    # for i in range(xtrain.shape[0]):
    #     plt.plot(xtrain[i,:])
    #plt.show()
    plt.semilogx(fpr, tpr, lw=2)
    # plt.semilogx(fpr, tpr, lw=2, label=str_label)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, label='guess', linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    #plt.legend(loc="lower right")
    plt.draw()
    plt.waitforbuttonpress()

def svm_classify(features):

    if type(features) is list:
        #convert featuers to list.
        features = np.array(features)
    #define positive features and negative ones by the type
    negative_features = features[features[:,2]==0]
    positive_features = features[features[:,2]==1]
    neg_num = negative_features.shape[0]
    pos_num = positive_features.shape[0]
    #choosing the positive and negative samples for training.
    train_pos_idx = np.random.randint(0, pos_num, int(pos_num/2))
    train_neg_idx = np.random.randint(0, neg_num, int(neg_num/4))
    # train_features = np.concatenate([positive_features[train_pos_idx,:],
    #                                  negative_features[train_neg_idx,:]])
    train_pos_num = 30
    train_neg_num = 60
    train_features = np.concatenate([positive_features[0:train_pos_num,:],
                                     negative_features[0:train_neg_num,:]])
    # test_pos_idx = []
    # test_neg_idx = []
    # for i in range(pos_num):
    #     if i not in train_pos_idx:
    #         test_pos_idx.append(i)
    # for j in range(neg_num):
    #     if j not in train_neg_idx:
    #         test_neg_idx.append(j)
    #
    # test_features  = np.concatenate([positive_features[test_pos_idx,:],
    #                                  negative_features[test_neg_idx,:]])
    # test_features = np.concatenate([positive_features[train_pos_num::,:],
    #                                  negative_features[train_neg_num::,:]])
    test_features = np.concatenate([positive_features,negative_features])
    x_train = train_features[:, 3::]
    y_train = train_features[:,2]
    x_test  = test_features[:, 3::]
    y_test_gt  = test_features[:,2]

    clf = LinearSVC(C=100, loss="hinge", random_state=42, max_iter=20000)
    clf.fit(x_train, y_train)
    y_score = clf.decision_function(x_test)
    fpr, tpr, thresh = roc_curve(y_test_gt[:], y_score[:])

    plt.semilogx(fpr, tpr, lw=2)
    # plt.semilogx(fpr, tpr, lw=2, label=str_label)
    # plt.plot([0, 1], [0, 1], color='navy', lw=2, label='guess', linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic curve')
    plt.legend(loc="lower right")
    plt.draw()
    plt.waitforbuttonpress()
    plt.show()

def get_sub_image_features():
    '''
    divide the 2^17 size image into small subimages.
    :return:
    '''
    fileprefix = '/Users/yizhou/Radar_Datasets/IPIX/'
    Nf = 512  # 128#256#512*2
    xpix = int(2**17/Nf)
    sub_num  = 4
    samples_info = pd.read_csv('/Users/yizhou/Radar_Datasets/IPIX/filename_labels.csv',
                               index_col='Label', sep=",")
    file_names = samples_info['file_name'].tolist()
    primaries = samples_info['primary'].tolist()
    secondary_st = samples_info['sec_start'].tolist()
    secondary_nd = samples_info['sec_end'].tolist()
    features = []

    for findx, fileName in enumerate(file_names):
        if findx >= 10:  # current not handle anstep files
            continue
        print(fileName)
        nc_file = fileprefix + fileName
        nc = Dataset(nc_file, mode='r')
        # turn off the maskarray set.
        [nc.variables[k].set_auto_mask(False) for k in nc.variables]
        # I,Q, meanIQ, stdIQ, inbal = ipixload(fh, pol='hh', rangebin=0, mode='auto')
        ranges = nc['range'][:]
        range_num = len(ranges)
        tfmat_list = []

        for rangebin in range(range_num):
            tfmat = tdoppl(nc, 'hh', rangebin, Nf=Nf)
            tfmat_list.append(tfmat)
            #if rangebin == (primaries[findx]-1):
            if (secondary_st[findx]-1)<=rangebin<=(secondary_nd[findx]-1):
                bpos = True
            else:
                bpos = False
            for si in range(sub_num):
                step = int(tfmat.shape[1]/sub_num)
                submat = tfmat[:, (si*step):((si+1)*step)]
                feature = make_feature_label_pair(findx, Nf, submat, bpos)
                features.append(feature)
    svm_classify(features)
    plt.show()

from skimage import transform
def segmentation(frame, lbp_contrast_select = False, kval=1, least_wh = (3,3), min_area=32, max_area=200, nref=25, mguide=18):
    # kval decide the false alarm rate of cfar.
    # least_wh is the least window size of the segmentation
    # using contrast intensity or not. Should not be used, this is not adaptive threshold
    cfar_cs    = cfar_model.CFAR(kval=kval, nref=nref, mguide=mguide)
    bin_image  = cfar_cs.cfar_seg(frame)
    (contours, _) = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    #(contours, _) = cv2.findContours(bin_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    blob_bb_list  = []  #blob bounding box list

    if lbp_contrast_select: # Eliminate blobs with low contrast intensities
        inv_var = local_binary_pattern(frame, P=24, R=3, method='var')
        inv_var[np.isnan(inv_var)] = 0
        int_img = transform.integral_image(inv_var)

    for id, contour in enumerate(contours):
        x, y, w, h= cv2.boundingRect(contour)  # only using the bounding box information
        bb_rect = [x,y,w,h]
        # omit small rectangles.
        if (w*h <= min_area) or (w*h>=max_area)or (w<=least_wh[0]) or (h<=least_wh[1]):
            continue
        if lbp_contrast_select:
            iiv = transform.integrate(int_img, (y, x), (h + y - 1, x + w - 1))
            # omit the sparse density of variance image.
            if (iiv[0] / (w * h)) < 500:
                continue
        blob_bb_list.append(bb_rect)
    return blob_bb_list, bin_image

def test_cfar_tf():
    fileprefix = '/Users/yizhou/Radar_Datasets/IPIX/'
    Nf = 512  # 64#128#256#512*2
    Nt = 2**17 # 60000 for 1998file,  #131072 for 1993file
    samples_info = pd.read_csv('/Users/yizhou/Radar_Datasets/IPIX/filename_labels.csv',
                               index_col='Label', sep=",")
    file_names   = samples_info['file_name'].tolist()
    primaries    = samples_info['primary'].tolist()
    secondary_st = samples_info['sec_start'].tolist()
    secondary_nd = samples_info['sec_end'].tolist()

    features = []
    figall, axall = plt.subplots()
   #plt.close(figall)
    Ncorrect_minscore = 0
    Ncorrect_labels   = 0
    Nfiles            = len(file_names)
    bdraw = False
    for findx, fileName in enumerate(file_names):
        # if findx >= 13:  # current not handle anstep files
        #     continue
        filekey = fileName.split('_')[1]
        fileyear= fileName.split('_')[0][:4]
        # if filekey not in ['191449', '162155','174259', '163113', '164055', '173950', '184537']:
        #     continue
        if bdraw:
            figh, axh = plt.subplots()
            figtd, axtd= plt.subplots(2,1)
            axtd[0].set_title(fileName)
            axh.set_title(fileName)


        if fileyear == '1993':
            Nt = 2**17
        if fileyear == '1998':
            Nt = 60000
        ##'1635'
        print(fileName + 'target range index is %d'%(primaries[findx]-1))
        nc_file = fileprefix + fileName
        nc = Dataset(nc_file, mode='r')
        # turn off the maskarray set.
        [nc.variables[k].set_auto_mask(False) for k in nc.variables]
        # I,Q, meanIQ, stdIQ, inbal = ipixload(fh, pol='hh', rangebin=0, mode='auto')
        ranges = nc['range'][:]
        range_num = len(ranges)

        PRF = nc['PRF'][0]
        tfmat_list = []

        xfeatures = []
        ylabels   = []
        for rangebin in range(range_num):
            # if rangebin !=8:
            #     continue
            #omit the secondary range cells
            if (secondary_st[findx] - 1) <= rangebin <= (secondary_nd[findx] - 1):
                if rangebin != primaries[findx]-1:
                    continue

            I, Q, meanIQ, stdIQ, inbal = ipixmodel.ipixload(nc, 'vv', rangebin, 'auto')

            ## ndimages.conv_mean_filter 128 length sliding window
            # weights = np.ones(32)/32
            # aI = ndimage.convolve1d(I, weights)
            # aQ = ndimage.convolve1d(Q, weights)
            # am = np.abs(aI + 1j * aQ)
            #am, am_ave = preprocessing(I, Q, kval=1.0, nref=8, mguide=4)
            am = np.abs(I + 1j * Q)
            tdmat  = get_tdoppl_mat(am, rangebin, Nf=Nf, Nt=Nt, PRF=1000)
            #td_lbp = local_binary_pattern(tdmat, P=24, R=3, method='var')
            # blob_bb_list, bin_image = segmentation(tdmat, lbp_contrast_select = False,
            #             kval=1., least_wh = (2,5), min_area=10, max_area=400, nref=16, mguide=8)

            #canvas = cv2.applyColorMap((bin_image*255).astype(np.uint8), cv2.COLORMAP_JET)
            # #canvas = cv2.cvtColor((tdmat).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            # lbp_hist_mat = []
            # #Get LBP features in each cfar blobs.
            # for bid, blob in enumerate(blob_bb_list):
            #     x, y, w, h = blob[:4]
            #     ROI = tdmat[y:y + h, x:x + w]
            #     lbp = local_binary_pattern(ROI, P=24, R=3, method='uniform')
            #     lbp[np.isnan(lbp)] = 0
            #     # Calculate the histogram
            #     bhist, bins = np.histogram(lbp.ravel(), bins=8,  density=True)
            #     lbp_hist_mat.append(bhist)
            #     #uti.draw_rect(canvas, blob, (0, 255, 255), 1)  # draw target contained blob in yellow
            #
            # center_row = int(tdmat.shape[0]/2)
            # center_blob_idx = []
            # center_label = -1
            # kmeans = KMeans(n_clusters=2, random_state=0).fit(lbp_hist_mat)
            # labels = kmeans.labels_
            # for index, label in enumerate(labels):
            #     blob = blob_bb_list[index]
            #     x, y, w, h = blob[:4]
            #     if y<=center_row<=(y+h): #blob contains center row of tdmat.
            #         center_blob_idx.append(index)
            #     if label==1:
            #         uti.draw_rect(canvas, blob, (0, 0, 255), 1)
            #     if label==2:
            #         uti.draw_rect(canvas, blob, (255, 0, 0), 1)
            #     if label==0:
            #         uti.draw_rect(canvas, blob, (0, 255, 0), 1)
            # center_labels = labels[center_blob_idx]
            # print('range cell %d center_labels has %d label 1, and %d label 0'
            #       %(rangebin, np.sum(center_labels==0), np.sum(center_labels==1)))
            # if np.sum(center_labels==0) > np.sum(center_labels==1):
            #     center_label = 0
            # else:
            #     center_label = 1
            # #cv2.imwrite(fileprefix+'/results/'+fileName+'_rangeid_%d.png'%rangebin,
            # #            canvas, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            #
            #
            # #Get the blob list which is not belonging to the center_label
            # #Extend the side lbp mat to a long vector for computing the histogram.
            # lbp_tdmat = local_binary_pattern(tdmat, P=24, R=3, method='uniform')
            # side_lbp_vector = np.array([])
            # for index, label in enumerate(labels):
            #     blob = blob_bb_list[index]
            #     x, y, w, h = blob[:4]
            #     #if label!=center_label:
            #     if index not in center_blob_idx:
            #         side_blob_lbp = lbp_tdmat[y:y + h, x:x + w]
            #         xv  = side_blob_lbp.ravel()
            #         xv[np.isnan(xv)] = 0
            #         side_lbp_vector = np.concatenate([side_lbp_vector, xv])
            #side_lbp_hist, lpb_bins = np.histogram(side_lbp_vector, bins=10, range = (0,26), density=True)
            # side_tdmat = (tdmat*(bin_image==0))
            # center_row = int(tdmat.shape[0] / 2)
            # side_tdmat[center_row-5:center_row+5, : ] = 0
            #side_lbp   = local_binary_pattern(tdmat, P=24, R=3, method='var')  or 'default'
            #side_lbp = local_binary_pattern(tdmat, P=8, R=1, method='default')
            side_lbp_var = local_binary_pattern(tdmat, P=8, R=1, method='var')

            side_lbp_var[np.isnan(side_lbp_var)] = 0
            side_lbp_var_hist, lpb_var_bins = np.histogram(side_lbp_var.ravel(), bins=10, range=(0, 0.045), density=False)
            side_lbp_var_hist = side_lbp_var_hist/np.sum(side_lbp_var<=0.045)
            #lbp_hist = side_lbp_var_hist
            # side_lbp_var_hist, lpb_var_bins = np.histogram(side_lbp_var.ravel(), bins=10, range=(np.min(side_lbp_var+np.spacing(1)),
            #                                                                                      np.max(side_lbp_var)), density=False)
            # side_lbp_var_hist = side_lbp_var_hist/np.sum(side_lbp_var>0)

            side_lbp = local_binary_pattern(tdmat, P=8, R=3, method='default')
            side_lbp[np.isnan(side_lbp)] = 0
            # # #side_lbp_hist, lpb_bins = np.histogram(side_lbp.ravel(), bins=10, density=True)
            side_lbp_hist, lpb_bins = np.histogram(side_lbp.ravel(), bins=10, density=False)
            side_lbp_hist = side_lbp_hist /side_lbp.size

            lbp_hist = np.concatenate([side_lbp_hist, side_lbp_var_hist])
            # image_hist, lpb_bins  = np.histogram(tdmat.ravel(), bins=10, density=False)
            # image_hist = image_hist/tdmat.size
            # lbp_hist = np.concatenate([image_hist, side_lbp_var_hist])
            #xfeatures.append(side_lbp_hist)
            xfeatures.append(lbp_hist)
            if rangebin == primaries[findx]-1:
                ylabels.append(-1) # Abnormal Mark for the primary cell with target
            else:
                ylabels.append(1)
            if bdraw:
                axtd[0].imshow(tdmat)
                axtd[1].imshow(side_lbp_var)
                figtd.savefig(fileprefix + '/results/' + fileName + '_cfar_rangebin_%d.png' % rangebin)

            hist_label = 'range %d'%rangebin
            if bdraw:
                if rangebin == primaries[findx] - 1: # primary range cell with target
                    # axh.plot(lpb_bins[:-1], side_lbp_hist, 'r.-',label=hist_label)
                    # axall.plot(lpb_bins[:-1], side_lbp_hist, '-',label=filekey)
                    axh.plot(lbp_hist, 'r.-',label=hist_label)
                    axall.plot(lbp_hist, '-',label=filekey)
                    #plt.waitforbuttonpress()
                else:
                    #axh.plot(lpb_bins[:-1], side_lbp_hist, label=hist_label)
                    axh.plot(lbp_hist, label=hist_label)
                axh.legend(loc="upper right")
            #axtd.imshow(canvas)
        if bdraw:
            figh.savefig(fileprefix+'/results/'+fileName+'.png')
        clf = OneClassSVM(nu=0.1, kernel="poly",degree = 3, gamma='auto')
        #clf = OneClassSVM(nu=0.1,kernel='rbf', gamma='auto')
        clf.fit(xfeatures)
        y_score = clf.score_samples(xfeatures)
        y_test  = clf.predict(xfeatures)
        predict_cells = np.where(y_test==-1)[0]
        n_errors = np.sum(y_test != ylabels)
        if n_errors==0:
            print('correct classification')
            Ncorrect_labels +=1
        else:
            print('wrong classification times: %d'%n_errors)
        strcid = ''
        for cid in predict_cells:
            strcid = strcid + str(cid) + ' '
        print('classified target cells are %s'% strcid )
        strscore=''
        # for score in y_score:
        #     sc = '%.4f '%score
        #     strscore += str(sc)
        print('one class min score %d'% (np.argmin(y_score)))
        if(ylabels[np.argmin(y_score)] == -1):
            print('correct classification by minimize the score!')
            Ncorrect_minscore +=1
        print('')
        #print((np.argmin(y_score), np.argmax(y_score)))
        #print(y_score.T)
        if bdraw:
            plt.close(figh)
            plt.close(figtd)
    if bdraw:
        axall.legend(loc='best')
        figall.savefig(fileprefix+'/results/'+'all_target_cell_hist'+'.png')
    print('correct labels %d in %d files' %(Ncorrect_labels, Nfiles))
    print('correct min score %d in %d files, detection rate is %.2f'
          %(Ncorrect_minscore, Nfiles, Ncorrect_minscore*1./Nfiles))
    plt.show()
if __name__=='__main__':
    make_tdmat_image()
    exit(0)
    test_cfar_tf()
    #get_sub_image_features()
    fileprefix = '/Users/yizhou/Radar_Datasets/IPIX/'
    Nf  = 128#64#128#256#512*2
    samples_info = pd.read_csv('/Users/yizhou/Radar_Datasets/IPIX/filename_labels.csv',
                               index_col='Label',sep=",")
    file_names = samples_info['file_name'].tolist()
    primaries  = samples_info['primary'].tolist()
    secondary_st= samples_info['sec_start'].tolist()
    secondary_nd = samples_info['sec_end'].tolist()

    features = []
    for findx, fileName in enumerate(file_names):
        if findx>=13: # current not handle anstep files
            continue
        print(fileName)
        nc_file = fileprefix + fileName
        nc = Dataset(nc_file, mode='r')
        #turn off the maskarray set.
        [nc.variables[k].set_auto_mask(False) for k in nc.variables]
        #I,Q, meanIQ, stdIQ, inbal = ipixload(fh, pol='hh', rangebin=0, mode='auto')
        ranges = nc['range'][:]
        range_num = len(ranges)
        tfmat_list = []

        for rangebin in range(range_num):
            tfmat = tdoppl(nc, 'hh', rangebin, Nf=Nf)
            tfmat_list.append(tfmat)
            if rangebin in range(int(secondary_st[findx] - 1), int(secondary_nd[findx] + 1)):
                feature = make_feature_label_pair(findx, Nf, tfmat, bpostive=True)
            else:
                feature = make_feature_label_pair(findx,Nf,tfmat,bpostive=False)
            features.append(feature)
        nc.close()
        # tfms = np.array(tfmat_list)
        # u_tf = np.mean(tfms,0)
        # sig_tf=np.std(tfms,0)
        # tfms_norm_list = []
        # for i,tfmat in enumerate(tfmat_list):
        #     #tfmat = (tfmat - u_tf)#/(sig_tf+np.spacing(1))
        #     tfms_norm_list.append(tfmat)
        #     # lbp = local_binary_pattern(tfmat, P=24, R=3, method='uniform')
        #     # lbp[np.isnan(lbp)] = 0
        #     # bhist, bins = np.histogram(lbp.ravel(), bins=8, range=(0, 8), density=True)
        #     # #
        #     # # # ax1.imshow(tfmat)
        #     # # # ax2.imshow(tfmat-u_tf)
        #     # # #if i==8:
        #     # plt.plot(bhist)
        #     # plt.title('range bin %d'%i)
        #     # if i == primaries[findx]:
        #     #     plt.title('object range bin %d'%i)
        #     #     plt.plot(bhist, 'r-.')
        #     #     #plt.waitforbuttonpress()
        #     # print('Range bin %d, diff sum %.2f' % (i,np.sum(tfmat**2)))
        #     # key_press = False
        #     # while not key_press:
        #     #     key_press = plt.waitforbuttonpress()
        #
        # primary_id = primaries[findx]
        # positive_bins = [primary_id-1]
        # for id in range(int(secondary_st[findx]-1), int(secondary_nd[findx]+1)):
        #        assert(secondary_nd[findx]>primary_id>secondary_st[findx])
        #        positive_bins.append(id-1) #in python, the index is starts from 0, not 1 in matlab.
        # onefile_features = make_feature_label_pairs(findx, Nf, tfms_norm_list, positive_bins)
        # features.extend(onefile_features)
        # #svm_1class(features)
        #
    svm_classify(features)
    plt.figure()
    plt.show()