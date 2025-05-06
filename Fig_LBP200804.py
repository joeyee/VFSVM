'''
This file contains the figure making code for the TAES2020 LBP section.
'''
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
import numpy as np
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
import glob
from statistics import mode
import Load_IPIX_200701         as ipixmodel
import cfar_segmentation_200527 as cfar_model     # cfar
import MCF_20200603             as mcf_model
import utilities_200611         as uti            # personal tools

LBP_FIG_PATH = '/Users/yizhou/code/inesa_it_radar_singal_process/result_images/lbp/'
def fig_target_TF(frame, str_name):
    frame_jet = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    frame_jet = cv2.cvtColor(frame_jet, cv2.COLOR_BGR2RGB)
    sframe = Image.fromarray(frame_jet)
    # cv2.imshow('t',frame_jet)
    # cv2.waitKey()
    if not os.path.exists(LBP_FIG_PATH):
         os.makedirs(LBP_FIG_PATH)

    sframe.save('%s/%s.png' % (LBP_FIG_PATH, str_name), compress_level=0, dpi=(600,600))

    tfig, tax = plt.subplots(dpi=200)
    h,w = frame.shape[:2]
    tax.imshow(frame, cmap='jet')#, extent=[0, 131, -500, 500])
    # change imshow labels refer to :
    # https://moonbooks.org/Articles/How-to-change-imshow-axis-values-labels-in-matplotlib-/
    y_ticklabel_list = [500, 0, -500]
    x_ticklabel_list = np.int0(np.arange(0,131,16.375)) # 8 labels
    tax.set_xticks(np.int0(np.arange(0, w, w/8))) # 8 xticks
    tax.set_yticks([0, int(h/2), h-1])
    tax.set_xticklabels(x_ticklabel_list)
    tax.set_yticklabels(y_ticklabel_list)
    tax.set_xlabel('Time(s)')
    tax.set_ylabel('Frequency(Hz)')
    tfig.savefig(os.path.join(LBP_FIG_PATH, '%s_tds_units.png' % str_name), bbox_inches='tight',
                 pad_inches=0, dpi=300)


def fig_lbp_hist(hist_list, ylabels, str_name):
    '''
    Plot all the histogram of the range cells in one figure.
    ylabels given  the clue of primary cell (labelled -1) and clutter only cell (labelled 1)
    :param hist_list:
    :param ylabels:
    :return:
    '''
    fig, ax       = plt.subplots(figsize=(6,2), dpi=300, frameon=True)
    #figsub, axsub = plt.subplots(figsize=(6,2), dpi=300, frameon=True)
    primary_hist  = hist_list[ylabels.index(-1)]
    clutter_hists = hist_list.copy()
    clutter_hists.pop(ylabels.index(-1))

    clutter_hists     = np.array(clutter_hists)
    clutter_mean_hist = np.mean(clutter_hists, 0)
    clutter_std_hist  = np.std(clutter_hists, 0)
    bins = np.arange(0, 9)

    nlen = len(hist_list)
    #ax.errorbar(bins, clutter_mean_hist,  yerr=clutter_std_hist, fmt='k', label='clutter-only')
    ax.plot(bins, clutter_mean_hist, 'k', label='clutter only')
    ax.plot(primary_hist, 'r-.', label='target contained')
    plt.ylim(0, 0.7)
    ax.set_xticks(bins)
    ax.set_yticks(np.arange(0, 0.7, 0.1))
    ax.set_xlabel('LBP code')
    ax.set_ylabel('Probability density')

    std_mean = np.mean(clutter_std_hist)
    # axsub.errorbar([2,3,4,5], clutter_mean_hist[2:6],  yerr=clutter_std_hist[2:6], fmt='k', label='clutter-only')
    # axsub.plot([2,3,4,5], primary_hist[2:6], 'r-.', label='target-contained')
    # # axsub.errorbar(bins, clutter_mean_hist,  yerr=clutter_std_hist, fmt='k', label='clutter-only')
    # # axsub.plot( primary_hist, 'r-.', label='target-contained')
    # axsub.set_xticks(bins)
    # axsub.set_yticks(np.arange(0,0.1,0.05))
    # for i in range(nlen):
    #     if ylabels[i]!=-1:
    #         ax.plot(hist_list[i])
        # if ylabels[i] == -1:
        #     ax.plot(hist_list[i], 'r-.')

    ax.legend(loc='upper center')
    plt.grid(True)
    #plt.show()
    fig.savefig(os.path.join(LBP_FIG_PATH, '%s_lbp_hist_stdave%.4f.png' % (str_name, std_mean)),
                    bbox_inches='tight', pad_inches=0, dpi=300)

    #plt.figure()
    print('std average %.4f for clutters'%std_mean)
    print('')

def fig_lbp_spot(frame, str_name):
    '''
    Compute a spot's LBP in the primary cell.
    :param frame:
    :return:
    '''
    lbp_frame = local_binary_pattern(frame, P=8, R=1, method='uniform')
    lbp_frame = lbp_frame.astype(np.uint8)

    roi_lbp = lbp_frame[7:12, 43:48]
    roi_frm = frame[7:12, 43:48]

    # sframe = Image.fromarray(roi_lbp)
    # if not os.path.exists(LBP_FIG_PATH):
    #     os.makedirs(LBP_FIG_PATH)
    # sframe.save(os.path.join(LBP_FIG_PATH,'%s_lbp_roi.png'  % str_name), compress_level=0)


    # roi_lbp = lbp_frame[7:12+20, 43:48+20]
    # roi_frm = frame[7:12+20, 43:48+20]
    fig, ax =  plt.subplots(1,2, dpi=100)
    ax[0].imshow(roi_frm, cmap='gray',extent=[42.5, 47.5, 6.5, 11.5])

    # xticks = np.arange(43,48)
    # yticks = np.arange(7,12)

    ax[1].imshow(roi_lbp, extent=[42.5, 47.5, 6.5, 11.5])
    # ax[1].set_yticks(yticks)
    # ax[1].set_xticks(xticks)
    fig_roi_hist, axhist = plt.subplots(dpi=150)
    roi_hist, bins = np.histogram(roi_lbp.ravel(), bins=9)
    axhist.bar(bins[:-1], roi_hist, width=0.1)
    axhist.set_xticks(np.arange(0,9))
    axhist.set_xticklabels(np.arange(0,9), fontweight='bold')
    axhist.set_yticklabels(np.arange(0,7), fontweight='bold')
    axhist.set_xlabel('LBP code', fontweight='bold')
    axhist.set_ylabel('Number of times', fontweight='bold')
    fig_roi_hist.savefig(os.path.join(LBP_FIG_PATH,'%s_lbp_roi_hist.png'  % str_name), bbox_inches='tight', pad_inches=0, dpi=300)
    print(str_name)

    import matplotlib.patches as mpatches
    tfig, tax = plt.subplots(dpi=200)
    mask = ((lbp_frame==4)+(lbp_frame==3))
    h,w = mask.shape[:2]

    enfrm = frame.copy()
    enfrm[mask]=255
    tax.imshow(frame, cmap='jet')#, extent=[0, 131, -500, 500])
    # change imshow labels refer to :
    # https://moonbooks.org/Articles/How-to-change-imshow-axis-values-labels-in-matplotlib-/
    y_ticklabel_list = [500, 0, -500]
    x_ticklabel_list = np.int0(np.arange(0,131,16.375)) # 8 labels
    tax.set_xticks(np.int0(np.arange(0, w, w/8))) # 8 xticks
    tax.set_yticks([0, int(h/2), h-1])
    tax.set_xticklabels(x_ticklabel_list)
    tax.set_yticklabels(y_ticklabel_list)
    tax.set_xlabel('Time(s)')
    tax.set_ylabel('Frequency(Hz)')
    #tax[1].imshow(frame, cmap='gray')
    for i in range(h):
        for j in range(w):
            if mask[i,j] == 1:
                circ = mpatches.Circle((j,i), 1, ec='w', fc='None', alpha=0.5)
                tax.add_artist(circ)
    tfig.savefig(os.path.join(LBP_FIG_PATH, '%s_tds_mark34.png' % str_name), bbox_inches='tight',
                         pad_inches=0, dpi=300)

    plt.show()

    # sframe = Image.fromarray(lbp_frame)
    # if not os.path.exists(LBP_FIG_PATH):
    #     os.makedirs(LBP_FIG_PATH)
    # sframe.save(os.path.join(LBP_FIG_PATH,'%s_lbp.png'  % str_name), compress_level=0)
    # sframe = Image.fromarray(frame)
    # sframe.save(os.path.join(LBP_FIG_PATH,'%s_tds_gray.png' % frame), compress_level=0, dpi=300)

def rayleigh():
    x =  np.arange(1,100,0.01)
    sigma = 10
    fx = (2*x/sigma)*np.exp(-x**2/sigma)
    plt.plot(x,fx)
    plt.show()

def frame_texture_extract():
    '''
    extract the frame texture with pre-processing method, monitor the lbp histogram changing.
    :return:
    '''
    print('Without the secondary cell, Check the target range cell detection rate!')
    fileprefix = '/Users/yizhou/Radar_Datasets/IPIX/'
    Nf = 512  # 1024#512  # 64#128#256#512*2
    Nt = 2 ** 17  # 60000 for 1998file,  #131072 for 1993file

    # loading csv file for listing the wanted filenames, primary and secondary cells
    samples_info = pd.read_csv('/Users/yizhou/Radar_Datasets/IPIX/filename_labels.csv',
                               index_col='Label', sep=",")
    file_names = samples_info['file_name'].tolist()
    primaries = samples_info['primary'].tolist()
    secondary_st = samples_info['sec_start'].tolist()
    secondary_nd = samples_info['sec_end'].tolist()

    #fig, axs = plt.subplots(2, 1)
    #fig_frm, frm_axs = plt.subplots(nrows=3, sharex=True)
    fig_frm, frm_axs = plt.subplots(nrows=2, sharex=True)
    for txmode in ['hh','hv', 'vh', 'vv']:
        correct_rate     = 0.
        correct_rate_93  = 0.
        correct_rate_98  = 0.
        correct_var_rate = 0.
        correct_lbp_rate = 0.
        test_files       = 0.
        test_files_93    = 0
        test_files_98    = 0


        for findx, fileName in enumerate(file_names): # for files in ipix1993
            filekey = fileName.split('_')[1]
            fileyear = fileName.split('_')[0][:4]
            primary_rangebin = primaries[findx] - 1
            # if fileyear != '1998': #omit 1998 for comparision
            #     continue
            if filekey != '135603':
                 continue

            # print(fileyear, filekey, txmode)
            image_path = '%s/%s_%s/%s/' % (fileprefix, fileyear, filekey, txmode)
            image_names = glob.glob(image_path + '*.png')
            image_names.sort()
            ylabels = []
            xfeatures     = []
            hist_list = [] #list contains all the histograms of all the cells in one txmode
            fig_hist, hist_ax = plt.subplots(nrows=2, sharex=False)
            for iname in image_names: #14 frames for each range cell
                fname_split = iname.split('/')
                frame_no    = int(fname_split[-1].split('.')[0])

                # omit the secondary range cells
                if (secondary_st[findx] - 1) <= frame_no <= (secondary_nd[findx] - 1):
                    if frame_no != primary_rangebin:
                        continue

                frame = cv2.imread(iname, 0)

                # for 93hh, 93vv, and 98hv
                Plbp = 8
                Rlbp = 1
                lbp_frame = local_binary_pattern(frame, P=Plbp, R=Rlbp, method='uniform')
                bin_lbp = int(np.max(lbp_frame))
                lbp_hist, lpb_bins = np.histogram(lbp_frame.ravel(), bins=bin_lbp, density=False)
                weights = np.array([2, 1, 1, 3, 3, 1, 1, 1, 2])
                lbp_hist_wt = lbp_hist * weights
                lbp_hist = lbp_hist_wt / np.sum(lbp_hist_wt)
                #lbp_hist = lbp_hist / lbp_frame.size

                if frame_no == (primary_rangebin-2):
                    str_name = '%s_%s_%02d_clutter' % (filekey, txmode, frame_no)
                    fig_target_TF(frame, str_name)
                if frame_no  == primary_rangebin:
                    str_name = '%s_%s_%02d_target' % (filekey, txmode, frame_no)
                    fig_target_TF(frame, str_name)
                    roi = lbp_frame [18:(18+5), 86:(86+5)]
                    rroi= np.flipud(np.fliplr(roi))
                    frroi = np.fft.fft2(rroi, lbp_frame.shape)
                    cr_frm= np.real(np.fft.ifft2(frroi*np.fft.fft2(lbp_frame)))
                    plt.figure()
                    plt.imshow(cr_frm, cmap='jet')
                    #plt.show()
                dn_frame = frame.copy()
                #dn_frame = cv2.medianBlur(dn_frame,3)
                kernel   = np.ones((3, 3), np.uint8)
                dn_frame  = cv2.erode(dn_frame, kernel, iterations=1)
                #dn_frame = cv2.fastNlMeansDenoising(dn_frame, None, 30.0, 7, 21)
                dn_lbp_frame = local_binary_pattern(dn_frame, P=Plbp, R=Rlbp, method='uniform')
                dn_bin_lbp = int(np.max(dn_lbp_frame))
                dn_lbp_hist, dn_lpb_bins = np.histogram(dn_lbp_frame.ravel(), bins=dn_bin_lbp, density=False)
                dn_lbp_hist = dn_lbp_hist / dn_lbp_frame.size

                feature_str = 'lbp P %d, R %d, bin %d' % (Plbp, Rlbp, bin_lbp)
                feature_hist = lbp_hist
                #feature_hist = np.concatenate([lbp_var_hist, lbp_hist])
                xfeatures.append(feature_hist)
                # xfeatures_var.append(lbp_var_hist)
                # xfeatures_lbp.append(lbp_hist)
                if frame_no == primary_rangebin:
                    ylabels.append(-1)  # Abnormal Mark for the primary cell with target
                else:
                    ylabels.append(1)
                hist_list.append(lbp_hist)

                # #Visualize discriminative lbp paterns
                # mask = (lbp_frame == 3) + (lbp_frame == 4) + (lbp_frame == 5)
                # mask_frame = frame
                # mask_frame[mask] = 255
                #
                # dn_mask = (dn_lbp_frame == 3) + (dn_lbp_frame == 4) + (dn_lbp_frame == 5)
                # dn_mask_frame = dn_frame
                # dn_mask_frame[dn_mask] = 255
                # #frm_axs[0].imshow(frame)
                # frm_axs[0].imshow(mask_frame)
                # frm_axs[1].imshow(dn_mask_frame)
                # #frm_axs[2].imshow(lbp_var)

                if frame_no == primary_rangebin:
                    fig_lbp_spot(frame, '%s_%s_%s_%02d'%(fileyear, filekey, txmode, frame_no))
                    frm_axs[0].set_title('%s_%d'%(filekey,frame_no), color='red')
                    hist_ax[0].plot(lpb_bins[:-1], lbp_hist, 'r-.', label=frame_no)
                    hist_ax[1].plot(dn_lpb_bins[:-1], dn_lbp_hist, 'r-.',label=frame_no)
                else:
                    frm_axs[0].set_title('%s_%d'%(filekey,frame_no), color='black')
                    hist_ax[0].plot(lpb_bins[:-1],lbp_hist, label=frame_no)
                    hist_ax[1].plot(dn_lpb_bins[:-1], dn_lbp_hist, label=frame_no)
                plt.draw()
                # figfrm_path = '%s/lbp/%s_%s/%s/' % (fileprefix, fileyear, filekey, txmode)
                # # if not os.path.exists(figfrm_path):
                # #     os.makedirs(figfrm_path)
                # # fig_frm.savefig(os.path.join(figfrm_path, '%04d_lbp.png' % (frame_no)),
                # #                 bbox_inches='tight', pad_inches=2, dpi=200)
                # sframe = Image.fromarray(mask_frame)
                # if not os.path.exists(figfrm_path):
                #     os.makedirs(figfrm_path)
                # sframe.save('%s/%02d.png' % (figfrm_path, frame_no), compress_level=0)

            hist_ax[0].set_title(filekey)
            fig_lbp_hist(hist_list, ylabels, '%s_%s_%s'%(fileyear, filekey, txmode))
            #hist_ax[1].set_title('lbp_var')
            #plt.legend()
            plt.draw()

            # fighist_path = '%s/lbp/' % fileprefix
            # if not os.path.exists(fighist_path):
            #     os.makedirs(fighist_path)
            # #fig_frm.savefig(os.path.join(fig_path,  '%04d_lbp.png'  % frame_no),  bbox_inches='tight', pad_inches=0, dpi=200)
            # fig_hist.savefig(os.path.join(fighist_path, '%s_%s_%s_hist.png' % (fileyear, filekey, txmode)),
            #                  bbox_inches='tight', pad_inches=2, dpi=200)
            plt.close(fig_hist)
            #abnormal frame detection
            # #clf = OneClassSVM(nu=0.1, kernel="poly", degree=3, gamma='auto')
            clf = OneClassSVM(nu=0.4, kernel='rbf', gamma='auto')
            clf.fit(xfeatures)
            y_score = clf.score_samples(xfeatures)
            y_test  = clf.predict(xfeatures)
            least_votes  = np.argsort(y_score)
            minvote = least_votes[0]
            maxvote = least_votes[-1]
            # print(least_votes)
            str_res = ''
            test_files +=1
            if fileyear == '1993':
                test_files_93 += 1
            if fileyear == '1998':
                test_files_98 += 1
            if (ylabels[minvote] == -1) or (ylabels[maxvote]==-1):
                correct_rate +=1
                str_res = 'correct'
                if fileyear == '1993':
                    correct_rate_93 +=1
                if fileyear == '1998':
                    correct_rate_98 +=1
            else:
                str_res = 'wrong'


        print(fileyear, filekey, txmode, str_res)
        # Compute the detection rate for one mode
        print('----\nWithout secondary cell, feature %s' % feature_str)
        print('Mode %s detection rate is %.2f' % (txmode, (correct_rate / test_files)))
        print('Mode %s detection rate is %.2f in 1993' % (txmode, (correct_rate_93 / test_files_93)))
        print('Mode %s detection rate is %.2f in 1998' % (txmode, (correct_rate_98 / test_files_98)))
    plt.close(fig_frm)


def deviation_LBP_histograms_table():
    '''
    1 In this file, we compute the averaged LBP histogram for the clutter-only cells.
    2 computed the deviation of the clutter-only cells. and the deviation between the target and averaged hist.
    3 compute  the sum of the deviation
    '''
    print('Without the secondary cell, Check the target range cell detection rate!')
    fileprefix = '/Users/yizhou/Radar_Datasets/IPIX/'
    Nf = 512  # 1024#512  # 64#128#256#512*2
    Nt = 2 ** 17  # 60000 for 1998file,  #131072 for 1993file

    # loading csv file for listing the wanted filenames, primary and secondary cells
    samples_info = pd.read_csv('/Users/yizhou/Radar_Datasets/IPIX/filename_labels.csv',
                               index_col='Label', sep=",")
    file_names = samples_info['file_name'].tolist()
    primaries = samples_info['primary'].tolist()
    secondary_st = samples_info['sec_start'].tolist()
    secondary_nd = samples_info['sec_end'].tolist()

    #fig, axs = plt.subplots(2, 1)
    #fig_frm, frm_axs = plt.subplots(nrows=3, sharex=True)
    fig_frm, frm_axs = plt.subplots(nrows=2, sharex=True)
    for txmode in ['hh','hv', 'vh', 'vv']:
        correct_rate     = 0.
        correct_rate_93  = 0.
        correct_rate_98  = 0.
        correct_var_rate = 0.
        correct_lbp_rate = 0.
        test_files       = 0.
        test_files_93    = 0
        test_files_98    = 0
        adc_93           = 0
        adc_98           = 0
        adt_93           = 0
        adt_98           = 0
        adc = 0 # averaged deviation to among the clutter-only cells
        adt = 0 # averaged deviation between clutter and target cells.


        for findx, fileName in enumerate(file_names): # for files in ipix1993
            filekey = fileName.split('_')[1]
            fileyear = fileName.split('_')[0][:4]
            primary_rangebin = primaries[findx] - 1
            # if fileyear != '1998': #omit 1998 for comparision
            #     continue
            # if filekey != '135603':
            #      continue

            # print(fileyear, filekey, txmode)
            image_path = '%s/%s_%s/%s/' % (fileprefix, fileyear, filekey, txmode)
            image_names = glob.glob(image_path + '*.png')
            image_names.sort()
            ylabels = []
            xfeatures     = []
            hist_list     = [] #list contains all the histograms of all the cells in one txmode
            clutter_list  = []
            target_hist   = np.array([])
            #fig_hist, hist_ax = plt.subplots(nrows=2, sharex=False)
            for iname in image_names: #14 frames for each range cell
                fname_split = iname.split('/')
                frame_no    = int(fname_split[-1].split('.')[0])

                # omit the secondary range cells
                if (secondary_st[findx] - 1) <= frame_no <= (secondary_nd[findx] - 1):
                    if frame_no != primary_rangebin:
                        continue

                frame = cv2.imread(iname, 0)

                # for 93hh, 93vv, and 98hv
                Plbp = 8
                Rlbp = 1
                lbp_frame = local_binary_pattern(frame, P=Plbp, R=Rlbp, method='uniform')
                bin_lbp = int(np.max(lbp_frame))
                lbp_hist, lpb_bins = np.histogram(lbp_frame.ravel(), bins=bin_lbp, density=False)
                weights = np.array([2, 1, 1, 3, 3, 1, 1, 1, 2])
                lbp_hist_wt = lbp_hist * weights
                lbp_hist = lbp_hist_wt / np.sum(lbp_hist_wt)
                #lbp_hist = lbp_hist / lbp_frame.size

                if frame_no  == primary_rangebin:
                    target_hist = lbp_hist.copy()

                feature_hist = lbp_hist
                feature_str = 'lbp P %d, R %d, bin %d' % (Plbp, Rlbp, bin_lbp)
                #feature_hist = np.concatenate([lbp_var_hist, lbp_hist])
                xfeatures.append(feature_hist)
                # xfeatures_var.append(lbp_var_hist)
                # xfeatures_lbp.append(lbp_hist)
                if frame_no == primary_rangebin:
                    ylabels.append(-1)  # Abnormal Mark for the primary cell with target
                else:
                    ylabels.append(1)
                    clutter_list.append(lbp_hist)
                hist_list.append(lbp_hist)

            #abnormal frame detection
            clf = OneClassSVM(nu=0.4, kernel='rbf', gamma='auto')
            clf.fit(xfeatures)
            y_score = clf.score_samples(xfeatures)
            y_test  = clf.predict(xfeatures)
            least_votes  = np.argsort(y_score)
            minvote = least_votes[0]
            # print(least_votes)
            str_res = ''
            #output the informations for each file
            ave_clutter_hist = np.mean(clutter_list, axis=0)
            std_cluster_hist = np.std(clutter_list,  axis=0)
            adc  = std_cluster_hist
            adt  = np.abs(target_hist - ave_clutter_hist)

            test_files +=1
            if fileyear == '1993':
                test_files_93 += 1
            if fileyear == '1998':
                test_files_98 += 1
            if (ylabels[minvote] == -1):
                correct_rate +=1
                str_res = 'correct'
                if fileyear == '1993':
                    correct_rate_93 +=1
                    adc_93  +=adc
                    adt_93  +=adt
                if fileyear == '1998':
                    correct_rate_98 +=1
                    adc_98 += adc
                    adt_98 += adt
            else:
                str_res = ' wrong '
            # print('%s %s %s %s sum_adc:%.5f - sum_adt%.5f mean_adc:%.5f - mean_adt%.5f，\n\t max_adc:%.5f - max_adt%.5f '
            #       %(fileyear, filekey, txmode, str_res, np.sum(adc), np.sum(adt),
            #         np.mean(adc), np.mean(adt), np.max(adc), np.max(adt)))
            print('%s %s %s %s TCR: %.5f '%(fileyear, filekey, txmode, str_res,10*np.log10(np.dot(adt,adt)/np.dot(adc,adc))))
            if filekey == '135603':
                print('adc:  ',adc)
                print('adt:  ',adt)
            # print('%s %s %s %s mean_adc:%.5f mean_adt%.5f'
            #       %(fileyear, filekey, txmode, str_res, np.mean(adc), np.mean(adt)))

        #print(fileyear, filekey, txmode, str_res)
        # Compute the detection rate for one mode
        print('----\nWithout secondary cell, feature %s' % feature_str)
        #print('Mode %s detection rate is %.2f' % (txmode, (correct_rate / test_files)))
        print('Mode %s detection rate is %.2f in 1993' % (txmode, (correct_rate_93 / test_files_93)))
        adt_93 = adt_93/test_files_93
        adc_93 = adc_93/test_files_93
        print('sum adt_93 is:%.5f' %  np.sum(adt_93))

        #print(format_floatvector_string(adt_93))
        print('sum adc_93 is:%.5f' %  np.sum(adc_93))
        #print(format_floatvector_string(adc_93))

        adt_98 = adt_98/test_files_98
        adc_98 = adc_98/test_files_98
        print('Mode %s detection rate is %.2f in 1998' % (txmode, (correct_rate_98 / test_files_98)))
        print('sum adt_98 is:%.5f' % np.sum(adt_98))
        print('sum adc_98 is:%.5f' % np.sum(adc_98))

        print('mean adt_93 is:%.5f' % np.mean(adt_93))
        print('mean adc_93 is:%.5f' % np.mean(adc_93))
        print('mean adt_98 is:%.5f' % np.mean(adt_98))
        print('mean adc_98 is:%.5f' % np.mean(adc_98))

        print('10log(dt^2/dc^2)_93: %.5f ' % (10 * np.log10(np.dot(adt_93, adt_93) / np.dot(adc_93, adc_93))))
        print('10log(dt^2/dc^2)_98: %.5f ' % (10 * np.log10(np.dot(adt_98, adt_98) / np.dot(adc_98, adc_98))))
        # print('adt_98 is:')
        # print(format_floatvector_string(adt_98))
        # print('adc_98 is:')
        # print(format_floatvector_string(adc_98))

    plt.close(fig_frm)

def format_floatvector_string(fv):
    string  = []
    for fl in fv:
        str = '%.5f'%fl
        string.append(str)
    return string



if __name__=="__main__":
    #rayleigh()
    #frame_texture_extract()
    deviation_LBP_histograms_table()