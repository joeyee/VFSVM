'''
Learn the Local Binary Patterns of each cfar_blobs.
Try to distinguish the clutter from the object.
KMeans(2 object), Merge(KDE),
SVM(labeled training on single frame)
Also pay attention to fast computing of LBPs.
'''

import sys
sys.path.append("../segmentation/")  #for import the utility in up-directory/segmention/
import cfar_segmentation_200527 as cfar_segentation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import utility as uti
import cv2
from PIL import Image
import glob
import time

from sklearn.cluster import KMeans

# Local Binary Pattern function
from skimage.feature import local_binary_pattern
# To calculate a normalized histogram
from scipy.stats import itemfreq
from scipy import ndimage
from skimage import transform

#verify the lbp parameters

#TODO distingguish the clutter and target by lbp histogram.
# 8 kmeans cluster the target into the same kind. (Done 5.31)
# Label the target and save the corresponding hist.
#
# cfar done on the local_var_image.

# def lbp_top(bing_image, echo_3d):
#


def get_lbp_hist(bin_image, echos):

    #echos[0:20,:] = 0
    blobs_lbp_dict ={}
    canvas = cv2.cvtColor(echos, cv2.COLOR_GRAY2BGR)
    (contours, _) = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(canvas, contours, contourIdx=-1, color=[0, 255, 0])
    # plt.imshow(canvas)
    # plt.show()

    #fechos = uti.frame_normalize(echos)
    tb = time.clock()
    lbp_blob_num = 0
    hist_data = []
    radius = 3
    # Number of points to be considered as neighbourers
    no_points = 8 * radius
    inv_var = local_binary_pattern(echos, no_points, radius, method='var')
    inv_var[np.isnan(inv_var)] = 0
    bin_var = cfar.cfar_seg(inv_var)
    plt.imshow(bin_var)
    plt.figure()
    plt.imshow(bin_image)
    plt.show()

    int_img = transform.integral_image(inv_var)

    target_hist = []
    for id, contour in enumerate(contours):
        x, y, w, h= cv2.boundingRect(contour)
        # omit the 20meters under the radar
        if (y <= 10):
            continue
        # omit small rectangle.
        if (w*h < 32) or (w<3) or (h<3):
            continue

        iiv = transform.integrate(int_img, (y, x), (h+y-1, x+w-1))
        #omit the sparse density of variance image.
        if (iiv[0]/(w*h)) < 500:
            continue
        ROI = echos[y:y + h, x:x + w]

        # Uniform LBP is used
        #lbp  = local_binary_pattern(echos, no_points, radius, method='uniform')
        lbp = local_binary_pattern(ROI, no_points, radius, method='uniform')

        lbp[np.isnan(lbp)]=0
        #echos_2ndf = ndimage.gaussian_laplace(lbp, sigma=2)
        #nlbp = local_binary_pattern(lbp, no_points, radius, method='var')

        #wld = 1 / (1 + np.exp(-nlbp))
        # plt.imshow(lbp>1000)#, cmap='jet')
        # plt.show()
        # Calculate the histogram
        bhist, bins = np.histogram(lbp.ravel(), bins=26, range=(0,25), density=True)

        bT = 0
        name = ''
        cys = [706, 717, 684, 969, 247, 249, 246, 249]
        cxs = [458, 508, 579, 813,  63,  1276, 1075, 1916]
        names=['Tri1', 'Tri2', 'Tri3', 'falao', 'Tail', 'Jiant', 'Blink', 'Titan']

        for cy, cx, nick in zip(cys, cxs, names):
            #pos of Tri1 in frame1
            if (y<cy<y+h) and (x< cx < x+w):
                bT = 1
                name = nick
                print('target ', nick, ' included', 'index: ', lbp_blob_num)
                target_hist.append(bhist)

        blobs_lbp_dict[str(lbp_blob_num)] = {'lbp_hist':bhist, 'label':bT, 'name':name, 'pos':[x,y,w,h]}
        hist_data.append(bhist)
        lbp_blob_num += 1


    score_matrix = np.zeros((lbp_blob_num, lbp_blob_num))
    for i in range(lbp_blob_num):
        for j in range(i, lbp_blob_num):
            score = cv2.compareHist(hist_data[i].astype(np.float32), hist_data[j].astype(np.float32), cv2.HISTCMP_CHISQR)
            #symetric matrix
            score_matrix[i, j] = round(score, 3)
            score_matrix[j, i] = round(score, 3)

    ntargets = len(target_hist)
    target_matrix = np.zeros((ntargets,ntargets))
    for i in range(ntargets):
        for j in range(i, ntargets):
            score = cv2.compareHist(target_hist[i].astype(np.float32), target_hist[j].astype(np.float32), cv2.HISTCMP_CHISQR)
            #symetric matrix
            target_matrix[i, j] = round(score, 3)
            target_matrix[j, i] = round(score, 3)


    plt.imshow(score_matrix)
    plt.figure()
    plt.imshow(target_matrix)
    plt.figure()
    rows = [0,1,2,3,14,15,17,18]
    for row in rows:
        plt.plot(score_matrix[row, :])
    plt.legend(rows)
    plt.show()
    tb = time.clock()
    # 8 kinds of kmeans cluster all target into one group. It indicates that there is strong clue of
    # texture different between object and sea clutter.
    # May be we can check the mean power of the clutter and target.
    kmeans = KMeans(n_clusters=3, random_state=0).fit(hist_data)
    print('%d blobs, lbp kmeans cost %.2fs' % (lbp_blob_num, time.clock()-tb))
    labels = kmeans.labels_

    # Create some random colors
    colors = np.random.randint(0, 255, (8, 3))

    for index, label in enumerate(labels):
        x,y,w,h = blobs_lbp_dict[str(index)]['pos']
        if label == 1:
            cv2.rectangle(canvas, (x,y), (x+w, y+h), color=(255,0,0))#colors[label, :].tolist())
        if label == 0:
             cv2.rectangle(canvas, (x, y), (x + w, y + h), color=(0, 255, 0))
        if label == 2:
             cv2.rectangle(canvas, (x, y), (x + w, y + h), color=(0, 0, 255))

    plt.imshow(canvas)
    plt.show()

    tcost = time.clock() - tb

    print('')





    #cv2.drawContours(canvas, contours, contourIdx=-1, color=[0, 255, 0])
    # cv2.imshow('cfar_seg', canvas)
    # cv2.waitKey()
    return canvas


if __name__=='__main__':
    file_prefix = '/Users/yizhou/Radar_Datasets/RecordEcho/2018-01-24-19_05_19-1/'
    #test_frame = np.array(Image.open('%s/%02d.png' % (file_prefix, frame_no)))
    file_names = glob.glob(file_prefix+'*.png')
    file_names.sort()
    # cv2.namedWindow('inesa', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('inesa', 1024, 768)
    # cv2.moveWindow('inesa', 200, 100)
    cfar = cfar_segentation.CFAR(kval=1)
    bGo=True
    while(bGo):
        for img_file in file_names:
            frame_no = img_file.split('/')[-1].split('.')[0]
            frame = np.array(Image.open(img_file))
            bin_img = cfar.cfar_seg(frame)
            get_lbp_hist(bin_img, frame)
            #cv2.putText(frame, frame_no, (500, 100), 2,2, (0,255,0))
            cv2.setWindowTitle('inesa', str(frame_no))
            #cv2.imshow('inesa', canvas)
            if(cv2.waitKey()& 0xFF == ord('q')):
                bGo=False
                break