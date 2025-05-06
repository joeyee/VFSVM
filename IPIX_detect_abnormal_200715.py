'''
Using the saved ipix images to detect abnormal range cells.
Two clues:
1 - suddenly appeared blobs in the two side way.
2 - booming effect of the center blobs.
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
import glob
from statistics import mode
import MCF_20200603             as mcf_model      # Fuse multiple correlated trackers for a target


def watershed_seg(frame):
    '''
    segment the blobs by watershed segmentation
    :param frame:
    :return:
    '''
    ret, thresh = cv2.threshold(frame, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # sure background area
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    ret, sure_fg   = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers + 1

    # Now, mark the region of unknown with zero
    markers[unknown == 255] = 0

    markers = cv2.watershed(frame, markers)
    frame[markers == -1] = [255, 0, 0]



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

def get_target_trajectory(frameno_list, rect_list):
    '''
    Make a trajectory dict by the frame_no key and corresponding rect_list.
    :param frameno_list:
    :param rect_list:
    :return:
    '''
    traj_dict = {}
    for frameno, rect in zip(frameno_list, rect_list):
        frame_key = '%02d'%frameno
        traj_dict[frame_key] = rect
    return traj_dict

def draw_mcf_trackers(mcf_tracker_list,blob_bb_list, frame_no, canvas):
    voted_blob_id = []
    assigned_tid = []
    with_text = False
    only_tail = False
    # each blob has a mcf_tracker, each mcf_tracker holds multiple mosse trackers.
    for mcf_tracker in mcf_tracker_list:
        mcf_fuse_rect = mcf_tracker.fuse_rect
        if mcf_tracker.voted_blob_id is not None:
            voted_blob_id.append(mcf_tracker.voted_blob_id)

        assigned_tid.append(mcf_tracker.tid)
        # draw rect_angles. text id for observing.

        if not only_tail:
            [x, y, w, h] = mcf_fuse_rect[:4]
            cv2.rectangle(canvas, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=1)
            if with_text:
                cv2.putText(canvas, str(mcf_tracker.tid), (int(x + w / 2), int(y + h)), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.3, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            #continue
            # mark out the stable tracker in Magenta:
            # record it in frame_target_list and save it in the csv file in the end.
            # if mcf_tracker.tracked_Frames >= 2:
            #     cv2.circle(canvas, (int(x + w / 2), int(y + h / 2)), max(int(w), int(h)), color=(255, 0, 255), thickness=2)
                #frame_target_list.append([frame_no, mcf_tracker.tid, mcf_fuse_rect, 'vessel'])
            # # draw voted blob in yellow
            #if mcf_tracker.voted_blob_id is not None:
            #    uti.draw_rect(canvas, blob_bb_list[mcf_tracker.voted_blob_id], (200, 200, 0), 1)
            # # draw new initialized blob in red
            # if mcf_tracker.init_blob_id is not None:  # blob are voted for initialize the new tracker
            #     uti.draw_rect(canvas, blob_bb_list[mcf_tracker.voted_blob_id], (0, 0, 200), 2)
            # # cv2.putText(canvas, str(mcf_tracker.tid), (int(x + w / 2), int(y + h / 2)), 1, 2, (255, 255, 255))
            # # draw component mosse tracker
            # for cid, comp_tracker in enumerate(mcf_tracker.trckList):
            #     px, py, pw, ph = comp_tracker.target_rect
            #     pu = (int(px), int(py - 5 + ph / 2))
            #     pc = (int(px + pw / 2), int(py + ph / 2))
            #     if comp_tracker.psr >= mcf_tracker.votePsrThreash:
            #         # draw less light yellow for high_psr_tracker's bb
            #         uti.draw_rect(canvas, comp_tracker.target_rect, (0, 100, 100), 2)
            #         # cv2.putText(canvas, str(cid), org=pc, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255),
            #         #             thickness=1, lineType=cv2.LINE_AA)
            #         if comp_tracker.target_rect in mcf_tracker.votable_tbox_list:
            #             # draw highlight yellow for votable boundingbox
            #             uti.draw_rect(canvas, comp_tracker.target_rect, (0, 255, 255), 1)
            #             if (mcf_model.DETAIL_MODE == True):
            #                 print('....voting tracker...psr%d, life %d' % (comp_tracker.psr, comp_tracker.tracked_Frames))
            #     else:
            #         # tracker' tbb is drawn in GRAY, if it has no right to vote!
            #         uti.draw_rect(canvas, comp_tracker.target_rect, (125, 125, 125), 1)
            #         txt = '%d-%d' % (cid, int(comp_tracker.psr))  # ,np.sum(comp_tracker.highpsr_container))
            #         if with_text:
            #             cv2.putText(canvas, txt, org=pu,
            #                         fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255),
            #                         thickness=1, lineType=cv2.LINE_AA)

        # draw history tail position
        bound_h, bound_w = canvas.shape[:2]
        for tail_rect in mcf_tracker.tail_rect_list:
            tx, ty, tw, th = tail_rect[:4]
            px = max(0,min(int(tx + tw / 2), bound_w-1))
            py = max(0,min(int(ty + th / 2), bound_h-1))
            canvas[py,px,:] = (0,255,255)
            #cv2.circle(canvas, (px, py), 2, color=(255, 255, 255), thickness=2)
            #cv2.circle(canvas, (px, py), 1, color=(255, 255, 255), thickness=1)
    return canvas

def blob_seg(tdmat):
    blob_bb_list, bin_image = segmentation(tdmat, lbp_contrast_select = False,
                 kval=.8, least_wh = (2,5), min_area=10, max_area=400, nref=8, mguide=4)

    frame_cfar = (bin_image*tdmat)
    canvas = cv2.cvtColor(frame_cfar, cv2.COLOR_BGR2RGB)
    for blob in blob_bb_list:
        uti.draw_rect(canvas, blob, color=(0,255,0))

    fig, axs = plt.subplot(2,1)
    axs[0].imshow(tdmat)
    axs[1].imshow(frame_cfar)
    return bin_image

def get_grid(frame):
    '''
    Divide the frame into sub grids.
    :param frame:
    :return:
    '''
    #canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    # canvas = np.zeros_like(frame)
    # fig, axs = plt.subplots(2,1)
    # fighist, axh = plt.subplots(2,1)
    h,w  = frame.shape[:2]  # frame's height and width
    #testify  the   frame  is symetric
    # half_h = int(h/2)
    # assert(np.sum(frame[0:half_h,:])==np.sum(frame[half_h:h,:]))
    # ghs = [10, 10, 12, 12, 10, 10] # grid's height and width
    # ys  = [0,  10, 20, 32, 43, 53]
    ghs = [10, 10, 10, 10] # side grid's height and width
    ys  = [0,  10, 43, 53]
    gw  = 10
    step_w     = int(w/gw)
    grids = []
    for i in range(len(ghs)):
        for j in range(step_w):
            x = j*gw
            y = ys[i]
            gh= ghs[i]
            grids.append([x, y, gw, gh])
            #uti.draw_rect(canvas,[x,y,gw,gh],color=(255,255,255))
    center_lane_grid = [0,20,w,24]
    grids.append(center_lane_grid)
    # for grid in grids:
    #     bx, by, bw, bh = grid[:4]
    #     roi = frame[by:by + bh, bx:bx + bw]
    #     lbp = local_binary_pattern(roi, P=8, R=1, method='uniform')
    #     canvas[by:by + bh, bx:bx + bw] = lbp
    #     #lbp[np.isnan(lbp)] = 0
    #     bhist, bins = np.histogram(lbp.ravel(), bins=9, density=False)
    #     bhist_l1 = bhist/np.sum(bhist)
    #     bhist_l2 = bhist/np.sqrt(np.dot(bhist, bhist))
    #     if by in [20, 32]:
    #         axh[0].plot(bhist_l1)
    #     else:
    #         axh[1].plot(bhist_l1)
    # axs[0].imshow(frame)
    # axs[1].imshow(canvas)
    #lbp = local_binary_pattern(frame, P=8, R=1, method='uniform')
    # plt.figure()
    # plt.imshow(lbp)
    #plt.show()
    return grids

def get_grid_feature(grid, frame, norm='l2'):
    '''
    Get features from the grid
    :param grid:
    :param frame:
    :return:
    '''
    bx, by, bw, bh = grid[:4]
    roi = frame[by:by + bh, bx:bx + bw]
    lbp = local_binary_pattern(roi, P=8, R=1, method='uniform')
    assert(np.sum(np.isnan(lbp))==0)
    bhist, bins = np.histogram(lbp.ravel(), bins=9, density=False)
    bhist_l1 = bhist / np.sum(bhist)
    bhist_l2 = bhist / np.sqrt(np.dot(bhist, bhist))
    if norm == 'l1':
        return bhist_l1
    else:
        return bhist_l2

def get_slice_feature(center_grid, frame, norm='l2'):
    '''
    Get the slice feature based on the center_grid.
    :param center_grid:
    :param frame:
    :param norm:
    :return:
    '''
    gx, gy, gw, gh = center_grid[:4]
    imh, imw = frame.shape[:2]
    #print(gx, gw, imh)
    #assert(gx>=0 and gw > 0  and imh>0)

    slice_grid = [gx, 0, gw, imh]
    lbphist = get_grid_feature(slice_grid, frame, norm)
    return lbphist



def get_center_blobs(blob_list, frame):
    '''
    get the slices(blob_tlx, 0,  blobwidth,imageHeight) for center blobs.
    :return:
    '''
    h,w = frame.shape[:2]
    half_h = int(h/2)
    #slices = [] # grid containers.
    center_blobs = []
    for blob in blob_list:
        bx, by, bw, bh = blob[:4]
        if by<=half_h<=by+bh:
            #blob contains the center line of the frame
            #slices.append([bx, 0, bw, h])
            center_blobs.append(blob)
    return center_blobs


def detect_target_cell(xfeatures, ylabels, bprint = True):
        #abnormal frame detection
        #clf = OneClassSVM(nu=0.1, kernel="poly", degree=3, gamma='auto')
        bCorrect_Classify = False
        clf = OneClassSVM(nu=0.4, kernel='rbf', gamma='auto')
        clf.fit(xfeatures)
        y_score = clf.score_samples(xfeatures)
        y_test  = clf.predict(xfeatures)
        least_votes  = np.argsort(y_score)
        minvote = least_votes[0]
        maxvote = least_votes[-1]
        # print(least_votes)
        str_res = ''
        if (ylabels[minvote] == -1) or (ylabels[maxvote]==-1):
            bCorrect_Classify = True
            str_res = 'correct'
        #     print('correct vote in %s-%s: %d'%(filekey, txmode, ylabels.index(-1)))
        else:
            str_res = 'wrong'
        #     print('wrong vote in %s-%s: least %d - highest %d - gt %d'
        #           %(filekey, txmode, least_votes[0],least_votes[-1], ylabels.index(-1)))
        #print(fileyear, filekey, txmode, str_res)
        if bprint:
            if bCorrect_Classify==False:
                print(least_votes)
                print('gt cell %s' % ylabels.index(-1))
        return bCorrect_Classify

def mcf_track_ipix():
    '''
    Using mcf tracker to tracking the blobs in the center lane of the Time-Frequency Matrix.
    The PSR and texture feature in each blob-container slices are extracted for abnormal detection.
    :return:
    '''
    fileprefix = '/Users/yizhou/Radar_Datasets/IPIX/'
    Nf = 256  # 1024#512  # 64#128#256#512*2
    Nt = 2 ** 17  # 60000 for 1998file,  #131072 for 1993file

    # loading csv file for listing the wanted filenames, primary and secondary cells
    samples_info = pd.read_csv('/Users/yizhou/Radar_Datasets/IPIX/filename_labels.csv',
                               index_col='Label', sep=",")
    file_names = samples_info['file_name'].tolist()
    primaries = samples_info['primary'].tolist()
    secondary_st = samples_info['sec_start'].tolist()
    secondary_nd = samples_info['sec_end'].tolist()

    fig, axs = plt.subplots(2, 1)

    for findx, fileName in enumerate(file_names):
        # if findx >= 13:  # current not handle anstep files
        #     continue
        filekey = fileName.split('_')[1]
        fileyear = fileName.split('_')[0][:4]
        primary_rangebin = primaries[findx] - 1
        if fileyear != '1993':
            continue
        for txmode in ['hv', 'vh', 'hh', 'vv']:
            print(fileyear, filekey, txmode)
            image_path = '%s/%s_%s/%s/' % (fileprefix, fileyear, filekey, txmode)
            image_names = glob.glob(image_path + '*.png')
            image_names.sort()
            ylabels = []
            slice_feature_dict = {}
            vert_intensity_dict = {}
            horn_intersity_dict = {}
            watch_psr_dict = {}
            for iname in image_names:
                fname_split = iname.split('/')
                frame_no = int(fname_split[-1].split('.')[0])
                frame = cv2.imread(iname, 0)
                #frame = cv2.medianBlur(frame, 5)
                frame = cv2.fastNlMeansDenoising(frame, None, h=9, templateWindowSize = 7, searchWindowSize = 15)
                if frame_no == 0:  # init_frame
                    # lbp_var = local_binary_pattern(frame, P=8, R=1, method='var')
                    # lbp_var[np.isnan(lbp_var)] = 0

                    blob_bb_list, bin_image = segmentation(frame, lbp_contrast_select=False,
                                                           kval=.8, least_wh=(2, 5), min_area=10, max_area=400, nref=8,
                                                           mguide=4)
                    center_blob_list = get_center_blobs(blob_bb_list, frame)
                    print('Frame %02d cfar segments %4d blobs and %4d slices' % (frame_no, len(blob_bb_list), len(center_blob_list)))
                    mcf_tracker_list = []
                    tracker_txt_dict = {}
                    # save the qualified [mctracker.tracked_Frames > nums_thresh] target trajectory in the dict.
                    # This dict's format is the same as the gt_dict in 'simulate_clutter_target_*.py'
                    # {'target_name':{'frame_no':[rect_x, y, w,h]}}
                    target_trajectory_dict = {}
                    mcf_model.DETAIL_MODE = False  # decrease the log text
                    for cid, cblob_bb in enumerate(center_blob_list):
                        # initialize mcftracker for each blob.
                        mcfTracker = mcf_model.MCF_Tracker(frame, frame_no, cblob_bb, cid)
                        mcf_tracker_list.append(mcfTracker)
                        center_blob = mcfTracker.fuse_rect
                        cx = center_blob[0] + center_blob[2] / 2
                        cy = center_blob[1] + center_blob[3] / 2
                        tracker_txt = axs[1].text(cx, cy, '%d'%mcfTracker.tid, color='white')
                        tracker_txt_dict[mcfTracker.tid] = tracker_txt
                    print('Frame %02d mcf inits %4d mcf_trackers' % (frame_no, len(mcf_tracker_list)))
                    frame_cfar = (bin_image * frame)
                    canvas = cv2.cvtColor(frame_cfar, cv2.COLOR_BGR2RGB)
                    for blob in blob_bb_list:
                        uti.draw_rect(canvas, blob, color=(255, 255, 255))
                    for cbb in center_blob_list:
                        uti.draw_rect(canvas, cbb, color=(0, 255, 0))
                    #plt.show()

                if frame_no > 0 : #tracking frame
                    #canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                    # var_mat = local_binary_pattern(frame, P=8, R=1, method='var')
                    # var_mat[np.isnan(var_mat)] = 0
                    # var_mat = uti.frame_normalize(var_mat)*255
                    # canvas = cv2.cvtColor(var_mat.astype(np.uint8), cv2.COLOR_GRAY2BGR)

                    blob_bb_list, bin_image = segmentation(frame, lbp_contrast_select=False,
                                                           kval=.8, least_wh=(2, 5), min_area=10, max_area=400, nref=8,
                                                           mguide=4)

                    # blob_bb_list, bin_image = segmentation(frame, lbp_contrast_select=False,
                    #                                        kval=.8, least_wh=(2, 4), min_area=8, max_area=400, nref=4,
                    #                                        mguide=2)
                    canvas = cv2.cvtColor(frame*bin_image, cv2.COLOR_GRAY2BGR)
                    for blob in blob_bb_list:
                        uti.draw_rect(canvas, blob, color=(255, 255, 255), thick=2)

                    center_blob_list = get_center_blobs(blob_bb_list, frame)
                    print('Frame %02d cfar segments %4d blobs and %4d slices' %
                          (frame_no, len(blob_bb_list), len(center_blob_list)))

                    voted_blob_id = []
                    assigned_tid = []
                    # each blob has a mcf_tracker, each mcf_tracker holds multiple mosse trackers.
                    for mcfTracker in mcf_tracker_list:
                        mcf_fuse_rect, psr = mcfTracker.update(frame, frame_no, center_blob_list)
                        if mcfTracker.voted_blob_id is not None:
                            voted_blob_id.append(mcfTracker.voted_blob_id)
                        assigned_tid.append(mcfTracker.tid)

                    # delete mcf_tracker which has low ave_psr value.
                    del_mcf_nums = 0
                    for mcfTracker in mcf_tracker_list:
                        if mcfTracker.ave_psr < mcfTracker.votePsrThreash / 2:
                            # save the qualified long_term tracker in the target_trajectory_dict, before delete the tracker
                            if mcfTracker.tracked_Frames >= 2:
                                target_trajectory_dict[mcfTracker.tid] = get_target_trajectory(
                                    mcfTracker.tail_frameno_list, mcfTracker.tail_rect_list)
                            mcf_tracker_list.remove(mcfTracker)
                            # print('delete %d tracker' % mcfTracker.tid)
                            assigned_tid.remove(mcfTracker.tid)
                            del_mcf_nums += 1

                    assigned_tid.sort()
                    ini_mcf_nums = 0
                    # initialize unlabelled blob for new tracker
                    for bid, blob_bb in enumerate(center_blob_list):
                        if bid not in voted_blob_id:
                            tid = assigned_tid[-1] + 1  # new target id
                            # initialized with new target id
                            new_mcf_tracker = mcf_model.MCF_Tracker(frame, frame_no, blob_bb, tid)
                            mcf_tracker_list.append(new_mcf_tracker)
                            assigned_tid.append(tid)
                            ini_mcf_nums += 1
                    print('Frame %02d delete %04d weak mcf_trackers with less psr, while init %04d new mcf_trackers'
                          % (frame_no, del_mcf_nums, ini_mcf_nums))
                    canvas = draw_mcf_trackers(mcf_tracker_list, center_blob_list, frame_no, canvas)
                # print ave_psr and tracked_frames for each mcftracker
                for mcf_tracker in mcf_tracker_list:
                    print('Tracker %d has ave_psr %.2f, holds %d frames '
                          %(mcf_tracker.tid, mcf_tracker.ave_psr, mcf_tracker.tracked_Frames))
                    # if mcf_tracker.voted_blob_id!=None:
                    #     print('vote_id %d'%mcf_tracker.voted_blob_id)
                    #     center_blob = center_blob_list[mcf_tracker.voted_blob_id]
                    # else:
                    center_blob = mcf_tracker.fuse_rect
                    if (center_blob[0]<0 or (center_blob[0]+center_blob[2])>frame.shape[1]): #assuming the moving from right to the right
                         print('omited center blob, tracker %d is across boundary '%mcf_tracker.tid)
                         #jump feature computing
                         continue
                    # if (center_blob[2]*center_blob[3]<100):
                    #     print('omited center blob, center blob is less than 100')
                    #     continue
                    cx = center_blob[0]+center_blob[2]/2
                    cy = center_blob[1]+center_blob[3]/2
                    #slice_grid = [center_blob[0], 1, center_blob[2], frame.shape[0]-2]
                    side_grid = [center_blob[0], 0, center_blob[2], 20] #side blob on the top  of center blob
                    uti.draw_rect(canvas, side_grid, color=(255,0,0))
                    #set the text position
                    if mcf_tracker.tid in tracker_txt_dict:
                        tracker_txt_dict[mcf_tracker.tid].set_position((cx, cy))
                    else:
                        tracker_txt_dict[mcf_tracker.tid] = axs[1].text(cx,cy, '%d'%mcf_tracker.tid, color='white')
                    print(center_blob)
                    slice_feature = get_slice_feature(side_grid, frame, 'l2')
                    lbp_var = local_binary_pattern(frame, P=8, R=1, method='var')
                    lbp_var[np.isnan(lbp_var)] = 0
                    grid_sum = np.sum(lbp_var[side_grid[1]:(side_grid[1]+side_grid[3]),
                                            side_grid[0]:(side_grid[0]+side_grid[2])])
                    vert_intensity = np.sum(lbp_var[side_grid[1]:(side_grid[1]+side_grid[3]),
                                            side_grid[0]:(side_grid[0]+side_grid[2])], 0)
                    horn_intersity = np.sum(lbp_var[side_grid[1]:(side_grid[1]+side_grid[3]),
                                            side_grid[0]:(side_grid[0]+side_grid[2])], 1)
                    if mcf_tracker.tid in slice_feature_dict:
                        slice_feature_dict[mcf_tracker.tid].append(slice_feature)
                        vert_intensity_dict[mcf_tracker.tid].append(vert_intensity)
                        horn_intersity_dict[mcf_tracker.tid].append(horn_intersity)
                        watch_psr_dict[mcf_tracker.tid].append(grid_sum)
                    else:
                        slice_feature_dict[mcf_tracker.tid] = [slice_feature]
                        watch_psr_dict[mcf_tracker.tid] = [grid_sum]
                        vert_intensity_dict[mcf_tracker.tid] = [vert_intensity]
                        horn_intersity_dict[mcf_tracker.tid] = [horn_intersity]

                str_mark = ''
                str_color = 'black'
                # for gid, grid in enumerate(grids):
                #     gxf = get_grid_feature(grid, frame, 'l2')  # feature in a grid
                #     grid_feature_dict[gid].append(gxf)
                if primary_rangebin == frame_no:
                    str_mark = 'in primary'
                    str_color = 'red'
                    ylabels.append(-1)
                    #canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                else:
                    ylabels.append(1)
                    if (secondary_st[findx] - 1) <= frame_no <= (secondary_nd[findx] - 1):
                        str_mark = 'in second'
                    else:
                        str_mark = ''
                #show the image
                im = axs[0].imshow(frame)
                canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                #axs[1].imshow(canvas, im.get_cmap())
                if frame_no == primary_rangebin:
                    #axs[1].plot(frame[:,3], 'r-.')
                    axs[1].clear()
                    axs[1].contour(frame, 10, color='red')
                else:
                    #axs[1].plot(frame[:,3])
                    axs[1].clear()
                    axs[1].contour(frame, 10, color='white')
                axs[0].set_title('%s_%s_%d' % (fileyear, filekey, frame_no), color=str_color)
                plt.waitforbuttonpress()
            ##check the psr change
            # plt.figure()
            # costs = np.linspace(0,13,14) # the rangceil on the first gets low costs
            # range_cts= np.zeros(14)
            # for key in watch_psr_dict:
            #     if(len(watch_psr_dict[key])==14):
            #         #plt.plot(np.sign(np.diff(watch_psr_dict[key])), label=key)
            #         plt.plot(watch_psr_dict[key], label=key)
            #         #print(np.argmin(watch_psr_dict[key]))
            #         range_rank = np.argsort(watch_psr_dict[key])
            #         range_cts[range_rank]   = range_cts[range_rank] + costs
            #         print(np.argsort(watch_psr_dict[key]))
            # plt.legend()
            # print('vote results')
            # print(np.argsort(range_cts))
            # if np.argmin(range_cts) == primary_rangebin:
            #     print('correct vote %d'%np.argmin(range_cts))
            # else:
            #     print('wrong vote %d - %d'%(np.argmin(range_cts), primary_rangebin))
            # plt.draw()
            #plt.show()

            # for key in vert_intensity_dict:
            #     vinten = vert_intensity_dict[key]  # xfeatures should get the dimensions of 10x14
            #     hinten = horn_intersity_dict[key]
            #     if len(vinten) == 14:  # features taken from the 14 TF frames.
            #         sf, sax  = plt.subplots(2,1)
            #         for i in range(14):
            #             if i == primary_rangebin:
            #                 sax[0].plot(vinten[i],'r-.', label=i)
            #                 sax[1].plot(hinten[i], 'r-.', label=i)
            #             else:
            #                 sax[0].plot(vinten[i], label=i)
            #                 sax[1].plot(hinten[i], label=i)
            #         plt.title('tid %d'%key)
            #         plt.legend()

            # for key in slice_feature_dict:
            #     xfeatures = slice_feature_dict[key]  # xfeatures should get the dimensions of 10x14
            #     if len(xfeatures) == 14:  # features taken from the 14 TF frames.
            #         plt.figure()
            #         for i in range(14):
            #             if i == primary_rangebin:
            #                 plt.plot(xfeatures[i],'r.', label=i)
            #             else:
            #                 plt.plot(xfeatures[i], label=i)
            #         plt.title('tid %d'%key)
            #         plt.legend()

            #plt.show()
            #detect_abnormal_rangebin(slice_feature_dict, 14)

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
            # if findx >= 13:  # current not handle anstep files
            #     continue
            filekey = fileName.split('_')[1]
            fileyear = fileName.split('_')[0][:4]
            primary_rangebin = primaries[findx] - 1
            # if fileyear != '1998': #omit 1998 for comparision
            #     continue
            # if filekey != '135603':
            #     continue
            # print(fileyear, filekey, txmode)
            image_path = '%s/%s_%s/%s/' % (fileprefix, fileyear, filekey, txmode)
            image_names = glob.glob(image_path + '*.png')
            image_names.sort()
            ylabels = []
            xfeatures     = []
            fig_hist, hist_ax = plt.subplots(nrows=1, sharex=False)
            for iname in image_names: #14 frames for each range cell
                fname_split = iname.split('/')
                frame_no    = int(fname_split[-1].split('.')[0])

                # omit the secondary range cells
                if (secondary_st[findx] - 1) <= frame_no <= (secondary_nd[findx] - 1):
                    if frame_no != primary_rangebin:
                        continue

                frame = cv2.imread(iname, 0)
                #frame = cv2.medianBlur(frame, 5)
                #frame = cv2.fastNlMeansDenoising(frame, None, h=9, templateWindowSize = 7, searchWindowSize = 15)
                # lbp_var = local_binary_pattern(frame, P=8, R=1, method='var')
                # lbp_var[np.isnan(lbp_var)] = 0
                # cfar parameters for 1993
                # blob_bb_list, bin_image = segmentation(frame, lbp_contrast_select=False,
                #                                        kval=1., least_wh=(4, 4), min_area=16, max_area=400, nref=4,
                #                                        mguide=2)
                # lbp_var = local_binary_pattern(frame, P=64, R=5, method='var')
                # lbp_var[np.isnan(lbp_var)] = 0
                # filter_var = uti.frame_normalize(lbp_var[bin_image>0])
                # lbp_var_hist, lpb_var_bins = np.histogram(filter_var.ravel(), bins=500, density=False)
                # lbp_var_hist =  lbp_var_hist / filter_var.size

                # cfar parameters for 1998
                # blob_bb_list, bin_image = segmentation(frame, lbp_contrast_select=False,
                #                                        kval=1., least_wh=(4, 4), min_area=16, max_area=400, nref=4,
                #                                        mguide=2)

                #for joint LBP/Var(LBP/Contrast)
                # Pvar = 80
                # Rvar = 6
                # bin_var = 8  # 500
                # lbp_var = local_binary_pattern(frame, P=Pvar, R=Rvar, method='var')
                # # lbp_var = local_binary_pattern(frame, P=64, R=5, method='var')
                # lbp_var[np.isnan(lbp_var)] = 0
                # lbp_var_hist, lpb_var_bins = np.histogram(lbp_var.ravel(), bins=bin_var, density=False)
                # lbp_frame = local_binary_pattern(frame, P=8, R=1, method='uniform')
                # bin_lbp = int(np.max(lbp_frame))
                # lbp_hist, lpb_bins = np.histogram(lbp_frame.ravel(), bins=bin_lbp, density=False)
                # lbp_c_jd = np.outer(lbp_var_hist, lbp_hist)
                # lbp_c_hist = lbp_c_jd.ravel() / np.sum(lbp_c_jd)


                # for 93hh, 93vv, and 98hv
                Plbp = 8
                Rlbp = 1
                lbp_frame = local_binary_pattern(frame, P=Plbp, R=Rlbp, method='uniform')
                bin_lbp = Plbp+2
                lbp_hist, lpb_bins = np.histogram(lbp_frame.ravel(), bins=bin_lbp, density=False)
                lbp_hist = lbp_hist / lbp_frame.size
                # if frame_no  == primary_rangebin:
                #     roi = lbp_frame [18:(18+5), 86:(86+5)]
                #     rroi= np.flipud(np.fliplr(roi))
                #     frroi = np.fft.fft2(rroi, lbp_frame.shape)
                #     cr_frm= np.real(np.fft.ifft2(frroi*np.fft.fft2(lbp_frame)))
                #     plt.figure()
                #     plt.imshow(cr_frm, cmap='jet')
                #     plt.show()

                # dn_frame = frame.copy()
                # #dn_frame = cv2.medianBlur(dn_frame,3)
                # kernel   = np.ones((3, 3), np.uint8)
                # dn_frame  = cv2.erode(dn_frame, kernel, iterations=1)
                # #dn_frame = cv2.fastNlMeansDenoising(dn_frame, None, 30.0, 7, 21)
                # dn_lbp_frame = local_binary_pattern(dn_frame, P=Plbp, R=Rlbp, method='uniform')
                # dn_bin_lbp = int(np.max(dn_lbp_frame))
                # dn_lbp_hist, dn_lpb_bins = np.histogram(dn_lbp_frame.ravel(), bins=dn_bin_lbp, density=False)
                # dn_lbp_hist = dn_lbp_hist / dn_lbp_frame.size

                feature_str = 'lbp P %d, R %d, bin %d' % (Plbp, Rlbp, bin_lbp)
                feature_hist = lbp_hist
                #feature_hist = lbp_c_hist
                #feature_hist = np.concatenate([lbp_var_hist, lbp_hist])
                xfeatures.append(feature_hist)
                # xfeatures_var.append(lbp_var_hist)
                # xfeatures_lbp.append(lbp_hist)
                if frame_no == primary_rangebin:
                    ylabels.append(-1)  # Abnormal Mark for the primary cell with target
                else:
                    ylabels.append(1)

                #Visualize discriminative lbp paterns
                mask = (lbp_frame == 3) + (lbp_frame == 4) + (lbp_frame == 5)
                mask_frame = frame
                mask_frame[mask] = 255

                # dn_mask = (dn_lbp_frame == 3) + (dn_lbp_frame == 4) + (dn_lbp_frame == 5)
                # dn_mask_frame = dn_frame
                # dn_mask_frame[dn_mask] = 255
                # #frm_axs[0].imshow(frame)
                # frm_axs[0].imshow(mask_frame)
                # frm_axs[1].imshow(dn_mask_frame)
                #frm_axs[2].imshow(lbp_var)

                if frame_no == primary_rangebin:
                    frm_axs[0].set_title('%s_%d'%(filekey,frame_no), color='red')
                    hist_ax.plot(lpb_bins[:-1], lbp_hist, 'r-.', label=frame_no)
                    # hist_ax[0].plot(lpb_bins[:-1], lbp_hist, 'r-.', label=frame_no)
                    # hist_ax[1].plot(dn_lpb_bins[:-1], dn_lbp_hist, 'r-.',label=frame_no)
                else:
                    frm_axs[0].set_title('%s_%d'%(filekey,frame_no), color='black')
                    hist_ax.plot(lpb_bins[:-1], lbp_hist, label=frame_no)
                    # hist_ax[0].plot(lpb_bins[:-1],lbp_hist, label=frame_no)
                    # hist_ax[1].plot(dn_lpb_bins[:-1], dn_lbp_hist, label=frame_no)
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

            hist_ax.set_title(filekey)
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
        #     bvar_res = detect_target_cell(xfeatures_var, ylabels, bprint=False)
        #     blbp_res = detect_target_cell(xfeatures_lbp, ylabels, bprint=False)
        #     classify_res = ['W', 'C']
        #     if bvar_res:
        #         correct_var_rate += 1
        #     if blbp_res:
        #         correct_lbp_rate += 1
        #     print(fileyear, filekey, txmode,
        #           'var - lbp ', classify_res[bvar_res],' - ', classify_res[blbp_res])
        # print('===\n mode %s dr_var %.2f, %d/%d\n==='
        #       %(txmode, (correct_var_rate/test_files), correct_var_rate, test_files))
        # print('===\n mode %s dr_lbp %.2f, %d/%d\n==='
        #       % (txmode, (correct_lbp_rate / test_files), correct_lbp_rate, test_files))
    plt.close(fig_frm)
from skimage.registration import optical_flow_tvl1
from skimage.feature import hog
from skimage import  exposure
def optical_flow_extract():
    '''
    extract the frame texture with pre-processing method, monitor the lbp histogram changing.
    :return:
    '''
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
    bdraw = False
    fig_frm, frm_axs = plt.subplots(nrows=3, sharex=True)
    for txmode in ['hh','hv', 'vh', 'vv']:
        correct_rate = 0.
        correct_var_rate = 0.
        correct_lbp_rate = 0.
        test_files   = 0.

        for findx, fileName in enumerate(file_names): # for files in ipix1993

            filekey = fileName.split('_')[1]
            fileyear = fileName.split('_')[0][:4]
            primary_rangebin = primaries[findx] - 1
            if fileyear != '1993': #omit 1998 for comparision
                continue
            # print(fileyear, filekey, txmode)

            image_path = '%s/%s_%s/%s/' % (fileprefix, fileyear, filekey, txmode)
            image_names = glob.glob(image_path + '*.png')
            image_names.sort()
            ylabels = []
            slice_feature_dict = {}
            vert_intensity_dict = {}
            horn_intersity_dict = {}
            watch_psr_dict = {}
            xfeatures_var = []
            xfeatures_lbp = []
            xfeatures     = []
            if bdraw:
                fig_hist, hist_ax = plt.subplots(nrows=2, sharex=False)
            frame_pre = cv2.imread(image_names[0], 0)
            hsv = np.zeros_like(cv2.cvtColor(frame_pre, cv2.COLOR_GRAY2RGB)) #3 channel hsv
            hsv[..., 1] = 255
            for iname in image_names: #14 frames for each range cell
                fname_split = iname.split('/')
                frame_no    = int(fname_split[-1].split('.')[0])

                # omit the secondary range cells
                # if (secondary_st[findx] - 1) <= frame_no <= (secondary_nd[findx] - 1):
                #     if frame_no != primary_rangebin:
                #         continue

                if iname == image_names[0]: #jump the first frame for the already frame_pre
                    continue
                frame = cv2.imread(iname, 0)

                # --- Compute the optical flow
                # v, u = optical_flow_tvl1(frame_pre, frame)
                # frame_pre = frame
                # mag, ang = cv2.cartToPolar(u,v)
                # ang = ang*180/np.pi/2
                # bin_lens = 8
                # hog_opfl = np.zeros(bin_lens)
                # ang_step = 180/bin_lens
                # ang_quant= np.int0(ang/ang_step)
                # for bin in range(bin_lens):
                #     mask = (ang_quant == bin)
                #     #hog_opfl[bin] = np.sum(mask*mag)
                #     hog_opfl[bin] = np.sum(mask)
                # #hog_l2sum = np.sqrt(np.dot(hog_opfl, hog_opfl))
                # #hog_opfl = hog_opfl/hog_l2sum
                # hog_opfl = hog_opfl/hog_opfl.size
                # hsv[..., 0] = ang * 180 / np.pi / 2
                # hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                # rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

                ## compute hog information.
                # fd, hog_image = hog(frame, orientations=8, pixels_per_cell=(16, 16),
                #                     cells_per_block=(1, 1), visualize=True, multichannel=False)
                # ffd = np.abs(np.fft.fft(fd))
                # sel_ffd = ffd[1:int(len(ffd)/2)]
                # Rescale histogram for better display
                #hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
                # cfar parameters for 1998
                # blob_bb_list, bin_image = segmentation(frame, lbp_contrast_select=False,
                #                                        kval=1., least_wh=(4, 4), min_area=16, max_area=400, nref=4,
                #                                        mguide=2)

                #for 93hh, 93vv, and 98hv
                lbp_var = local_binary_pattern(frame, P=80, R=10, method='var')
                lbp_var[np.isnan(lbp_var)] = 0
                lbp_var_hist, lpb_var_bins = np.histogram(lbp_var.ravel(), bins=500, density=False)
                lbp_var_hist = lbp_var_hist / lbp_var.size

                # for 93hh, 93vv, and 98hv
                # divide the frame into sub grid.
                lbp_frame = local_binary_pattern(frame, P=8, R=1, method='nri_uniform')
                #lbp_frame = local_binary_pattern(frame, P=64, R=5, method='default')
                # # #side_lbp_hist, lpb_bins = np.histogram(side_lbp.ravel(), bins=10, density=True)
                lbp_hist, lpb_bins = np.histogram(lbp_frame.ravel(), bins=59, density=False)
                #norm_val = np.sqrt(np.dot(lbp_hist, lbp_hist))
                lbp_hist = lbp_hist / lbp_frame.size
                # filter_lbp = frame[bin_image>0]
                # lbp_hist, lpb_bins = np.histogram(filter_lbp.ravel(), bins=500, density=False)
                # lbp_hist = lbp_hist / filter_lbp.size
                feature_hist = lbp_var_hist
                #feature_hist = hog_opfl
                #feature_hist = lbp_hist
                #feature_hist = np.concatenate([lbp_var_hist, lbp_hist])
                #feature_hist = np.concatenate([hog_opfl, lbp_var_hist])
                xfeatures.append(feature_hist)
                # xfeatures_var.append(lbp_var_hist)
                # xfeatures_lbp.append(lbp_hist)
                if frame_no == primary_rangebin:
                    ylabels.append(-1)  # Abnormal Mark for the primary cell with target
                else:
                    ylabels.append(1)

                #Visualize feature
                if bdraw:
                    frm_axs[0].imshow(frame)
                    frm_axs[1].imshow(lbp_var)
                    frm_axs[2].cla()
                    frm_axs[2].imshow(frame)
                    # for j in range(0,frame.shape[0],5):
                    #     for i in range(0,frame.shape[1],5):
                    #         dy = v[j, i]
                    #         dx = u[j, i]
                    #         #frm_axs[2].arrow(i, j, dy, dx, head_width=0.3, head_length=0.2, color='red')
                    #         frm_axs[2].arrow(i, j, dx, dy, head_width=0.3, head_length=0.2, color='white')

                    if frame_no == primary_rangebin:
                        frm_axs[0].set_title('%s_%d'%(filekey,frame_no), color='red')
                        #hist_ax[0].plot(ffd[1:-1], 'r-.', label=frame_no)
                        #hist_ax[0].plot(hog_opfl, 'r-.', label=frame_no)
                        hist_ax[1].plot(lpb_var_bins[:-1], lbp_var_hist, 'r-.',label=frame_no)
                    else:
                        frm_axs[0].set_title('%s_%d'%(filekey,frame_no), color='black')
                        #hist_ax[0].plot(ffd[1:-1], label=frame_no)
                        #hist_ax[0].plot(hog_opfl, label=frame_no)
                        hist_ax[1].plot(lpb_var_bins[:-1], lbp_var_hist, label=frame_no)
                    hist_ax[0].set_title(filekey)
                    hist_ax[1].set_title('lbp_var')
                    hist_ax[0].legend()
                    plt.draw()
            clf = OneClassSVM(nu=0.4, kernel='rbf', gamma='auto')
            clf.fit(xfeatures)
            y_score = clf.score_samples(xfeatures)
            y_test = clf.predict(xfeatures)
            least_votes = np.argsort(y_score)
            sort_yscore = np.sort(y_score)
            minvote = least_votes[0]
            maxvote = least_votes[-1]
            print(least_votes)
            #print(ylabels)
            # for id in range(len(ylabels)):
            #     print('%2d %2d %.20f'%(ylabels[id], least_votes[id], sort_yscore[id]))
            str_res = ''
            test_files += 1
            if (ylabels[minvote] == -1) or (ylabels[maxvote] == -1):
            #if least_votes[ylabels.index(-1)]==ylabels.index(-1):
                correct_rate += 1
                str_res = 'correct'
                #print('correct vote in %s-%s: %d'%(filekey, txmode, ylabels.index(-1)))
            else:
                str_res = 'wrong'
                #print('wrong vote in %s-%s: least %d - highest %d - gt %d'
                #      %(filekey, txmode, least_votes[0],least_votes[-1], ylabels.index(-1)))
            print(fileyear, filekey, txmode, 'gt_%2d'%ylabels.index(-1), str_res)
        # Compute the detection rate for one mode
        print('Mode %s detection rate is %.2f'% (txmode, (correct_rate/test_files)))

def monitor_cell_change():
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

    # fig, axs = plt.subplots(2, 1)
    bdraw = False
    fig_frm, frm_axs = plt.subplots(nrows=2, sharex=True, sharey=True)
    for txmode in ['hh', 'hv', 'vh', 'vv']:
        correct_rate = 0.
        correct_var_rate = 0.
        correct_lbp_rate = 0.
        test_files = 0.
        for findx, fileName in enumerate(file_names):  # for files in ipix1993

            filekey = fileName.split('_')[1]
            fileyear = fileName.split('_')[0][:4]
            primary_rangebin = primaries[findx] - 1
            if fileyear != '1993':  # omit 1998 for comparision
                continue
            # print(fileyear, filekey, txmode)

            image_path = '%s/%s_%s/%s/' % (fileprefix, fileyear, filekey, txmode)
            image_names = glob.glob(image_path + '*.png')
            image_names.sort()
            ylabels = []
            xfeatures = []
            if bdraw:
                fig_hist, hist_ax = plt.subplots(nrows=2, sharex=False)
            for iname in image_names:  # 14 frames for each range cell
                fname_split = iname.split('/')
                frame_no = int(fname_split[-1].split('.')[0])

                # omit the secondary range cells
                # if (secondary_st[findx] - 1) <= frame_no <= (secondary_nd[findx] - 1):
                #     if frame_no != primary_rangebin:
                #         continue

                if frame_no != primary_rangebin:
                    continue
                frame = cv2.imread(iname, 0)
                lbp_frame = local_binary_pattern(frame, P=8, R=1, method='nri_uniform')
                frm_axs[0].imshow(frame)
                frm_axs[1].imshow(lbp_frame)
                plt.show()

def detect_abnormal_cellid(xfeatures, range_cell_ids):
    '''
    Detect the abnormal rangebin from the xfeatures.
    Return the sorted range_cell_id, the least score's range_cell_id ranks first.
    Also returned the range_score for each range_rank.
    :param slice_feature_dict:
    :return:
    '''
    #clf = OneClassSVM(nu=0.1, kernel="poly", degree=3, gamma='auto')
    clf = OneClassSVM(nu=0.4, kernel='rbf', gamma='auto')
    clf.fit(xfeatures)
    y_score = clf.score_samples(xfeatures)
    least_votes  = np.argsort(y_score)
    range_cell_ids_array = np.array(range_cell_ids)
    range_rank   = range_cell_ids_array[least_votes]
    range_score  = np.sort(y_score)
    return range_rank, range_score


def collect_frame_features(frame_names, Pvar = 80, Rvar=10, Plbp=8, Rlbp=1):
    '''
    Extract features from the frames in the frame_names list.
    :param frame_names:
    :param range_cell_ids:
    :param Pvar:
    :param Rvar:
    :param Plbp:
    :param Rlbp:
    :return:
    '''
    lbp_features = []
    var_features = []
    range_cell_ids = []

    for iname in frame_names:
        fname_split = iname.split('/')
        frame_no = int(fname_split[-1].split('.')[0])
        frame = cv2.imread(iname, 0)

        dn_frame = frame.copy()
        from skimage.filters import gabor
        from scipy import ndimage as ndi
        from skimage.filters import gabor_kernel
        # kernel = np.real(gabor_kernel(0.1, theta=0,
        #                               sigma_x=1, sigma_y=1))
        # kernel = np.real(gabor_kernel(0.1, bandwidth=0.1))
        #frame = ndimage.gaussian_filter(dn_frame, sigma=0.05)
        #frame = ndi.convolve(dn_frame, kernel, mode='wrap')
        # kernel = np.ones((1, 1), np.uint8)
        # frame = cv2.erode(dn_frame, kernel, iterations=1)

        #frame = frame[:, :150]

        lbp_var = local_binary_pattern(frame, P=Pvar, R=Rvar, method='var')
        lbp_var[np.isnan(lbp_var)] = 0
        lbp_var_hist, lpb_var_bins = np.histogram(lbp_var.ravel(), bins=500, density=False)
        lbp_var_hist = lbp_var_hist / lbp_var.size
        var_features.append(lbp_var_hist)

        lbp_frame = local_binary_pattern(frame, P=Plbp, R=Rlbp, method='uniform')
        # # #side_lbp_hist, lpb_bins = np.histogram(side_lbp.ravel(), bins=10, density=True)
        bin_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        #weights   = np.array([2, 1, 1, 3, 3, 1, 1, 1, 2])
        weights   = np.array([2, 1, 1, 3, 3, 1, 1, 1, 2])
        #weights = np.array([1.5, 1, 1, 3, 3, 1, 1, 1, 1.5])
        weights = weights/np.sum(weights)
        lbp_hist, lpb_bins = np.histogram(lbp_frame.ravel(), bins=int(np.max(lbp_frame)), density=False)
        #lbp_hist = lbp_hist / lbp_frame.size
        lbp_hist_wt = lbp_hist * weights
        lbp_hist = lbp_hist_wt/np.sum(lbp_hist_wt)

        #
        # lbp_hist = np.concatenate((lbp_hist, lbp_hist2, lbp_hist3))
        lbp_features.append(lbp_hist)
        range_cell_ids.append(frame_no)
    # dev_lbp_features = np.array(lbp_features)
    # mean_lbp_features = np.mean(dev_lbp_features, 0)
    # dev_lbp_features = np.abs(dev_lbp_features - mean_lbp_features)

    return lbp_features, var_features, range_cell_ids

def ave_rank_score(rank_mat, score_mat, range_cell_num):
    '''
    For each ceil id, compute the average scores based on the score_mat [range_cell_num-2, range_cell_num-2]
    :param rank_mat:
    :param score_mat:
    :return:
    '''

    ave_score_list = []
    for i in range(range_cell_num):
        mask = (rank_mat==i)
        ave_score = np.sum(score_mat[mask])/mask.size
        if ave_score == 0:
            # i is not in the range_mat
            ave_score = float('inf')
        ave_score_list.append(ave_score)
    return np.array(ave_score_list)




if __name__=="__main__":
    #frame_texture_extract()
    #monitor_cell_change()
    #exit(0)
    #optical_flow_extract()
    #mcf_track_ipix()
    #exit(0)
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

    # fig, axs = plt.subplots(2, 1)
    fig_frm, frm_axs = plt.subplots(nrows=3, sharex=True)
    for txmode in ['hh', 'hv', 'vh', 'vv']:
        correct_rate = 0.
        correct_var_rate = 0.
        correct_lbp_rate = 0.
        test_files = 0.
        for findx, fileName in enumerate(file_names):
            filekey = fileName.split('_')[1]
            fileyear = fileName.split('_')[0][:4]
            primary_rangebin = primaries[findx] - 1
            # if fileyear != '1998': #omit 1998 for comparision
            #     continue
            #print(fileyear, filekey, txmode, 'gt_cell: %2d'%primary_rangebin)
            image_path = '%s/%s_%s/%s/' % (fileprefix, fileyear, filekey, txmode)
            image_names = glob.glob(image_path + '*.png')
            image_names.sort()
            #setup the ground_truth labels
            ylabels = np.ones(len(image_names))
            ylabels[primary_rangebin] = -1

            range_cell_num = len(image_names)

            lbp_range_rank_list  = []
            lbp_range_score_list = []
            var_range_rank_list  = []
            var_range_score_list = []
            #each time select one range and omit the two neighbour cell.
            # if fileyear == '1993':
            #     range_cell_up = range_cell_num-1
            # if fileyear == '1998':
            #     range_cell_up = range_cell_num-2
            # for fid in range(1,range_cell_up):
            #     select_frames = image_names.copy()
            #     del select_frames[fid-1:fid+2:2]  #delete the fid's before and after frame.
            #     if fileyear=='1998':
            #         del select_frames[fid] # delete one more cell in 1998
            #
            #     lbp_features, var_features, range_cell_ids = \
            #         collect_frame_features(select_frames,Pvar=80, Rvar=8, Plbp=8, Rlbp=1)
            #     #print(range_cell_ids)
            #     lbp_range_rank, lbp_range_score = detect_abnormal_cellid(lbp_features, range_cell_ids)
            #     var_range_rank, var_range_score = detect_abnormal_cellid(var_features, range_cell_ids)
            #     lbp_range_rank_list.append(lbp_range_rank)
            #     lbp_range_score_list.append(lbp_range_score)
            #     var_range_rank_list.append(var_range_rank)
            #     var_range_score_list.append(var_range_score)

            # # omit the secondary range cells
            secondary_cells = []
            for delid in range(secondary_st[findx] - 1, secondary_nd[findx]):
                if delid != primary_rangebin:
                    secondary_cells.append(delid)

            selected_images = []
            ylabels         = []
            for id, image_name in enumerate(image_names):
                if id not in secondary_cells:
                    selected_images.append(image_name)
                    if id == primary_rangebin:
                        ylabels.append(-1)
                    else:
                        ylabels.append(1)

            lbp_features, var_features, range_cell_ids = \
                collect_frame_features(selected_images, Pvar=80, Rvar=8, Plbp=8, Rlbp=1)
            # lbp_features, var_features, range_cell_ids = \
            #     collect_frame_features(image_names, Pvar=80, Rvar=8, Plbp=8, Rlbp=1)
            # print(range_cell_ids)
            lbp_range_rank, lbp_range_score = detect_abnormal_cellid(lbp_features, range_cell_ids)
            var_range_rank, var_range_score = detect_abnormal_cellid(var_features, range_cell_ids)

            lbp_range_rank_list.append(lbp_range_rank)
            lbp_range_score_list.append(lbp_range_score)
            var_range_rank_list.append(var_range_rank)
            var_range_score_list.append(var_range_score)

            lbp_range_rank_mat  = np.array(lbp_range_rank_list)
            lbp_range_score_mat = np.array(lbp_range_score_list)
            var_range_rank_mat  = np.array(var_range_rank_list)
            var_range_score_mat = np.array(var_range_score_list)

            #compute the abnormal scores for each range cell.
            lbp_ave_score = ave_rank_score(lbp_range_rank_mat, lbp_range_score_mat, range_cell_num)
            var_ave_score = ave_rank_score(var_range_rank_mat, var_range_score_mat, range_cell_num)
            lbp_cell_id   = np.argmin(lbp_ave_score)
            var_cell_id   = np.argmin(var_ave_score)

            weighted_score = 0.1*lbp_ave_score + 0.9*var_ave_score # [hh, hv, vh, vv]  = [0.7   0.9   0.8  0.7]
            weighted_score = 0.9*lbp_ave_score + 0.1*var_ave_score # [hh, hv, vh, vv]  = [0.75, 0.85, 0.9, 0.7]
            weighted_score = lbp_ave_score
            weighted_score_rank = np.argsort(weighted_score)
            target_cell_id   = np.argmin(weighted_score)

            # select the most voted cell as the target cell
            # vstack_rank_mat = np.concatenate((lbp_range_rank_mat, var_range_rank_mat))
            # vote_rank_vec   = vstack_rank_mat[:,0]
            # target_cell_id  = np.argmax(np.bincount(vote_rank_vec))

            # print(fileyear, filekey, primary_rangebin)
            # print('lbp', lbp_range_rank)
            # print('var', var_range_rank)
            test_files += 1
            str_res = ['W', 'C']

            btarget = False
            blbp_target = False
            bvar_target = False
            target_cells = np.arange(secondary_st[findx]-1, secondary_nd[findx])
            if (target_cell_id in target_cells):
                correct_rate += 1
                btarget = True
            if (lbp_cell_id in target_cells):
                correct_lbp_rate += 1
                blbp_target =True
            if (var_cell_id in target_cells):
                correct_var_rate += 1
                bvar_target = True

            # if (target_cell_id == primary_rangebin):
            #     correct_rate += 1
            #     btarget = True
            # if (lbp_cell_id==primary_rangebin):
            #     correct_lbp_rate +=1
            #     blbp_target =True
            # if (var_cell_id==primary_rangebin):
            #     correct_var_rate +=1
            #     bvar_target = True
            print(fileyear, filekey, txmode, '(gt,lbp_id, var_id, tid: %2d %2d %2d %2d)'
                  %(primary_rangebin, lbp_cell_id, var_cell_id, target_cell_id),
                  str_res[blbp_target],
                  str_res[bvar_target],
                  str_res[btarget])
            # if (ylabels[target_cell_id] == -1):
            #     correct_rate += 1
            # if (ylabels[lbp_cell_id]==-1):
            #     correct_lbp_rate +=1
            # if (ylabels[var_cell_id]==-1):
            #     correct_var_rate +=1
            # print(fileyear, filekey, txmode, '(gt,lbp_id, var_id, tid: %2d %2d %2d %2d)'
            #       %(primary_rangebin, lbp_cell_id, var_cell_id, target_cell_id),
            #       str_res[ylabels[lbp_cell_id]==-1],
            #       str_res[ylabels[var_cell_id]==-1],
            #       str_res[ylabels[target_cell_id] == -1])
            # #print('')
            # Compute the detection rate for one mode
        print('Mode %s detection rate is %.2f-%.2f-%.2f' % (txmode,
                                                            (correct_lbp_rate / test_files),
                                                            (correct_var_rate / test_files),
                                                            (correct_rate / test_files)))
        print('')