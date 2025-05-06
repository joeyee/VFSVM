import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_curve, auc
# Local Binary Pattern function
from skimage.feature import local_binary_pattern
import time
import utilities_200611 as uti
import cfar_segmentation_200527 as cfar_model     # cfar
import load_json_200616         as ld_json


from skimage.feature import hog
from skimage import transform
import glob
import cv2

def train_svm_clf(positive_sample_feature, positive_sample_weights,
                  negative_sample_feature, negative_sample_weights, Cval=10):
    '''
    train a linear svm by specified positive_sample_features and neg.
    :param positive_sample_feature: [header+body], [header [frame_no, int(subkey), bx, by, bw, bh, ntype, bins_length 26]]:
                                                   [body: 26 length lbp_hist]
    :param negative_sample_feature:
    :return: svm classifier
    '''


    # lbp_neg_mat      = np.array(negative_sample_feature)
    # lbp_pos_mat      = np.array(positive_sample_feature)

    lbp_neg_mat      = negative_sample_feature
    lbp_pos_mat      = positive_sample_feature

    print('Training SVM with %04d positive - %04d negative samples, C value is %f'
          %(lbp_pos_mat.shape[0], lbp_neg_mat.shape[0],Cval))

    lbp_neg_hist_mat = lbp_neg_mat[:, 8:]
    lable_neg        = lbp_neg_mat[:, 6] * 0  # mark negative type as 0.

    #lbp_pos_file = open(feature_path + lbp_pos_filename, 'rb')
    #lbp_pos_dat = np.fromfile(lbp_pos_file, 'float')
    #lbp_pos_mat = lbp_pos_dat.reshape(int(lbp_pos_dat.shape[0] / 34), 34)
    #print('pos samples matrix shape ', lbp_pos_mat.shape)


    lbp_pos_hist_mat = lbp_pos_mat[:, 8:]
    lable_pos        = lbp_pos_mat[:, 6] * 1  # mark positive type as 1

    lbp_hist_mat = np.concatenate((lbp_neg_hist_mat, lbp_pos_hist_mat))
    lbp_labels   = np.concatenate((lable_neg, lable_pos))
    sample_weights = np.concatenate((negative_sample_weights, positive_sample_weights))

    # train the linearSVC classifier
    clf = LinearSVC(C=Cval, loss="hinge", random_state=42,
                    max_iter=2000).fit(lbp_hist_mat, lbp_labels, sample_weight=sample_weights)
    return clf
def select_blobs(svm_clf, over_load_blob_list, frame, frame_no, bthresh = -0.5):
    # select the blob which is classified as the target
    classified_blob_list = []
    score_list = []
    for bid, blob in enumerate(over_load_blob_list):
        bx, by, bw, bh = blob[:4]
        # uti.draw_rect(canvas, blob, color=(255, 255, 255), thick=1)  # draw negative samples in white.
        # cv2.putText(canvas, str(bid), (int(bx + bw / 2), int(by + bh + 15)),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255),
        #             thickness=1, lineType=cv2.LINE_AA)
        # Get the roi of the target from the frame.
        roi = frame[by:by + bh, bx:bx + bw]

        lbp = local_binary_pattern(roi, P=24, R=3, method='uniform')
        lbp[np.isnan(lbp)] = 0
        bhist, bins = np.histogram(lbp.ravel(), bins=26, range=(0, 25), density=True)
        label = svm_clf.predict(np.array([bhist]))
        score = svm_clf.decision_function(np.array([bhist]))
        score_list.append(score)
        #if label[0] == 1: # for target
        if score[0] >= bthresh:  #standard 0, bias -0.5  for little positive samples.
            classified_blob_list.append(blob)
    return classified_blob_list, score_list



def load_feature_labels():
    '''
    Get the freature and labels from the prepared dat file.
    the format of the dat file is as follows:
    # writing the lbp features.
    # frame_no, tid, rect(x,y,w,h), type(0 for vessel), histogram dimensions]
    target_ini = np.array([frame_no, int(subkey), bx, by, bw, bh, ntype, 26]).astype('float')
    lbp = local_binary_pattern(troi, P=24, R=3, method='uniform')
    lbp[np.isnan(lbp)] = 0
    bhist, bins = np.histogram(lbp.ravel(), bins=26, range=(0, 25), density=True)
    lbp_feature = np.concatenate((target_ini, bhist)) # len = 8 + 26
    :return:
    '''
    feature_path = '/Users/yizhou/code/inesa_it_radar_singal_process/inesa_template/'
    #hog_filename = 'inesa_201801_hog.dat'
    lbp_neg_filename = 'inesa_201801_neg_lbp_P24xR3.dat'
    lbp_pos_filename = 'inesa_201801_lbp_P24xR3.dat'

    lbp_neg_file = open(feature_path + lbp_neg_filename, 'rb')
    lbp_neg_dat = np.fromfile(lbp_neg_file, 'float')
    lbp_neg_mat = lbp_neg_dat.reshape(int(lbp_neg_dat.shape[0] / 34), 34)
    print('negative samples matrix shape ', lbp_neg_mat.shape)

    lbp_neg_hist_mat = lbp_neg_mat[:, 8:]
    lable_negs       = lbp_neg_mat[:, 6]*0 # mark negative type as 0.

    lbp_pos_file = open(feature_path + lbp_pos_filename, 'rb')
    lbp_pos_dat = np.fromfile(lbp_pos_file, 'float')
    lbp_pos_mat = lbp_pos_dat.reshape(int(lbp_pos_dat.shape[0] / 34), 34)
    print('pos samples matrix shape ', lbp_pos_mat.shape)

    lbp_pos_hist_mat = lbp_pos_mat[:, 8:]
    lable_pos        = lbp_pos_mat[:, 6]*1 # mark positive type as 1

    lbp_hist_mat = np.concatenate((lbp_neg_hist_mat, lbp_pos_hist_mat))
    lbp_labels   = np.concatenate((lable_negs, lable_pos))
    # TODO Visualize the False Positive Samples and False Negative Samples. (Done)
    # samples_mat  = np.concatenate((lbp_neg_mat, lbp_pos_mat))
    # # Train a linear svm
    # ts = time.clock()
    # clf = LinearSVC(C=10, loss="hinge", random_state=42, max_iter=2000).fit(lbp_hist_mat, lbp_labels)
    # print('svm training time is %f s' % (time.clock() - ts))
    # decision_function = clf.decision_function(lbp_hist_mat)
    # # we can also calculate the decision function manually
    # # decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]
    # # if negative samples get lables 0.
    # false_positive = ((lbp_labels == 0) * (decision_function > 0))
    # false_positive_indexs = np.where(false_positive > 0)
    # false_negative = ((lbp_labels == 1) * (decision_function < 0))
    # false_negative_indexs = np.where(false_negative > 0)
    #
    # fp_feature_mat = samples_mat[false_positive_indexs]
    # fn_feature_mat = samples_mat[false_negative_indexs]
    #
    # file_prefix = '/Users/yizhou/Radar_Datasets/RecordEcho/2018-01-24-19_05_19-1/'
    # file_names = glob.glob(file_prefix + '*.png')
    # file_names.sort()
    # file_len = len(file_names)
    # for i in range(8, file_len):
    #     fname_split = file_names[i].split('/')
    #     frame_no = int(fname_split[-1].split('.')[0])
    #     if frame_no in [20, 21]:
    #         print('Omit anormal frame %d' % frame_no)
    #         continue
    #     print('frame no %d' % frame_no)
    #     frame = cv2.imread(file_names[i], 0)  # gray_scale image
    #     # canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    #     canvas = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
    #     canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)  # for plt.imshow()
    #     draw_samples(lbp_neg_mat, frame_no, canvas, color=(0,0,255), thick=2) # draw labeled sea clutter in blue
    #     draw_samples(lbp_pos_mat, frame_no, canvas, color=(0,255,0), thick=2) # draw labeled target in green
    #     draw_samples(fp_feature_mat, frame_no, canvas, color=(255,255,0), thick=1) # draw false  positive in yellow
    #     draw_samples(fn_feature_mat, frame_no, canvas, color=(255,0,0), thick=1) #draw false negative in red.
    #     plt.imshow(canvas)
    #     plt.pause(0.001)
    #     plt.waitforbuttonpress()
    return lbp_hist_mat, lbp_labels


def draw_samples(feature_mat, frame_no, canvas, color = (255,255,255), thick=1):
    '''
    draw the rect in given color in the current frame_no on canvas
    :param feature_mat: [frame_no, tid, x,y,w,h, ntype, bin_nums]
    :param frame_no:
    :param canvas:
    :param color:
    :return:
    '''
    frame_index = feature_mat[:,0].astype('int') # get the frame_index column
    rect_index = np.where(frame_index==frame_no)[0]
    if len(rect_index)==0: # if the rect_index is empty array
        return canvas
    for rid in rect_index:
        x,y,w,h = feature_mat[rid, 2:6].astype('int')
        rect = [x, y, w, h]
        canvas = uti.draw_rect(canvas, rect, color=color, thick=thick)
    return canvas

def text_scores(feature_mat, scores, frame_no, canvas):
    frame_index = feature_mat[:,0].astype('int') # get the frame_index column
    rect_index = np.where(frame_index==frame_no)[0]
    if len(rect_index)==0: # if the rect_index is empty array
        return canvas
    for rid in rect_index:
        x,y,w,h = feature_mat[rid, 2:6].astype('int')
        rect = [x, y, w, h]
        score = '%.2f'%scores[rid]
        cv2.putText(canvas, score, (int(x + w / 2), int(y + h + 15)),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255),
                    thickness=1, lineType=cv2.LINE_AA)
    return canvas

def segmentation(frame):
    cfar_cs    = cfar_model.CFAR(kval=1)
    bin_image  = cfar_cs.cfar_seg(frame)
    (contours, _) = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    blob_bb_list  = []  #blob bounding box list

    #computing the local binary pattern for an image.
    # inv_var = local_binary_pattern(frame, P=24, R=3, method='var')
    # inv_var[np.isnan(inv_var)] = 0
    #int_img = transform.integral_image(inv_var)

    for id, contour in enumerate(contours):
        x, y, w, h= cv2.boundingRect(contour)
        bb_rect = [x,y,w,h]
        # omit the 20meters under the radar
        if (y <= 10):
            continue
        # omit small rectangle.
        if (w*h < 32) or (w<3) or (h<3):
            continue
        # iiv = transform.integrate(int_img, (y, x), (h + y - 1, x + w - 1))
        # # omit the sparse density of variance image.
        # if (iiv[0] / (w * h)) < 500:
        #     continue
        blob_bb_list.append(bb_rect)
    return blob_bb_list

def compute_feature(frame, frame_no):
    '''
    Compute featuer matrix from a frame, by the region segmented by cfar.
    :param frame:
    :return:
    '''
    blob_bb_list = segmentation(frame)
    num_blob     = len(blob_bb_list)
    lbp_feature_mat = np.zeros((num_blob, 8+26), dtype='float')

    # select the blob which is not intersected with the target as negative samples.
    for bid, blob in enumerate(blob_bb_list):
        bx, by, bw, bh = blob[:4]
        # uti.draw_rect(canvas, blob, color=(255, 255, 255), thick=1)  # draw negative samples in white.
        # cv2.putText(canvas, str(bid), (int(bx + bw / 2), int(by + bh + 15)),
        #             fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(255, 255, 255),
        #             thickness=1, lineType=cv2.LINE_AA)
        # Get the roi of the target from the frame.
        clutter_roi = frame[by:by + bh, bx:bx + bw]

        ntype = 1  # for sea clutter
        # writing the lbp features.
        # frame_no, blob id, rect(x,y,w,h), type(0 for vessel, 1 for sea clutter), histogram dimensions]
        target_ini = np.array([frame_no, bid, bx, by, bw, bh, ntype, 26]).astype('float')
        lbp = local_binary_pattern(clutter_roi, P=24, R=3, method='uniform')
        lbp[np.isnan(lbp)] = 0
        bhist, bins = np.histogram(lbp.ravel(), bins=26, range=(0, 25), density=True)
        lbp_feature = np.concatenate((target_ini, bhist))  # len = 8 + 26
        lbp_feature = lbp_feature.astype('float64')
        lbp_feature_mat[bid, :] = lbp_feature
        # # Resize the frame to different scale for computing features.
        # # frame_no, tid, rect(x,y,w,h), type(0 for vessel, 1 for sea clutter), histogram dimensions]
        # target_ini = np.array([frame_no, bid, bx, by, bw, bh, ntype, 108]).astype('float')
        # sample_template = transform.resize(clutter_roi, (16, 8))
        # fd, hog_image = hog(sample_template, orientations=9, pixels_per_cell=(4, 4),
        #                     cells_per_block=(2, 2), feature_vector=True, visualize=True, multichannel=False)
        # hog_feature = np.concatenate((target_ini, fd))  # len = 8 + 36 [8x8], len=8+ 36*3 =116 [16x8]
        # hog_feature = hog_feature.astype('float64')
    return lbp_feature_mat

def test_feature():
    # load featuers from pre_training
    X, y = load_feature_labels()
    # train the linearSVC classifier
    clf = LinearSVC(C=10, loss="hinge", random_state=42, max_iter=2000).fit(X, y)

    file_prefix = '/Users/yizhou/Radar_Datasets/RecordEcho/2018-01-24-19_05_19-1/'
    file_names = glob.glob(file_prefix + '*.png')
    file_names.sort()
    file_len = len(file_names)
    for i in range(8, file_len):
        fname_split = file_names[i].split('/')
        frame_no = int(fname_split[-1].split('.')[0])
        if frame_no in [20, 21]:
            print('Omit anormal frame %d' % frame_no)
            continue
        print('frame no %d' % frame_no)
        frame = cv2.imread(file_names[i], 0)  # gray_scale image
        # canvas = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        canvas = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)  # for plt.imshow()

        lbp_feature_mat = compute_feature(frame, frame_no)
        lbp_features    = lbp_feature_mat[:, 8:]
        lables = clf.predict(lbp_features)
        scores = clf.decision_function(lbp_features)

        positive_feature_mat = lbp_feature_mat[lables==1]
        negative_feature_mat= lbp_feature_mat[lables==0]
        positive_scores = scores[lables==1]

        draw_samples(negative_feature_mat, frame_no, canvas, color=(0, 0, 255), thick=2)  # draw labeled sea clutter in blue
        draw_samples(positive_feature_mat, frame_no, canvas, color=(0, 255, 0), thick=2)  # draw labeled target in green
        text_scores(positive_feature_mat, positive_scores, frame_no, canvas) #text the positive socres on the canvas
        plt.imshow(canvas)
        plt.title('frame ' + str(frame_no))
        plt.draw()
        plt.pause(0.0001)
        key_press = False
        while not key_press:
            key_press = plt.waitforbuttonpress()

def get_tp_np(target_dict, clf_blob_list):
    '''
    Get the true positive rectangles from the blob_list
    Get the false positive rectangles from the blob_list
    :param target_dict:  ground_truth in dict, {[nick name]:rect}
    :param clf_blob_list: classified target blob
    :return:
    '''

    tp_rects = [] # True positive rects
    fp_rects = [] # False positive rects

    for blob in clf_blob_list:
        blob_intersect = False
        for key in target_dict:
            tbb = target_dict[key]
            iou = uti.intersection_rect(tbb, blob)
            if iou > 0.5:
                tp_rects.append(blob)
                blob_intersect = True
                break
        if blob_intersect==False:
            fp_rects.append(blob)

    return tp_rects, fp_rects

def test_dr_fr(clf, p_sam_framenos, neg_sample_frame_nos, bthresh):
    '''
    compute the detection_rate (true positive rate in sklearn) and false_alarm_rate (false positive rate) in test frames
    :return:
    '''
    file_prefix = '/Users/yizhou/Radar_Datasets/RecordEcho/2018-01-24-19_05_19-1/'
    file_names = glob.glob(file_prefix + '*.png')
    file_names.sort()
    file_len = len(file_names)

    test_times = 0
    ave_detect_rate = 0
    ave_false_alarm_rate = 0
    for i in range(0, file_len):
        fname_split = file_names[i].split('/')
        frame_no = int(fname_split[-1].split('.')[0])
        frame = cv2.imread(file_names[i], 0)
        canvas = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        if (frame_no not in p_sam_framenos) and \
                (frame_no not in neg_sample_frame_nos):  # testing frame no should not be the training frame_no.
            test_times += 1
            over_load_blob_list = cfar_model.segmentation(frame, lbp_contrast_select=False)
            # classify testing frame
            clf_blob_list, score_list = select_blobs(clf, over_load_blob_list, frame, frame_no, bthresh=bthresh)
            json_file = file_prefix + '/Json/%02d.json' % frame_no
            target_gt_dict = ld_json.get_gt_rect(json_file)
            # get true positive rects, false positive rects
            tps, fps = get_tp_np(target_gt_dict, clf_blob_list)

            test_pos_nums = len(target_gt_dict)
            test_neg_nums = len(over_load_blob_list) - test_pos_nums

            detection_rate   = len(tps) * 1.0 / test_pos_nums
            false_alarm_rate = len(fps) * 1.0 / (len(fps)+test_neg_nums)


            ave_detect_rate += detection_rate
            ave_false_alarm_rate += false_alarm_rate
            # for key in target_gt_dict:
            #     rect = target_gt_dict[key]
            #     uti.draw_rect(canvas, rect, color=(255, 255, 255), thick=2)  # draw gt samples in white
            # for blob in tps:
            #     uti.draw_rect(canvas, blob, color=(0,255,0), thick=2) #draw true positive samples in green
            # for blob in fps:
            #     uti.draw_rect(canvas, blob, color=(0, 255, 255), thick=2)  # draw false positive samples in yellow
            #
            # # print('Frame No. %02d train positive samples: %03d, negative samples: %04d, Cvalue %02d, bthresh %.2f' \
            # #       '\n\t detection rate %.3f, false alarm rate  %.3f, '
            # #       %(frame_no, trained_pos_nums, neg_sample_nums, cvalue, bthresh, detection_rate, false_alarm_rate))
            # canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            # plt.imshow(canvas)
            # plt.title('frame %02d' % frame_no)
            # plt.draw()
            # plt.pause(0.0001)
            # key_press = False
            # while not key_press:
            #     key_press = plt.waitforbuttonpress()
    ave_detect_rate = ave_detect_rate / test_times
    ave_false_alarm_rate = ave_false_alarm_rate / test_times
    return ave_detect_rate, ave_false_alarm_rate, test_times


def test_svm_parameters():
    '''
    In this function, test the svm_clf's parameters for best detection rate and least false alarm.
    :return:
    '''
    #load all the positive samples info and features
    positive_samples_features = ld_json.get_positive_feature_from_json()
    positive_sample_nums      = len(positive_samples_features)

    file_prefix = '/Users/yizhou/Radar_Datasets/RecordEcho/2018-01-24-19_05_19-1/'
    file_names  = glob.glob(file_prefix + '*.png')
    file_names.sort()
    file_len    = len(file_names)

    fname_split = file_names[0].split('/')
    init_frame_no = int(fname_split[-1].split('.')[0])

    neg_sample_frame_nos = [init_frame_no]
    neg_samples_features     = ld_json.get_negative_features(frame_no = neg_sample_frame_nos[0])
    neg_sample_nums          = len(neg_samples_features)
    neg_sample_weights       = np.ones(neg_sample_nums)

    #tested parameters
    pos_sample_nums_training = [8, 16, 32, 64]
    Cvalues                  = [1, 20, 50, 100]
    Bthreshs                 = [0, -0.5, -1.0, -1.5]

    for trained_pos_nums in pos_sample_nums_training:
        row_index = np.random.randint(low=0, high=positive_sample_nums-1, size=trained_pos_nums)
        p_sam_features = positive_samples_features[row_index, :]
        p_sam_weights  = np.ones(trained_pos_nums)
        p_sam_framenos = np.unique(p_sam_features[:, 0].astype('int')) # get the frame_nos for positive samples.

        for cvalue in Cvalues:
            for bthresh in Bthreshs:
                clf = train_svm_clf(p_sam_features, p_sam_weights,
                                    neg_samples_features, neg_sample_weights,
                                    Cval=cvalue)
                ave_detect_rate, ave_false_alarm_rate, test_times = test_dr_fr(clf, p_sam_framenos,
                                                                               neg_sample_frame_nos,
                                                                               bthresh = bthresh)
                print('Train positive samples: %03d, negative samples: %04d, Cvalue %02d, bthresh %.2f' \
                      '\n\t Test %03d frames, ave detection rate %.3f, ave false alarm rate  %.3f '\
                      %(trained_pos_nums, neg_sample_nums, cvalue, bthresh,
                        test_times, ave_detect_rate, ave_false_alarm_rate))

def get_radar_tpr_fpr(threshs, yscores, ytests, test_samples_features):
    '''
    compute classifier's true positive rate (tpr) and false positive rate (fpr) by set different thresholds
    yscores is the classifier's test score.
    ytests  is the ground_truth labels
    test_samples_features include's the samples' information (0-7) (frame_no, bid, rect_x, y, w, h, type, bin_length)
    and lbp feature vectors (26 elements)

    fpr = fp/(fp+tn) = fp/gt_negative
    tpr = tp/(tp+fn) = tp/gt_positive
    '''

    gt_positive_idx = (ytests==1)
    gt_positive_in_pixel  = np.sum(test_samples_features[gt_positive_idx, 4] * test_samples_features[gt_positive_idx, 5])
    gt_positive_in_rect = np.sum(gt_positive_idx)
    gt_negative_idx = (ytests==0)
    gt_negtaive_frames = np.unique(test_samples_features[gt_negative_idx, 0])
    # true negative cover all the frame size(2048*1024), including cfar's classification.
    gt_negative_in_pixel  = gt_negtaive_frames.shape[0] * 2048*1024 - gt_positive_in_pixel
    #gt_negative_in_pixel  = np.sum(test_samples_features[gt_negative_idx, 4] * test_samples_features[gt_negative_idx, 5])
    gt_negative_in_rect   = np.sum(gt_negative_idx)
    fpr = []
    tpr = []
    for th in threshs:
        classify_labels = (yscores >= th)*1
        fp_index = (classify_labels == 1)*(ytests==0)
        fp_in_pixel = np.sum(test_samples_features[fp_index, 4] * test_samples_features[fp_index, 5])
        fp_in_rect  = np.sum(fp_index)
        tp_index = (classify_labels == 1)*(ytests==1)
        tp_in_pixel = np.sum(test_samples_features[tp_index, 4] * test_samples_features[tp_index, 5])
        tp_in_rect = np.sum(tp_index)
        fpr.append(fp_in_pixel / gt_negative_in_pixel)
        tpr.append(tp_in_pixel / gt_positive_in_pixel)
        # print('tpr_in_pixel %.2e, tpr_in_rect %.2e' % ((tp_in_pixel/gt_positive_in_pixel), (tp_in_rect / gt_positive_in_rect)))
        # print('fpr_in_pixel %.2e, fpr_in_rect %.2e' % ((fp_in_pixel/gt_negative_in_pixel), (fp_in_rect / gt_negative_in_rect)))
    return np.array(fpr), np.array(tpr)

def train_test_inesa():
    '''
    Train and test  inesa data with varying imbalance positive and negative samples.
    Try different threshold for AUC test.
    :return:
    Need test parameters for the performance of SVM classifiers.

    '''
    feature_file_prefix = '/Users/yizhou/code/inesa_it_radar_singal_process/inesa_template/'
    gt_pos_lbp_features_file = open(feature_file_prefix + 'inesae_gt_pos.dat','rb')
    gt_neg_lbp_features_file = open(feature_file_prefix + 'inesae_gt_neg.dat','rb')

    positive_sample_feature = np.fromfile(gt_pos_lbp_features_file)
    negative_sample_feature = np.fromfile(gt_neg_lbp_features_file)

    positive_sample_feature = positive_sample_feature.reshape(int(positive_sample_feature.shape[0] / 34), 34)
    negative_sample_feature = negative_sample_feature.reshape(int(negative_sample_feature.shape[0] / 34), 34)

    #validate the data format is correct
    print(np.unique(positive_sample_feature[:,0]))
    print(np.unique(negative_sample_feature[:, 0]))
    #positive_sample_feature, negative_sample_feature = ld_json.get_pos_neg_feature_from_json()

    pos_sample_nums  = len(positive_sample_feature)
    trained_pos_nums = [8,  64, 128,256]
    Cvalues          = [10, 20, 30]

    # Get training negative samples based on the frame_no.
    neg_train_frame_no = np.random.randint(low=1, high=98, size=1)[0]
    neg_train_frame_no = 1
    neg_train_row_index= np.where(np.int0(negative_sample_feature[:,0])==neg_train_frame_no)[0]
    neg_train_nums = neg_train_row_index.size

    # sam_feature include samples info and feature vector
    n_train_sam_features     = negative_sample_feature[neg_train_row_index, :]
    # feature mat only include features.
    n_train_features_mat     = n_train_sam_features[:, 8:]
    n_train_weights          = np.ones(neg_train_nums)
    n_train_labels           = np.zeros(neg_train_nums)

    #Get testing negative samples by exclude the frame_no
    neg_test_row_index = np.where(np.int0(negative_sample_feature[:,0])!=neg_train_frame_no)[0]
    neg_test_nums      = neg_test_row_index.size
    n_test_sam_features     = negative_sample_feature[neg_test_row_index, :]
    # feature mat only include features.
    n_test_features_mat     = n_test_sam_features[:, 8:]
    n_test_weights          = np.ones(neg_test_nums)
    n_test_labels           = np.zeros(neg_test_nums)

    for used_pos_num in trained_pos_nums:
        #pos_train_row_index  = np.random.randint(low = 0, high = pos_sample_nums - 1, size = used_pos_num)
        pos_train_row_index  = [id for id in range(used_pos_num)]
        train_pos_nums       = len(pos_train_row_index)
        p_train_sam_features = positive_sample_feature [pos_train_row_index, :]
        p_train_features_mat = p_train_sam_features[:, 8:]
        p_train_weights      = np.ones(train_pos_nums)
        p_train_labels       = np.ones(train_pos_nums)

        pos_test_row_index = [i for i in range(pos_sample_nums) if i not in pos_train_row_index]
        test_pos_nums       = len(pos_test_row_index)
        p_test_sam_features = positive_sample_feature [pos_test_row_index, :]
        p_test_features_mat = p_test_sam_features[:, 8:]
        p_test_weights      = np.ones(test_pos_nums)
        p_test_labels       = np.ones(test_pos_nums)

        #prepare X_train, y_train and X_test and y_test
        X_train = np.concatenate((p_train_features_mat, n_train_features_mat))
        y_train = np.concatenate((p_train_labels, n_train_labels))
        train_sam_weights = np.concatenate((p_train_weights, n_train_weights))

        test_samples_features = np.concatenate((p_test_sam_features, n_test_sam_features))
        X_test = np.concatenate((p_test_features_mat, n_test_features_mat))
        y_test = np.concatenate((p_test_labels, n_test_labels))
        test_sam_weights = np.concatenate((p_test_weights, n_test_weights))

        # train the linearSVC classifier
        for Cval in Cvalues:
            clf = LinearSVC(C=Cval, loss="hinge", random_state=42,max_iter=2000)
            clf.fit(X_train, y_train, sample_weight=train_sam_weights)
            y_score = clf.decision_function(X_test)
            fpr, tpr, thresh     = roc_curve(y_test[:], y_score[:])
            radar_thresh  = np.random.normal(-1, 1, thresh.size)
            radar_thresh  =  np.array(sorted(radar_thresh, reverse=True))
            radar_fpr, radar_tpr = get_radar_tpr_fpr(radar_thresh, y_score, y_test, test_samples_features)

            ideal_thresh_id_skl  = thresh[tpr>0.9][0]
            ideal_thresh_id_radar= radar_thresh[radar_tpr>0.9][0]
            # roc_auc_skl          = auc(fpr, tpr)
            # roc_auc_radar        = auc(radar_fpr, radar_tpr)
            print('pos_train_nums %d, cval %d, tpr>0.9 thresh_skl %.3f'
                  % (used_pos_num, Cval, ideal_thresh_id_skl))
            print('fpr %e, when tpr > 0.9' % (fpr[tpr>0.9][0]))
            #print('auc_skl %.4f, auc_radar %.4f' % (roc_auc_skl, roc_auc_radar))
            str_label = 'C=%d, pnum=%2d, tpr>0.9thresh %.2f'%(Cval, used_pos_num, ideal_thresh_id_radar)
            plt.semilogx(radar_fpr, radar_tpr, lw=2, label=str_label)
            #plt.semilogx(fpr, tpr, lw=2, label=str_label)
            #plt.plot([0, 1], [0, 1], color='navy', lw=2, label='guess', linestyle='--')
            #plt.xlim([0.0, 1.0])
            #plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver operating characteristic curve')
            plt.legend(loc="lower right")
            plt.draw()
            #plt.show()
            plt.waitforbuttonpress()



if __name__=='__main__':

    train_test_inesa()
    #test_svm_parameters()

    print(__doc__)

    import numpy as np
    import matplotlib.pyplot as plt
    from itertools import cycle

    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
    from sklearn.multiclass import OneVsRestClassifier
    from scipy import interp
    from sklearn.metrics import roc_auc_score

    # Import some data to play with
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    # Binarize the output
    y = label_binarize(y, classes=[0, 1, 2])
    n_classes = y.shape[1]

    # Add noisy features to make the problem harder
    random_state = np.random.RandomState(0)
    n_samples, n_features = X.shape
    X = np.c_[X, random_state.randn(n_samples, 200 * n_features)]

    # shuffle and split training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.5,
                                                        random_state=0)

    # Learn to predict each class against the other
    classifier = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True,
                                             random_state=random_state))
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    thresh = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], thresh[i] = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, label='guess',linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
