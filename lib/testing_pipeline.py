# data_creation.py
# Arnav Ghosh, Arun Pidugu, Zhao Shen
# 3rd Dec. 2018

import cv2
from keras.models import model_from_json
import numpy as np
import tensorflow as tf

from model.config import cfg, cfg_from_file, cfg_from_list
from model.test import _get_blobs
from nets.resnet_fusion import resnet_fusion

# ============ CONSTANTS ============ #
AUGMENTED = "A"
RGB_N = "R"

COPY_MOVE = "CM"
REMOVAL = "REM"
SPLICED = "SP"

# ============ TESTING PARAM. ============ #
MODEL_JSON_PATH = "" #TODO: Add
MODEL_WEIGHTS_PATH = "" #TODO: Add
KEEP_FIRST_K_CHANNELS = 5 #TODO: Change
IS_DIST = True #TODO: Change

# ============ AUG-RGB-N SETUP FUNC. ============ #
""" Assumes that the model takes a list of exactly two inputs """
def load_model():
    model = model_from_json(MODEL_JSON_PATH)
    model.load_weights(MODEL_WEIGHTS_PATH)
    return model

# roi_features: num_rois x w x h x channels (eg: 100 x 7 x 7 x 1024)
def run_aug_model(model, roi_features):
    predictions = get_all_model_comparisons(roi_features)

    if IS_DIST:
        # TODO: Use softmax or sigmoids?
        pass
    else:
        # If already probabilitiies, does it need to be squashed in anyway?
        pass

    # TODO: Assign Probs to one of the feature maps

""" Returns a list of tuples where each tuple is as follows:
    ((i, j), score) where i, j are the rois being compared and
    the score is a distance or probability value indicating how sim. they are """
def get_all_model_comparisons(model, roi_features):
    num_rois = roi_features.shape[0]

    predictions = [] #[((i, j), dist/probs) ... ]
    for i in range(num_rois):
        for j in range(i + 1, num_rois):
            inputs = [roi_features[i : i + 1, :, :, :KEEP_FIRST_K_CHANNELS], 
                      roi_features[j : j + 1, :, :, :KEEP_FIRST_K_CHANNELS]]
            predictions.append(((i, j), model.predict(inputs)))

    return predictions

# ============ RGB-N RUN FUNC. ============ #
def setup_paper_model(model_chkpt_path, sess):
    net = resnet_fusion(batch_size=1, num_layers=101) # Using code from lib/model/test.py

    # Load Model (instead of .json)
    net.create_architecture(sess, "TEST", 2, tag='default',
                            anchor_scales=cfg.ANCHOR_SCALES,
                            anchor_ratios=[8, 16, 32, 64])

    saver = tf.train.Saver()
    saver.restore(sess, model_chkpt_path) # Load weights
    return net

def run_paper_model(sess, net, im):
    blobs, im_scales = _get_blobs(im)

    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']
    blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    roi_features, rois, classes, bb_delta = net.get_roi_and_preds(sess, blobs['data'],blobs['noise'], blobs['im_info']) #TODO
    return roi_features, rois, classes, bb_delta

# ============ TESTING FUNC. ============ #
""" Passes the image through the original
    and augmented network and obtains the 'predictions'
    from each. For RGB-N and augmented-RGB-N, this is 
    the class labels and bbox for each region. 

    Returns (Aug-RGB-N, RGB-N), where Aug-RGB-N
    is [(cls, bbox), ...]. Similarly for RGB-N"""
def get_all_predictions(sess, rgb_net, aug_model, ims):
    #TODO
    roi_features, rois, classes, bboxes = run_paper_model(sess, net, ims)
    run_aug_model(aug_model, roi_features) #TODO

    return 


""" Refines the predictions from the Aug-RGB-N and RGB-N
    methods by performing NMS, Thresholding. Collects necessary stats. 

    Returns [(cls, bbox) ..], stats --> TBD Dictionary"""
def refine_model_predictions(aug_preds, rgb_model_preds):
    # TODO: Add stats calculations here for different things
    #       (eg: )
    # keep tracking of model 
    ann_aug_preds = list(map(lambda x : (x[0], x[1], AUGMENTED), aug_preds))
    ann_rgb_preds = list(map(lambda x : (x[0], x[1], RGB), rgb_model_preds))
    sorted_preds = ann_aug_preds + ann_rgb_preds
    sorted_preds.sort(key = lambda t : t[0]) #sort by class scores

    #TODO non-max suppression, add thresholds

def run_image(sess, rgb_net, aug_model, ims):
    aug_model, rgb_model = get_all_predictions(sess, rgb_net, aug_model, ims)
    preds, stats = refine_model_predictions(aug_model, rgb_model)

def main(model_chkpt_path):
    tfconfig = tf.ConfigProto(allow_soft_placement = True)
    tfconfig.gpu_options.allow_growth = True

    with tf.Session(config = tfconfig) as sess:
        net = setup_paper_model(model_chkpt_path, sess)
        aug_model = load_model()
        pass

# ============ METRICS ============ #
""" Returns the pixel-wise F1 score.
    
    Output Mask and ground truth must be 2D arrays of 1, 0s
    with 1s (tampered) indicate a positive class and 0s indicating 
    the negative class (background). """
def calculate_pixel_F1(output_mask, gt_mask):
    true_positive_regions = gt_mask == 1
    true_positives = np.sum(output_mask[true_positive_regions])

    false_negatives = len(np.where(output_mask[true_positive_regions] == 0))

    false_positive_regions = output_mask == 1
    false_positives = len(np.where(gt_mask[false_positive_regions] == 0))

    f1 = (2.0 * true_positives) / ((2.0 * true_positives) + false_positives + false_negatives)
    return f1

# ============ DATASET HELPERS ============ #
def load_casia(casia_test_dir_path):
    all_files = os.listdir(casia_test_dir_path)
    # use a list because no set image size
    dataset = []
    metadata = {"op" : [], "processing" : [], "tmp_size" : [], "from_cat" : [], "to_cat" : []} #CASIA SPECIFIC
    dataset_gt = []

    # "Tp_D_NRN_S_N_sec00100_ani00005_00708.tif" is showing a tampered 
    # image no.00708 that generated from copy a contour of a bird from an animal authentic set (no. 00005) 
    # then resized and paste to a authentic image in scene category (no.10139) 
    # without rotation and deformation without blurring operation.
    for file_name in all_files:
        img = load_image(os.path.join(casia_test_dir_path, file_name))

        split_name = file_name.split("_")
        assert split_name[0] == "Tp", "Make sure all the images used are actually tampered images."

        dataset.append(img)
        metadata["op"].append(COPY_MOVE if split_name[1] == "S" else SPLICED)
        metadata["processing"].append(split_name[2])
        metadata["tmp_size"].append(split_name[3])

        # assign cats
        metadata["from_cat"].append(split_name[6][:3])
        metadata["to_cat"].append(split_name[5][:3])

        # get ground truths (must be precomputed)
        pass #TODO

    return dataset_gt, gt, metadata

def load_cover(cover_test_dir_path):
    all_files = os.listdir(os.path.join("images", cover_test_dir_path))
    # use a list because no set image size
    dataset = []
    metadata = {} #COVER SPECIFIC
    dataset_gt = []

    for file_name in all_files:
        fname, fext = os.path.splitext(file_name)

        if fname[-1] == "t":
            img = load_image(os.path.join(os.path.join("images", cover_test_dir_path), file_name))
            gt = load_image(os.path.join(os.path.join("mask", cover_test_dir_path), "{0}forged.tif".format(fname[:-1])))

            #TODO Check if gt needs to be compressed
            assert gt.ndim == 2

            dataset.append(img)
            dataset_gt.append(gt)

    return dataset, gt, metadata

# ============ HELPERS ============ #
def load_image(file_name):
    return cv2.imread(file_name)

def find_ious(rois, bbox):
    broad_bbox = np.repeat(np.array(bbox).reshape(1, len(bbox)), rois.shape[0], axis = 0)

    broad_bbox_area = (broad_bbox[:, 2:3] - broad_bbox[:, 0:1]) * (broad_bbox[:, 3:] - broad_bbox[:, 1:2])
    rois_area = (rois[:, 2:3] - rois[:, 0:1]) * (rois[:, 3:] - rois[:, 1:2])

    #intersection
    x_left = np.maximum(broad_bbox[:, 0], rois[:, 0]).reshape(rois.shape[0], 1)
    y_top = np.maximum(broad_bbox[:, 1], rois[:, 1]).reshape(rois.shape[0], 1)
    x_right = np.minimum(broad_bbox[:, 2], rois[:, 2]).reshape(rois.shape[0], 1)
    y_bot = np.minimum(broad_bbox[:, 3], rois[:, 3]).reshape(rois.shape[0], 1)

    intersection_area = (x_right - x_left) * (y_bot - y_top)

    ious = np.divide(intersection_area.reshape(-1), (broad_bbox_area + rois_area - intersection_area).reshape(-1))

    #make all horizontal or vertical rois = 0:
    non_rect_rois = np.logical_or(rois[:,0] == rois[:, 2], rois[:,1] == rois[:, 3])
    ious[non_rect_rois] = 0

    #ensure that two disjoint boxes have zero iou
    no_overlap = np.logical_or(x_right < x_left, y_bot < y_top).reshape(-1)
    ious[no_overlap] = 0

    #what about tie breaking
    return ious

def locate_truth_distance(true_idx, tmp_idx, preds):
    idxs = [true_idx, tmp_idx]
    idxs.sort()

    idxs = tuple(idx)
    truth_idx = None
    for i in range(len(preds)):
        if preds[i][0] == idxs:
            return preds[i]

#siam_model = load_model(mod_loc, custom_objects = {'contrastive_loss' : contrastive_loss, 'cont_accuracy' : cont_accuracy})



# TODO:
# 1. Take Image
# 2. Run Image through model
#    - grab the roi features --> pass these through our model
# 3. Obtain class scores from both model

# What's the policy to assign things

# TODO:
# Test on COCO Synthetic & CASIA (possibly Cover)
# Check what the values of the returned bboxes are (is the shortest part still 600)