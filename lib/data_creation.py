# data_creation.py
# Zhao Shen, Arun Pidugu, Arnav Ghosh
# 30th Nov. 2018

# TODO: Use in conjunction with code from RGB-N

import cv2
import json
import numpy as np
import time, os, sys
import tensorflow as tf

from datasets.factory import get_imdb
from model.config import cfg, cfg_from_file, cfg_from_list
from model.test import _get_blobs
from nets.resnet_fusion import resnet_fusion

import data_manipulation

# ============== DIRECTORIES ============== #
DATA_DIR = "data_copy_move"
Y_DATA_DIR = DATA_DIR
X_DATA_DIR = os.path.join(DATA_DIR, "feature_maps")
LABELS_PATH = "labels"

def create_dataset(image_dir, model_chkpt_path, img_processed_truth_paths):
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    with tf.Session(config=tfconfig) as sess:
        net = resnet_fusion(batch_size=1, num_layers=101) # Using code from lib/model/test.py

        # Load Model (instead of .json)
        net.create_architecture(sess, "TEST", 2, tag='default',
                                anchor_scales=cfg.ANCHOR_SCALES,
                                anchor_ratios=[8, 16, 32, 64])

        saver = tf.train.Saver()
        saver.restore(sess, model_chkpt_path) # Load weights

        all_image_files = os.listdir(image_dir)

        with open(img_processed_truth_paths, 'r') as fi_file:
            im_processed_truth = json.load(fi_file)

        img_rois = []
        for i, image_path in enumerate(all_image_files):
            try:
                print(i)
                if image_path in im_processed_truth:
                    roi_features, rois = run_image(sess, net, os.path.join(image_dir, image_path))
                    reduecd_roi_indices = reduce_data(rois, im_processed_truth[image_path])

                    if reduecd_roi_indices not None:
                        for roi_i, idx in enumerate(list(map(int, reduecd_roi_indices['gt'] + reduecd_roi_indices['other']))):
                            # save data
                            filename = "roi_feature_map_{0}_{1}".format(i, roi_i)
                            #print(os.path.join(X_DATA_DIR, filename))
                            np.save(os.path.join(X_DATA_DIR, filename), roi_features[idx, :, :, :])

                        # save rois
                        img_rois.append(rois)
                
            except Exception as e:
                print(e)
            
            #periodically save img_rois
            if i % 500 == 0:
                np.save(os.path.join(Y_DATA_DIR, LABELS_PATH), img_rois)

def run_image(sess, net, file_name):
    im = cv2.imread(file_name)
    blobs, im_scales = _get_blobs(im)

    assert len(im_scales) == 1, "Only single-image batch implemented"

    im_blob = blobs['data']
    blobs['im_info'] = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)

    roi_features, rois = net.get_roi_features(sess, blobs['data'],blobs['noise'], blobs['im_info'])
    return roi_features, rois

# {"roi_featre_map_0" : {"gt" : [true, tamp], "other" : [1,2,7,8,9,11]}}
def reduce_data(rois, roi_truths):
    true_ious = find_ious(roi, roi_truths['truth'][0]).reshape(-1)
    tamp_ious = find_ious(roi, roi_truths['truth'][1]).reshape(-1)

    true_fm_idx = np.argmax(true_ious)
    tamp_fm_idx = np.argmax(tamp_ious)

    if true_fm_idx != tamp_fm_idx:
        exclude_idx = [true_fm_idx, tamp_fm_idx]
        keep_true_idxs = data_manipulation.sample_values(true_ious, 4, exclude_idx)
        keep_tamp_idxs = data_manipulation.sample_values(tamp_ious, 4, exclude_idx + keep_true_idxs)
        other_idx = keep_true_idxs + keep_tamp_idxs
        assert true_fm_idx not in other_idx
        assert tamp_fm_idx not in other_idx
        assert len(set(other_idx)) == len(other_idx)

        return {"gt" : [str(true_fm_idx), str(tamp_fm_idx)], "other" : list(map(str, other_idx))}
    else:
        return None

# ============== DEBUGGING ============== #
def test_net_creation(file_name, model_chkpt_path):
    #imdb = get_imdb(imdb_name)
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True

    with tf.Session(config=tfconfig) as sess:
        net = resnet_fusion(batch_size=1, num_layers=101)

        # Load Model (instead of .json)
        net.create_architecture(sess, "TEST", 2, tag='default',
                                anchor_scales=cfg.ANCHOR_SCALES,
                                anchor_ratios=[8, 16, 32, 64])

        saver = tf.train.Saver()
        saver.restore(sess, model_chkpt_path) # Load weights
        run_image(sess, net, file_name)

# data_creation.create_dataset("../filter_tamper_copy_move", "../output/coco_flip_0001_bilinear_new/coco_train_filter_single/EXP_DIR_coco_flip_0001_bilinear_new/res101_fusion_faster_rcnn_iter_60000.ckpt", "../new_data/new_labels.json")

