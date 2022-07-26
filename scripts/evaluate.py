#!/usr/bin/env python3

import argparse
from cProfile import label
import json
from statistics import mean
import os
import sys
from numpy import append
import numpy as np
from sklearn import metrics
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import torch
import torch.nn as nn

from dataset_interface.siamese_net.model import SiameseNetwork
from dataset_interface.siamese_net.utils import get_image_tensor
from inference import KNN, get_label, load_data

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-m', '--model_path', type=str,
                           help='Path to a trained model',
                           default='/home/dlrv-ss22-face-recognition/workspace/face-recognition-models/model_9.pt')
    argparser.add_argument('-d', '--dataset_path', type=str,
                           help='Path to a trained model',
                           default='/home/dlrv-ss22-face-recognition/workspace/data/person-images2')

    argparser.add_argument('-t', '--testset_path', type=str,
                        help='Path to a trained model',
                        default='/home/dlrv-ss22-face-recognition/workspace/data/person-images2')

    args = argparser.parse_args()
    model_path = args.model_path
    dataset_path = args.dataset_path
    testset_path = args.testset_path

    knn = KNN(dataset_path, model_path)

    testset, test_labels = load_data(testset_path)
    class_names = np.unique(test_labels).tolist()
    eval = {}
    for class_name in class_names:
        eval[class_name] = {'TP': 0, 'FN': 0, 'FP': 0, 'recall': 0, 'precision': 0}

    for test_img in testset:
        prediction = knn.predict(test_img, 3, 0.1)
        label = get_label(test_img)
        if label == prediction:
            eval[label]['TP'] += 1

        else:
            eval[label]["FN"] += 1
            if prediction != "unknown":
                eval[prediction]["FP"] += 1
    
    for key in eval:
        tp = eval[key]['TP']
        fn = eval[key]['FN']
        fp = eval[key]['FP']
        eval[key]['precision'] =  tp / (tp + fp)
        eval[key]['recall'] =  tp / (tp + fn)

    print(eval)

