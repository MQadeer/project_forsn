#!/usr/bin/env python3
import argparse
from statistics import mean
import os
import sys
import numpy as np
from numpy import append
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import torch
import torch.nn as nn

from siamese_net.model import SiameseNetwork
from siamese_net.utils import get_image_tensor, get_transform_norm
from torchvision import datasets
from siamese_net.utils import get_transforms
import matplotlib.pyplot as plt

class KNN():
    def __init__(self, dataset_path, testset_path, model_path, means, std):
        self.model = SiameseNetwork()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model.to(device)
        # get the dataset
        self.dataset, self.dataset_labels, _ = self.__load_data(dataset_path, means, std)
        self.testset, labels, _ = self.__load_data(testset_path, means, std)
        self.class_names = np.unique(labels.numpy()).astype(str)
        self.labels = labels.numpy().astype(str)
        
    def predict(self, k, thresh):
        predictions = []
        for i in range(self.testset.size(dim=0)):
            distances = nn.functional.pairwise_distance(self.testset[i], self.dataset)
            distances = distances.detach().cpu().numpy()
            smallest_indices = np.argpartition(distances, k)[:k]
            # apply threshold
            if distances[smallest_indices[-1]] > thresh:
                predictions.append('u')
            
            else:
                predicted_labels = self.dataset_labels.numpy()[smallest_indices].tolist()
                predictions.append(max(set(predicted_labels), key = predicted_labels.count))
        predictions = np.array(predictions).astype(str)
        # print(predictions)
        return predictions

    
    def __load_data(self, data_path, means, std):
        dataset = datasets.ImageFolder(data_path, transform=get_transform_norm(means, std))
        
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=dataset.__len__(), shuffle=False) 
        images, labels = next(iter(dataloader))
        images = images.cuda()
        latent_vectors = self.model.forward_once(images)
        return latent_vectors, labels, dataset.class_to_idx
    
    def evaluate(self, predictions):

        eval = {}
        for class_name in self.class_names:
            eval[class_name] = {'TP': 0, 'FN': 0, 'FP': 0, 'recall': 0, 'precision': 0}
        
        for i in range(self.labels.shape[0]):
            if self.labels[i] == predictions[i]:
                eval[self.labels[i]]['TP'] += 1

            else:
                eval[self.labels[i]]["FN"] += 1
                if predictions[i] != 'u':
                    eval[predictions[i]]["FP"] += 1
    
        for key in eval:
            tp = eval[key]['TP']
            fn = eval[key]['FN']
            fp = eval[key]['FP']
            eval[key]['precision'] =  tp / ((tp + fp)+1e-5)    
            eval[key]['recall'] =  tp / ((tp + fn)+1e-5)
        return eval
    
    def __AP(self, precision_recall_curve):
        AP = []
        for key in precision_recall_curve:
            precision = np.array(precision_recall_curve[key]['precision'])
            recall = np.array(precision_recall_curve[key]['recall'])
            AP.append((np.diff(recall) * precision[1:]).sum()+ (recall[0] * precision[0]))
        return AP
    
    def calculate_mAP(self, thresh_max, thresh_min, num_samples, k):
        thresh_arr = np.linspace(thresh_min, thresh_max, num_samples)
        precision_recall_curve = {}
        for class_name in self.class_names:
            precision_recall_curve[class_name] = {'precision':[], 'recall': []}
            
        for thresh in thresh_arr:
            predictions = self.predict(k, thresh)
            eval = self.evaluate(predictions)
            for key in eval:
                precision_recall_curve[key]['precision'].append(eval[key]['precision'])
                precision_recall_curve[key]['recall'].append(eval[key]['recall'])
        
        AP = self.__AP(precision_recall_curve)
        mAP = np.mean(AP)
        return mAP
                   
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-m', '--model_path', type=str,
                           help='Path to a trained model',
                           default='/home/dlrv-ss22-face-recognition/workspace/face-recognition-models/e_15_b_16_l_1e2/model_14.pt')
    argparser.add_argument('-d', '--dataset_path', type=str,
                           help='Path to dataset',
                           default='/home/dlrv-ss22-face-recognition/workspace/data/expressions3')

    argparser.add_argument('-i', '--testset_path', type=str,
                        help='Path to testset',
                        default='/home/dlrv-ss22-face-recognition/workspace/data/final_test')
    args = argparser.parse_args()
    model_path = args.model_path
    dataset_path = args.dataset_path
    testset_path = args.testset_path
    means = torch.Tensor([0.5240, 0.3407, 0.2634])
    std = torch.Tensor([0.2024, 0.1820, 0.1643])
    knn = KNN(dataset_path, testset_path, model_path, means, std)
    predictions = knn.predict(1, 0.01)
    evals = knn.evaluate(predictions)
    print(evals)
    mAP = knn.calculate_mAP(0.1, 0.01, 11, 1)
    print(mAP)

