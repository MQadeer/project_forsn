#!/usr/bin/env python3
import os
import argparse
from posixpath import dirname
from statistics import mean, stdev

import numpy as np
import torchvision.transforms as transforms
import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    pass

import torch
import torchvision
from torch.utils.data import DataLoader

from siamese_net.model import SiameseNetwork, initialize_weights
from siamese_net.dataset import SiameseNetworkDataset
from siamese_net.loss import ContrastiveLoss
from siamese_net.utils import get_transforms, get_transform_norm

if __name__ == '__main__':
    
    lr= 1e-2
    epochs = 15
    batch_size = 64
    margin = 0.1
    gamma = 0.1
    step_size = 5
    
    new_folder = "lr_" + str(lr) + "_e_" + str(epochs) + "_b_" + str(batch_size) + "_m_" + str(margin) + "_g_" + str(gamma) + "_s_" + str(step_size)
    dir = '/home/dlrv-ss22-face-recognition/workspace/face-recognition-models'
    loss_file = 'train_loss.log'
    model_dir = os.path.join(dir, new_folder)
    loss_file_dir = os.path.join(model_dir, loss_file)
    
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-d', '--data_path', type=str,
                           help='Directory containing training data',
                           default='/home/dlrv-ss22-face-recognition/workspace/data/augmented-faces')
    argparser.add_argument('-m', '--model_path', type=str,
                           help='Path to a directory where the trained models (one per epoch) should be saved',
                           default=model_dir)
    argparser.add_argument('-e', '--num_epochs', type=int,
                           help='Number of training epochs',
                           default=epochs)
    argparser.add_argument('-lr', '--learning_rate', type=float,
                           help='Initial learning rate',
                           default=lr)
    argparser.add_argument('-b', '--training_batch_size', type=int,
                           help='Training batch size',
                           default=batch_size)
    argparser.add_argument('-l', '--train_loss_file_path', type=str,
                           help='Path to a file in which training losses will be saved',
                           default=loss_file_dir)

    # we read all arguments
    args = argparser.parse_args()
    data_path = args.data_path
    model_path = args.model_path
    train_loss_file_path = args.train_loss_file_path
    num_epochs = args.num_epochs
    training_batch_size = args.training_batch_size
    learning_rate = args.learning_rate

    print('\nThe following arguments were read:')
    print('------------------------------------')
    print('data_path:               {0}'.format(data_path))
    print('model_path:              {0}'.format(model_path))
    print('train_loss_file_path:    {0}'.format(train_loss_file_path))
    print('num_epochs:              {0}'.format(num_epochs))
    print('training_batch_size:     {0}'.format(training_batch_size))
    print('learning_rate:           {0}'.format(learning_rate))
    print('------------------------------------')
    print('Proceed with training (y/n)')
    proceed = input()
    if proceed != 'y':
        print('Aborting training')
        sys.exit(1)

    # we create a data loader by instantiating an appropriate
    # dataset class depending on the annotation type
    folder_dataset = torchvision.datasets.ImageFolder(data_path, transform=get_transforms())
    dataloader = torch.utils.data.DataLoader(folder_dataset, batch_size=folder_dataset.__len__(), shuffle=False)
    images, labels = next(iter(dataloader))
    means = torch.mean(images, dim=[0, 2, 3])
    std = torch.std(images, dim=[0, 2, 3])
    print(means)
    print(std)
    folder_dataset = torchvision.datasets.ImageFolder(root=data_path)
    siamese_dataset = SiameseNetworkDataset(image_folder_dataset=folder_dataset,
                                            transform=get_transform_norm(means, std),
                                            should_invert=False)

    train_dataloader = DataLoader(siamese_dataset,
                                  shuffle=True,
                                  num_workers=8,
                                  batch_size=training_batch_size)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = SiameseNetwork()
    model.apply(initialize_weights)
    criterion = torch.nn.TripletMarginLoss(margin=margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    # we move the model to the correct device before training
    model.to(device)

    # we create the model path directory if it doesn't exist
    if not os.path.isdir(model_path):
        print('Creating model directory {0}'.format(model_path))
        os.mkdir(model_path)

    # we clear the files in which the training and validation losses are saved
    open(train_loss_file_path, 'w').close()

    print('Training model for {0} epochs'.format(num_epochs))
    for epoch in range(num_epochs):
        losses = []
        for i, data in enumerate(train_dataloader):
            # print(data[0].size())
            # print(torch.mean(torch.stack(data),dim=[1,2]))
            # normalized_data = transforms.Normalize(torch.mean(torch.stack(data)),torch.std(torch.stack(data)))
            anchor, positive, negative = data
            anchor, positive, negative = anchor.cuda(), positive.cuda(), negative.cuda()
            optimizer.zero_grad()
            output1, output2, output3 = model(anchor, positive, negative)
            loss_contrastive = criterion(output1, output2, output3)
            loss_contrastive.backward()
            optimizer.step()
            losses.append(loss_contrastive.item())
            # print("loss in batch ",loss_contrastive.item())
        avg_loss = np.mean(losses)
        print("no of losses",len(losses))
        print('Epoch number {}\n Average loss {}\n'.format(epoch, avg_loss))

        if train_loss_file_path:
            with open(train_loss_file_path, 'a+') as loss_file:
                loss_file.write(str(avg_loss).split(' ')[0] + '\n')

        lr_scheduler.step()
    torch.save(model.state_dict(), os.path.join(model_path, 'model.pt'))
    print('Training done')
