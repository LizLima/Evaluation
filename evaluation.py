# Implementation  Experiment LightCNN 256-vector
# 1.- Get array of identity vector for each class of Dataset
# 2.- Get 256 vector for image to test
# 3.- Calculate cosine distance between image vector and indentity vector
# 4.- Sort array 
# 5.- Select and valida according Rank-n 
#  format list test label | path

# Enviroment coda: pytorch
from __future__ import print_function
import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import numpy as np
import cv2

from light_cnn import LightCNN_9Layers, LightCNN_29Layers, LightCNN_29Layers_v2


URL_CHCKPOINT = "/home/liz/Documents/Thesis/LightCNN/models"

parser = argparse.ArgumentParser(description='PyTorch  Feature Extracting')
parser.add_argument('--cuda', '-c', default=True)
parser.add_argument('--model', default='', type=str, metavar='Model',
                    help='model type: LightCNN-9, LightCNN-29')
parser.add_argument('--img_list', default='', type=str, metavar='PATH', 
                    help='list of face images for feature extraction (default: none).')
parser.add_argument('--img_test', default='', type=str, metavar='PATH', 
                    help='list of face images test for feature extraction (default: none).')
parser.add_argument('--num_classes', default=79077, type=int,
                    metavar='N', help='mini-batch size (default: 79077)')
parser.add_argument('--save_path', default='', type=str, metavar='PATH', 
                    help='save root path for features of face images.')

# v1 and v2 : shape [256]
def dis_cosine(v1, v2):
    # 1 = + similarity
    # 0 = not similarity
    xy = torch.dot(v1, v2)
    x_euc = torch.norm(v1)
    y_euc = torch.norm(v2)
    return xy/(x_euc * y_euc)

def dis_euclidean(v1, v2):
    # 0 = + similarity
    return torch.norm(v1, v2)

def dis_euclideanv2(v1, v2):
    result = v1 - v2
    result_square = np.square(result)
    result_sum = np.sum(result_square)
    result_sqrt = np.sqrt(result_sum)
    
    return result_sqrt 



def read_list(list_path):
    img_list = []
    with open(list_path, 'r') as f:
        for line in f.readlines()[0:]:
            img_path = line.strip().split("|")
            img_list.append({"label": img_path[0], "path": img_path[1]})
    print('There are {} images..'.format(len(img_list)))
    return img_list


def getFeattures (title, model, img_list):

    img_features = []
    transform = transforms.Compose([transforms.ToTensor()])
    count     = 0
    input     = torch.zeros(1, 1, 128, 128)
    for img_name in img_list:
        count = count + 1
        img   = cv2.imread(img_name["path"], cv2.IMREAD_GRAYSCALE)
        img   = np.reshape(img, (128, 128, 1))
        img   = transform(img)
        input[0,:,:,:] = img

        start = time.time()
        if args.cuda:
            input = input.cuda()
        input_var   = torch.autograd.Variable(input, volatile=True)
        _, features = model(input_var)
        end         = time.time() - start
        # Add feature tu list
        
        img_features.append({"label":img_name["label"], "path": img_name["path"], "feat": torch.squeeze(features) })
        print("{} {}({}/{}). time: {}".format(title, img_name["path"], count, len(img_list), end))
        # save_feature(args.save_path, img_name, features.data.cpu().numpy()[0])

    return img_features

def getRankN(list, rank):
    rank_list = list[0: rank]

    for item in rank_list:
        if(item["label_target"] == item["label_test"]):
            return True
    return False

def main():
    global args
    args = parser.parse_args()
    model= LightCNN_9Layers(num_classes=args.num_classes)
    print(args)
    if args.model == 'LightCNN-9':
        model = LightCNN_9Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29':
        model = LightCNN_29Layers(num_classes=args.num_classes)
    elif args.model == 'LightCNN-29v2':
        model = LightCNN_29Layers_v2(num_classes=args.num_classes)
    else:
        print('Error model type\n')

    model.eval()
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    
    if args.model:
        checkpoint = URL_CHCKPOINT + "/LightCNN_9Layers_checkpoint.pth.tar"
        if args.model == 'LightCNN-9':
            checkpoint = URL_CHCKPOINT + "/LightCNN_9Layers_checkpoint.pth.tar"
        elif args.model == 'LightCNN-29':
            checkpoint = URL_CHCKPOINT + "/LightCNN_29Layers_checkpoint.pth.tar"
        elif args.model == 'LightCNN-29v2':
            checkpoint = URL_CHCKPOINT + "/LightCNN_29Layers_V2_checkpoint.pth.tar"

        if os.path.isfile(checkpoint):
            print("=> loading checkpoint '{}'".format(checkpoint))
            checkpoint = torch.load(checkpoint)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            print("=> no checkpoint found at '{}'".format(checkpoint))


    img_list  = read_list(args.img_list)
    img_test  = read_list(args.img_test)
    img_features = []
    test_features = []
    

    #######################################
    # Get feature of target/real identities
    #######################################
    img_features = getFeattures("Target features", model, img_list)
 
    #######################################
    # Get features for each test
    #######################################
    test_features = getFeattures("Test features", model, img_test)

    #######################################
    # Calculate Rank 
    #######################################
    count_rank = 0
    count_error = 0
    rank=1
    for feat_test in test_features:
        dist_array = []
        for target_test in img_features:
            dist = dis_cosine(feat_test["feat"], target_test["feat"])
            dist_array.append({"label_target": target_test["label"], "label_test": feat_test["label"], "dist": dist })

        # Sort array dist_array max-to-min
        dist_array.sort(key=lambda x: x["dist"], reverse=True)
    
        if(getRankN(dist_array, rank)):
            count_rank += 1
        else:
            count_error += 1

    print("Correct: {} / Error: {}".format(count_rank, count_error))



if __name__ == "__main__":
    main()