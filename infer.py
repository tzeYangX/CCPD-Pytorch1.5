#encoding:utf-8
import sys
import cv2
import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import argparse
import numpy as np
from os import path, mkdir
from time import time
from net import *

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']

numClasses = 4
numPoints = 4
weight="../fh02.pth"

model_conv = fh02(numClasses, numPoints)
model_conv = torch.nn.DataParallel(model_conv, device_ids=range(torch.cuda.device_count()))
model_conv.load_state_dict(torch.load(weight))
model_conv = model_conv.cuda()
model_conv.eval()


img = cv2.imread(sys.argv[1])
resizedImage = cv2.resize(img, (480, 480))
resizedImage = np.transpose(resizedImage, (2,0,1))
resizedImage = resizedImage.astype('float32')
resizedImage /= 255.0
 
im_as_ten = torch.from_numpy(resizedImage).float()
im_as_ten = im_as_ten.view(1, 3, 480, 480)
x = Variable(im_as_ten, requires_grad=True)

fps_pred, y_pred = model_conv(x)

outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
labelPred = [t[0].index(max(t[0])) for t in outputY]

[cx, cy, w, h] = fps_pred.data.cpu().numpy()[0].tolist()

left_up = [(cx - w/2)*img.shape[1], (cy - h/2)*img.shape[0]]
right_down = [(cx + w/2)*img.shape[1], (cy + h/2)*img.shape[0]]
cv2.rectangle(img, (int(left_up[0]), int(left_up[1])), (int(right_down[0]), int(right_down[1])), (0, 0, 255), 2)

lpn = provinces[labelPred[0]]+alphabets[labelPred[1]] + ads[labelPred[2]] + ads[labelPred[3]] + ads[labelPred[4]] + ads[labelPred[5]] + ads[labelPred[6]]

print(lpn)

cv2.putText(img, lpn, (int(left_up[0]), int(left_up[1])-20),cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),4)
cv2.imwrite("result.jpg", img)

