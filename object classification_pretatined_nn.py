# -*- coding: utf-8 -*-
"""
Created on Sun Feb 19 16:58:58 2023

@author: Omen
"""
import torch
import torchvision.transforms as transforms 
import torchvision.models as models 
import requests 
from matplotlib import pyplot as plt 
import cv2
import warnings

warnings.filterwarnings("ignore")
im = cv2.imread('dog.jpg', 1)

im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

plt.imshow(im)

model = models.vit_b_16(pretrained=True).eval()
     

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

preprocess = transforms.Compose([
   transforms.ToPILImage(),
   transforms.Resize((224, 224)),
   transforms.CenterCrop(224),
   transforms.ToTensor(),
   normalize
])

t = torch.tensor(im)
t = t.type(torch.FloatTensor)/255

t = t.permute(2, 0, 1)

t = preprocess(t)

t = t.unsqueeze(0)

t = t.type(torch.FloatTensor)

results = torch.nn.Softmax()(model(t))
index = torch.argmax(results).item()
labels = requests.get('https://s3.amazonaws.com/deep-learning-models/image-models/imagenet_class_index.json').json()
result=labels[str(index)]

print("The image is " +result[1])
