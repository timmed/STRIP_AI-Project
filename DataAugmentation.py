import os
import cv2
from PIL import Image
import torch
import torchvision
import torchvision.transforms as T
import numpy as np

transform = T.Compose([T.ToTensor()])
augmentedImages = []
labels = []
w = 3000
h = 3000
for i, file in enumerate(os.listdir(os.getcwd() + '\STRIP_AI\\processed_data' + '\\' + 'LAA')):
        labels.append('LAA')
        image = cv2.imread(os.getcwd() + '\STRIP_AI\\processed_data' + '\\' + 'LAA' + '\\' + file)
        center = image.shape
        x = center[1] / 2 - w / 2
        y = center[0] / 2 - h / 2
        crop_img = image[int(y):int(y + h), int(x):int(x + w)]
        im = Image.fromarray(crop_img)
        im.save(f'STRIP_AI/processed_data/AIG/aug_image_{i}.png')

for i, file in enumerate(os.listdir(os.getcwd() + '\STRIP_AI\\processed_data' + '\\' + 'LAA')):
        labels.append('LAA')
        image = cv2.imread(os.getcwd() + '\STRIP_AI\\processed_data' + '\\' + 'LAA' + '\\' + file)
        rotated_image = cv2.rotate(image, cv2.ROTATE_180)
        im = Image.fromarray(rotated_image)
        im.save(f'STRIP_AI/processed_data/AIG/aug_image_{i + 207}.png')

