
import os
import cv2
from torch.utils.data import Dataset
import numpy as np
from skimage.transform import estimate_transform, warp
import json
import re
import torch
import torch.nn.functional as F

class AlignDataset(Dataset):
    def __init__(self, imgs_path, annotations_path, transforms = None, input_size = 224):
        '''
            imgs_path (str) : path to images
            masks_path : path to masks
            annotations_path : path to annotations (bbox (x, y, w, h) 68 keypoints (x, y, visibility)
            transforms : transforms to apply
        '''
        self.imgs_path = imgs_path
        self.annotations_path = annotations_path
        f = open(annotations_path)
        self.anns = json.load(f)
        self.valid_labels = [1, 2, 4, 5, 6, 7, 10, 11, 12]

        self.transforms = transforms
        self.files = [x[:-4] for x in os.listdir(self.imgs_path)]
        self.input_size = input_size
        self.valid_labels = [1, 2, 4, 5, 6, 7, 10, 11, 12]
        self.regex = "^0+(?!$)"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # loads keypoints
        new_file = re.sub(self.regex, "", self.files[index])
        keypoints = np.array(self.anns[new_file]["image"]["face_landmarks"]) / 4
    
        # loads image
        img_file = os.path.join(self.imgs_path, self.files[index] + ".png")
        image = cv2.imread(img_file)[...,::-1]
        image = cv2.resize(image, (256, 256), cv2.INTER_LINEAR)


        left = np.min(keypoints[:,0]); right = np.max(keypoints[:,0]); 
        top = np.min(keypoints[:,1]); bottom = np.max(keypoints[:,1])
        size = (right - left + bottom - top)/2*1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        size = size * 1.25
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.input_size - 1], [self.input_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        warped_image = warp(image, tform.inverse, output_shape=(self.input_size, self.input_size), preserve_range=True).astype(int)
        kpts = np.dot(tform.params, np.hstack([keypoints, np.ones([keypoints.shape[0],1])]).T).T # np.linalg.inv(tform.params)

        #kpts[:,:2] = kpts[:,:2]/self.input_size * 2  - 1

        warped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(int)
        kpts = np.dot(tform.params, np.hstack([keypoints, np.ones([keypoints.shape[0],1])]).T).T # np.linalg.inv(tform.params)

        #kpts[:,:2] = kpts[:,:2]/self.input_size * 2  - 1


        
        warped_image = torch.FloatTensor(warped_image.copy()).permute(2, 0, 1) / 255
        image = torch.FloatTensor(image.copy()).permute(2, 0, 1) / 255
        tform = torch.FloatTensor(tform.params)

        return image, warped_image, keypoints, tform



class AlignDataset1(Dataset):
    def __init__(self, imgs_path, annotations_path, mask_path, transforms = None, input_size = 224):
        '''
            imgs_path (str) : path to images
            masks_path : path to masks
            annotations_path : path to annotations (bbox (x, y, w, h) 68 keypoints (x, y, visibility)
            transforms : transforms to apply
        '''
        self.imgs_path = imgs_path
        self.annotations_path = annotations_path
        f = open(annotations_path)
        self.anns = json.load(f)
        self.valid_labels = [1, 2, 4, 5, 6, 7, 10, 11, 12]
        self.mask_path = mask_path
        self.transforms = transforms
        self.files = [x[:-4] for x in os.listdir(self.imgs_path)]
        self.input_size = input_size
        self.valid_labels = [1, 2, 4, 5, 6, 7, 10, 11, 12]
        self.regex = "^0+(?!$)"

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # loads keypoints
        new_file = re.sub(self.regex, "", self.files[index])
        keypoints = np.array(self.anns[new_file]["image"]["face_landmarks"]) / 4
    
        # loads image
        img_file = os.path.join(self.imgs_path, self.files[index] + ".png")
        image = cv2.imread(img_file)

        mask_file = os.path.join(self.mask_path, self.files[index] + ".png")
        mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)

        left = np.min(keypoints[:,0]); right = np.max(keypoints[:,0]); 
        top = np.min(keypoints[:,1]); bottom = np.max(keypoints[:,1])
        size = (right - left + bottom - top)/2*1.1
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])
        size = size * 1.25
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.input_size - 1], [self.input_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        warped_image = warp(image, tform.inverse, output_shape=(self.input_size, self.input_size), preserve_range=True).astype(int)
        kpts = np.dot(tform.params, np.hstack([keypoints, np.ones([keypoints.shape[0],1])]).T).T # np.linalg.inv(tform.params)

        #kpts[:,:2] = kpts[:,:2]/self.input_size * 2  - 1

        warped_image = warp(image, tform.inverse, output_shape=(224, 224), preserve_range=True).astype(int)
        kpts = np.dot(tform.params, np.hstack([keypoints, np.ones([keypoints.shape[0],1])]).T).T # np.linalg.inv(tform.params)

        #kpts[:,:2] = kpts[:,:2]/self.input_size * 2  - 1


        
        warped_image = torch.FloatTensor(warped_image.copy()).permute(2, 0, 1) / 255
        image = torch.FloatTensor(image.copy()).permute(2, 0, 1) / 255
        tform = torch.FloatTensor(tform.params)
        mask = torch.FloatTensor(mask).unsqueeze(0)
        return image, warped_image, keypoints, tform, mask, new_file