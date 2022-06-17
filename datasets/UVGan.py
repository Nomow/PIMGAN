
import os
import cv2
from torch.utils.data import Dataset
import numpy as np
from skimage.transform import estimate_transform, warp
import torch

class UVGanDataset(Dataset):
    def __init__(self, npy_path):
        '''
            imgs_path (str) : path to images
            masks_path : path to masks
            annotations_path : path to annotations (bbox (x, y, w, h) 68 keypoints (x, y, visibility)
            transforms : transforms to apply
        '''
        self.npy_path = npy_path
        self.files = [x[:-4] for x in os.listdir(self.npy_path)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # loads bboxes and keypoints
        data = np.load(os.path.join(self.npy_path, self.files[index] + ".npy"), allow_pickle=True)
        data = data.item()
        visibility_mask = data['visibility_mask']
        grid = data['grid']
        shading_mask = data['shading_mask']
        face_eye_mask = data['face_eye_mask']
        dst_lmks = data['dst_lmks']
        src_lmks = data['src_lmks']
        texcode = data['texcode']
        img = data['img']
        return img, visibility_mask, face_eye_mask, shading_mask, grid, dst_lmks, src_lmks, texcode



import os
import cv2
from torch.utils.data import Dataset
import numpy as np
from skimage.transform import estimate_transform, warp
import torch

class UVGanDataset1(Dataset):
    def __init__(self, npy_path):
        '''
            imgs_path (str) : path to images
            masks_path : path to masks
            annotations_path : path to annotations (bbox (x, y, w, h) 68 keypoints (x, y, visibility)
            transforms : transforms to apply
        '''
        self.npy_path = npy_path
        self.files = [x[:-4] for x in os.listdir(self.npy_path)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # loads bboxes and keypoints
        data = np.load(os.path.join(self.npy_path, self.files[index] + ".npy"), allow_pickle=True)
        data = data.item()
        visibility_mask = data['face_eye_mask']
        grid = data['grid']
        shading_mask = data['shading_mask']
        face_eye_mask = data['mask']
        dst_lmks = data['dst_lmks']
        src_lmks = data['src_lmks']
        texcode = data['texcode']
        img = data['img']
        return img, visibility_mask, face_eye_mask, shading_mask, grid, dst_lmks, src_lmks, texcode
