
import os
import cv2
from torch.utils.data import Dataset
import numpy as np
from skimage.transform import estimate_transform, warp


class PIMGANDataset(Dataset):
    def __init__(self, imgs_path, transforms = None):
        '''
            imgs_path (str) : path to images
            masks_path : path to masks
            annotations_path : path to annotations (bbox (x, y, w, h) 68 keypoints (x, y, visibility)
            transforms : transforms to apply
        '''
        
        self.imgs_path = imgs_path
        self.transforms = transforms
        self.files = [x[:-4] for x in os.listdir(self.imgs_path)]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # loads image
        img_file = os.path.join(self.imgs_path, self.files[index] + ".png")
        image = cv2.imread(img_file)[...,::-1]


        if self.transforms is not None:
            transformed = self.transforms(image=image)
            image = transformed["image"]

        return image
