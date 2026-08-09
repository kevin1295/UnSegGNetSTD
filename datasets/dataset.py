import os
import deeplake
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path


DATASET_ROOT = Path(__file__).resolve().parent

class Dataset:
    def __init__(self, dataset):
        if dataset not in ["CUB", "ECSSD", "DUTS", "CVC"]:
            raise ValueError(f'Dataset: {dataset} is not supported')
        self.dataset = dataset
        if dataset == "CUB":
            self.images, self.masks = load_cub()
        elif dataset == "ECSSD":
            ds = deeplake.load("hub://activeloop/ecssd")
            self.images = ds["images"]
            self.masks = ds["masks"]
        elif dataset == "DUTS":
            self.images, self.masks = load_duts()
        elif dataset == "CVC":
            self.images, self.masks = load_cvc()
        self.loader = len(self.images)

    def load_samples(self):
        for imagep, true_maskp in zip(self.images, self.masks):
            if self.dataset == "CUB":
                img = np.asarray(Image.open(imagep))
                seg = np.asarray(Image.open(true_maskp).convert('L'))
                true_mask = np.where(seg >= 200, 1, 0)
            elif self.dataset == "ECSSD":
                img = np.asarray(imagep)
                seg = np.asarray(true_maskp)
                true_mask = np.where(seg, 1, 0)
            elif self.dataset == "DUTS":
                img = np.asarray(Image.open(imagep))
                seg = np.asarray(Image.open(true_maskp).convert('L'))
                true_mask = np.where(seg == 255, 1, 0).astype(np.uint8)
            elif self.dataset == "CVC":
                img = np.asarray(Image.open(imagep))
                seg = np.asarray(Image.open(true_maskp).convert('L'))
                true_mask = np.where(seg > 0, 1, 0).astype(np.uint8)
            yield img, true_mask

           
def load_cub():
    cp = DATASET_ROOT
    with open(cp / 'CUB_200_2011/train_test_split.txt') as f:
        pretest = set()
        for line in f:
            image_id, split = line.split()
            if split == "0":
                pretest.add(image_id)

    with open(cp / 'CUB_200_2011/images.txt') as u:
        test = []
        for line in u:
            x,y  = line.split()[0],line.split()[1]
            if x in pretest:
                test.append(y)

    masks = sorted([str(cp / 'segmentations' / (x[:-3] + 'png')) for x in test])
    test = sorted([str(cp / 'CUB_200_2011/images' / x) for x in test])
    missing = [path for path in test + masks if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f'CUB dataset is incomplete; missing {missing[0]}')

    return test, masks


def load_duts():
    cp = DATASET_ROOT
    image_dir = cp / 'DUTS-TE/DUTS-TE-Image'
    mask_dir = cp / 'DUTS-TE/DUTS-TE-Mask'
    test = sorted(str(path) for path in image_dir.iterdir() if path.is_file())
    masks = sorted(str(path) for path in mask_dir.iterdir() if path.is_file())
    if len(test) != len(masks):
        raise ValueError(f'DUTS image/mask count mismatch: {len(test)} images, {len(masks)} masks')
    if [Path(path).stem for path in test] != [Path(path).stem for path in masks]:
        raise ValueError('DUTS image and mask filenames are not aligned')

    return test, masks

def load_cvc():
    cp = DATASET_ROOT
    
    # 读取metadata.csv文件获取文件路径
    metadata_path = os.path.join(cp, 'CVC/metadata.csv')
    
    # 使用pandas读取CSV文件
    df = pd.read_csv(metadata_path)
    
    # 根据frame_id排序
    df = df.sort_values('frame_id')
    
    images = []
    masks = []
    
    # 遍历每一行，使用png_image_path和png_mask_path
    for _, row in df.iterrows():
        png_image_path = row['png_image_path']
        png_mask_path = row['png_mask_path']
        
        # 构建完整路径
        full_image_path = os.path.join(cp, 'CVC', png_image_path)
        full_mask_path = os.path.join(cp, 'CVC', png_mask_path)
        
        # 检查文件是否存在
        if os.path.exists(full_image_path) and os.path.exists(full_mask_path):
            images.append(full_image_path)
            masks.append(full_mask_path)
    
    if not images:
        raise FileNotFoundError(f'No complete CVC image/mask pairs found under {cp / "CVC"}')
    return images, masks
