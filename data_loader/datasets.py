# 导入相关模块
from torch.utils.data import Dataset
import cv2
import os
import torch
from torchvision import transforms
import numpy as np


class ATMData(Dataset):
    def __init__(self, data_dir, transform=None):  # __init__是初始化该类的一些基础参数
        self.transform = transform  # 变换
        self.data_dir = data_dir
        fh = open(os.path.join('./', data_dir, 'index.txt'), 'r')
        img_labels = []
        for line in fh:
            line = line.strip('\n')
            words = line.split()
            label = int(words[1])
            img_path = os.path.join(self.data_dir, words[0])
            img_labels.append((img_path, label))
        self.img_labels = img_labels

    def __len__(self):  # 返回整个数据集的大小
        return len(self.img_labels)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        img_path, label = self.img_labels[index]  # 根据索引index获取该图片
        img = cv2.imread(img_path)  # 读取该图片
        if self.transform:
            img = self.transform(img)  # 对样本进行变换
        return img, label  # 返回该样本
