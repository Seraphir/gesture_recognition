# 导入相关模块
from torch.utils.data import Dataset
import cv2
import os
import torch
from torchvision import transforms
import numpy as np


class GestureData(Dataset):
    def __init__(self, data_dir, index_file="index.txt", transform=None):  # __init__是初始化该类的一些基础参数
        self.transform = transform  # 变换
        self.data_dir = data_dir
        self.index_file = index_file
        self.resize_height = 112
        self.resize_width = 112
        self.img_video_labels = self._read_index_text()

    def __len__(self):  # 返回整个数据集的大小
        return len(self.img_video_labels)

    def __getitem__(self, index):  # 根据索引index返回dataset[index]
        image_path, video_path, label = self.img_video_labels[index]  # 根据索引index获取该图片
        img = cv2.imread(image_path)  # 读取该图片
        video = self._get_video(video_path)
        video = video.transpose((3, 0, 1, 2))
        if self.transform:
            img = self.transform(img)  # 对样本进行变换
        return img, video, label  # 返回该样本

    def _read_index_text(self):
        fh = open(os.path.join('./', self.data_dir, self.index_file), 'r')
        img_video_labels = []
        for line in fh:
            line = line.strip('\n')
            words = line.split(' ')
            image_path = os.path.join(self.data_dir, words[0])
            video_path = os.path.join(self.data_dir, words[1])
            label = int(words[2])
            img_video_labels.append((image_path, video_path, label))
        return img_video_labels

    def _get_video(self, video_path):
        frame_paths = sorted([os.path.join(video_path, img) for img in os.listdir(video_path)])
        frame_count = len(frame_paths)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_path in enumerate(frame_paths):
            frame = cv2.imread(frame_path)
            frame = cv2.resize(frame, (self.resize_height, self.resize_width))
            frame = np.array(frame).astype(np.float64)
            buffer[i] = frame
        return buffer
