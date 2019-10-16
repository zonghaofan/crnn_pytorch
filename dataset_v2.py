#coding:utf-8
"""
fzh created on 2019/10/15
数据读取
"""
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import math
import os
import torch
from skimage.transform import resize
import utils
import params
from PIL import Image
import utils

class baiduDataset(Dataset):
    def __init__(self, img_root, label_path, alphabet, isBaidu, Resize, transforms=None):
        super(baiduDataset, self).__init__()
        self.img_root = img_root
        self.isBaidu = isBaidu
        self.labels = self.get_labels(label_path)
        print('self.labels[:10]:', self.labels[:10])
        self.alphabet = alphabet
        self.transforms = transforms
        self.width, self.height = Resize

    # print(list(self.labels[1].values())[0])
    def get_labels(self, label_path):
        # return text labels in a list
        if self.isBaidu:
            with open(label_path, 'r', encoding='utf-8') as file:
                # {"image_name":"chinese_text"}
                content = [[{c.split('\t')[2]: c.split('\t')[3][:-1]}, {"w": c.split('\t')[0]}] for c in
                           file.readlines()];
            labels = [c[0] for c in content]
        # self.max_len = max([int(list(c[1].values())[0]) for c in content])
        else:
            with open(label_path, 'r', encoding='utf-8') as file:
                labels = [{c.split(' ')[0]: c.split(' ')[-1][:-1]} for c in file.readlines()]
        return labels

    def __len__(self):
        return len(self.labels)

    # def compensation(self, image):
    # 	h, w = image.shape # (48,260)
    # 	image = cv2.resize(image, (0,0), fx=280/w, fy=32/h, interpolation=cv2.INTER_CUBIC)
    # 	# if w>=self.max_len:
    # 	# 	image = cv2.resize(image, (0,0), fx=280/w, fy=32/h, interpolation=cv2.INTER_CUBIC)
    # 	# else:
    # 	# 	npi = -1*np.ones(self.max_len-)
    # 	return image
    def preprocessing(self, image):
        ## already have been computed
        image = image.astype(np.float32) / 255.
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image.sub_(params.mean).div_(params.std)

        return image

    def __getitem__(self, index):
        # print(self.labels[index].keys())
        image_name = list(self.labels[index].keys())[0]
        # print('image_name:',image_name)
        # label = list(self.labels[index].values())[0]
        image = cv2.imread(self.img_root + '/' + image_name)
        # print(self.img_root+'/'+image_name)
        # print('image.shape:',image.shape)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ori_h, ori_w = image.shape

        if ori_h > ori_w:
            image = np.rot90(image)
            # cv2.imwrite('test.jpg', image)
            # h,w=image.shape
            # image = cv2.resize(image, (0,0), fx=self.width/w, fy=self.height/h, interpolation=cv2.INTER_CUBIC)
            # cv2.imwrite('test1.jpg', image)
            # image = cv2.resize(image, (self.width,self.height))
        # else:
        #     # image = cv2.resize(image, (0, 0), fx=self.width / ori_w, fy=self.height / ori_h, interpolation=cv2.INTER_CUBIC)
        #     # cv2.imwrite('test2.jpg', image)
        #     image = cv2.resize(image, (self.width, self.height))
        # image = np.reshape(image, (32, self.width, 1)).transpose(2, 0, 1)
        new_h, new_w = image.shape
        ratio = self.height/new_h
        resize_w = new_w * ratio
        image_resize = cv2.resize(image, (int(resize_w), self.height))
        # cv2.imwrite('hah.jpg', image_resize)
        if resize_w < self.width:#200
            length = (self.width - resize_w) / 2
            image = np.pad(np.array(image_resize), ((0, 0), (int(length), int(length))),
                               'constant', constant_values=(225, 225))
        # cv2.imwrite('test1.jpg',image)
        image = cv2.resize(image, (self.width, self.height))
        image = np.reshape(image, (self.height, self.width,1)).transpose(2, 0, 1)
        image = self.preprocessing(image)

        return image, index


if __name__ == '__main__':
    dataset = baiduDataset("./data_chinese_tra/images_add_fake", "./data_chinese_tra/label_add_fake/train_add_fake.txt", params.alphabet, False,
                           (params.imgW, params.imgH))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    converter = utils.strLabelConverter(dataset.alphabet)
    for i_batch, (images, index) in enumerate(dataloader):
        # if i_batch < 1:
            print('images.shape:', images.shape)
            for j, image in enumerate(images):
                img = image.numpy()
                img = np.transpose(img,(1,2,0))
                img = (img*params.std+params.mean)*255
                cv2.imwrite('img'+str(j)+'.jpg',img)
            # print('index:', index)
            # label = utils.get_batch_label(dataset, index)
            # print('label:', label)
            # print(type(label[0]))
            # text, length = converter.encode(label)
            # print('text:', text)
            # print('length:', length)
        # print(params.alphabet.index(label[0][0]))
