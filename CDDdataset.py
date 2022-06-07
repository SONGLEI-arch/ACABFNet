from PIL import Image
import os
import glob
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data
import random
import torchvision.transforms.functional as tf

root_path = r'D:\CDD\ChangeDetectionDataset\Real\subset\CDD'

def read_images(root):
    imgs_At = glob.glob(root + '/train/A/*.jpg')
    imgs_Bt = glob.glob(root + '/train/B/*.jpg')
    labelst = glob.glob(root + '/train/OUT/*.jpg')
    imgs_Att = glob.glob(root + '/test/A/*.jpg')
    imgs_Btt = glob.glob(root + '/test/B/*.jpg')
    labelstt = glob.glob(root + '/test/OUT/*.jpg')
    return imgs_At, imgs_Bt, labelst, imgs_Att, imgs_Btt, labelstt

# """
# 划分训练集和测试集
# """
# def split_dataset(before, after, change):
#     train_size = int(0.8 * len(before))
#     random.seed(train_size)
#     random.shuffle(before)
#     random.seed(train_size)
#     random.shuffle(after)
#     random.seed(train_size)
#     random.shuffle(change)

#     random.seed(len(before)-train_size)
#     train_dataset_before = random.sample(before, train_size)
#     random.seed(len(before)-train_size)
#     train_dataset_after = random.sample(after, train_size)
#     random.seed(len(before)-train_size)
#     train_dataset_change = random.sample(change, train_size)

#     test_dataset_before = [i for i in before if i not in train_dataset_before]
#     test_dataset_after = [i for i in after if i not in train_dataset_after]
#     test_dataset_change = [i for i in change if i not in train_dataset_change]
#     return train_dataset_before, train_dataset_after, train_dataset_change, test_dataset_before, test_dataset_after, test_dataset_change

def imagestransforms(before, after, change):
    if random.random() > 0.7:
        before = tf.hflip(before)
        after = tf.hflip(after)
        change = tf.hflip(change)
    if random.random() < 0.3:
        before = tf.vflip(before)
        after = tf.vflip(after)
        change = tf.vflip(change)
    angle = transforms.RandomRotation.get_params([-180, 180])
    before = before.rotate(angle)
    after = after.rotate(angle)
    change = change.rotate(angle)

    before = tf.to_tensor(before)
    after = tf.to_tensor(after)
    change = tf.to_tensor(change)
    return before, after, change

def images_transforms(before, after, change):
    before, after, change = imagestransforms(before, after, change)
    return before, after, change

def images_transforms_(before, after, change):
    before = tf.to_tensor(before)
    after = tf.to_tensor(after)
    change = tf.to_tensor(change)
    return before, after, change

class CDDDataset(torch.utils.data.Dataset):
    def __init__(self, mode='train'):
        self.mode = mode
        # self.before_list, self.after_list, self.change_list = 
        self.train_dataset_before, self.train_dataset_after, self.train_dataset_change, self.test_dataset_before, self.test_dataset_after, self.test_dataset_change = read_images(root_path)
        if mode == 'train':
            print('训练集加载了' + str(len(self.train_dataset_before)) + '张图片')
        elif mode == 'test':
            print('测试集加载了' + str(len(self.test_dataset_before)) + '张图片')

    def __getitem__(self, item):
        if self.mode == 'train':
            before = Image.open(self.train_dataset_before[item]).convert('RGB')
            after = Image.open(self.train_dataset_after[item]).convert('RGB')
            change = Image.open(self.train_dataset_change[item]).convert('L')
            before, after, change = images_transforms(before, after, change)
            return before, after, change
        elif self.mode == 'test':
            before = Image.open(self.test_dataset_before[item]).convert('RGB')
            after = Image.open(self.test_dataset_after[item]).convert('RGB')
            change = Image.open(self.test_dataset_change[item]).convert('L')
            before, after, change = images_transforms_(before, after, change)
            return before, after, change

    def __len__(self):
        if self.mode == 'train':
            return len(self.train_dataset_before)
        elif self.mode == 'test':
            return len(self.test_dataset_before)

if __name__ == '__main__':
    train_data = CDDDataset(mode='train')
    test_data = CDDDataset(mode='test')