import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse

import numpy as np
import scipy.io
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import util
from util import *

images_folder_path = os.path.join('ESP-ImageSet', 'images')
datamap_path = os.path.join('misc', 'esp_data.mat')
mat = scipy.io.loadmat(datamap_path)
espgame_url = 'http://hunch.net/~learning/ESP-ImageSet.tar.gz'

def download_espgame(root):

    espgame_path = os.path.join(root, 'ESP-ImageSet')

    # create directory
    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.exists(espgame_path):
        parts = urlparse(espgame_url)
        filename = os.path.basename(parts.path)
        tmp_path = os.path.join(root, 'tmp')
        cached_file = os.path.join(tmp_path, filename)

        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(espgame_url, cached_file))
            util.download_url(espgame_url, cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

def get_classes():
    classes = []
    keywords = mat['dict']
    for keyword in keywords:
        keyword = keyword[0][0]
        classes.append(keyword)
    return classes

def get_labelled_data(setname, path):
    images = []
    dataset = mat[setname][0]
    num_img = len(mat['data'][0])
    num_classes = len(mat['dict'])
    cnt = 0
    mx = 20000
    # annot = np.zeros(shape = (num_classes, num_img), dtype=np.int64)
    data = mat['data'][0]
    for val in dataset:
        img = data[val-1]
        keywords = img['keywords'][0]
        name = img['file'][0]
        img_path = os.path.join(path, name)
        if not os.path.exists(img_path): 
            continue 
        labels = np.full(shape = (num_classes, ), fill_value=-1, dtype=np.float32)
        for keyword in keywords:
            labels[keyword-1] = 1
        labels = torch.from_numpy(labels)
        item = (name, labels)
        images.append(item)
        cnt += 1
        if cnt > mx:
            break
    return images

class ESPGAME(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, inp_name=None, adj=None):
        self.root = root
        self.path_images = os.path.join(root, images_folder_path)
        self.set = set
        self.transform = transform
        self.target_transform = target_transform
        self.classes = get_classes()
        # download dataset
        download_espgame(self.root)
        if self.set == 'trainval':
            setname = 'train'
        else:
            setname = 'test'

        self.images = get_labelled_data(setname, self.path_images)

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

        print('[dataset] ESPGAME classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, path, self.inp), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)
