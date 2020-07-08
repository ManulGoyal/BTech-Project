import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
import pickle
import util
from util import *

path_to_dictionary = 'data/iaprtc/annotation/iaprtc12_dictionary.txt'
images_folder_path = os.path.join('iaprtc12', 'images')
annotation_folder_path = 'annotation'

iaprtc_url = 'http://www-i6.informatik.rwth-aachen.de/imageclef/resources/iaprtc12.tgz'

def download_iaprtc12(root):

    iaprtc12_path = os.path.join(root, 'iaprtc12')

    # create directory
    if not os.path.exists(root):
        os.makedirs(root)

    if not os.path.exists(iaprtc12_path):
        parts = urlparse(iaprtc_url)
        filename = os.path.basename(parts.path)
        tmp_path = os.path.join(root, 'tmp')
        cached_file = os.path.join(tmp_path, filename)

        if not os.path.exists(tmp_path):
            os.makedirs(tmp_path)

        if not os.path.exists(cached_file):
            print('Downloading: "{}" to {}\n'.format(iaprtc_url, cached_file))
            util.download_url(iaprtc_url, cached_file)

        # extract file
        print('[dataset] Extracting tar file {file} to {path}'.format(file=cached_file, path=root))
        cwd = os.getcwd()
        tar = tarfile.open(cached_file, "r:gz")
        os.chdir(root)
        tar.extractall()
        tar.close()
        os.chdir(cwd)
        print('[dataset] Done!')

    
def read_file(file):
    words = []
    with open(file) as f:
        for l in f:
            s = l.split('\n')
            words.append(s[0])
    return words

def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row                    # TODO: doubt
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]
                labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                labels = torch.from_numpy(labels)
                item = (name, labels)
                images.append(item)
            rownum += 1
    return images

def read_annot_csv(file, header=False):
    images = []
    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for i, row in enumerate(reader):
            if header and rownum == 0:
                header = row                    # TODO: doubt
            else:
                # if num_categories == 0:
                #     num_categories = len(row)
                
                images.append((np.asarray(row[0:])).astype(np.float32))
                # labels = torch.from_numpy(labels)
                # item = (name, labels)
                
            rownum += 1
    return images

def write_object_labels_csv(images, file, labels, names):
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(labels)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, name in enumerate(names):
            example = {'name' : name}
            for j, tag in enumerate(images[i]):
                if tag == 0:
                    example[labels[j]] = -1
                else:
                    example[labels[j]] = 1
            writer.writerow(example)
    csvfile.close()

class IAPRTC12Classification(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None, inp_name=None, adj=None):
        self.root = root
        # self.path_devkit = os.path.join(root, 'VOCdevkit')
        self.path_images = os.path.join(root, images_folder_path)
        self.set = set
        self.transform = transform
        self.target_transform = target_transform
        self.classes = read_file(path_to_dictionary)
        # download dataset
        download_iaprtc12(self.root)

        # define path of csv file
        # path_csv = os.path.join(self.root, 'files', 'VOC2007')
        # define filename of csv file
        file_csv = os.path.join(self.root, 'classification_' + set + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            print('doin')
            # if not os.path.exists(path_csv):  # create dir if necessary
            #     os.makedirs(path_csv)
            # generate csv file
            if self.set == 'trainval':
                setname = 'train'
            else:
                setname = 'test'
            path_to_names = os.path.join(self.root, 'annotation', 'iaprtc12_'+setname+'_list.txt')
            path_to_annot = os.path.join(self.root, 'annotation', 'iaprtc12_'+setname+'_annot.csv')
            names = read_file(path_to_names)
            labeled_data = read_annot_csv(path_to_annot)    
            # write csv file
            write_object_labels_csv(labeled_data, file_csv, self.classes, names)

       
        self.images = read_object_labels_csv(file_csv)

        with open(inp_name, 'rb') as f:
            self.inp = pickle.load(f)
        self.inp_name = inp_name

        print('[dataset] IAPR-TC 12 classification set=%s number of classes=%d  number of images=%d' % (
            set, len(self.classes), len(self.images)))

    # open the required image on the go (when item is accessed)
    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path + '.jpg')).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return (img, path, self.inp), target

    def __len__(self):
        return len(self.images)

    def get_number_classes(self):
        return len(self.classes)
