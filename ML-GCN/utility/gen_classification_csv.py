import csv
import os
import os.path
import tarfile
from urllib.parse import urlparse
import pickle
import numpy as np

num_categories = 291

path_to_dictionary = 'data/iaprtc/annotation/iaprtc12_dictionary.txt'
path_to_train_list = 'data/iaprtc/annotation/iaprtc12_train_list.txt'
path_to_test_list = 'data/iaprtc/annotation/iaprtc12_test_list.txt'
path_to_train_annot = 'data/iaprtc/annotation/iaprtc12_train_annot.csv'
path_to_test_annot = 'data/iaprtc/annotation/iaprtc12_test_annot.csv'
save_path_train_labels = 'data/iaprtc/classification_trainval.csv'
save_path_test_labels = 'data/iaprtc/classification_test.csv'

def read_object_labels_csv(file, header=False):
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

def read_file(file):
    words = []
    with open(file) as f:
        for l in f:
            s = l.split('\n')
            words.append(s[0])
    return words

def write_csv_labels(images, file, labels, names):
    print('[dataset] write file %s' % file)
    with open(file, 'w') as csvfile:
        fieldnames = ['name']
        fieldnames.extend(labels)
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for i, name in enumerate(names):
            example = {'name' : name}
            for j, tag in enumerate(images[i]):
                example[labels[j]] = int(tag)
            writer.writerow(example)
    csvfile.close()

labels = read_file(path_to_dictionary)
names_test = read_file(path_to_test_list)
images_test = read_object_labels_csv(path_to_test_annot)

write_csv_labels(images_test, save_path_test_labels, labels, names_test)


names_train = read_file(path_to_train_list)
images_train = read_object_labels_csv(path_to_train_annot)

write_csv_labels(images_train, save_path_train_labels, labels, names_train)