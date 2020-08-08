import os
import math
import csv
from tqdm import tqdm
from sampling import *
import argparse

parser = argparse.ArgumentParser(description='Sampling for Binary GCN')
parser.add_argument('data', metavar='PATH_TO_DATASET', type=str, help='Path to the dataset')
parser.add_argument('--samples', '-s', default=2000, type=int, metavar='SAMPLES',
                        help='Upper limit on number of positive and negative samples for each label')

train_features_file = 'iaprtc12_data_vggf_pca_train.txt'
test_features_file = 'iaprtc12_data_vggf_pca_test.txt'
train_annot_file = 'annotation/iaprtc12_train_annot.csv'
test_annot_file = 'annotation/iaprtc12_test_annot.csv'
remove_images_train_file = 'remove_images_train.csv'
remove_images_test_file = 'remove_images_test.csv'
leaf_nodes_file = 'semantic_files/leaf_nodes.csv'
sp_file = 'semantic_files/semantic_paths.txt'
node_layer_file = 'semantic_files/node_layer_of_paths.txt'
train_annot_aug_file = 'annotation/train_annot_SH_augmented.csv'
test_annot_aug_file = 'annotation/test_annot_SH_augmented.csv'
singleton_nodes_file = 'semantic_files/singleton_nodes.csv'

# positive_samples_file = 'iaprtc12_positive_samples_2000.csv'
# negative_samples_file = 'iaprtc12_negative_samples_2000.csv'

# num_positive_negative_samples = 2000

def read_features(file):
    features = []
    with open(file) as f:
        for l in f:
            row = l.split(',')
            features.append(row)
    f.close()
    return np.asarray(features, dtype='float')

def read_remove_indices(file):
    rem_ind = []
    with open(file) as f:
        csv_reader = csv.reader(f)
        row = list(csv_reader)[0]
        for i in row:
            rem_ind.append(int(i)-1)
    f.close()
    return rem_ind

def read_annot(file):
    annot = []
    with open(file) as f:
        for l in f:
            row = l.split(',')
            annot.append(row)
    f.close()
    return np.asarray(annot, dtype='int')

def get_leaf_nodes(file):
    with open(file) as f:
        row = f.readline()
        row = row.split(',')
        for i in range(len(row)):
            row[i] = int(row[i])-1
        return row

def get_semantic_paths_or_layers(file):
    sp = []
    with open(file) as f:
        for l in f:
            row = l.split(' ')
            path = [int(i) for i in row]
            sp.append(path)
    f.close()
    return sp

def main_dataset_prep():
    global args
    args = parser.parse_args()

    train_features = read_features(os.path.join(args.data, train_features_file))
    test_features = read_features(os.path.join(args.data, test_features_file))

    train_annot_aug = read_annot(os.path.join(args.data, train_annot_aug_file)).T
    test_annot_aug = read_annot(os.path.join(args.data, test_annot_aug_file)).T

    leaves = get_leaf_nodes(os.path.join(args.data, leaf_nodes_file))
    singleton_nodes = get_leaf_nodes(os.path.join(args.data, singleton_nodes_file))

    semantic_paths = get_semantic_paths_or_layers(os.path.join(args.data, sp_file))
    node_layers = get_semantic_paths_or_layers(os.path.join(args.data, node_layer_file))

    save_path = os.path.join(args.data, 'GCN_ML_Binary_Datasets')

    if not os.path.exists(save_path):
        os.makedirs(save_path)


    save_file_pos = open(os.path.join(save_path, 'iaprtc12_positive_samples_' + str(args.samples) + '.csv'), 'w')
    save_file_neg = open(os.path.join(save_path, 'iaprtc12_negative_samples_' + str(args.samples) + '.csv'), 'w')
    pos_writer = csv.writer(save_file_pos)
    neg_writer = csv.writer(save_file_neg)

    for lbl in tqdm(range(train_annot_aug.shape[1])):
        positive_samples, negative_samples = positive_negative_split(lbl, 
                                    args.samples, semantic_paths, 
                                    node_layers, train_annot_aug, singleton_nodes)
        pos_writer.writerow(positive_samples)
        neg_writer.writerow(negative_samples)

    save_file_neg.close()
    save_file_pos.close()

if __name__ == '__main__':
    main_dataset_prep()