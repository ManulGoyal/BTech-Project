import numpy as np
import pickle
import os
import csv
from tqdm import tqdm

path_to_model = '../data/iaprtc12/GCN_ML_Binary_Datasets/model/model.pkl'
path_to_features = '../data/iaprtc12/GCN_ML_Binary_Datasets/agg_features_test_2000.pkl'
path_to_annot = '../data/iaprtc12/annotation/iaprtc12_test_annot_del.csv'

def read_csv(file, dtype):
    with open(file) as f:
        csvf = csv.reader(f)
        return [[dtype(x) for x in row] for row in csvf]

def write_csv(file, matrix):
    with open(file, 'w') as f:
        csvw = csv.writer(f)
        for row in matrix:
            csvw.writerow(row)

def read_model_weights(file):
    weights = {}
    with open(file, 'rb') as f:
        model = pickle.load(f)
        for i, j in model.items():
            weights[i] = (j['W'], j['Z'])
    f.close()
    return weights

test_annot = read_csv(path_to_annot, int)

with open(path_to_features, 'rb') as f:
    test_features_agg = pickle.load(f)
f.close()

weights = read_model_weights(path_to_model)

test_scores = np.zeros((1957,291))
for i in tqdm(range(0, 291)):
    # print(test_features_agg[i].shape, weights[i][0].shape, weights[i][1].shape)
    X1 = np.dot(test_features_agg[i], weights[i][0])
    # print(X1.shape)
    X11 = 1 / (1 + np.exp(-(X1)))
    X2 = np.dot(X11, weights[i][1])
    # print(X2.shape)
    X22 = 1 / (1 + np.exp(-(X2)))

    X22 = X22.reshape((1957,))
    test_scores[:, i] = X22

write_csv('test_scores_2.csv', test_scores)


