from GCN import *
import argparse
import csv
import os
import numpy as np
import math
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--samples', type=int, default=2000, metavar='SAMPLES',
                    help='approx. upper bound on positive and negative samples for each label')
parser.add_argument('--start', type=int, default=0, metavar="START",
                    help='index of label to start aggregation at')
parser.add_argument('--end', type=int, default=290, metavar="END",
                    help='index of label to end aggregation at')

def read_csv(file, dtype):
    with open(file) as f:
        csvf = csv.reader(f)
        return [[dtype(x) for x in row] for row in csvf]

def main_aggregate():
    args = parser.parse_args()
    num_classes = 291

    pos_file = os.path.join(args.data, 'GCN_ML_Binary_Datasets', f'iaprtc12_positive_samples_{args.samples}.csv')
    neg_file = os.path.join(args.data, 'GCN_ML_Binary_Datasets', f'iaprtc12_negative_samples_{args.samples}.csv')
    train_feature_file = os.path.join(args.data, 'iaprtc12_data_vggf_pca_train.txt')
    test_feature_file = os.path.join(args.data, 'iaprtc12_data_vggf_pca_test.txt')

    test_agg_save_file = os.path.join(args.data, 'GCN_ML_Binary_Datasets', f'agg_features_test_{args.samples}.pkl')
    train_agg_save_file = os.path.join(args.data, 'GCN_ML_Binary_Datasets', f'agg_features_train_{args.samples}.pkl')

    pos_samples = read_csv(pos_file, dtype=int)
    neg_samples = read_csv(neg_file, dtype=int)

    train_features = np.asarray(read_csv(train_feature_file, dtype=float), dtype=float)
    test_features = np.asarray(read_csv(test_feature_file, dtype=float), dtype=float)

    feat_dim = train_features.shape[1]

    for i in range(train_features.shape[0]):
        train_features[i] = train_features[i] / (math.sqrt(sum(train_features[i] ** 2)))
    
    for i in range(test_features.shape[0]):
        test_features[i] = test_features[i] / (math.sqrt(sum(test_features[i] ** 2)))

    agg_features_train = {}
    agg_features_test = {}

    if os.path.exists(test_agg_save_file):
        tmp = open(test_agg_save_file, 'rb')
        agg_features_test = pickle.load(tmp)
        tmp.close()
    if os.path.exists(train_agg_save_file):
        tmp = open(train_agg_save_file, 'rb')
        agg_features_train = pickle.load(tmp)
        tmp.close() 

    for i in range(args.start, args.end+1):
        pos_samples_i = pos_samples[i]
        neg_samples_i = neg_samples[i]
        train_features_i = train_features[pos_samples_i+neg_samples_i]
        total_features_i = np.concatenate((train_features_i, test_features), axis=0)
        train_features_agg_i = findNeighbourAggregation(train_features_i)
        test_features_agg_i = doAggregation(total_features_i, test_features)
        
        agg_features_test[i] = test_features_agg_i
        agg_features_train[i] = train_features_agg_i

    test_agg_file = open(test_agg_save_file, 'wb')
    train_agg_file = open(train_agg_save_file, 'wb')

    pickle.dump(agg_features_test, test_agg_file)
    pickle.dump(agg_features_train, train_agg_file)

    test_agg_file.close()
    train_agg_file.close()

if __name__ == '__main__':
    main_aggregate()