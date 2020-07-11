from GCN import *
import argparse
import csv
import os
import numpy as np
import math

parser = argparse.ArgumentParser()
parser.add_argument('data', metavar='DIR',
                    help='path to dataset (e.g. data/')
parser.add_argument('--samples', type=int, default=2000, metavar='SAMPLES',
                    help='approx. upper bound on positive and negative samples for each label')
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.5, type=float,
                    metavar='LR', help='learning rate')                    

def read_csv(file, dtype):
    with open(file) as f:
        csvf = csv.reader(f)
        return [[dtype(x) for x in row] for row in csvf]

def main_iaprtc12():
    args = parser.parse_args()

    pos_file = os.path.join(args.data, 'GCN_ML_Binary_Datasets', f'iaprtc12_positive_samples_{args.samples}.csv')
    neg_file = os.path.join(args.data, 'GCN_ML_Binary_Datasets', f'iaprtc12_negative_samples_{args.samples}.csv')
    train_feature_file = os.path.join(args.data, 'iaprtc12_data_vggf_pca_train.txt')
    test_feature_file = os.path.join(args.data, 'iaprtc12_data_vggf_pca_test.txt')
    test_annot_file = os.path.join(args.data, 'annotation', 'iaprtc12_test_annot.csv')

    pos_samples = read_csv(pos_file, dtype=int)
    neg_samples = read_csv(neg_file, dtype=int)
    test_annot = np.asarray(read_csv(test_annot_file, dtype=int), dtype=int)
    train_features = np.asarray(read_csv(train_feature_file, dtype=float), dtype=float)
    test_features = np.asarray(read_csv(test_feature_file, dtype=float), dtype=float)

    num_classes = test_annot.shape[1]
    feat_dim = train_features.shape[1]

    for i in range(train_features.shape[0]):
        train_features[i] = train_features[i] / (math.sqrt(sum(train_features[i] ** 2)))
    
    for i in range(test_features.shape[0]):
        test_features[i] = test_features[i] / (math.sqrt(sum(test_features[i] ** 2)))

    print(len(pos_samples))


    # total_features = np.concatenate((train_features, test_features), axis=0)

    # train_features_agg = findNeighbourAggregation(train_features)
    # test_features_agg = doAggregation(total_features, test_features)

    for i in range(1):
        pos_samples_i = pos_samples[i]
        neg_samples_i = neg_samples[i]
        train_features_i = train_features[pos_samples_i+neg_samples_i]
        total_features_i = np.concatenate((train_features_i, test_features), axis=0)
        train_features_agg_i = findNeighbourAggregation(train_features_i)
        test_features_agg_i = doAggregation(total_features_i, test_features)
        train_labels_i = [[1] for _ in pos_samples_i]
        train_labels_i += [[0] for _ in neg_samples_i]
        train_labels_i = np.asarray(train_labels_i, dtype=int)
        test_labels_i = [[tag] for tag in test_annot[:, i]]
        test_labels_i = np.asarray(test_labels_i, dtype=int)
        W, Z = trainGCN(train_features_agg_i, train_labels_i, feat_dim, epochs, lr)
        trainAcc = computeAcc(W, Z, train_features_agg_i, train_labels_i)
        testAcc = computeAcc(W, Z, test_features_agg_i, test_labels_i)

        print('Train acc: ' + str(trainAcc))
        print('Test acc: ' + str(testAcc))

if __name__ == '__main__':
    main_iaprtc12()