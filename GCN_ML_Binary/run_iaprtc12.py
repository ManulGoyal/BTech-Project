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
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--lr', '--learning-rate', default=0.5, type=float,
                    metavar='LR', help='learning rate')                    
parser.add_argument('--start', type=int, default=0, metavar="START",
                    help='index of label to start training at')
parser.add_argument('--end', type=int, default=290, metavar="END",
                    help='index of label to end training at')
parser.add_argument('--model', type=str, default='GCN_ML_Binary_Datasets/model/model.pkl',
                    metavar="PATH", help='path to file to save model in')
parser.add_argument('--resume', dest='resume', action='store_true',
                    help='set this flag to resume training from model file last epoch')

def read_csv(file, dtype):
    with open(file) as f:
        csvf = csv.reader(f)
        return [[dtype(x) for x in row] for row in csvf]

def write_csv(file, matrix):
    with open(file) as f:
        csvw = csv.writer(f)
        for row in matrix:
            csvw.writerow(row)

def main_iaprtc12():
    args = parser.parse_args()
    
    pos_file = os.path.join(args.data, 'GCN_ML_Binary_Datasets', f'iaprtc12_positive_samples_{args.samples}.csv')
    neg_file = os.path.join(args.data, 'GCN_ML_Binary_Datasets', f'iaprtc12_negative_samples_{args.samples}.csv')
    # train_feature_file = os.path.join(args.data, 'iaprtc12_data_vggf_pca_train.txt')
    # test_feature_file = os.path.join(args.data, 'iaprtc12_data_vggf_pca_test.txt')
    test_annot_file = os.path.join(args.data, 'annotation', 'iaprtc12_test_annot.csv')
    delete_test_file = os.path.join(args.data, 'remove_images_test.csv')

    test_agg_save_file = os.path.join(args.data, 'GCN_ML_Binary_Datasets', f'agg_features_test_{args.samples}.pkl')
    train_agg_save_file = os.path.join(args.data, 'GCN_ML_Binary_Datasets', f'agg_features_train_{args.samples}.pkl')
    model_save_file = os.path.join(args.data, args.model)

    pos_samples = read_csv(pos_file, dtype=int)
    neg_samples = read_csv(neg_file, dtype=int)
    test_annot = np.asarray(read_csv(test_annot_file, dtype=int), dtype=int)
    # train_features = np.asarray(read_csv(train_feature_file, dtype=float), dtype=float)
    # test_features = np.asarray(read_csv(test_feature_file, dtype=float), dtype=float)
    delete_test_images = read_csv(delete_test_file, dtype=int)
    test_annot = np.delete(test_annot, delete_test_images, 0)

    test_agg_file = open(test_agg_save_file, 'rb')
    train_agg_file = open(train_agg_save_file, 'rb')

    test_agg_features = pickle.load(test_agg_file)
    train_agg_features = pickle.load(train_agg_file)

    num_classes = test_annot.shape[1]
    feat_dim = 536
    # print(train_agg_features[2].shape)
    # test_agg_features[2] = test_agg_features[2].reshape(1957, 536)
    # for i in range(train_features.shape[0]):
    #     train_features[i] = train_features[i] / (math.sqrt(sum(train_features[i] ** 2)))
    
    # for i in range(test_features.shape[0]):
    #     test_features[i] = test_features[i] / (math.sqrt(sum(test_features[i] ** 2)))

    # print(len(pos_samples))
    # print(test_annot.shape[0])

    # total_features = np.concatenate((train_features, test_features), axis=0)

    # train_features_agg = findNeighbourAggregation(train_features)
    # test_features_agg = doAggregation(total_features, test_features)
    basedir_model = os.path.split(model_save_file)[0]
    if not os.path.exists(basedir_model):
        os.makedirs(basedir_model)

    models = {}

    if os.path.exists(model_save_file):
        f = open(model_save_file, 'rb')
        models = pickle.load(f)
        f.close()
    else:
        print(f"No file at {model_save_file} found")

    
    for i in range(args.start, args.end+1):
        print('Training for label ' + str(i+1))
        pos_samples_i = pos_samples[i]
        neg_samples_i = neg_samples[i]
        # train_features_i = train_features[pos_samples_i+neg_samples_i]
        # total_features_i = np.concatenate((train_features_i, test_features), axis=0)
        train_features_agg_i = train_agg_features[i]
        # print(train_features_agg_i.shape)
        test_features_agg_i = test_agg_features[i]
        # print(test_features_agg_i.shape)
        train_labels_i = [[1] for _ in pos_samples_i]
        train_labels_i += [[0] for _ in neg_samples_i]
        train_labels_i = np.asarray(train_labels_i, dtype=int)
        test_labels_i = [[tag] for tag in test_annot[:, i]]
        test_labels_i = np.asarray(test_labels_i, dtype=int)
        
        W = Z = None
        last_epochs = 0
        if args.resume:
            if i in models.keys():
                W, Z, last_epochs = models[i]['W'], models[i]['Z'], models[i]['epochs']
        if args.epochs > last_epochs:
            W, Z = trainGCNBest(train_features_agg_i, train_labels_i, feat_dim, args.epochs - last_epochs, i, 
            test_features_agg_i, test_labels_i, last_epochs, lr=args.lr, W=W, Z=Z)
            train_pred, trainAcc = computeAcc(W, Z, train_features_agg_i, train_labels_i)
            test_pred, testAcc = computeAcc(W, Z, test_features_agg_i, test_labels_i)
            model_i = {'W' : W, 'Z' : Z, 'epochs' : args.epochs, 'train_pred' : train_pred,
                       'train_acc' : trainAcc, 'test_pred' : test_pred, 'test_acc' : testAcc}
            print('Train acc: ' + str(trainAcc))
            print('Test acc: ' + str(testAcc))
            models[i] = model_i
        else:
            print('Epochs less than or equal to before!')

        print('Saving model...')
        f = open(model_save_file, 'wb')
        pickle.dump(models, f)
        f.close()
    

if __name__ == '__main__':
    main_iaprtc12()