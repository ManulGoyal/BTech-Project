import pickle
import os
import numpy as np 
import scipy.io

path_to_word_embeddings = os.path.join('data', 'glove_word_embeddings', 'glove.6B.300d.txt')
path_to_dataset_dictionary = os.path.join('misc', 'esp_data.mat')
path_to_save_embeddings_of_dictionary = os.path.join('data', 'espgame', 'espgame_glove_word2vec.pkl')
mat = scipy.io.loadmat(path_to_dataset_dictionary)
mat = mat['dict']
dictionary = []
embeddings = {}

for keyword in mat:
    keyword = keyword[0][0]
    dictionary.append(keyword)

f = open(path_to_word_embeddings, 'r', encoding="utf-8")
for l in f:
    s = l.split(' ')
    embeddings[s[0]] = np.array(s[1:], dtype='float32')
f.close()

num_classes = len(dictionary)
dim_embedding = 300
dict_embeddings = np.empty((num_classes, dim_embedding), dtype='float32')

for i, label in enumerate(dictionary):
    if label not in embeddings.keys():
        print (label)
    else:
        dict_embeddings[i] = embeddings[label]

with open(path_to_save_embeddings_of_dictionary, 'wb') as f:
    pickle.dump(dict_embeddings, f)
f.close()