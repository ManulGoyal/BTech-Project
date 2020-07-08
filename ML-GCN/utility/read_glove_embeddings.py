import pickle
import os
import numpy as np

path_to_word_embeddings = os.path.join('data', 'glove_word_embeddings', 'glove.6B.300d.txt')
path_to_dataset_dictionary = os.path.join('data', 'iaprtc', 'annotation', 'iaprtc12_dictionary.txt')
path_to_save_embeddings_of_dictionary = os.path.join('data', 'iaprtc', 'iaprtc_glove_word2vec.pkl')
dictionary = []
embeddings = {}

synonym_words = {
    'bedcover' : 'bedspread',
    'table-cloth' : 'tablecloth',
    'tee-shirt' : 't-shirt'
}

with open(path_to_dataset_dictionary) as f:
    for l in f:
        s = l.split('\n')
        dictionary.append(s[0])

f.close()

with open(path_to_word_embeddings) as f:
    for l in f:
        s = l.split(' ')
        embeddings[s[0]] = np.array(s[1:], dtype='float32')

f.close()

num_classes = len(dictionary)
dim_embedding = 300
dict_embeddings = np.empty((num_classes, dim_embedding), dtype='float32')

for i, label in enumerate(dictionary):
    if label not in embeddings.keys():
        dict_embeddings[i] = embeddings[synonym_words[label]]
    else:
        dict_embeddings[i] = embeddings[label]

with open(path_to_save_embeddings_of_dictionary, 'wb') as f:
    pickle.dump(dict_embeddings, f)
f.close()