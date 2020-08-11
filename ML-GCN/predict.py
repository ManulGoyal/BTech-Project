import pickle
import numpy as np
import csv

def write_csv(file, matrix):
    with open(file, 'w') as f:
        csvw = csv.writer(f)
        for row in matrix:
            csvw.writerow(row)

f = open('iaprtc12_mlgcn_scores.pkl', 'rb')
scores1 = pickle.load(f)
scores2 = scores1.copy()
f.close()

scores1[scores1 > 0] = 1
scores1[scores1 < 0] = 0
scores1 = np.array(scores1, dtype=int)

write_csv('iaprtc12_mlgcn_bin.csv', scores1)

sorts = np.zeros(scores2.shape)
for i in range(scores2.shape[0]):
    row = scores2[i]
    row = row.argsort()[::-1]
    sorts[i, row[0:5]] = 1

print(sorts.sum(axis=1))
write_csv('iaprtc12_mlgcn_top5.csv', sorts)
