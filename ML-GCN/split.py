import numpy as np
import csv
import math

# labels_to_consider = [1,3,4,6,7,10,11,13,14,17,19,24,29,31,31,37,38,40,42,
# 44,45,46,46,47,48,49,49,50,52,53,57,58, 59, 61, 62, 64, 66, 67, 68, 69, 72, 74, 74, 75, 
# 76, 76, 77, 81, 83, 83, 84, 85, 87, 89, 91, 92, 94, 94, 95, 97, 98, 100,102,103,107,107,
# 108,110,110,111,113,118,119,120,121,122,124,125,126,127,128,129,131,132,133,134,135,136,
# 137,139,139,140,141,141,142,143,147,147,149,159,160,160,161,163,164,165,172,173,175,176,
# 179,185,189,189,192,193,195,198,201,202,205,207,209,210,211,214,215,216,218,221,222,224,
# 227,230,232,235,236,238,239,242,243,244,244,247,252,253,254,255,256,257,257,258,260,261,
# 262,264,265,268,268,269,271,273,274,275,276,276,277,278,279,279,282,283,285,288,288,289
# ]

# def get_min_max_ocuurence(dataset,val_index):
#     smallest = 1.0
#     largest = 0.0 
#     # print(val_index[0])
#     num_labels = np.size(dataset,0)
#     num_images = np.size(val_index)
#     label_occur = np.sum(dataset,axis = 1)
    
#     label_in_val = np.zeros(num_labels)

#     for i in range(num_images):
#         for j in range(num_labels):
#             if dataset[j][int(val_index[i])] == 1:
#                 label_in_val[j] = label_in_val[j] + 1
    
#     # print(label_in_val)
#     # print(label_occur)
#     countg = 0
#     countl = 0
#     for i in range(num_labels):
#         smallest = min(smallest,label_in_val[i]/label_occur[i])
#         largest = max(largest,label_in_val[i]/label_occur[i])
#         if(label_in_val[i]/label_occur[i]) > 0.30 :
#             countg = countg + 1
#         if(label_in_val[i]/label_occur[i]) < 0.10 :
#             countl = countl + 1
            
#     print(countg)
#     print(countl)
#     print('smallest :')
#     print(smallest)
#     print('largest :')
#     print(largest)
#     return

# def train_set_index(dataset,start):

#     num_labels = np.size(dataset,0)
#     num_images = np.size(dataset,1)
#     label_occur = np.sum(dataset,axis = 1)
#     label_occur = (label_occur * start)
#     # print(label_occur)

#     image_map = np.zeros(num_images)   
#     label_map = np.zeros(num_labels)
#     for x in labels_to_consider:
#         label_map[x] = 1

#     for i in range(num_labels):
#         if label_map[i] == 0:
#             continue
#         for k in range(num_images):
#             if dataset[i][k] == 1 and image_map[k] == 0:
#                 image_map[k] = 1
#                 label_occur[i] = label_occur[i] - 1
#             if label_occur[i] <= 0:
#                 break

#     train_index = np.array([])
#     val_index = np.array([])

#     for x in range(num_images):
#         if int(image_map[x]) != 1:
#             train_index = np.append(train_index,[x])
#         else:
#             val_index = np.append(val_index,[x])
    
#     get_min_max_ocuurence(dataset,val_index)
#     return train_index


# def run_local():
#     dataset = np.arange(5140515).reshape(291,17665)
#     with open('train.txt') as csv_file:
#         csv_reader = csv.reader(csv_file, delimiter=',')
#         i = 0
#         j = 0
#         for row in csv_reader:
            
#             for c in row:
#                 dataset[i][j] = c 
#                 i = i+1
#             j = j+1
#             i = 0

#     # print(np.size(dataset,axis = 0))
#     # print(np.size(dataset,axis = 1))
#     """
#     nearly divides datset in 20-80 weightage because of the fact that
#     each image has 4 labels on average so taking 5% of each label takes 
#     20% of images 
#     """
#     v = train_set_index(dataset,0.05)   
#     print(v)
#     print(np.size(v))


"""
trick = 1 :A[i][j] = adj_A[i][j]*(s[i]/s[j])^scalar 
trick = 2 :A[i][j] = (s[i]/s[j])*scalar
trick = 3 :A[i][j] = adj_A[i][j]*(s[i]-s[j])
"""
def gen_A_with_semantic_weights_iaprtc12(adj_A, trick, scalar = 1):
    num_classes = 291
    sz = num_classes*num_classes
    semantic_weight = [0]*num_classes

    # gen_A = np.arrange(sz).reshape(num_classes,num_classes)
    gen_A = []
    gen_A = [ [ 0 for y in range( num_classes ) ] 
             for x in range( num_classes) ] 
    with open('iaprtc12_weights.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        j = 0
        for row in csv_reader:
            for col in row:
                semantic_weight[j] = float(col)
                j = j + 1
    # num_classes = 2
    # print(j)
    # return 1
    # print(semantic_weight[0])
    # print(semantic_weight[1])
    for i in range(num_classes):
        for j in range(num_classes):
            if trick == 1:
                gen_A[i][j] = adj_A[i][j]*pow(semantic_weight[i]/semantic_weight[j],scalar)
            elif trick == 2:
                gen_A[i][j] = pow(semantic_weight[i]/semantic_weight[j],scalar)
            elif trick == 3:
                # print(type(adj_A[i][j]))
                # print(type(semantic_weight[i]))
                gen_A[i][j] = adj_A[i][j]*(semantic_weight[i]-semantic_weight[j])
            else:
                gen_A[i][j] = adj_A[i][j]  
    # print(gen_A)

    return np.asarray(gen_A, dtype=float)

# A = [[1,1],[2,1]]
# B = gen_A_with_semantic_weights_iaprtc12(A,2,2)
# print(B[0][0])
# print(B[0][1])
# print(B[1][0])
# print(B[1][1])

