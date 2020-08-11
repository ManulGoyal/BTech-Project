### Approachs
All our models are explained in our report

### Models 
Baseline Approach -> KNN
ML-GCN -> ML-GCN
BIN-GCN -> GCN-ML-Binary
Metrics Calculation -> Parameters


### Requirements
Please, install the following packages
- numpy
- torch
- torchnet
- torchvision
- tqdm

### Options
- `lr`: learning rate
- `lrp`: factor for learning rate of pretrained layers. The learning rate of the pretrained layers is `lr * lrp`
- `batch-size`: number of images per batch
- `image-size`: size of the image
- `epochs`: number of training epochs
- `evaluate`: evaluate model on validation set
- `resume`: path to checkpoint
- `semantic`: generate adjacency matrix using semantic weights
- `trick`: method used for generating semantic matrix (1, 2 or 3)

### Demo IAPR-TC 12 ML-GCN
```sh
cd ML-GCN
python3 demo_iaprtc12_gcn.py ../data/iaprtc12 --image-size 448 --batch-size 32 --resume checkpoint/iaprtc/checkpoint.pth.tar
```
### Demo IAPR-TC 12 of ML-GCN using Semantic Matrix
```sh
cd ML-GCN
python3 demo_iaprtc12_gcn.py ../data/iaprtc12 --image-size 448 --batch-size 32 --semantic --resume checkpoint/iaprtc_semantic/checkpoint.pth.tar
```

### Demo VOC 2007 ML-GCN
```sh
python3 ML-GCN/demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 32 --resume checkpoint/voc2007/checkpoint.pth.tar
```

