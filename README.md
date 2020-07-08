# ML-GCN.pytorch
PyTorch implementation of [Multi-Label Image Recognition with Graph Convolutional Networks](https://arxiv.org/abs/1904.03582), CVPR 2019.



### Requirements
Please, install the following packages
- numpy
- torch
- torchnet
- torchvision
- tqdm

### Download pretrain models
checkpoint/coco ([GoogleDrive](https://drive.google.com/open?id=1ivLi1Rc-dCUmN1ProcMk76zxF1DSvlIk))

checkpoint/voc ([GoogleDrive](https://drive.google.com/open?id=1lhbmW5g-Mo9KgI07nmc1kwSbEnb6t-YA))

or

[Baidu](https://pan.baidu.com/s/17j3lTjMRmXvWHT86zhaaVA)

### Options
- `lr`: learning rate
- `lrp`: factor for learning rate of pretrained layers. The learning rate of the pretrained layers is `lr * lrp`
- `batch-size`: number of images per batch
- `image-size`: size of the image
- `epochs`: number of training epochs
- `evaluate`: evaluate model on validation set
- `resume`: path to checkpoint

### Demo IAPR-TC 12
```sh
python3 ML-GCN/demo_iaprtc12_gcn.py data/iaprtc12 --image-size 448 --batch-size 32 --resume checkpoint/iaprtc/checkpoint.pth.tar
```

### Demo VOC 2007
```sh
python3 ML-GCN/demo_voc2007_gcn.py data/voc --image-size 448 --batch-size 32 --resume checkpoint/voc2007/checkpoint.pth.tar
```

### Demo COCO 2014
```sh
python3 ML-GCN/demo_coco_gcn.py data/coco --image-size 448 --batch-size 32 --resume checkpoint/coco/coco_checkpoint.pth.tar
```

## Citing this repository
If you find this code useful in your research, please consider citing us:

```
@inproceedings{ML-GCN_CVPR_2019,
author = {Zhao-Min Chen and Xiu-Shen Wei and Peng Wang and Yanwen Guo},
title = {{Multi-Label Image Recognition with Graph Convolutional Networks}},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
year = {2019}
}
```
## Reference
This project is based on https://github.com/durandtibo/wildcat.pytorch

## Tips
If you have any questions about our work, please do not hesitate to contact us by emails.
