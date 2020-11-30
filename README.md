# Disentangling Label Distribution for Long-tailed Visual Recognition

- This codebase is built on [Causal Norm](https://github.com/KaihuaTang/Long-Tailed-Recognition.pytorch).

## Install

```
conda create -n longtail pip python=3.7 -y
source activate longtail
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install pyyaml tqdm matplotlib sklearn h5py tensorboard
```

## Training

### Preliminaries

- Download pretrained caffe resnet152 model for Places-LT: please refer to [link](https://github.com/zhmiao/OpenLongTailRecognition-OLTR#download-caffe-pre-trained-models-for-places_lt-stage_1-training).

- Prepare dataset: CIFAR-100, Places-LT, ImageNet-LT, iNaturalist 2018
  - Please download those datasets following [Decoupling](https://github.com/facebookresearch/classifier-balancing#dataset).

### CIFAR-100 training

For CIFAR-100 with imbalance ratio 0.01, using LADE:

```
python main.py --seed 1 --cfg config/CIFAR100_LT/lade.yaml --exp_name lade2021/cifar100_imb0.01_lade --cifar_imb_ratio 0.01 --remine_lambda 0.01 --alpha 0.1 --gpu 0
```

### Places-LT training

For PC Softmax:

```
python main.py --seed 1 --cfg config/Places_LT/ce.yaml --exp_name lade2021/places_pc_softmax --lr 0.05 --gpu 0,1,2,3
```

For LADE:

```
python main.py --seed 1 --cfg config/Places_LT/lade.yaml --exp_name lade2021/places_lade --lr 0.05 --remine_lambda 0.1 --alpha 0.005 --gpu 0,1,2,3
```

### ImageNet-LT training

For LADE:

```
python main.py --seed 1 --cfg config/ImageNet_LT/lade.yaml  --exp_name lade2021/imagenet_lade --lr 0.05 --remine_lambda 0.5 --alpha 0.05 --gpu 0,1,2,3
```

### iNaturalist18 training

For LADE:

```
python main.py --seed 1 --cfg ./config/iNaturalist18/lade.yaml --exp_name lade2021/inat_lade --lr 0.1 --alpha 0.05 --gpu 0,1,2,3
```

## Evaluate on shifted test set & Confidence calibration
For Imagenet (Section 4.3, 4.4):
```
./notebooks/imagenet-shift-calib.ipynb
```

For CIFAR-100 (Supplementary material):
```
./notebooks/cifar100-shift-calib.ipynb
```
