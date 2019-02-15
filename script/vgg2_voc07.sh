# !/usr/bin/env bash
gpu=${1:0:1}
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
python train_end2end.py --gpu $1 --network vgg2 --prefix model/vgg2
python test.py --gpu $gpu --network vgg2 --prefix model/vgg2
