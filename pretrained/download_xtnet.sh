#!/bin/sh

#
# script to download pretrained facebook neural nets
# resnet-18
# resnet-34
# resnet-50
# resnet-101
# resnet-152
# resnet-200
#
if [ '$1' == '' ]; then
   echo 'Missing neural net'
   exit 1
fi

wget https://s3.amazonaws.com/resnext/imagenet_models/$1"_32x4d".t7 -O ./pretrained/$1"_32x4d".t7