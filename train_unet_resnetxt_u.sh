
#######
# First parameter is output path
########
GPU=2,3
SAVE_PATH=./unet_segmenter_rnxtu_50x224_bce_v7
CONTINUE=""
LearningRateDecay=1e-4
LearningRate=1e-4
MODEL=image224_segmenter_resnet

iter=0
if [ "$1" == "continue" ]; then
   CONTINUE="--continue $SAVE_PATH/checkpoint.t7"
   iter=1
fi


 

while :
do
if [ $iter -gt 0 ]; then
  if [ -f $SAVE_PATH/train.log ]; then
     RC=$(cat $SAVE_PATH/test.log | wc -l)
     RC=$(($RC-1))
     tail -$RC $SAVE_PATH/test.log > $SAVE_PATH/train.log
  else
     cp $SAVE_PATH/test.log $SAVE_PATH/train.log
  fi
  if [ "$CONTINUE" == "" ]; then
     CONTINUE="--continue $SAVE_PATH/checkpoint.t7" 
  fi
fi
# start train
export CUDA_VISIBLE_DEVICES=$GPU; th ./train.lua \
 -i ./data/unet_tsx224/ \
 -s $SAVE_PATH \
 -b 6 \
 -r $LearningRate \
 --learningRateDecay $LearningRateDecay \
 --model $MODEL \
 --net_config "{cinput_planes=3, image_size=224, class_count=3, resnet='xt_50U', baseWidth=4, cardinality=32}" \
 --provider_config "{provider='datasets/h5-mask-provider', image_size=224}" \
 --use_optnet 0 \
 --epoch_step 100 \
 --max_epoch 100000 \
 --optim adam \
 --criterion SpatialBCE \
 --backend cudnn $CONTINUE \
 --checkpoint ./checkpoints 
 #--continue ./VGG_LUNG_AUG_SLarge/checkpoint.t7 \
 #--min_save_error -2.08

iter=$(($iter+1))
done
