
#######
# First parameter is output path
########
GPU=1,3
SAVE_PATH=./unet_segmenter
CONTINUE=""
LearningRateDecay=1e-4
LearningRate=0.01
MODEL=unet_with_resnet

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
 -i ./data2/unet_ts/ \
 -s $SAVE_PATH \
 -b 6 \
 -r $LearningRate \
 --learningRateDecay $LearningRateDecay \
 --model $MODEL \
 --net_config "{cinput_planes=3, image_size=512, class_count=3}" \
 --provider_config "{provider='datasets/h5-mask-provider', image_size=512}" \
 --use_optnet 0 \
 --epoch_step 100 \
 --max_epoch 100000 \
 --optim sgd \
 --criterion Dice \
 --backend cudnn $CONTINUE 
 #--continue ./VGG_LUNG_AUG_SLarge/checkpoint.t7 \
 #--min_save_error -2.08

iter=$(($iter+1))
done
