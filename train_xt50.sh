
#######
# First parameter is output path
########
GPU=2
SAVE_PATH=./cervix_classifier_transfer_learning_50xt
RESNET=resnext_50_32x4d
CONTINUE=""
LearningRateDecay=1e-7
LearningRate=1e-3
MODEL=resnet-xxx-fb
#_spatial_transformer
FC_CONFIG=',fc={{size=2048,bn=true,lrelu=0.1,dropout=0.3}}'
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
 -i ./data/nn_ts_rest/ \
 -s $SAVE_PATH \
 -b 10 \
 -r $LearningRate \
 --learningRateDecay $LearningRateDecay \
 --lr_decay_sheduler '{[25]=0.5}' \
 --model $MODEL \
 --net_config "{cinput_planes=3, image_size=224, class_count=3, model_file='$RESNET.t7', gradiend_decrease=0, localization_resnet=false $FC_CONFIG}" \
 --provider_config "{provider='datasets/h5-dir-provider', image_size=224}" \
 --use_optnet 0 \
 --epoch_step 100 \
 --max_epoch 100000 \
 --optim sgd \
 --backend cudnn $CONTINUE
 #--checkpoint ./checkpoints
 #--continue ./VGG_LUNG_AUG_SLarge/checkpoint.t7 \
 #--min_save_error -2.08

iter=$(($iter+1))
done
