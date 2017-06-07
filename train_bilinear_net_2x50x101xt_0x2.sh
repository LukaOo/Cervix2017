
#######
# First parameter is output path
########
GPU=0,1
SAVE_PATH=./bilinear_net_2x50x101xt_0x2
RESNET=resnet-50
RESNET2=resnext_101_64x4d
CONTINUE=""
LearningRateDecay=1e-4
weightDecay=5e-3
LearningRate=5e-2
MODEL=bilinear_net
#_spatial_transformer
# FC_CONFIG=',fc={{size=2048,bn=true,lrelu=0.1,dropout=0.3},{size=1024,bn=true,lrelu=0.1,dropout=0.3},{size=512,bn=true,lrelu=0.1,dropout=0.3}}'
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
 -i ./data2/nn_ts_x224.merged/ \
 -s $SAVE_PATH \
 -b 10 \
 -r $LearningRate \
 --learningRateDecay $LearningRateDecay \
 --weightDecay $weightDecay \
 --lr_decay_sheduler '{[70]=0.5, [120]=0.5, [160]=0.5}' \
 --model $MODEL \
 --net_config "{cinput_planes=3, image_size=224, class_count=3, model_file='$RESNET.t7', model_file1='$RESNET2.t7', gradiend_decrease=0.0, fc_dropout=0.0 }" \
 --provider_config "{provider='datasets/h5-dir-provider', image_size=224, dual_target=true, bilinear=true, siames_input=true}" \
 --use_optnet 0 \
 --epoch_step 100 \
 --max_epoch 100000 \
 --optim sgd \
 --criterion CrossEntropy \
 --backend cudnn $CONTINUE 
 # --crit_config "{weights={0.1, 1}}" \
 # --checkpoint ./checkpoints
 #--continue ./VGG_LUNG_AUG_SLarge/checkpoint.t7 \
 #--min_save_error -2.08

iter=$(($iter+1))
done