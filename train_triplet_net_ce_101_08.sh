
#######
# First parameter is output path
########
GPU=3
SAVE_PATH=./triplet_net_ce_101_08
RESNET=resnet-101
CONTINUE=""
LearningRateDecay=1e-4
weightDecay=1e-3
LearningRate=1e-4
MODEL=siames_net
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
 --model $MODEL \
 --net_config "{cinput_planes=3, image_size=224, class_count=3, model_file='$RESNET.t7', fc_dropout=0.8, tripletnet=true }" \
 --provider_config "{provider='datasets/h5-dir-provider', image_size=224, siames_input=true, dual_target=true, triplets=true}" \
 --use_optnet 0 \
 --epoch_step 100 \
 --max_epoch 100000 \
 --optim sgd \
 --criterion DistanceRatio \
 --crit_config "{cross_entropy=true}" \
 --backend cudnn $CONTINUE 
 # --crit_config "{weights={0.1, 1}}" \
 # --checkpoint ./checkpoints
 #--continue ./VGG_LUNG_AUG_SLarge/checkpoint.t7 \
 #--min_save_error -2.08

iter=$(($iter+1))
done
