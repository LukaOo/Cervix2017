
#######
# First parameter is output path
########
GPU=1
SAVE_PATH=./cervix_classifier_transfer_learning_50
RESNET=resnet-50
CONTINUE=""
LearningRateDecay=1e-3
LearningRate=0.01
MODEL=resnet-xxx-fb
#_spatial_transformer
FC_CONFIG=',fc={{size=2048,bn=true,lrelu=0.1,dropout=0.3},{size=1024,bn=true,lrelu=0.1,dropout=0.3},{size=512,bn=true,lrelu=0.1,dropout=0.3}}'
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
 -i ./data2/nn_ts/ \
 -s $SAVE_PATH \
 -b 10 \
 -r $LearningRate \
 --learningRateDecay $LearningRateDecay \
 --model $MODEL \
 --net_config "{cinput_planes=3, image_size=224, class_count=3, model_file='$RESNET.t7', localization_resnet=false $FC_CONFIG}" \
 --provider_config "{provider='datasets/h5-dir-provider', image_size=224}" \
 --use_optnet 0 \
 --epoch_step 100 \
 --max_epoch 100000 \
 --optim sgd \
 --backend cudnn $CONTINUE 
 # --checkpoint ./checkpoints
 #--continue ./VGG_LUNG_AUG_SLarge/checkpoint.t7 \
 #--min_save_error -2.08

iter=$(($iter+1))
done
