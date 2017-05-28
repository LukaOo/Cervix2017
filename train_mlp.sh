
#######
# First parameter is output path
########
GPU=1
SAVE_PATH=./mlp
CONTINUE=""
LearningRateDecay=1e-4
LearningRate=1e-5
MODEL=mlp
#_spatial_transformer
FC_CONFIG=',fc={{size=2048,bn=true,lrelu=0.1,dropout=0.9},{size=1024,bn=true,lrelu=0.1,dropout=0.9},{size=512,bn=true,lrelu=0.1,dropout=0.9}}'
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
 -i ./data/nn_ts_emb/ \
 -s $SAVE_PATH \
 -b 128 \
 -r $LearningRate \
 --learningRateDecay $LearningRateDecay \
 --model $MODEL \
 --net_config "{class_count=3, inputsize=2048 $FC_CONFIG }" \
 --provider_config "{provider='datasets/h5-mlp-provider' }" \
 --use_optnet 0 \
 --epoch_step 200 \
 --max_epoch 100000 \
 --optim adam \
 --criterion CrossEntropy \
 --backend cudnn $CONTINUE 
 # --crit_config "{weights={0.1, 1}}" \
 # --checkpoint ./checkpoints
 #--continue ./VGG_LUNG_AUG_SLarge/checkpoint.t7 \
 #--min_save_error -2.08

iter=$(($iter+1))
done
