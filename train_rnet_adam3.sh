
#######
# First parameter is output path
########
GPU=3
SAVE_PATH=./restorenet_rnxt_50.v5
CONTINUE=""
LearningRateDecay=1e-4
LearningRate=0.01
MODEL=image_restoration_resnet

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
 -i ./data/rnet_ts/ \
 -s $SAVE_PATH \
 -b 6 \
 -r $LearningRate \
 --learningRateDecay $LearningRateDecay \
 --lr_decay_sheduler '{}' \
 --model $MODEL \
 --net_config "{cinput_planes=3, image_size=224, class_count=3, resnet='xt_50', baseWidth=40, cardinality=2}" \
 --provider_config "{provider='datasets/h5-aenc-provider', image_size=224}" \
 --use_optnet 1 \
 --epoch_step 100 \
 --max_epoch 100000 \
 --optim sgd \
 --criterion PL \
 --perceptual_config  '{ normalize_grad=true, calc_only_target=false, model = "perceptual_loss_model", model_prototxt="./pretrained/VGG_ILSVRC_19_layers_deploy.prototxt", model_file="./pretrained/VGG_ILSVRC_19_layers.caffemodel", layer_name={["relu4_1"]=1, ["relu3_2"]=1, ["relu2_1"]=1}, error_weight=1.0 } ' \
 --backend cudnn $CONTINUE
 #--checkpoint ./checkpoints 
 #--continue ./VGG_LUNG_AUG_SLarge/checkpoint.t7 \
 #--min_save_error -2.08

iter=$(($iter+1))
done
