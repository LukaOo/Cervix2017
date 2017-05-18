
INPUT_PATH="$1"
OUTPUT_PATH="$2"
GPU=2
MODEL="$3"
if [ ! -d "$OUTPUT_PATH" ]
then
  mkdir "$OUTPUT_PATH"
fi


ls -1 $INPUT_PATH | while read f 
do  
   if [ ! -d "$OUTPUT_PATH/$f" ]
   then
      mkdir "$OUTPUT_PATH/$f"
   fi
   export CUDA_VISIBLE_DEVICES=$GPU; th scripts/restore_cervix_image.lua -i $INPUT_PATH/$f -o $OUTPUT_PATH/$f --model $MODEL --backend cudnn --use_optnet 1
done
