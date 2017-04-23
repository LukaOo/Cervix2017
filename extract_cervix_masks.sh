
INPUT_PATH="$1"
OUTPUT_PATH="$2"
GPU=3
MODEL="$3"

ls -1 $INPUT_PATH | while read f 
do  
   if [ ! -d "$OUTPUT_PATH/$f" ]
   then
      mkdir "$OUTPUT_PATH/$f"
   fi
   export CUDA_VISIBLE_DEVICES=$GPU; th scripts/extract_cervix_mask.lua -i $INPUT_PATH/$f -o $OUTPUT_PATH/$f --model $MODEL --backend cudnn
done
