MODEL_FLAGS="--attention_resolutions 16,8 --channel 200 --channel_mult 1,2,2 --class_cond False --diffusion_steps 1000 --image_size 48 --noise_schedule cosine --num_channels 128 --num_head_channels -1 --num_res_blocks 2 --learn_sigma True" # --dropout 0.1 --resblock_updown True --use_fp16 True --use_scale_shift_norm True
DATASET="Indian_Pines" 

python feature_extract.py --dataset ${DATASET} --exp configs/${DATASET}.json $MODEL_FLAGS $1 $2