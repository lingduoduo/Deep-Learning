export PATH=/data/zhengwj/anaconda3/bin:$PATH
FORCE_TORCHRUN=4 CUDA_VISIBLE_DEVICES=0,2,4,5 llamafactory-cli train ./qwen3_14B_full_ds_3_sft.yaml
