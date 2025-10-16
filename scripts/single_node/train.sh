
CUDA_VISIBLE_DEVICES=4,5,6,7 accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=4 --main_process_port 29502 train_wan2_1_vace.py --config config/dgx.py:vace

# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 29502 train_wan2_1_vace.py --config config/dgx.py:vace
