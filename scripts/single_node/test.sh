
CUDA_VISIBLE_DEVICES=7 accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 test_wan2_1_vace.py --config config/dgx.py:vace