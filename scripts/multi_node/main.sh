# #!/bin/bash
# # Common part for all nodes
# export NCCL_IB_DISABLE=0
# export NCCL_IB_HCA=mlx5
# export NCCL_DEBUG=WARN
# export NCCL_IB_GID_INDEX=3

# # Conda activation (must be executed on all nodes)
# source /m2v_intern/liujie/miniconda3/etc/profile.d/conda.sh
# conda deactivate
# conda activate /m2v_intern/liujie/miniconda3/envs/flow_grpo

# # Project root directory (modify according to actual path)
# PROJECT_ROOT="/m2v_intern/liujie/research/flow_grpo"
# cd $PROJECT_ROOT

# MASTER_PORT=19001
# RANK=0
# MASTER_ADDR=10.82.139.22
# # Launch command (parameters automatically read from accelerate_multi_node.yaml)
# accelerate launch --config_file $PROJECT_ROOT/scripts/accelerate_configs/multi_node.yaml \
#     --num_machines 3 --num_processes 24 \
#     --machine_rank ${RANK} --main_process_ip ${MASTER_ADDR} --main_process_port ${MASTER_PORT} \
#     scripts/train_sd3.py \
#     --config config/dgx.py:geneval_sd3
