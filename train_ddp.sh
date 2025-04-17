#!/bin/bash
#SBATCH --job-name=train_sam2
#SBATCH --partition=i64m1tga40u
#SBATCH -N 1                     
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4             
#SBATCH --qos=priority
#SBATCH -o Cityscapes_sam2_l.out       
#SBATCH -e Cityscapes_sam2_l.err            

export MASTER_PORT=$((10000 + $RANDOM % 20000))
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_TIMEOUT=7200

srun torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --master_port=$MASTER_PORT \
  train_ddp.py \
  --batch_size 4 \
  --lr=6e-5 \
  --num_epochs=10 \
  --dataset='Cityscapes' \
  --use_mem_bank \
  --exp_name "exp1" \
  --backbone "sam2_l"
echo "Done."
