CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
scripts/train_pytorch.py pi05_libero \
--exp-name=pi05_libero \
--num-action-heads=1 \
--batch-size=32 \
--pytorch-weight-path=pi05_base_torch \
--num-train-steps=30000 \
--seed=42