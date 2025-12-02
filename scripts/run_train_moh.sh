exp_name=pi05_moh_test_libero
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --master_port=25980 --nnodes=1 --nproc_per_node=4 \
scripts/train_pytorch_moh.py pi05_libero \
--exp-name=${exp_name} \
--batch-size=32 \
--pytorch-weight-path=pi05_base_torch \
--num-train-steps=30000 \
--horizons 3 6 9 12 15 18 21 24 27 30 \
--seed=42 \
> logs/${exp_name}.txt 2>&1 &