EXP_NAME="pi05_test_moh_dynamic_inference"
RANKS=(0 1 2 3)
scale_ratio_list=(1.4 1.3 1.2 1.1)
VIDEO_OUT_PATH="./videos_dynamic/${EXP_NAME}_"
CONFIG="pi05_libero"
# download from https://huggingface.co/Timsty/mixture_of_horizons/tree/main
CKPT_DIR="pi05_moh_libero_3to30stride3"

for i in "${!scale_ratio_list[@]}"; do
    TORCHDYNAMO_DISABLE=1 python examples/libero_scripts/pi_moh_dynamic.py \
        --task_suite_name="libero_10" \
        --rank="${RANKS[$i]}" \
        --video_out_path="${VIDEO_OUT_PATH}${scale_ratio_list[$i]}" \
        --config="${CONFIG}" \
        --checkpoint_dir="${CKPT_DIR}" \
        --scale_ratio="${scale_ratio_list[$i]}" \
        > "eval_dynamic_logs/${EXP_NAME}_ratio${scale_ratio_list[$i]}.txt" 2>&1 &
done