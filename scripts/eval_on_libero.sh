EXP_NAME="pi05_moh_test"
SUITES=("libero_object" "libero_spatial" "libero_goal" "libero_10")
RANKS=(0 1 2 3)
VIDEO_OUT_PATH="./videos/${EXP_NAME}_"
CONFIG="pi05_libero"

# download from https://huggingface.co/Timsty/mixture_of_horizons/tree/main
CKPT_DIR="pi05_moh_libero_3to30stride3"

for i in "${!SUITES[@]}"; do
    TORCHDYNAMO_DISABLE=1 python examples/libero_scripts/pi_moh.py \
        --task_suite_name="${SUITES[$i]}" \
        --rank="${RANKS[$i]}" \
        --video_out_path="${VIDEO_OUT_PATH}" \
        --config="${CONFIG}" \
        --checkpoint_dir="${CKPT_DIR}" \
        --horizons 3 6 9 12 15 18 21 24 27 30 \
        > "logs/${EXP_NAME}_${SUITES[$i]}.txt" 2>&1 &
done