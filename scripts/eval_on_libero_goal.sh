EXP_NAME="pi05_moh_test"
# SUITES=("libero_object" "libero_spatial" "libero_goal" "libero_10")
SUITES=("libero_goal")
# RANKS=(0 0 0 0)
RANKS=(0)
VIDEO_OUT_PATH="./videos/${EXP_NAME}_"
RESULTS_DIR="./results/${EXP_NAME}_"
CONFIG="pi05_libero"

# download from https://huggingface.co/Timsty/mixture_of_horizons/tree/main
# CKPT_DIR="pi05_moh_libero_3to30stride3"
# CKPT_DIR="/home/txs/Code/Policy_Eval_Done_Right/MixtureOfHorizons/checkpoints/pi05_moh_libero_3to30stride3"
CKPT_DIR="/share/data/ripl/vincenttann/Code/MixtureOfHorizons/checkpoints/pi05_moh_libero_3to30stride3"

for i in "${!SUITES[@]}"; do
    TORCHDYNAMO_DISABLE=1 python examples/libero_scripts/pi_moh.py \
        --task_suite_name="${SUITES[$i]}" \
        --rank="${RANKS[$i]}" \
        --results_dir="${RESULTS_DIR}" \
        --config="${CONFIG}" \
        --checkpoint_dir="${CKPT_DIR}" \
        --horizons 3 6 9 12 15 18 21 24 27 30
done
