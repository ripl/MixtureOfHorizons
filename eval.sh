#!/bin/bash

SET_ID=4

export LIBERO_CONFIG_PATH="/share/data/ripl/vincenttann/Code/Policy_Eval_Done_Right/LIBERO/libero_configs/set_${SET_ID}"
echo "LIBERO_CONFIG_PATH: ${LIBERO_CONFIG_PATH}"

bash scripts/eval_on_libero_goal.sh

exit