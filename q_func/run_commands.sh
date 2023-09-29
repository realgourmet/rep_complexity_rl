#!/bin/bash

# Define an array of rollout_path values
# rollout_paths=("sac_ant" "sac_hopper" "sac_walker" "sac_pendulum" "sac_cheetah")

# rollout_paths=("sac_hopper" "sac_walker" "sac_pendulum" "sac_cheetah")

rollout_paths=("ac_Ant-v4" "ac_Hopper-v4" "ac_HalfCheetah-v4" "ac_InvertedPendulum-v4" "ac_Walker2d-v4")


# Define an array of task values
tasks=("reward" "model" "value")

# Number of times to run each experiment
num_runs=3

# Loop through the rollout_paths and tasks and execute the commands 5 times
for path in "${rollout_paths[@]}"; do
  for task in "${tasks[@]}"; do
    for ((i=1; i<=num_runs; i++)); do
      python cs285/scripts/approx_ac.py --rollout_path "$path" --task "$task"
    done
  done
done
