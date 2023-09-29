#!/bin/bash

# Define an array of env_name values
env_names=("Ant-v4" "Hopper-v4" "HalfCheetah-v4" "InvertedPendulum-v4" "Walker2d-v4")

# Loop through the env_names and execute the commands
for env_name in "${env_names[@]}"; do
  if [ "$env_name" == "InvertedPendulum-v4" ]; then
    ep_len=1000
  else
    ep_len=200
  fi

  python cs285/scripts/run_hw3_actor_critic.py --env_name "$env_name" --ep_len "$ep_len" --discount 0.95 --scalar_log_freq 1 -n 200 -l 3 -s 128 -b 30000 -eb 1500 -lr 0.001 --exp_name "ac_$env_name" --save_params
done
