#!/bin/bash

#SBATCH --output=outputs/%j.out
#SBATCH --error=outputs/%j.err

#SBATCH --nodes=1

#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=60GB
#SBATCH --time=8:00:00

#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sortur.n@northeastern.edu

#SBATCH --exclude=c[2160,2162-2175,2176-2183],d1008,d1026
##SBATCH --gres=gpu:v100-sxm2

# export PYTHONPATH="/scratch/wang.dian/python_packages/helping_hands_rl_envs"

echo "node: $SLURM_NODELIST"
env="task=reacher_easy"

# probably don't need to do that (parallel m)
# load_sub is passed from the other file

m_run=1
for (( i=1; i<=$m_run; i++ ))
do
    log=${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${i}
    if [ "$LOAD_SUB" = "None" ]; then
    	load="None"
    else

    	load=${LOAD_SUB}_${i}
      log=load
    fi
    seed=$(($SEED+($i)))
    python train.py ${env} ${log} $@ &
    PIDS+=($!)
done

# wait for all processes to finish, and store each process's exit code into array STATUS[].
for pid in ${PIDS[@]}; do
  echo "pid=${pid}"
  wait ${pid}
  STATUS+=($?)
done

# after all processed finish, check their exit codes in STATUS[].
i=0
for st in ${STATUS[@]}; do
  if [[ ${st} -ne 0 ]]; then
    echo "$i failed"
    exit 1
  else
    echo "$i finish"
  fi
  ((i+=1))
done
