sbatch --job-name=$1 --dependency=afternotok:$2 --export=ALL,START_DIRNAME=$3 $4
