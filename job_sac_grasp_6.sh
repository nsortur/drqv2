run()
{
# jn=${env_abbr}_${alg}_${model}_planner${planner}_buffer${buffer}_N${N}_${view_type}_gamma${gamma}_${date}
jn=testing
# args="
# --robot=panda
# --num_process=1
# --num_eval_processes=1
# --perlin=0
# --env=${env}
# --max_episode_steps=50
# --num_objects=${num_objects}
# --alg=${alg}
# --batch_size=32
# --model=${model}
# --gamma=${gamma}
# --buffer=${buffer}
# --per_expert_eps=1
# --max_train_step=20000
# --planner_episode=${planner}
# --fixed_eps
# --random_orientation=t
# --action_sequence=pxyzr
# --dpos=0.03
# --drot_n=8
# --view_type=${view_type}
# --heightmap_size=128
# --crop_size=128
# --aug=f
# --buffer_aug_type=${buffer_aug_type}
# --aug_type=so2
# --buffer_aug_n=0
# --expert_aug_n=0
# --eval_freq=-1
# --equi_n=${N}
# --training_iter=1
# --fix_set=f
# "
slurm_args=""

jn1=${jn}_1
jid[1]=$(sbatch --test-only ${slurm_args} --job-name=${jn1} --export=SEED=${seed},LOAD_SUB='None' ${script} | tr -dc '0-9')
for j in {2..2}
do
jid[${j}]=$(sbatch --test-only ${slurm_args} --job-name=${jn}_${j} --dependency=afterok:${jid[$((j-1))]} --export=SEED=${seed},LOAD_SUB=${jn}_$((j-1))_${jid[$((j-1))]} ${script} ${args} | tr -dc '0-9')
done
}

runall_2()
{
script=run_2.sbatch
seed=0
run
# seed=2
# run
}

runall_1()
{
script=run_1.sbatch
seed=0
run
seed=1
run
seed=2
run
seed=3
run
}

date=0314

N=4

planner=100
env=close_loop_clutter_picking
num_objects=5
env_abbr=grasp_5_p100
gamma=0.9
buffer=normal
buffer_aug_type=dqn_c4

alg=sdqfd_fac

view_type=camera_fix_rgbd

model=equi_d_w_enc
runall_2

# model=equi_d_w_fcn
# runall_2
# 
# model=equi_d
# runall_2
# 
# model=cnn
# runall_2
# 
# view_type=camera_fix
# 
# model=equi_d_w_enc
# runall_2
# 
# model=equi_d_w_fcn
# runall_2
# 
# model=equi_d
# runall_2
# 
# model=cnn
# runall_2
