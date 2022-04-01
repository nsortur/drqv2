jn="rot8_equi_2"

start_dir=${jn}
script=run_drqv2.sbatch

jid[1]=$(./run_help.sh ${jn} ${start_dir} ${script} | tr -dc '0-9')
echo ${jid[1]}

for j in {2..3}
do
  echo ${jid[$((j-1))]}
  jid[${j}]=$(./run_help2.sh ${jn}_${j} ${jid[$((j-1))]} ${start_dir} ${script} | tr -dc '0-9')
done
