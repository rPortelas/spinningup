#!/bin/sh
#SBATCH --mincpus 24
#SBATCH -p hack
#SBATCH -t 24:00:00
#SBATCH -e un_ortest.sh.err
#SBATCH -o un_ortest.sh.out

export EXP_INTERP='/home/rportela/anaconda3/envs/tfGPU/bin/python' ;
nb_gpus=4
exp_name="bashest2"

# Create expe directory
mkdir data/$exp_name


# Copy itself in experimental dir
cp $SLURM_JOB_NAME data/$exp_name/$exp_name.sh

# Add info about git status to exp dir
current_commit="$(git log -n 1)"
echo "${current_commit}" > data/$exp_name/git_status.txt
# Same for gym_flowers repo
cd ../gym_flowers/
current_commit="$(git log -n 1)"
cd ../spinningup/
echo "${current_commit}" >> data/$exp_name/git_status.txt

# Launch !
for i in {0..11}
do
   $EXP_INTERP spinup/algos/sac/sac.py --env flowers-Walker-v2 --exp_name $exp_name \
    --seed $(($i+0)) --epochs 500 --gpu_id $(($i%$nb_gpus)) --max_ep_len 2000 --env_babbling oracle --max_stump_h 2.0 \
    --ent_coef 0.0005 --env_param_input 0 --batch_size 1000 --train_freq 10 --lr 0.001 --buf_size 2000000 &
done
wait
