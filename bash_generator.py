#!/usr/bin/python
# -*- coding: utf-8 -*-
PATH_TO_INTERPRETER = "/home/rportela/anaconda3/envs/tfGPU/bin/python"  # plafrim
logdir_name = "sacsu_hc"
filename = "run_" + logdir_name + ".sh"
save_dir = '.'
trial_id = list(range(0, 8))

with open(filename, 'w') as f:
    f.write('#!/bin/sh\n')
    f.write('#SBATCH --mincpus 20\n')
    f.write('#SBATCH -p court_sirocco \n')
    f.write('#SBATCH -t 4:00:00\n')
    f.write('#SBATCH -e ' + filename[1:] + '.err\n')
    f.write('#SBATCH -o ' + filename[1:] + '.out\n')
    f.write('rm log.txt; \n')
    f.write("export EXP_INTERP='%s' ;\n" % PATH_TO_INTERPRETER)

    gpu_id = 0
    for seed in range(len(trial_id)):
        t_id = trial_id[seed]
        f.write("$EXP_INTERP spinup/algos/sac/sac.py --exp_name %s --seed %s --epochs 10 --gpu_id %s&\n" % (logdir_name, t_id, gpu_id))
        gpu_id += 1
        if gpu_id == 4:
            gpu_id = 0
    f.write('wait')
