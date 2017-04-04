"""
This is the script I use to launch the experiment jobs on SLURM

"""
import os
import itertools
import numpy as np
import subprocess
import argparse
parser = argparse.ArgumentParser()
# TODO: analyze mode
parser.add_argument('--backend', type=str, default='print', choices=['analyze', 'print', 'run', 'test'], help="analyze completed jobs, print job strings, run jobs, or run one job")
# TODO: all of these args
parser.add_argument('--exp_script', type=str, default='print', choices=['analyze', 'print', 'run', 'test'], help="analyze completed jobs, print job strings, run jobs, or run one job")
parser.add_argument('--gpu_mem', type=str, default='print', choices=['analyze', 'print', 'run', 'test'], help="analyze completed jobs, print job strings, run jobs, or run one job")
parser.add_argument('--mode', type=str, default='run', choices=['analyze', 'print', 'run', 'test'], help="analyze completed jobs, print job strings, run jobs, or run one job")
parser.add_argument('--run_time', type=int, default=24, help="expected run time, in hours")
locals().update(parser.parse_args().__dict__)

#--------------------------------------------------
# TODO: move this elsewhere?
def grid_search(args_vals):
    """ arg_vals: a list of lists, each one of format (argument, list of possible values) """
    lists = []
    for arg_vals in args_vals:
        arg, vals = arg_vals
        ll = []
        for val in vals:
            ll.append(" --" + arg + "=" + str(val))
        lists.append(ll)
    return ["".join(item) for item in itertools.product(*lists)]


#--------------------------------------------------
# set-up dispatch commands
# TODO: tensorflow...
# TODO: use only one dispatcher

job_prefix = ""

# Check which cluster we're using
if subprocess.check_output("hostname").startswith("hades"):
    #launch_str = "smart-dispatch --walltime=48:00:00 --queue=@hades launch THEANO_FLAGS=device=gpu,floatX=float32"
    job_prefix += "smart-dispatch --walltime=" + str(run_time) + ":00:00 --queue=@hades launch THEANO_FLAGS=device=gpu,floatX=float32"
    job_prefix += " python"
elif subprocess.check_output("hostname").startswith("helios"):
    job_prefix += "jobdispatch --gpu --queue=gpu_1 --duree=" + str(run_time) + ":00H --env=THEANO_FLAGS=device=gpu,floatX=float32 --project=jvb-000-ag "
    job_prefix += " python"
else: # TODO: SLURM
    nhours = run_time
    print "running at MILA, assuming job takes about", nhours, "hours"
    #job_prefix += 'sbatch --gres=gpu -C"gpu6gb|gpu12gb" --mem=4000 -t 0-' + str(nhours)
    job_prefix += 'sbatch --gres=gpu  --mem=4000 -t 0-' + str(nhours)


#--------------------------------------------------
# specify experiments (grid-search, etc.)
exp_script = ' $HOME/projects/doina_class/HW2_DNNs.py'
job_prefix += exp_script


grid = [] 

grid2 = [] 
#grid2 += [['dropout_p', [0.0, 0.5]]]
grid2 += [['environment', ['rand_mdp', 'rand_grid']]]
grid2 += [["num_layers", [1,2,4]]]
grid2 += [['size', [3, 5]]]#, 10]]]
grid2 += [["gamma", [0.9, 0.5]]]


launcher_name = os.path.basename(__file__)
# TODO: log the launch?
#https://stackoverflow.com/questions/12842997/how-to-copy-a-file-using-python
#print os.path.abspath(__file__)
#import shutil

#--------------------------------------------------
# construct job_strs
job_strs = []
for settings in grid_search(grid) + grid_search(grid2):
    job_str = job_prefix + settings
    job_str += ' --num_epochs=500'
    job_str += ' --num_train=10000'
    job_str += " --save=1"
    job_str += " --training=1"
    print job_str
    job_strs.append(job_str)




#--------------------------------------------------
# call os.system() 

if mode == 'analyze':
    # TODO
    for job_str in job_strs:
        pass


if mode == 'run':
    for job_str in job_strs:
        os.system(job_str)
elif mode == 'test':
    for job_str in job_strs[-1:]:
        os.system(job_str)



