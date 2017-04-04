"""
This is just a sloppy script for plotting the results of my experiments
"""


import os
import numpy
np = numpy
from pylab import *

hhs = {}

results_path = './results/'

for path in os.listdir(results_path):
    if path.startswith('slurm_script'):
        try:
            hhs[path] = np.load(results_path + path + '/hh.npy')[()]
        except:
            print path, "FAILED!"


# GRID WORLD MDPS
envs = []
envs.append(['grid', 'size=3', 'gamma=0.5'])
envs.append(['grid', 'size=5', 'gamma=0.5'])
envs.append(['grid', 'size=3', 'gamma=0.9'])
envs.append(['grid', 'size=5', 'gamma=0.9'])
envs_strs = []
envs_strs.append(['9 states, gamma=0.5'])
envs_strs.append(['25 states, gamma=0.5'])
envs_strs.append(['9 states, gamma=0.9'])
envs_strs.append(['25 states, gamma=0.9'])

data_paths = os.listdir('./HW2_data')
def lookup_greedy(env):
    data_path = [path for path in data_paths if all([el in path for el in env])][0]
    data_= np.load('./HW2_data/' + data_path)[()]
    return data_['greedy_acc']



figure()
suptitle('gridworld results; MSE of Q-function (left), and policy accuracy (right)')
for nn, (env, env_str) in enumerate(zip(envs, envs_strs)):
    print nn
    for path in hhs:
        # get env
        if all([el in path for el in env]):
            for layers in [1,2,4]:
                if "layers=" + str(layers) in path:
                    print path, layers
                    #
                    subplot(4,2,1 + 2*nn)
                    ylabel(env_str[0])
                    plot(hhs[path]['loss'], label='train layers=' + str(layers))
                    plot(hhs[path]['val_loss'], label='valid layers=' + str(layers))
                    if 'gamma=0.5' in path:
                        ylim(0, 1)
                    else:
                        ylim(0,10)
                    #
                    subplot(4,2,2 + 2*nn)
                    #ylabel('policy accuracy')
                    plot(hhs[path]['<lambda>'], label='train layers=' + str(layers))
                    plot(hhs[path]['val_<lambda>'], label='valid layers=' + str(layers))
                    # GREEDY BASELINE
                    if layers == 4:
                        plot(500 * [lookup_greedy(env),], label='greedy')
    legend()

    figManager = get_current_fig_manager()
    figManager.window.showMaximized()
#savefig('HW2_gridworld.pdf')




# RANDOM MDPS
envs = []
envs.append(['mdp', 'size=3', 'gamma=0.5'])
envs.append(['mdp', 'size=5', 'gamma=0.5'])
envs.append(['mdp', 'size=3', 'gamma=0.9'])
envs.append(['mdp', 'size=5', 'gamma=0.9'])

data_paths = os.listdir('./HW2_data')
def lookup_greedy(env):
    data_path = [path for path in data_paths if all([el in path for el in env])][0]
    data_= np.load('./HW2_data/' + data_path)[()]
    return data_['greedy_acc']


# TODO

figure()
suptitle('random MDP results; MSE of Q-function (left), and policy accuracy (right)')
for nn, (env, env_str) in enumerate(zip(envs, envs_strs)):
    print nn
    for path in hhs:
        # get env
        if all([el in path for el in env]):
            for layers in [1,2,4]:
                if "layers=" + str(layers) in path:
                    print path, layers
                    #
                    subplot(4,2,1 + 2*nn)
                    ylabel(env_str[0])
                    plot(hhs[path]['loss'], label='train layers=' + str(layers))
                    plot(hhs[path]['val_loss'], label='valid layers=' + str(layers))
                    if 'gamma=0.5' in path:
                        ylim(0, 1)
                    else:
                        ylim(0,10)
                    #
                    subplot(4,2,2 + 2*nn)
                    #ylabel('policy accuracy')
                    plot(hhs[path]['<lambda>'], label='train layers=' + str(layers))
                    plot(hhs[path]['val_<lambda>'], label='valid layers=' + str(layers))
                    # GREEDY BASELINE
                    if layers == 4:
                        plot(500 * [lookup_greedy(env),], label='greedy')
    legend()

    figManager = get_current_fig_manager()
    figManager.window.showMaximized()
#savefig('HW2_random.pdf')





