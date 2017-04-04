#!/usr/bin/env python

"""
This is the script which actually runs experiments.
I use keras, and some extensions which are in my publicly available dk_keras repo.

"""


import numpy
np  =  numpy
import os
import sys

# keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop, Adam
from keras.utils import np_utils

# my code
from dk_keras.metrics import get_optimal_policy_recovered
from dk_keras.utils import plot_history

# my code
from doina_class.environments import grid_world, MDP

import argparse
parser = argparse.ArgumentParser()
# dataset hyperparameters
parser.add_argument('--environment', type=str, default='rand_grid', choices=['rand_mdp', 'rand_grid'])
parser.add_argument('--gamma', type=float, default=.9) #
parser.add_argument('--num_train', type=int, default=10000) #
parser.add_argument('--size', type=int, default=5)
# neural net hyperparameters
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--dropout_p', type=float, default=0.0)
parser.add_argument('--lr', type=float, default=0.002)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--num_layers', type=int, default=5)
parser.add_argument('--num_units', type=int, default=100)
# script configurations
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--save_dir', type=str, default="./")
parser.add_argument('--seed', type=int, default=1337)
parser.add_argument('--training', type=int, default=1)


args = parser.parse_args()
args_dict = args.__dict__
flags = [flag.lstrip('--') for flag in sys.argv[1:]]
flags = [ff for ff in flags if not ff.startswith('save_dir')]


# SET-UP SAVING (directories and paths)
save_dir = args_dict.pop('save_dir')
save_path = os.path.join(save_dir, os.path.basename(__file__) + '___' + '_'.join(flags))
args_dict['save_path'] = save_path
if args_dict['save']:
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open (os.path.join(save_path,'exp_settings.txt'), 'w') as f:
        for key in sorted(args_dict):
            f.write(key+'\t'+str(args_dict[key])+'\n')
    print( save_path)
locals().update(args_dict)


# PSEUDO-RANDOMNESS (optional)
if seed is not None:
    np.random.seed(seed)
    rng = numpy.random.RandomState(seed)
else:
    rng = numpy.random.RandomState(np.random.randint(2**32 - 1))



# ----------------------------------------------
# The functions I use to generate random MDPs
nS = size**2
nA = 4
def randP():
    return rng.dirichlet(1./nS * np.ones(nS), nS * nA).reshape((nS,nA,nS))

def randR():
    return rng.normal(0,1,(nS,nA))

def rand_mdp():
    return MDP(randP(), randR(), gamma=gamma)

# fixed dynamics
grid_P = grid_world(size).P[:,:,:-1] # remove terminal state

def rand_grid():
    return MDP(grid_P, randR(), gamma=gamma)

rand_grid()
rand_mdp()
assert False

# ----------------------------------------------
# GET DATA
data_path = os.path.join(save_dir, 'HW2_data', 'env=' + str(environment) + '_size=' + str(size) + '_num_train=' + str(num_train) + '_gamma=' + str(gamma))
try: # load data
    data = np.load(data_path)[()]
    locals().update(data)
except: # create and save data
    num_examples = int(num_train * 1.2)

    Y = []

    if environment == 'rand_mdp':
        envs = [rand_mdp() for nn in range(num_examples)]
    elif environment == 'rand_grid':
        envs = [rand_grid() for nn in range(num_examples)]


    print "solving for Q..."
    for nn, _ in enumerate(envs):
        print nn
        Y.append(envs[nn].get_Q())

    if environment == 'rand_mdp':
        X = [np.hstack( [env.P.flatten(), env.R.flatten()] ) for env in envs]
    elif environment == 'rand_grid':
        X = [env.R.flatten() for env in envs]

    pi_star = [yy.argmax(1) for yy in Y]
    pi_greedy = [env.R.argmax(1) for env in envs]
    greedy_acc = np.mean([np.all(Y[nn].argmax(1) == envs[nn].R.argmax(1)) for nn in range(len(envs))])

    X = numpy.array(X)
    Y = numpy.array(Y)
    X = X.reshape((num_examples, -1))
    Y = Y.reshape((num_examples, -1))
    # train/valid/test split
    trX, trY = X[:num_train], Y[:num_train]
    teX, teY = X[num_train:], Y[num_train:]
    vaX, vaY = teX[:len(teX)/2], teY[:len(teX)/2]
    teX, teY = teX[len(teX)/2:], teY[len(teX)/2:]
    # save dataset
    data = {}
    data['trX'] = trX
    data['vaX'] = vaX
    data['teX'] = teX
    data['trY'] = trY
    data['vaY'] = vaY
    data['teY'] = teY
    # greedy solutions
    data['pi_star'] = pi_star
    data['pi_greedy'] = pi_greedy
    data['greedy_acc'] = greedy_acc
    np.save(data_path, data)

print "done getting dataset"


if not training: # we just want to create the dataset
    assert False

# ----------------------------------------------
# TRAIN MODEL

model = Sequential()
if num_layers > 0:
    model.add(Dense(num_units, input_shape=trX[0].shape))
    model.add(Activation('relu'))
    model.add(Dropout(dropout_p))
    for layer in range(num_layers-1):
        model.add(Dense(num_units))
        model.add(Activation('relu'))
        model.add(Dropout(dropout_p))
    model.add(Dense(nS * nA))
else:
    model.add(Dense(ns * nA))
    

model.summary()

model.compile(loss='mse',
              optimizer=Adam(), 
              metrics=[get_optimal_policy_recovered(nS)])

history = model.fit(trX, trY,
                    batch_size=batch_size, 
                    nb_epoch=num_epochs,
                    verbose=1, 
                    validation_data=(vaX, vaY))

if save:
    model.save(os.path.join(save_path, 'model.h5'))
    hh = history.history
    np.save(os.path.join(save_path, 'hh.npy'), hh)
    plt_save_path = os.path.join(save_path, 'lcs.pdf')
else:
    plt_save_path = None








