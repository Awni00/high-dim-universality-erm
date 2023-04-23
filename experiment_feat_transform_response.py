# EXPERIMENT DESCRIPTION
# implements experiments where response function is linear function of feature transformation
# in original setting of Hu & Lu '22, the response function may take the form
# y = psi(<beta, z>) where z are the covariates and psi is some univariate function.
# here, we examine the setting where the the response function takes the form 
# y = <beta, psi(z)> for some univariate function psi applied elementwise to the covariates

import numpy as np
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm.auto import tqdm, trange

from universality_erm_utils import run_trial_gaussian_equiv_model, run_trial_rfmodel, get_metric_keys

import argparse
import datetime
import os

# region argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, default=5000)
parser.add_argument('--n_train', type=int, default=200)
parser.add_argument('--d', type=int, default=100)
parser.add_argument('--nu', type=float, default=0.1)
parser.add_argument('--lamda', type=float, default=0.0002)
parser.add_argument('--n_trials', type=int, default=10)
parser.add_argument('--n_epochs', type=int, default=1000)
parser.add_argument('--early_stopping', type=bool, default=False)
parser.add_argument('--gamma_start', type=float, default=0.05)
parser.add_argument('--gamma_end', type=float, default=1.4)
parser.add_argument('--n_ps', type=int, default=14, help='number of values of p to test (+/-)')
parser.add_argument('--psi', type=str, default='identity')
parser.add_argument('--rf_activation', type=str, default='tanh', choices=['tanh', 'relu', 'identity'])
parser.add_argument('--model_activation', type=str, default='tanh')
parser.add_argument('--wandb_project_name', type=str, default='high-dim-universality-erm')
parser.add_argument('--out_dir', type=str, default='results')
parser.add_argument('--out_dir_addtime', type=bool, default=True)
parser.add_argument('--mode', type=str, default='both', choices=['random_feat', 'gaussian_equiv', 'both'])

args = parser.parse_args()
print(f'\n\n args: {args} \n\n')

n = args.n
n_train = args.n_train
d = args.d
nu = args.nu
lamda = args.lamda
n_trials = args.n_trials
n_epochs = args.n_epochs
gamma_start = args.gamma_start
gamma_end = args.gamma_end
n_ps = args.n_ps
out_dir = args.out_dir
if args.out_dir_addtime:
    datetimestr = datetime.datetime.now().strftime("%Y-%m-%d-%H%M")
    out_dir = f'{out_dir}_{datetimestr}'

os.mkdir(f'results/{out_dir}')

# endregion

# region model set up
def create_callbacks(monitor='loss'):
    callbacks = [
        wandb.keras.WandbMetricsLogger(log_freq='epoch'),
        ]

    if args.early_stopping:
        callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10, mode='auto', restore_best_weights=False))

    return callbacks

metrics = ['mean_squared_error']

# endregion

# region Set Up Logging to W&B
import logging
logger = logging.getLogger("wandb")
logger.setLevel(logging.ERROR)

os.environ["WANDB_SILENT"] = "true"

import wandb
wandb.login()

wandb_project_name = args.wandb_project_name

# endregion

# region data set up

# ground truth function to be predicted: y = psi(<z, beta_star> + epsilon)
if args.psi == 'clip':
    psi = lambda t: np.clip(t, -1, 1)
elif args.psi == 'identity':
    psi = lambda t: t
elif args.psi == 'abs':
    psi = np.abs
elif args.psi == 'sigmoid':
    psi = lambda t: 1 / (1+ np.exp(-t))
elif args.psi == 'relu':
    psi = lambda t: np.maximum(0, t)
elif args.psi == 'square':
    psi = lambda t: t**2
elif args.psi == 'cube':
    psi = lambda t: t**3
elif args.psi == 'cuberoot':
    psi = lambda t: t**(1/3)
elif args.psi == 'exp':
    psi = np.exp
else:
    raise ValueError(f"psi argument `{args.psi}` is invalid")

# generate data

## generate covariates
# NOTE: for now, covariates are iid uncorrelated standard gaussian vectors
# but theory supports different mean / cov
z_mean = np.array([0]*d)
z_cov = np.identity(d)
Z = np.random.multivariate_normal(mean=z_mean, cov=z_cov, size=n)

## generate unknown model parameter
beta_star = np.random.normal(loc=0, scale=1, size=(d,1))
beta_star /= np.linalg.norm(beta_star, ord=2)

## generate response
epsilon = np.random.normal(loc=0, scale=nu, size=(n,1))
y = psi(Z) @ beta_star + epsilon


# print data schema
print(f'n = {n}, n_train = {n_train}, d = {d}, nu = {nu}, lambda = {lamda}')
print(f'Z.shape = {Z.shape}')
print(f'beta_star.shape = {beta_star.shape}')
print(f'y.shape = {y.shape}')

# get train-test split
Z_train = Z[:n_train]
y_train = y[:n_train]
Z_test = Z[n_train:]
y_test = y[n_train:]

data = Z_train, y_train, Z_test, y_test

# endregion

# region model setup

# create random features model
if args.rf_activation == 'tanh':
    rf_activation = np.tanh
elif args.rf_activation == 'relu':
    rf_activation = lambda x: np.maximum(0, x)
elif args.rf_activation == 'identity':
    rf_activation = lambda x: x
else:
    raise ValueError("`rf_activation` is invalid")


model_activation = args.model_activation

def create_model():
    out_layer = tf.keras.layers.Dense(
            1, activation=model_activation, use_bias=False,
            kernel_regularizer=tf.keras.regularizers.L2(l2=lamda))
    model = tf.keras.Sequential([out_layer], name='rand_feat_model')
    return model

# endregion

# region run experiment

# create sequence of p to test in terms of values of gamma
ps = np.arange(int(n_train*gamma_start), int(gamma_end*n_train), step=int(n_train*(gamma_end - gamma_start) / n_ps))

print(f'evaluating {len(ps)} values of p covering gamma = {gamma_start} to gamma = {gamma_end}.')
print(f'runnng {n_trials} trials for each value of p. Total # of trials = {n_trials * len(ps)}.\n')
print('evaluating a random features model and a gaussian-equivalent model')

metric_keys = get_metric_keys(create_model, Z_train, y_train, metrics)

if args.mode in ['random_feat', 'both']:
    print('\n' + '='*60 + '\n')
    print("STARTING RANDOM FEATURES MODEL TRIALS")

    results_dict = {key: np.zeros(shape=(len(ps), n_trials)) for key in metric_keys}
    for ip, p in enumerate(tqdm(ps, desc='p', position=0)):
        for trial in trange(n_trials, leave=False, desc='trial', position=1):
            trial_results = run_trial_rfmodel(
                create_model, p, rf_activation, data, 
                wandb_project_name, trial, metrics, 
                n_epochs, create_callbacks, verbose=False)

            for key in metric_keys:
                results_dict[key][ip, trial] = trial_results[key]

    # save results
    results_dict['ps'] = ps
    results_dict['args'] = args

    results_dict = {
        **results_dict,
        'Z_train': Z_train,
        'Z_test': Z_test,
        'y_train': y_train,
        'y_test': y_test
        }

    np.save(f'results/{out_dir}/random_feats', results_dict, allow_pickle=True)

    print('COMPLETED RANDOM FEATURES MODEL TRIALS')
    print(f'saved results to `results/{out_dir}/random_feats`')

if args.mode in ['gaussian_equiv', 'both']:
    print('\n' + '='*60 + '\n')
    print("STARTING GAUSSIAN EQUIVALENT MODEL TRIALS")

    results_dict = {key: np.zeros(shape=(len(ps), n_trials)) for key in metric_keys}
    for ip, p in enumerate(tqdm(ps, desc='p', position=0)):
        for trial in trange(n_trials, leave=False, desc='trial', position=1):
            trial_results = run_trial_gaussian_equiv_model(
                create_model, p, rf_activation, data,
                wandb_project_name, trial, metrics, 
                n_epochs, create_callbacks, verbose=False)

            for key in metric_keys:
                results_dict[key][ip, trial] = trial_results[key]

    # save results
    results_dict['ps'] = ps
    results_dict['args'] = args

    results_dict = {
        **results_dict,
        'Z_train': Z_train,
        'Z_test': Z_test,
        'y_train': y_train,
        'y_test': y_test
        }

    np.save(f'results/{out_dir}/gaussian_equiv', results_dict, allow_pickle=True)
    print('COMPLETED GAUSSIAN EQUIVALENT MODEL TRIALS')
    print(f'saved results to `results/{out_dir}/gaussian_equiv`')

# endregion