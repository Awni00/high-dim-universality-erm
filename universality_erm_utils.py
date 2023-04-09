import numpy as np
import tensorflow as tf
import wandb
import os

def get_rfmodel(d, p, rf_activation):
    """generate a random-features model.

    Args:
        d (int): dimension of original covariates.
        p (int): number of features to generate.

    Returns:
        Tuple[function, nd.array]: phi, W
    """
    ## sample weights uniformly from S^{d-1}(1)
    W = np.random.normal(loc=0, scale=1, size=(d, p))
    W /= np.linalg.norm(W, ord=2, axis=0)

    ## create random features model
    def phi(z):
        return rf_activation(z @ W)

    return phi, W

def get_gaussian_equiv_feats(Z, W, rf_activation, n_samples=int(1e6)):
    """generate gaussian-equivalent features of random features model.

    Args:
        Z (nd.array[float]): original feature matrix. shape=(n, d)
        W (nd.array[float]): linear transformation matrix of random features model. shape=(d,p)
        rf_activation (function): activation function of random features model
        n_samples (int, optional): number of samples to use for estimating parameters of 
            gaussian equivalent model.Defaults to int(1e6).

    Returns:
        nd.array[float]: gaussian-equivalent features. shape=(n,p)
    """

    # NOTE:this assumes original covariates Z are iid standard normal vectors
    d, p = np.shape(W)

    zs = [np.random.normal() for _ in range(n_samples)] # standard gaussian rvs
    mu_0 =  np.mean(rf_activation(zs))
    mu_1 = np.mean(zs * rf_activation(zs))
    act_z_sq = np.mean(rf_activation(zs)**2)
    mu_2 = np.sqrt(act_z_sq - mu_0**2 -mu_1**2)

    additional_gaussian_noise = np.random.multivariate_normal(mean=[0]*p, cov=np.identity(p), size=len(Z))

    X_gauss = mu_0 * np.ones(shape=(len(Z), p)) + mu_1 * Z @ W + mu_2 * additional_gaussian_noise

    return X_gauss

def run_trial_rfmodel(
    create_model, p, rf_activation, data,
    wandb_project_name, trial, metrics, 
    n_epochs, create_callbacks, verbose=False
    ):

    wandb.init(project=wandb_project_name, group='random features model', 
        name=f'p={p}, trial={trial}', config={'p': p, 'trial': trial, 'group': 'random features model'})

    Z_train, y_train, Z_test, y_test = data
    d = Z_train.shape[1]

    rfmodel_phi, W = get_rfmodel(d, p, rf_activation)
    # save weights matrix from random features model to wandb
    data_dict = {'phi': W}
    np.save(os.path.join(wandb.run.dir, 'data_p={p}_trial={trial}.npy'), data_dict, allow_pickle=True)

    X_train = rfmodel_phi(Z_train)
    X_test = rfmodel_phi(Z_test)

    data = (X_train, y_train, X_test, y_test)

    trial_metrics, model = fit_eval_model(create_model, data, metrics, n_epochs, create_callbacks, verbose)

    trial_results = {
        **trial_metrics,
        'model': model,
        'phi_W': W,
        'rfmodel_phi': rfmodel_phi
        }
    

    wandb.finish(quiet=True)

    return trial_results


def run_trial_gaussian_equiv_model(
    create_model, p, rf_activation, data,
    wandb_project_name, trial, metrics, 
    n_epochs, create_callbacks, verbose=False
    ):

    wandb.init(project=wandb_project_name, group='gaussian equivalent model', 
        name=f'p={p}, trial={trial}', config={'p': p, 'trial': trial, 'group': 'gaussian equivalent model'})

    Z_train, y_train, Z_test, y_test = data
    d = Z_train.shape[1]

    rfmodel_phi, W = get_rfmodel(d, p, rf_activation)
    # save weights matrix from random features model to wandb
    data_dict = {'phi': W}
    np.save(os.path.join(wandb.run.dir, 'data_p={p}_trial={trial}.npy'), data_dict, allow_pickle=True)

    X_train = get_gaussian_equiv_feats(Z_train, W, rf_activation)
    X_test = get_gaussian_equiv_feats(Z_test, W, rf_activation)

    data = (X_train, y_train, X_test, y_test)

    trial_metrics, model = fit_eval_model(create_model, data, metrics, n_epochs, create_callbacks, verbose)

    trial_results = {
        **trial_metrics,
        'model': model,
        'phi_W': W,
        'rfmodel_phi': rfmodel_phi
        }
    
    wandb.finish(quiet=True)

    return trial_results

# TODO: make loss/optimizer an input (same as create_model)
def fit_eval_model(create_model, data, metrics, n_epochs, create_callbacks, verbose=False):

    X_train, y_train, X_test, y_test = data

    model = create_model()
    model(X_train);

    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)
    history = model.fit(X_train, y_train, epochs=n_epochs, verbose=verbose, callbacks=create_callbacks())

    # save model weights (\hat{theta}) to wandb
    model.save_weights(os.path.join(wandb.run.dir, "model_p={p}_trial={trial}.h5"))

    train_metrics = model.evaluate(X_train, y_train, verbose=False, return_dict=True)
    test_metrics = model.evaluate(X_test, y_test, verbose=False, return_dict=True)

    trial_metrics = {
        **{f'train_{metric_name}': metric for metric_name, metric in train_metrics.items()},
        **{f'test_{metric_name}': metric for metric_name, metric in test_metrics.items()}
        }

    wandb.log(trial_metrics) # log metrics from this trial

    return trial_metrics, model

# TODO: make loss/optimizer an input (same as create_model)
def get_metric_keys(create_model, Z_train, y_train, metrics):
    """gets metric keys that would be returned by fit_eval_model"""
    
    model = create_model()
    model(Z_train)
    loss = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam()
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    metric_keys = model.evaluate(Z_train, y_train, return_dict=True, verbose=False).keys()

    metric_keys = [
    *[f'train_{metric_name}' for metric_name in metric_keys], 
    *[f'test_{metric_name}' for metric_name in metric_keys]
    ]

    return metric_keys